import os
import random
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# =========================================================
# CONFIG
# =========================================================
CSV_PATH = "fft_features/fft_features_pid_clean.csv"
OUTPUT_DIR = "vae_multi_latent_outputs_2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 80
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
TEST_SIZE = 0.2

# latent sizes to compare
LATENT_DIMS = [2, 4, 8, 16, 32]

# beta for beta-VAE style weighting
BETA = 3.0


# =========================================================
# REPRODUCIBILITY
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)


# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(CSV_PATH, dtype={"PID": str})
print("Loaded:", CSV_PATH)
print("Shape:", df.shape)

if "PID" not in df.columns:
    raise ValueError("Expected a PID column in the input CSV.")

feature_cols = [c for c in df.columns if c != "PID"]
if not feature_cols:
    raise ValueError("No feature columns found.")

X = df[feature_cols].copy()

# fill missing values
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed).astype(np.float32)

# split once for fair comparison across all latent dims
X_train, X_val = train_test_split(
    X_scaled,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    shuffle=True
)

train_ds = TensorDataset(torch.tensor(X_train))
val_ds = TensorDataset(torch.tensor(X_val))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_scaled.shape[1]

print("Device:", device)
print("Input dim:", input_dim)


# =========================================================
# VAE MODEL
# =========================================================
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        # encoder
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # decoder
        self.fc3 = nn.Linear(latent_dim, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, input_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        h = self.relu(self.fc3(z))
        h = self.relu(self.fc4(h))
        x_hat = self.fc5(h)
        return x_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z


# =========================================================
# LOSS FUNCTION
# =========================================================
def vae_loss(x, x_hat, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(x_hat, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


# =========================================================
# TRAIN FUNCTION
# =========================================================
def train_one_vae(latent_dim, train_loader, val_loader, input_dim, device):
    model = VAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    train_total_losses = []
    train_recon_losses = []
    train_kl_losses = []

    val_total_losses = []
    val_recon_losses = []
    val_kl_losses = []

    best_val_total = np.inf
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_total_sum = 0.0
        train_recon_sum = 0.0
        train_kl_sum = 0.0

        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)

            optimizer.zero_grad()
            batch_recon, mu, logvar, z = model(batch_x)
            total_loss, recon_loss, kl_loss = vae_loss(batch_x, batch_recon, mu, logvar, beta=BETA)

            total_loss.backward()
            optimizer.step()

            bs = batch_x.size(0)
            train_total_sum += total_loss.item() * bs
            train_recon_sum += recon_loss.item() * bs
            train_kl_sum += kl_loss.item() * bs

        epoch_train_total = train_total_sum / len(train_loader.dataset)
        epoch_train_recon = train_recon_sum / len(train_loader.dataset)
        epoch_train_kl = train_kl_sum / len(train_loader.dataset)

        model.eval()

        val_total_sum = 0.0
        val_recon_sum = 0.0
        val_kl_sum = 0.0

        with torch.no_grad():
            for (batch_x,) in val_loader:
                batch_x = batch_x.to(device)

                batch_recon, mu, logvar, z = model(batch_x)
                total_loss, recon_loss, kl_loss = vae_loss(batch_x, batch_recon, mu, logvar, beta=BETA)

                bs = batch_x.size(0)
                val_total_sum += total_loss.item() * bs
                val_recon_sum += recon_loss.item() * bs
                val_kl_sum += kl_loss.item() * bs

        epoch_val_total = val_total_sum / len(val_loader.dataset)
        epoch_val_recon = val_recon_sum / len(val_loader.dataset)
        epoch_val_kl = val_kl_sum / len(val_loader.dataset)

        train_total_losses.append(epoch_train_total)
        train_recon_losses.append(epoch_train_recon)
        train_kl_losses.append(epoch_train_kl)

        val_total_losses.append(epoch_val_total)
        val_recon_losses.append(epoch_val_recon)
        val_kl_losses.append(epoch_val_kl)

        if epoch_val_total < best_val_total:
            best_val_total = epoch_val_total
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"[latent={latent_dim}] Epoch {epoch:3d} | "
                f"train_total={epoch_train_total:.6f} | val_total={epoch_val_total:.6f} | "
                f"train_recon={epoch_train_recon:.6f} | val_recon={epoch_val_recon:.6f} | "
                f"train_kl={epoch_train_kl:.6f} | val_kl={epoch_val_kl:.6f}"
            )

    model.load_state_dict(best_state)

    return {
        "model": model,
        "best_val_total": best_val_total,
        "train_total_losses": train_total_losses,
        "train_recon_losses": train_recon_losses,
        "train_kl_losses": train_kl_losses,
        "val_total_losses": val_total_losses,
        "val_recon_losses": val_recon_losses,
        "val_kl_losses": val_kl_losses,
    }


# =========================================================
# RUN ALL LATENT SIZES
# =========================================================
results = []

for latent_dim in LATENT_DIMS:
    print("\n" + "=" * 70)
    print(f"Training VAE with latent_dim = {latent_dim}")
    print("=" * 70)

    out = train_one_vae(
        latent_dim=latent_dim,
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=input_dim,
        device=device
    )

    model = out["model"]

    model_path = os.path.join(OUTPUT_DIR, f"best_vae_latent_{latent_dim}.pt")
    torch.save(model.state_dict(), model_path)

    results.append({
        "latent_dim": latent_dim,
        "best_val_total": out["best_val_total"],
        "model_path": model_path,
        "train_total_losses": out["train_total_losses"],
        "train_recon_losses": out["train_recon_losses"],
        "train_kl_losses": out["train_kl_losses"],
        "val_total_losses": out["val_total_losses"],
        "val_recon_losses": out["val_recon_losses"],
        "val_kl_losses": out["val_kl_losses"],
    })


# =========================================================
# RESULTS TABLE
# =========================================================
results_df = pd.DataFrame([
    {
        "latent_dim": r["latent_dim"],
        "best_val_total_loss": r["best_val_total"],
        "best_val_recon_loss": min(r["val_recon_losses"]),
        "best_val_kl_loss": min(r["val_kl_losses"]),
        "model_path": r["model_path"],
    }
    for r in results
]).sort_values("best_val_total_loss", ascending=True)

results_csv = os.path.join(OUTPUT_DIR, "vae_latent_dim_comparison.csv")
results_df.to_csv(results_csv, index=False)

print("\nComparison results:")
print(results_df)

best_latent_dim = int(results_df.iloc[0]["latent_dim"])
print(f"\nBest latent_dim by validation total loss: {best_latent_dim}")


# =========================================================
# PLOT 1 - TOTAL LOSS COMPARISON BY LATENT DIM
# =========================================================
plt.figure(figsize=(8, 5))
plt.plot(results_df["latent_dim"], results_df["best_val_total_loss"], marker="o")
plt.xlabel("Latent Dimension")
plt.ylabel("Best Validation Total Loss")
plt.title("VAE Comparison by Latent Size")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "vae_latent_dim_comparison_total_loss.png"), dpi=300)
plt.show()


# =========================================================
# PLOT 2 - BEST RECON LOSS COMPARISON
# =========================================================
plt.figure(figsize=(8, 5))
plt.plot(results_df["latent_dim"], results_df["best_val_recon_loss"], marker="o")
plt.xlabel("Latent Dimension")
plt.ylabel("Best Validation Reconstruction Loss")
plt.title("VAE Reconstruction Loss by Latent Size")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "vae_latent_dim_comparison_recon_loss.png"), dpi=300)
plt.show()


# =========================================================
# PLOT 3 - BEST KL LOSS COMPARISON
# =========================================================
plt.figure(figsize=(8, 5))
plt.plot(results_df["latent_dim"], results_df["best_val_kl_loss"], marker="o")
plt.xlabel("Latent Dimension")
plt.ylabel("Best Validation KL Loss")
plt.title("VAE KL Loss by Latent Size")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "vae_latent_dim_comparison_kl_loss.png"), dpi=300)
plt.show()


# =========================================================
# PLOT 4 - VALIDATION TOTAL CURVES
# =========================================================
plt.figure(figsize=(10, 6))
for r in results:
    plt.plot(r["val_total_losses"], label=f"latent={r['latent_dim']}")
plt.xlabel("Epoch")
plt.ylabel("Validation Total Loss")
plt.title("Validation Total Loss Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "vae_validation_total_curves.png"), dpi=300)
plt.show()


# =========================================================
# PLOT 5 - TRAIN VS VALIDATION TOTAL LOSS
# =========================================================
plt.figure(figsize=(12, 7))
for r in results:
    plt.plot(r["train_total_losses"], linestyle="--", label=f"train latent={r['latent_dim']}")
    plt.plot(r["val_total_losses"], label=f"val latent={r['latent_dim']}")
plt.xlabel("Epoch")
plt.ylabel("Total Loss")
plt.title("Train vs Validation Total Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "vae_train_vs_val_total_loss.png"), dpi=300)
plt.show()


# =========================================================
# PLOT 6 - VALIDATION RECON CURVES
# =========================================================
plt.figure(figsize=(10, 6))
for r in results:
    plt.plot(r["val_recon_losses"], label=f"latent={r['latent_dim']}")
plt.xlabel("Epoch")
plt.ylabel("Validation Reconstruction Loss")
plt.title("Validation Reconstruction Loss Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "vae_validation_recon_curves.png"), dpi=300)
plt.show()


# =========================================================
# PLOT 7 - VALIDATION KL CURVES
# =========================================================
plt.figure(figsize=(10, 6))
for r in results:
    plt.plot(r["val_kl_losses"], label=f"latent={r['latent_dim']}")
plt.xlabel("Epoch")
plt.ylabel("Validation KL Loss")
plt.title("Validation KL Loss Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "vae_validation_kl_curves.png"), dpi=300)
plt.show()


# =========================================================
# LOAD BEST MODEL
# =========================================================
best_row = results_df.iloc[0]
best_model_path = best_row["model_path"]
best_latent_dim = int(best_row["latent_dim"])

best_model = VAE(input_dim=input_dim, latent_dim=best_latent_dim).to(device)
best_model.load_state_dict(torch.load(best_model_path, map_location=device))
best_model.eval()


# =========================================================
# ENCODE FULL DATASET
# =========================================================
X_tensor = torch.tensor(X_scaled).to(device)

with torch.no_grad():
    X_recon, mu, logvar, z = best_model(X_tensor)

Z = z.cpu().numpy()
MU = mu.cpu().numpy()
LOGVAR = logvar.cpu().numpy()
X_recon = X_recon.cpu().numpy()

# per-sample reconstruction error
recon_error = np.mean((X_scaled - X_recon) ** 2, axis=1)


# =========================================================
# SAVE LATENT REPRESENTATION
# =========================================================
latent_cols = [f"z{i+1}" for i in range(best_latent_dim)]
mu_cols = [f"mu{i+1}" for i in range(best_latent_dim)]
logvar_cols = [f"logvar{i+1}" for i in range(best_latent_dim)]

latent_df = pd.DataFrame(Z, columns=latent_cols)
mu_df = pd.DataFrame(MU, columns=mu_cols)
logvar_df = pd.DataFrame(LOGVAR, columns=logvar_cols)

full_latent_df = pd.concat([latent_df, mu_df, logvar_df], axis=1)
full_latent_df.insert(0, "PID", df["PID"].values)
full_latent_df["reconstruction_error"] = recon_error

latent_csv_path = os.path.join(OUTPUT_DIR, f"pid_vae_latent_best_{best_latent_dim}.csv")
full_latent_df.to_csv(latent_csv_path, index=False)


# =========================================================
# SAVE RECONSTRUCTION ERROR TABLE
# =========================================================
error_df = pd.DataFrame({
    "PID": df["PID"].values,
    "reconstruction_error": recon_error
}).sort_values("reconstruction_error", ascending=False)

error_csv_path = os.path.join(OUTPUT_DIR, "pid_vae_reconstruction_error.csv")
error_df.to_csv(error_csv_path, index=False)


# =========================================================
# SAVE TOP ANOMALIES
# =========================================================
top_anomalies_df = error_df.head(20).copy()
top_anomalies_csv = os.path.join(OUTPUT_DIR, "top_20_vae_anomalies.csv")
top_anomalies_df.to_csv(top_anomalies_csv, index=False)


# =========================================================
# PLOT 8 - RECONSTRUCTION ERROR HISTOGRAM
# =========================================================
plt.figure(figsize=(8, 5))
plt.hist(recon_error, bins=50)
plt.xlabel("Reconstruction Error")
plt.ylabel("Count")
plt.title("Distribution of VAE Reconstruction Error")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "vae_reconstruction_error_hist.png"), dpi=300)
plt.show()


# =========================================================
# PLOT 9 - TOP 20 ANOMALIES BARPLOT
# =========================================================
plt.figure(figsize=(12, 6))
plt.bar(top_anomalies_df["PID"].astype(str), top_anomalies_df["reconstruction_error"])
plt.xticks(rotation=90)
plt.xlabel("PID")
plt.ylabel("Reconstruction Error")
plt.title("Top 20 Highest VAE Reconstruction Errors")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "vae_top_20_anomalies_bar.png"), dpi=300)
plt.show()


# =========================================================
# PLOT 10 - LATENT SPACE
# =========================================================
if best_latent_dim >= 2:
    plt.figure(figsize=(7, 6))
    plt.scatter(Z[:, 0], Z[:, 1], s=20)
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("VAE Latent Space")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "vae_latent_space.png"), dpi=300)
    plt.show()
else:
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    Z_2d = pca.fit_transform(Z)

    plt.figure(figsize=(7, 6))
    plt.scatter(Z_2d[:, 0], Z_2d[:, 1], s=20)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("VAE Latent Space (PCA Projection)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "vae_latent_space_pca.png"), dpi=300)
    plt.show()


# =========================================================
# PLOT 11 - LATENT SPACE COLORED BY ERROR
# =========================================================
if best_latent_dim >= 2:
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=recon_error, s=20)
    plt.colorbar(sc, label="Reconstruction Error")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("VAE Latent Space Colored by Reconstruction Error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "vae_latent_space_colored.png"), dpi=300)
    plt.show()
else:
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    Z_2d = pca.fit_transform(Z)

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(Z_2d[:, 0], Z_2d[:, 1], c=recon_error, s=20)
    plt.colorbar(sc, label="Reconstruction Error")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("VAE Latent Space PCA Projection Colored by Reconstruction Error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "vae_latent_space_pca_colored.png"), dpi=300)
    plt.show()


# =========================================================
# PLOT 12 - LATENT SPACE COLORED BY PID ORDER
# =========================================================
if best_latent_dim >= 2:
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=np.arange(len(Z)), s=20)
    plt.colorbar(sc, label="Sample Index")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("VAE Latent Space Colored by Sample Order")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "vae_latent_space_sample_order.png"), dpi=300)
    plt.show()


# =========================================================
# OPTIONAL - SAVE RECONSTRUCTED DATASET
# =========================================================
recon_df = pd.DataFrame(X_recon, columns=feature_cols)
recon_df.insert(0, "PID", df["PID"].values)
recon_df["reconstruction_error"] = recon_error
recon_df.to_csv(os.path.join(OUTPUT_DIR, "vae_reconstructed_dataset.csv"), index=False)


# =========================================================
# SAVE SUMMARY TXT
# =========================================================
summary_path = os.path.join(OUTPUT_DIR, "vae_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(f"CSV_PATH: {CSV_PATH}\n")
    f.write(f"Input shape: {df.shape}\n")
    f.write(f"Input dim: {input_dim}\n")
    f.write(f"Latent dims tested: {LATENT_DIMS}\n")
    f.write(f"Best latent dim: {best_latent_dim}\n")
    f.write(f"Best validation total loss: {results_df.iloc[0]['best_val_total_loss']}\n")
    f.write(f"Best validation recon loss: {results_df.iloc[0]['best_val_recon_loss']}\n")
    f.write(f"Best validation kl loss: {results_df.iloc[0]['best_val_kl_loss']}\n")
    f.write(f"Beta: {BETA}\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Batch size: {BATCH_SIZE}\n")
    f.write(f"Learning rate: {LEARNING_RATE}\n")


# =========================================================
# FINAL PRINTS
# =========================================================
print("\nSaved comparison table to:", results_csv)
print("Saved best latent representation to:", latent_csv_path)
print("Saved reconstruction errors to:", error_csv_path)
print("Saved top anomalies to:", top_anomalies_csv)
print("Saved summary to:", summary_path)
print("Best latent dim:", best_latent_dim)
print("Done.")