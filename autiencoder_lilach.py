import os
import random
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# =========================================================
# CONFIG
# =========================================================
CSV_PATH = "fft_features/fft_features_pid_clean.csv"
OUTPUT_DIR = "autoencoder_multi_latent_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
TEST_SIZE = 0.2

# הגדלים שאת רוצה לבדוק
LATENT_DIMS = [2, 4, 8, 16, 32]


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

# split once so all models are compared fairly
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
# MODEL
# =========================================================
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


# =========================================================
# TRAIN FUNCTION
# =========================================================
def train_one_autoencoder(latent_dim, train_loader, val_loader, input_dim, device):
    model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    train_losses = []
    val_losses = []

    best_val_loss = np.inf
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0

        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)

            optimizer.zero_grad()
            batch_recon, _ = model(batch_x)
            loss = criterion(batch_recon, batch_x)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * batch_x.size(0)

        epoch_train_loss = train_loss_sum / len(train_loader.dataset)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for (batch_x,) in val_loader:
                batch_x = batch_x.to(device)
                batch_recon, _ = model(batch_x)
                loss = criterion(batch_recon, batch_x)
                val_loss_sum += loss.item() * batch_x.size(0)

        epoch_val_loss = val_loss_sum / len(val_loader.dataset)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 10 == 0 or epoch == 1:
            print(f"[latent={latent_dim}] Epoch {epoch:3d} | train={epoch_train_loss:.6f} | val={epoch_val_loss:.6f}")

    model.load_state_dict(best_state)

    return {
        "model": model,
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


# =========================================================
# RUN ALL LATENT SIZES
# =========================================================
results = []

for latent_dim in LATENT_DIMS:
    print("\n" + "=" * 60)
    print(f"Training Autoencoder with latent_dim = {latent_dim}")
    print("=" * 60)

    out = train_one_autoencoder(
        latent_dim=latent_dim,
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=input_dim,
        device=device
    )

    model = out["model"]
    best_val_loss = out["best_val_loss"]
    train_losses = out["train_losses"]
    val_losses = out["val_losses"]

    # save model
    model_path = os.path.join(OUTPUT_DIR, f"best_autoencoder_latent_{latent_dim}.pt")
    torch.save(model.state_dict(), model_path)

    results.append({
        "latent_dim": latent_dim,
        "best_val_loss": best_val_loss,
        "model_path": model_path,
        "train_losses": train_losses,
        "val_losses": val_losses,
    })


# =========================================================
# RESULTS TABLE
# =========================================================
results_df = pd.DataFrame([
    {
        "latent_dim": r["latent_dim"],
        "best_val_loss": r["best_val_loss"],
        "model_path": r["model_path"],
    }
    for r in results
]).sort_values("best_val_loss", ascending=True)

results_csv = os.path.join(OUTPUT_DIR, "latent_dim_comparison.csv")
results_df.to_csv(results_csv, index=False)

print("\nComparison results:")
print(results_df)

best_latent_dim = int(results_df.iloc[0]["latent_dim"])
print(f"\nBest latent_dim by validation loss: {best_latent_dim}")


# =========================================================
# PLOT COMPARISON
# =========================================================
plt.figure(figsize=(8, 5))
plt.plot(results_df["latent_dim"], results_df["best_val_loss"], marker="o")
plt.xlabel("Latent Dimension")
plt.ylabel("Best Validation Loss")
plt.title("Autoencoder Comparison by Latent Size")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "latent_dim_comparison.png"), dpi=300)
plt.show()


# =========================================================
# PLOT TRAINING CURVES FOR ALL
# =========================================================
plt.figure(figsize=(10, 6))
for r in results:
    plt.plot(r["val_losses"], label=f"latent={r['latent_dim']}")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Curves for Different Latent Dimensions")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "all_validation_curves.png"), dpi=300)
plt.show()


# =========================================================
# SAVE LATENT REPRESENTATION FOR BEST MODEL
# =========================================================
best_row = results_df.iloc[0]
best_model_path = best_row["model_path"]
best_latent_dim = int(best_row["latent_dim"])

best_model = Autoencoder(input_dim=input_dim, latent_dim=best_latent_dim).to(device)
best_model.load_state_dict(torch.load(best_model_path, map_location=device))
best_model.eval()

X_tensor = torch.tensor(X_scaled).to(device)

with torch.no_grad():
    X_recon, Z = best_model(X_tensor)

Z = Z.cpu().numpy()
X_recon = X_recon.cpu().numpy()

recon_error = np.mean((X_scaled - X_recon) ** 2, axis=1)

latent_cols = [f"z{i+1}" for i in range(best_latent_dim)]
latent_df = pd.DataFrame(Z, columns=latent_cols)
latent_df.insert(0, "PID", df["PID"].values)
latent_df["reconstruction_error"] = recon_error

latent_csv_path = os.path.join(OUTPUT_DIR, f"pid_autoencoder_latent_best_{best_latent_dim}.csv")
latent_df.to_csv(latent_csv_path, index=False)

print("\nSaved best latent representation to:", latent_csv_path)
print("Saved comparison table to:", results_csv)
print("Done.")
