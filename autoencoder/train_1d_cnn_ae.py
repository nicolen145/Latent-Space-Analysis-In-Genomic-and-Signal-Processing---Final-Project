# =========================================================
# 1D CNN AUTOENCODER - FINAL AE VERSION
# Latent dimension: 40
# =========================================================

import os
import glob
import random
import copy
import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = "/sise/nadav-group/nadavrap-group/Final_Project-Nicole_Gal_Lilach"

SOURCE_SEGMENTED_DIR = os.path.join(BASE_DIR, "data/segmented_10s")

OUTPUT_DIR = os.path.join(BASE_DIR, "Final_Project/Final_AE")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

SCRATCH_DIR = os.environ.get("SLURM_TMPDIR", os.path.join(OUTPUT_DIR, "tmp"))
SEGMENTED_DIR = os.path.join(SCRATCH_DIR, "segmented_10s")

SEGMENTS_CSV = os.path.join(SEGMENTED_DIR, "segments_with_trend.csv")

RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2

BATCH_SIZE = 128
EPOCHS = 60
PATIENCE = 10

LR = 1e-3
WEIGHT_DECAY = 1e-5

LATENT_DIM = 40
NUM_WORKERS = 4


# =========================================================
# SEED + DEVICE
# =========================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    
    
# =========================================================
# COPY SEGMENTED DATA TO SCRATCH
# =========================================================

def copy_segmented_to_scratch_if_needed():
    import shutil

    marker_file = os.path.join(SEGMENTED_DIR, ".copied")

    if os.path.exists(marker_file):
        print("Segmented data already copied:", SEGMENTED_DIR)
        return

    print("Copying segmented data to scratch:")
    print("From:", SOURCE_SEGMENTED_DIR)
    print("To:", SEGMENTED_DIR)

    if os.path.exists(SEGMENTED_DIR):
        shutil.rmtree(SEGMENTED_DIR)

    shutil.copytree(SOURCE_SEGMENTED_DIR, SEGMENTED_DIR)

    with open(marker_file, "w") as f:
        f.write("done")

    print("Copy done.")


copy_segmented_to_scratch_if_needed()


# =========================================================
# LOAD METADATA
# =========================================================

print("Using SEGMENTS_CSV:", SEGMENTS_CSV)

if not os.path.exists(SEGMENTS_CSV):
    raise FileNotFoundError(f"Segments CSV not found: {SEGMENTS_CSV}")
  
df = pd.read_csv(
    SEGMENTS_CSV,
    dtype={
        "PID": str,
        "file_base": str,
        "segment_id": str,
        "segment_path": str
    }
)

df = df.dropna(subset=["PID", "segment_path"]).reset_index(drop=True)

print("Loaded metadata:", df.shape)


# =========================================================
# FIX SEGMENT PATHS
# =========================================================

all_npz_files = glob.glob(
    os.path.join(SEGMENTED_DIR, "**", "*.npz"),
    recursive=True
)

print("NPZ files found:", len(all_npz_files))

npz_by_name = {
    os.path.basename(path): path
    for path in all_npz_files
}


def fix_segment_path(old_path):
    old_path = str(old_path)

    if os.path.exists(old_path):
        return old_path

    filename = os.path.basename(old_path)

    if filename in npz_by_name:
        return npz_by_name[filename]

    return old_path


df["segment_path"] = df["segment_path"].apply(fix_segment_path)

missing_paths = (~df["segment_path"].apply(os.path.exists)).sum()
print("Missing segment paths:", missing_paths)

df = df[df["segment_path"].apply(os.path.exists)].reset_index(drop=True)

print("Metadata after path fix:", df.shape)

if len(df) == 0:
    raise RuntimeError("No valid NPZ segment paths found.")


# =========================================================
# SPLIT BY PID - PREVENT DATA LEAKAGE
# =========================================================

unique_pids = df["PID"].unique()

train_val_pids, test_pids = train_test_split(
    unique_pids,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    shuffle=True
)

train_pids, val_pids = train_test_split(
    train_val_pids,
    test_size=VAL_SIZE,
    random_state=RANDOM_SEED,
    shuffle=True
)

train_df = df[df["PID"].isin(train_pids)].reset_index(drop=True)
val_df = df[df["PID"].isin(val_pids)].reset_index(drop=True)
test_df = df[df["PID"].isin(test_pids)].reset_index(drop=True)

print(f"Train PIDs: {len(train_pids)} | segments: {len(train_df)}")
print(f"Val PIDs:   {len(val_pids)} | segments: {len(val_df)}")
print(f"Test PIDs:  {len(test_pids)} | segments: {len(test_df)}")


# =========================================================
# LOAD ONE SEGMENT
# =========================================================

def load_segment_signal(segment_path, seq_len=None, n_leads=None):
    d = np.load(segment_path, allow_pickle=True)

    skip_keys = {
        "fs", "PID", "pid",
        "segment_id",
        "start_sample",
        "start_sec",
        "end_sec"
    }

    lead_keys = sorted([
        k for k in d.files
        if k not in skip_keys
        and isinstance(d[k], np.ndarray)
        and d[k].ndim == 1
        and d[k].size > 30
    ])

    if not lead_keys:
        raise ValueError(f"No ECG lead arrays found in {segment_path}")

    min_len = min(len(d[k]) for k in lead_keys)

    x = np.stack(
        [d[k][:min_len].astype(np.float32) for k in lead_keys],
        axis=0
    )

    if n_leads is not None:
        if x.shape[0] > n_leads:
            x = x[:n_leads, :]
        elif x.shape[0] < n_leads:
            x = np.pad(
                x,
                ((0, n_leads - x.shape[0]), (0, 0)),
                mode="constant"
            )

    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-8
    x = (x - mean) / std

    if seq_len is not None:
        if x.shape[1] > seq_len:
            x = x[:, :seq_len]
        elif x.shape[1] < seq_len:
            x = np.pad(
                x,
                ((0, 0), (0, seq_len - x.shape[1])),
                mode="constant"
            )

    return x.astype(np.float32)


sample = load_segment_signal(train_df.loc[0, "segment_path"])
N_LEADS, SEQ_LEN = sample.shape

print("N_LEADS:", N_LEADS)
print("SEQ_LEN:", SEQ_LEN)


# =========================================================
# DATASET
# =========================================================

class ECGSegmentDataset(Dataset):
    def __init__(self, meta_df, seq_len, n_leads):
        self.meta_df = meta_df.reset_index(drop=True)
        self.seq_len = seq_len
        self.n_leads = n_leads

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        path = self.meta_df.loc[idx, "segment_path"]

        x = load_segment_signal(
            path,
            seq_len=self.seq_len,
            n_leads=self.n_leads
        )

        return torch.tensor(x, dtype=torch.float32)


def make_loader(meta_df, shuffle):
    return DataLoader(
        ECGSegmentDataset(meta_df, SEQ_LEN, N_LEADS),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0)
    )


train_loader = make_loader(train_df, shuffle=True)
val_loader = make_loader(val_df, shuffle=False)
test_loader = make_loader(test_df, shuffle=False)

train_enc_loader = make_loader(train_df, shuffle=False)
val_enc_loader = make_loader(val_df, shuffle=False)
test_enc_loader = make_loader(test_df, shuffle=False)


# =========================================================
# MODEL BLOCKS
# =========================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7, stride=2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size=kernel,
                stride=stride,
                padding=kernel // 2
            ),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.net(x)


class ConvTransBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7, stride=2):
        super().__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose1d(
                in_ch,
                out_ch,
                kernel_size=kernel,
                stride=stride,
                padding=kernel // 2,
                output_padding=stride - 1
            ),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.net(x)


class CNNAutoEncoder(nn.Module):
    def __init__(self, n_leads, seq_len, latent_dim, base_ch=32):
        super().__init__()

        self.n_leads = n_leads
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        self.enc_convs = nn.Sequential(
            ConvBlock(n_leads, base_ch),
            ConvBlock(base_ch, base_ch * 2),
            ConvBlock(base_ch * 2, base_ch * 4),
            ConvBlock(base_ch * 4, base_ch * 8)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, n_leads, seq_len)
            enc_out = self.enc_convs(dummy)
            self.enc_shape = enc_out.shape[1:]
            self.flat_dim = enc_out.numel()

        self.fc_enc = nn.Linear(self.flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)

        self.dec_convs = nn.Sequential(
            ConvTransBlock(base_ch * 8, base_ch * 4),
            ConvTransBlock(base_ch * 4, base_ch * 2),
            ConvTransBlock(base_ch * 2, base_ch),
            nn.ConvTranspose1d(
                base_ch,
                n_leads,
                kernel_size=7,
                stride=2,
                padding=3,
                output_padding=1
            )
        )

    def encode(self, x):
        h = self.enc_convs(x)
        h = h.flatten(start_dim=1)
        z = self.fc_enc(h)
        return z

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(z.size(0), *self.enc_shape)

        x_hat = self.dec_convs(h)

        if x_hat.size(2) > self.seq_len:
            x_hat = x_hat[:, :, :self.seq_len]
        elif x_hat.size(2) < self.seq_len:
            x_hat = F.pad(x_hat, (0, self.seq_len - x_hat.size(2)))

        return x_hat

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


# =========================================================
# TRAIN MODEL FROM SCRATCH
# =========================================================

def train_ae():
    model = CNNAutoEncoder(
        n_leads=N_LEADS,
        seq_len=SEQ_LEN,
        latent_dim=LATENT_DIM
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters for latent={LATENT_DIM}: {num_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val = np.inf
    best_state = None
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_sum = 0.0

        for bx in train_loader:
            bx = bx.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                x_hat, _ = model(bx)
                loss = F.mse_loss(x_hat, bx)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_sum += loss.item() * bx.size(0)

        train_loss = train_sum / len(train_loader.dataset)

        model.eval()
        val_sum = 0.0

        with torch.no_grad():
            for bx in val_loader:
                bx = bx.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    x_hat, _ = model(bx)
                    loss = F.mse_loss(x_hat, bx)

                val_sum += loss.item() * bx.size(0)

        val_loss = val_sum / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"[latent={LATENT_DIM}] "
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train={train_loss:.6f} | "
            f"val={val_loss:.6f} | "
            f"patience={patience_counter}/{PATIENCE}",
            flush=True
        )

        if patience_counter >= PATIENCE:
            print("Early stopping")
            break

    model.load_state_dict(best_state)

    return model, best_val, train_losses, val_losses


print("\n" + "=" * 70)
print(f"Training final 1D CNN AE from scratch | latent_dim={LATENT_DIM}")
print("=" * 70)

model, best_val, train_losses, val_losses = train_ae()


# =========================================================
# SAVE MODEL
# =========================================================

model_path = os.path.join(
    OUTPUT_DIR,
    f"best_1d_ae_latent_{LATENT_DIM}.pt"
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "n_leads": N_LEADS,
        "seq_len": SEQ_LEN,
        "latent_dim": LATENT_DIM
    },
    model_path
)

print("Saved model:", model_path)


# =========================================================
# ENCODE SPLITS
# =========================================================

def encode_split(model, loader, meta_df, split_name):
    model.eval()

    all_z = []
    all_errors = []

    with torch.no_grad():
        for bx in loader:
            bx = bx.to(device, non_blocking=True)

            x_hat, z = model(bx)

            err = F.mse_loss(
                x_hat,
                bx,
                reduction="none"
            ).mean(dim=(1, 2))

            all_z.append(z.cpu().numpy())
            all_errors.append(err.cpu().numpy())

    Z = np.concatenate(all_z, axis=0)
    errors = np.concatenate(all_errors, axis=0)

    z_cols = [f"z{i+1}" for i in range(Z.shape[1])]
    z_df = pd.DataFrame(Z, columns=z_cols)

    out_df = pd.concat(
        [
            meta_df.reset_index(drop=True),
            z_df.reset_index(drop=True)
        ],
        axis=1
    )

    out_df["split"] = split_name
    out_df["reconstruction_error"] = errors

    return out_df


print("\nEncoding train / val / test...")

train_out = encode_split(model, train_enc_loader, train_df, "train")
val_out = encode_split(model, val_enc_loader, val_df, "val")
test_out = encode_split(model, test_enc_loader, test_df, "test")

latent_df = pd.concat(
    [train_out, val_out, test_out],
    axis=0
).reset_index(drop=True)


# =========================================================
# SAVE OUTPUTS
# =========================================================

latent_csv = os.path.join(
    OUTPUT_DIR,
    f"segments_1d_ae_latent_best_{LATENT_DIM}.csv"
)

error_csv = os.path.join(
    OUTPUT_DIR,
    f"segments_1d_ae_reconstruction_error_{LATENT_DIM}.csv"
)

top20_csv = os.path.join(
    OUTPUT_DIR,
    f"top_20_segments_1d_ae_anomalies_{LATENT_DIM}.csv"
)

training_json = os.path.join(
    OUTPUT_DIR,
    "training_results.json"
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "ae_1d_summary.txt"
)

latent_df.to_csv(latent_csv, index=False)

latent_df.sort_values(
    "reconstruction_error",
    ascending=False
).to_csv(error_csv, index=False)

latent_df.sort_values(
    "reconstruction_error",
    ascending=False
).head(20).to_csv(top20_csv, index=False)

with open(training_json, "w", encoding="utf-8") as f:
    json.dump(
        {
            "latent_dim": int(LATENT_DIM),
            "best_val_loss": float(best_val),
            "model_path": model_path,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "segments_csv": SEGMENTS_CSV,
            "segmented_dir": SEGMENTED_DIR
        },
        f,
        indent=2
    )

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("Model: 1D CNN Autoencoder\n")
    f.write("Training: from scratch on all available segmented data\n")
    f.write("Input: cleaned 10-second ECG segments [leads, time]\n")
    f.write("FFT features are NOT used as model input\n")
    f.write("TrendData is used only as metadata labels\n")
    f.write(f"Segments CSV: {SEGMENTS_CSV}\n")
    f.write(f"Segmented dir: {SEGMENTED_DIR}\n")
    f.write(f"N_LEADS: {N_LEADS}\n")
    f.write(f"SEQ_LEN: {SEQ_LEN}\n")
    f.write(f"Latent dim: {LATENT_DIM}\n")
    f.write(f"Best val loss: {best_val}\n")
    f.write(f"Train/Val/Test PIDs: {len(train_pids)}/{len(val_pids)}/{len(test_pids)}\n")
    f.write(f"Train/Val/Test segments: {len(train_df)}/{len(val_df)}/{len(test_df)}\n")
    f.write(f"Batch size: {BATCH_SIZE}\n")
    f.write(f"Epochs max: {EPOCHS}\n")
    f.write(f"Early stopping patience: {PATIENCE}\n")
    f.write(f"Learning rate: {LR}\n")
    f.write(f"Device: {device}\n")


# =========================================================
# PLOTS
# =========================================================

def save_plot(filename):
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved plot:", path)


plt.figure(figsize=(10, 5))
plt.plot(val_losses, label=f"latent={LATENT_DIM}")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss (MSE)")
plt.title("Final 1D CNN AE - Validation Loss During Training")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_plot("01_validation_loss_curve.png")


plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Final 1D CNN AE - Train vs Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_plot("02_train_vs_validation_loss.png")


recon = latent_df["reconstruction_error"].values
recon_clean = recon[np.isfinite(recon)]

x_max = np.percentile(recon_clean, 99.9)

plt.figure(figsize=(8, 5))
plt.hist(
    recon_clean[recon_clean <= x_max],
    bins=80,
    edgecolor="white"
)
plt.xlim(0, x_max)
plt.xlabel("Reconstruction Error (MSE)")
plt.ylabel("Count")
plt.title("Final 1D CNN AE - Reconstruction Error Distribution")
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_plot("03_reconstruction_error_distribution.png")


top20 = latent_df.sort_values(
    "reconstruction_error",
    ascending=False
).head(20)

plt.figure(figsize=(14, 5))
plt.bar(
    top20["segment_id"].astype(str),
    top20["reconstruction_error"]
)
plt.xticks(rotation=90, fontsize=7)
plt.xlabel("Segment ID")
plt.ylabel("Reconstruction Error (MSE)")
plt.title("Final 1D CNN AE - Top 20 Anomalous Segments")
plt.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
save_plot("04_top_20_anomalous_segments.png")


print("\nSaved files:")
print("Model:", model_path)
print("Latent CSV:", latent_csv)
print("Reconstruction errors CSV:", error_csv)
print("Top 20 anomalies CSV:", top20_csv)
print("Training JSON:", training_json)
print("Summary:", summary_path)
print("Plots dir:", PLOTS_DIR)
print("\nDone.")