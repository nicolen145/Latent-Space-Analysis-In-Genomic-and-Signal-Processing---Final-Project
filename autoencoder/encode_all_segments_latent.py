# =========================================================
# ENCODE ALL ECG SEGMENTS TO LATENT SPACE
# Uses trained 1D CNN AE model
# =========================================================

import os
import glob
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


BASE_DIR = "/sise/nadav-group/nadavrap-group/Final_Project-Nicole_Gal_Lilach"

SEGMENTED_DIR = os.path.join(BASE_DIR, "data/segmented_10s")
SEGMENTS_CSV = os.path.join(SEGMENTED_DIR, "segments_with_trend.csv")

MODEL_PATH = os.path.join(
    BASE_DIR,
    "Final_Project/Final_AE/best_1d_ae_latent_40.pt"
)

OUT_CSV = os.path.join(
    BASE_DIR,
    "Final_Project/Final_AE/all_segments_latent_40.csv"
)

BATCH_SIZE = 128
NUM_WORKERS = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# =========================================================
# LOAD METADATA
# =========================================================

df = pd.read_csv(
    SEGMENTS_CSV,
    dtype={
        "PID": str,
        "file_base": str,
        "segment_id": str,
        "segment_path": str
    }
)

df = df.dropna(subset=["segment_path"]).reset_index(drop=True)

# Fix paths if needed
all_npz_files = glob.glob(
    os.path.join(SEGMENTED_DIR, "**", "*.npz"),
    recursive=True
)

npz_by_name = {
    os.path.basename(p): p
    for p in all_npz_files
}

def fix_segment_path(path):
    path = str(path)

    if os.path.exists(path):
        return path

    filename = os.path.basename(path)

    if filename in npz_by_name:
        return npz_by_name[filename]

    return path

df["segment_path"] = df["segment_path"].apply(fix_segment_path)

df = df[df["segment_path"].apply(os.path.exists)].reset_index(drop=True)

print("Segments to encode:", len(df))


# =========================================================
# LOAD MODEL CHECKPOINT
# =========================================================

checkpoint = torch.load(MODEL_PATH, map_location=device)

N_LEADS = checkpoint["n_leads"]
SEQ_LEN = checkpoint["seq_len"]
LATENT_DIM = checkpoint["latent_dim"]

print("N_LEADS:", N_LEADS)
print("SEQ_LEN:", SEQ_LEN)
print("LATENT_DIM:", LATENT_DIM)


# =========================================================
# LOAD SEGMENT
# =========================================================

def load_segment_signal(segment_path, seq_len, n_leads):
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

    if x.shape[1] > seq_len:
        x = x[:, :seq_len]
    elif x.shape[1] < seq_len:
        x = np.pad(
            x,
            ((0, 0), (0, seq_len - x.shape[1])),
            mode="constant"
        )

    return x.astype(np.float32)


# =========================================================
# DATASET
# =========================================================

class ECGSegmentDataset(Dataset):
    def __init__(self, meta_df):
        self.meta_df = meta_df.reset_index(drop=True)

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        path = self.meta_df.loc[idx, "segment_path"]

        x = load_segment_signal(
            path,
            seq_len=SEQ_LEN,
            n_leads=N_LEADS
        )

        return torch.tensor(x, dtype=torch.float32)


loader = DataLoader(
    ECGSegmentDataset(df),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
    persistent_workers=(NUM_WORKERS > 0)
)


# =========================================================
# MODEL ARCHITECTURE
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

        self.seq_len = seq_len

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


model = CNNAutoEncoder(
    n_leads=N_LEADS,
    seq_len=SEQ_LEN,
    latent_dim=LATENT_DIM
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model loaded.")


# =========================================================
# ENCODE ALL SEGMENTS
# =========================================================

all_z = []
all_errors = []

with torch.no_grad():
    for batch_idx, bx in enumerate(loader, start=1):
        bx = bx.to(device, non_blocking=True)

        x_hat, z = model(bx)

        err = F.mse_loss(
            x_hat,
            bx,
            reduction="none"
        ).mean(dim=(1, 2))

        all_z.append(z.cpu().numpy())
        all_errors.append(err.cpu().numpy())

        if batch_idx % 50 == 0:
            print(
                f"Encoded batch {batch_idx}/{len(loader)}",
                flush=True
            )


Z = np.concatenate(all_z, axis=0)
errors = np.concatenate(all_errors, axis=0)

z_cols = [f"z{i+1}" for i in range(Z.shape[1])]
z_df = pd.DataFrame(Z, columns=z_cols)

out_df = pd.concat(
    [
        df.reset_index(drop=True),
        z_df.reset_index(drop=True)
    ],
    axis=1
)

out_df["reconstruction_error"] = errors

out_df.to_csv(OUT_CSV, index=False)

print("Saved latent CSV:")
print(OUT_CSV)
print("Done.")
