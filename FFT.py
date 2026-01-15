import os
import glob
import numpy as np
import pandas as pd


# =========================
# CONFIG
# =========================
SEGMENTS_DIR = "segmented_10s"              
OUT_DIR = "fft_features"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_WINDOWS_CSV = os.path.join(OUT_DIR, "fft_features_windows.csv")
OUT_PID_CSV = os.path.join(OUT_DIR, "fft_features_pid.csv")

FMIN = 0.5
FMAX = 40.0

# Bands for bandpower features (Hz)
BANDS = [
    (0.5, 4),
    (4, 8),
    (8, 12),
    (12, 20),
    (20, 30),
    (30, 40),
]


# =========================
# HELPERS
# =========================
def load_segment_npz(path):
    d = np.load(path, allow_pickle=True)

    fs = float(np.array(d["fs"]).reshape(-1)[0])
    pid = str(np.array(d["pid"]).reshape(-1)[0])
    segment_id = str(np.array(d["segment_id"]).reshape(-1)[0])
    start_sample = int(np.array(d["start_sample"]).reshape(-1)[0])

    lead_keys = [k for k in d.files if k not in ["fs", "pid", "segment_id", "start_sample"]]
    lead_keys = sorted([k for k in lead_keys if isinstance(d[k], np.ndarray) and d[k].ndim == 1])

    leads = {k: d[k].astype(np.float32) for k in lead_keys}
    return fs, pid, segment_id, start_sample, leads


def hann_rfft_logpower(x, fs):
    """
    x: 1D window (already cleaned + zscored)
    returns freqs, log_power
    """
    N = len(x)
    w = np.hanning(N).astype(np.float32)
    xw = x * w

    X = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(N, d=1/fs)

    power = (np.abs(X) ** 2) / N
    logp = np.log(power + 1e-12)
    return freqs, logp


def bandpower_features(freqs, logp, bands):
    """
    Integrate power (not logpower) per band.
    We'll convert back from logp -> power for integration:
      power = exp(logp)
    """
    power = np.exp(logp)
    feats = {}
    for (lo, hi) in bands:
        mask = (freqs >= lo) & (freqs < hi)
        feats[f"bp_{lo}_{hi}"] = float(power[mask].mean()) if mask.any() else 0.0
    return feats


# =========================
# MAIN
# =========================
def main():
    # all segment files (nested folders)
    seg_files = sorted(glob.glob(os.path.join(SEGMENTS_DIR, "*", "*.npz")))
    if not seg_files:
        raise SystemExit(f"No segment npz files found under {SEGMENTS_DIR}/<base>/*.npz")

    rows = []

    for i, path in enumerate(seg_files, start=1):
        fs, pid, segment_id, start_sample, leads = load_segment_npz(path)

        base = os.path.basename(os.path.dirname(path))  # 1002688_6025_0_0
        start_sec = start_sample / fs

        for lead_name, x in leads.items():
            freqs, logp = hann_rfft_logpower(x, fs)

            # keep only 0.5-40 Hz
            mask = (freqs >= FMIN) & (freqs <= FMAX)
            freqs2 = freqs[mask]
            logp2 = logp[mask]

            # bandpower features
            feats = bandpower_features(freqs2, logp2, BANDS)

            row = {
                "PID": pid,
                "file_base": base,
                "segment_id": segment_id,
                "lead": lead_name,
                "fs": fs,
                "start_sec": start_sec,
                "segment_path": path,
            }
            row.update(feats)
            rows.append(row)

        if i % 200 == 0:
            print(f"Processed {i}/{len(seg_files)} segment files...")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_WINDOWS_CSV, index=False)
    print(f"Saved window-level FFT features: {OUT_WINDOWS_CSV}  (rows={len(df)})")

    # =========================
    # PID-level embedding
    # =========================
    # Aggregate per PID and lead using median across segments
    group_cols = ["PID", "lead"]
    feat_cols = [c for c in df.columns if c.startswith("bp_")]

    pid_df = df.groupby(group_cols)[feat_cols].median().reset_index()

    # Optionally pivot leads into columns (PID one row)
    pid_pivot = pid_df.pivot(index="PID", columns="lead", values=feat_cols)
    pid_pivot.columns = [f"{feat}_{lead}" for feat, lead in pid_pivot.columns]
    pid_pivot = pid_pivot.reset_index()

    pid_pivot.to_csv(OUT_PID_CSV, index=False)
    print(f"Saved PID-level FFT embeddings: {OUT_PID_CSV}  (rows={len(pid_pivot)})")


if __name__ == "__main__":
    main()
