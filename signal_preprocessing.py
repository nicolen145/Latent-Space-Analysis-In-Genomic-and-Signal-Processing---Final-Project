import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt


# ----------------------------
# Configuration
# ----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(BASE_DIR, "parsed_signals_full")
CLEAN_DIR = os.path.join(BASE_DIR, "cleaned_signals_full")
OUTPUTS_DIR = os.path.join(BASE_DIR, "OUTPUTS")

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Cleaning settings
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 40.0
FILTER_ORDER = 4

# Plot settings (one example file)
PLOT_SECONDS = 10
PLOT_LEAD_INDEX = 0  # 0 => first lead (after sorting keys)


# ----------------------------
# Signal loading (NPZ format: fs + per-lead arrays)
# ----------------------------

def load_npz_leads(npz_path):
    """
    Load ECG from NPZ where:
      - 'fs' exists (scalar or length-1 array)
      - each lead is stored as its own 1D array (e.g., 'I', 'II', 'III')

    Returns:
      leads_matrix: (T, C) float32
      fs: float
      lead_names: list[str]
    """
    d = np.load(npz_path, allow_pickle=True)

    if "fs" not in d.files:
        raise ValueError("Missing 'fs' key in NPZ.")

    fs = float(np.array(d["fs"]).reshape(-1)[0])

    # Lead keys = all keys except 'fs'
    lead_keys = [k for k in d.files if k != "fs"]

    if not lead_keys:
        raise ValueError("No lead arrays found (only 'fs' exists).")

    # Keep only 1D arrays with meaningful length (avoid scalars/metadata)
    lead_keys = [k for k in lead_keys if isinstance(d[k], np.ndarray) and d[k].ndim == 1 and d[k].size > 30]

    if not lead_keys:
        raise ValueError("No valid lead arrays found (all too short).")

    # Sort lead keys for consistent channel order (I, II, III, V1...)
    lead_keys = sorted(lead_keys)

    # Ensure all leads have the same length (truncate to min if needed)
    lengths = [d[k].shape[0] for k in lead_keys]
    T = min(lengths)

    leads = np.stack([d[k][:T].astype(np.float32) for k in lead_keys], axis=1)  # (T, C)

    return leads, fs, lead_keys


def save_npz_leads(out_path, leads_matrix, fs, lead_names):
    """
    Save cleaned ECG in the same format:
      fs + per-lead arrays (1D)
    """
    payload = {"fs": np.array([fs], dtype=np.float32)}
    for i, name in enumerate(lead_names):
        payload[name] = leads_matrix[:, i].astype(np.float32)

    np.savez(out_path, **payload)


# ----------------------------
# Cleaning
# ----------------------------

def bandpass_filter(x, fs, low=0.5, high=40.0, order=4):
    """Zero-phase bandpass filter using filtfilt."""
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x, axis=0)


def zscore_per_channel(x, eps=1e-8):
    """Z-score normalize each channel independently."""
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return (x - mean) / (std + eps)


def clean_ecg(leads_matrix, fs, min_len=200):
    """
    Clean ECG:
      1) Remove DC offset
      2) Bandpass 0.5-40 Hz
      3) Z-score per channel

    Safety:
      - Reject signals that are too short for stable filtering.
    """
    x = leads_matrix.astype(np.float32)

    if x.shape[0] < min_len:
        raise ValueError(f"Signal too short for filtering: length={x.shape[0]} < {min_len}")

    # Remove DC offset
    x = x - x.mean(axis=0, keepdims=True)

    # Bandpass
    x = bandpass_filter(x, fs, BANDPASS_LOW, BANDPASS_HIGH, FILTER_ORDER)

    # Normalize
    x = zscore_per_channel(x)

    return x


# ----------------------------
# Plot one example (before vs after) and save
# ----------------------------

def save_before_after_plot(out_png_path, raw_mat, clean_mat, fs, lead_names, lead_index=0, seconds=10):
    """Save a two-panel plot (before/after) for one lead."""
    n = min(int(seconds * fs), raw_mat.shape[0], clean_mat.shape[0])
    t = np.arange(n) / fs

    lead_name = lead_names[lead_index] if 0 <= lead_index < len(lead_names) else f"lead{lead_index}"

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t, raw_mat[:n, lead_index])
    plt.title(f"Raw ECG (Before Cleaning) - {lead_name}")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t, clean_mat[:n, lead_index])
    plt.title(f"Cleaned ECG (After Bandpass + Z-score) - {lead_name}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (z-score)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_png_path, dpi=300, bbox_inches="tight")
    plt.close()


# ----------------------------
# Batch processing
# ----------------------------

def main():
    npz_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.npz")))
    print(f"Found {len(npz_files)} NPZ files in: {INPUT_DIR}")

    errors = []
    example_saved = False

    for i, path in enumerate(npz_files, start=1):
        base = os.path.splitext(os.path.basename(path))[0]
        print(f"[{i}/{len(npz_files)}] Processing: {base}")

        try:
            raw_mat, fs, lead_names = load_npz_leads(path)
            clean_mat = clean_ecg(raw_mat, fs)

            # Save cleaned NPZ (same structure)
            out_npz = os.path.join(CLEAN_DIR, f"{base}.npz")
            save_npz_leads(out_npz, clean_mat, fs, lead_names)

            # Save one example plot (first file that succeeds)
            if not example_saved:
                out_png = os.path.join(OUTPUTS_DIR, f"{base}_before_after.png")
                save_before_after_plot(
                    out_png_path=out_png,
                    raw_mat=raw_mat,
                    clean_mat=clean_mat,
                    fs=fs,
                    lead_names=lead_names,
                    lead_index=PLOT_LEAD_INDEX,
                    seconds=PLOT_SECONDS
                )
                print(f"Saved example plot to: {out_png}")
                example_saved = True

        except Exception as e:
            errors.append({"file": os.path.basename(path), "error": str(e)})

    # Write errors file if needed
    if errors:
        err_path = os.path.join(CLEAN_DIR, "errors_cleaning.csv")
        import csv
        with open(err_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file", "error"])
            w.writeheader()
            w.writerows(errors)
        print(f"Some files failed. See: {err_path}")

    print("Done.")


if __name__ == "__main__":
    main()
