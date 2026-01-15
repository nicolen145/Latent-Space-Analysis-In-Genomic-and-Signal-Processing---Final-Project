"""

Cleans ECG signals stored as NPZ files (fs + per-lead arrays) using:
  1) DC removal (per lead)
  2) Wavelet-based approximate bandpass filtering (~0.5–40 Hz) with db4
  3) Z-score normalization (per lead)

Outputs:
  - Cleaned full-length NPZ files in: cleaned_signals_full/
  - One example before/after plot in: OUTPUTS/
  - errors_cleaning.csv if failures occur

Notes:
  - This is NOT a clinical filter; wavelet band boundaries are approximate.
  - Use pip install pywavelets if needed.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.join(BASE_DIR, "parsed_signals_full")
CLEAN_DIR = os.path.join(BASE_DIR, "cleaned_signals_full")
OUTPUTS_DIR = os.path.join(BASE_DIR, "OUTPUTS")

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Wavelet / bandpass settings
WAVELET_NAME = "db4"
LOWCUT_HZ = 0.5
HIGHCUT_HZ = 40.0

# Plot settings (one example)
PLOT_SECONDS = 10
PLOT_LEAD_INDEX = 0  # first lead after sorting lead keys


# ============================================================
# NPZ LOADING/SAVING (format: fs + per-lead arrays)
# ============================================================

def load_npz_leads(npz_path):
    """
    Load ECG from NPZ where:
      - 'fs' exists (scalar or length-1 array)
      - each lead is stored as its own 1D array (e.g., I, II, III, ...)
    Returns:
      mat: (T, C) float32
      fs: float
      lead_names: list[str]
    """
    d = np.load(npz_path, allow_pickle=True)

    if "fs" not in d.files:
        raise ValueError("Missing 'fs' key in NPZ.")

    fs = float(np.array(d["fs"]).reshape(-1)[0])

    lead_keys = [k for k in d.files if k != "fs"]
    # Keep only 1D arrays with meaningful length (avoid scalars/metadata)
    lead_keys = [
        k for k in lead_keys
        if isinstance(d[k], np.ndarray) and d[k].ndim == 1 and d[k].size > 30
    ]
    if not lead_keys:
        raise ValueError("No valid lead arrays found in NPZ.")

    lead_keys = sorted(lead_keys)

    # Align lead lengths (truncate to minimum)
    lengths = [d[k].shape[0] for k in lead_keys]
    T = min(lengths)

    mat = np.stack([d[k][:T].astype(np.float32) for k in lead_keys], axis=1)
    return mat, fs, lead_keys


def save_npz_leads(out_path, mat, fs, lead_names):
    """
    Save cleaned ECG in NPZ with the same structure:
      - fs
      - each lead as a separate 1D array
    """
    payload = {"fs": np.array([fs], dtype=np.float32)}
    for c, name in enumerate(lead_names):
        payload[name] = mat[:, c].astype(np.float32)
    np.savez(out_path, **payload)


# ============================================================
# WAVELET BANDPASS (APPROX 0.5–40 Hz)
# ============================================================

def pick_level_for_lowcut(fs, lowcut_hz=0.5):
    """
    Pick decomposition level L such that A_L cutoff (~fs / 2^(L+1)) is close to lowcut_hz.
    This is approximate (wavelet bands are not brick-wall filters).
    """
    L = 1
    while (fs / (2 ** (L + 1))) > lowcut_hz and L < 12:
        L += 1
    return L


def wavelet_bandpass_1d(x, fs, wavelet="db4", lowcut_hz=0.5, highcut_hz=40.0):
    """
    Wavelet-based approximate bandpass by zeroing coefficients outside [lowcut_hz, highcut_hz].

    coeffs = [A_L, D_L, D_{L-1}, ..., D_1]
      - A_L corresponds to very low frequencies (< fs/2^(L+1))
      - D_j corresponds roughly to (fs/2^(j+1), fs/2^j)

    We remove A_L to enforce lowcut, and remove D_j bands that do not overlap the target range.
    """
    x = x.astype(np.float32)

    # Choose a level that reaches down to ~lowcut_hz
    L = pick_level_for_lowcut(fs, lowcut_hz=lowcut_hz)

    coeffs = pywt.wavedec(x, wavelet, level=L)
    new_coeffs = [coeffs[0].copy()]  # A_L

    # Remove very-low frequencies (<~lowcut) by zeroing A_L
    new_coeffs[0] = np.zeros_like(new_coeffs[0])

    # Keep only detail bands overlapping the desired range
    for idx in range(1, len(coeffs)):
        d = coeffs[idx]

        # idx=1 => D_L (lowest detail band), idx=2 => D_{L-1}, ..., idx=-1 => D_1
        j = L - (idx - 1)

        band_low = fs / (2 ** (j + 1))
        band_high = fs / (2 ** j)

        overlaps = (band_high >= lowcut_hz) and (band_low <= highcut_hz)

        if overlaps:
            new_coeffs.append(d)
        else:
            new_coeffs.append(np.zeros_like(d))

    y = pywt.waverec(new_coeffs, wavelet)
    return y[:len(x)].astype(np.float32)


def clean_ecg_wavelet(mat, fs, wavelet="db4", lowcut_hz=0.5, highcut_hz=40.0):
    """
    Cleaning pipeline:
      1) DC removal (per lead)
      2) Wavelet bandpass (~0.5–40 Hz) per lead
      3) Z-score normalization (per lead)
    """
    x = mat.astype(np.float32)

    # DC removal
    x = x - x.mean(axis=0, keepdims=True)

    # Wavelet bandpass per lead
    out = np.zeros_like(x, dtype=np.float32)
    for ch in range(x.shape[1]):
        out[:, ch] = wavelet_bandpass_1d(
            x[:, ch],
            fs=fs,
            wavelet=wavelet,
            lowcut_hz=lowcut_hz,
            highcut_hz=highcut_hz
        )

    # Z-score normalization per lead
    mean = out.mean(axis=0, keepdims=True)
    std = out.std(axis=0, keepdims=True) + 1e-8
    out = (out - mean) / std

    return out


# ============================================================
# PLOT EXAMPLE (BEFORE vs AFTER) AND SAVE
# ============================================================

def save_before_after_plot(out_png_path, raw_mat, clean_mat, fs, lead_names, lead_index=0, seconds=10):
    """
    Save a two-panel plot for one lead: raw vs cleaned.
    """
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
    plt.title(f"Cleaned ECG (Wavelet db4 bandpass {LOWCUT_HZ}-{HIGHCUT_HZ} Hz + Z-score) - {lead_name}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (z-score)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_png_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.npz")))
    print(f"Found {len(files)} raw files in: {RAW_DIR}")

    errors = []
    example_saved = False

    for i, path in enumerate(files, start=1):
        base = os.path.splitext(os.path.basename(path))[0]
        print(f"[{i}/{len(files)}] Cleaning: {base}")

        try:
            raw_mat, fs, lead_names = load_npz_leads(path)

            clean_mat = clean_ecg_wavelet(
                raw_mat,
                fs=fs,
                wavelet=WAVELET_NAME,
                lowcut_hz=LOWCUT_HZ,
                highcut_hz=HIGHCUT_HZ
            )

            out_clean = os.path.join(CLEAN_DIR, f"{base}.npz")
            save_npz_leads(out_clean, clean_mat, fs, lead_names)

            # Save one example plot (first successful file)
            if not example_saved:
                out_png = os.path.join(OUTPUTS_DIR, f"{base}_before_after_wavelet_bandpass.png")
                save_before_after_plot(out_png, raw_mat, clean_mat, fs, lead_names, PLOT_LEAD_INDEX, PLOT_SECONDS)
                print(f"Saved example plot to: {out_png}")
                example_saved = True

        except Exception as e:
            errors.append({"file": os.path.basename(path), "error": str(e)})
            print(f"  -> FAILED: {e}")

    if errors:
        err_path = os.path.join(CLEAN_DIR, "errors_cleaning.csv")
        pd.DataFrame(errors).to_csv(err_path, index=False)
        print(f"\nSome files failed. Errors saved to: {err_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
