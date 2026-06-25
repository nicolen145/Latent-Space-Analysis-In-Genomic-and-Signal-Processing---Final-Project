# ============================================================
# FILTER / CLEAN RAW NPZ SIGNALS - CLUSTER VERSION
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pywt


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path("/sise/nadav-group/nadavrap-group/Final_Project-Nicole_Gal_Lilach")

PREPROCESSING_DIR = BASE_DIR / "Final_Project/Preprocessing"

RAW_DIR = BASE_DIR / "data/parsed_signals_full"
CLEAN_DIR = BASE_DIR / "data/cleaned_signals_full"
OUTPUTS_DIR = PREPROCESSING_DIR / "OUTPUTS"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

WAVELET_NAME = "db4"
LOWCUT_HZ = 0.5
HIGHCUT_HZ = 40.0

PLOT_SECONDS = 10
PLOT_LEAD_INDEX = 0


# ============================================================
# HELPERS
# ============================================================

def load_npz_leads(npz_path):
    d = np.load(npz_path, allow_pickle=True)

    if "fs" not in d.files:
        raise ValueError("Missing 'fs' key in NPZ.")

    fs = float(np.array(d["fs"]).reshape(-1)[0])

    lead_keys = [
        k for k in d.files
        if k != "fs"
        and isinstance(d[k], np.ndarray)
        and d[k].ndim == 1
        and d[k].size > 30
    ]

    if not lead_keys:
        raise ValueError("No valid lead arrays found in NPZ.")

    lead_keys = sorted(lead_keys)

    T = min(d[k].shape[0] for k in lead_keys)

    mat = np.stack(
        [d[k][:T].astype(np.float32) for k in lead_keys],
        axis=1
    )

    return mat, fs, lead_keys


def save_npz_leads(out_path, mat, fs, lead_names):
    payload = {
        "fs": np.array([fs], dtype=np.float32)
    }

    for c, name in enumerate(lead_names):
        payload[name] = mat[:, c].astype(np.float32)

    np.savez_compressed(out_path, **payload)


def pick_level_for_lowcut(fs, lowcut_hz=0.5):
    L = 1

    while (fs / (2 ** (L + 1))) > lowcut_hz and L < 12:
        L += 1

    return L


def wavelet_bandpass_1d(
    x,
    fs,
    wavelet="db4",
    lowcut_hz=0.5,
    highcut_hz=40.0
):
    x = x.astype(np.float32)

    L = pick_level_for_lowcut(fs, lowcut_hz=lowcut_hz)

    coeffs = pywt.wavedec(x, wavelet, level=L)

    new_coeffs = [np.zeros_like(coeffs[0])]

    for idx in range(1, len(coeffs)):
        d = coeffs[idx]

        j = L - (idx - 1)

        band_low = fs / (2 ** (j + 1))
        band_high = fs / (2 ** j)

        overlaps = (
            band_high >= lowcut_hz
            and band_low <= highcut_hz
        )

        if overlaps:
            new_coeffs.append(d)
        else:
            new_coeffs.append(np.zeros_like(d))

    y = pywt.waverec(new_coeffs, wavelet)

    return y[:len(x)].astype(np.float32)


def clean_ecg_wavelet(
    mat,
    fs,
    wavelet="db4",
    lowcut_hz=0.5,
    highcut_hz=40.0
):
    x = mat.astype(np.float32)

    x = x - x.mean(axis=0, keepdims=True)

    out = np.zeros_like(x, dtype=np.float32)

    for ch in range(x.shape[1]):
        out[:, ch] = wavelet_bandpass_1d(
            x[:, ch],
            fs=fs,
            wavelet=wavelet,
            lowcut_hz=lowcut_hz,
            highcut_hz=highcut_hz
        )

    mean = out.mean(axis=0, keepdims=True)
    std = out.std(axis=0, keepdims=True) + 1e-8

    out = (out - mean) / std

    return out


def save_before_after_plot(
    out_png_path,
    raw_mat,
    clean_mat,
    fs,
    lead_names,
    lead_index=0,
    seconds=10
):
    n = min(
        int(seconds * fs),
        raw_mat.shape[0],
        clean_mat.shape[0]
    )

    t = np.arange(n) / fs

    lead_name = (
        lead_names[lead_index]
        if 0 <= lead_index < len(lead_names)
        else f"lead{lead_index}"
    )

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t, raw_mat[:n, lead_index])
    plt.title(f"Raw ECG - Before Filtering - {lead_name}")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t, clean_mat[:n, lead_index])
    plt.title(
        f"Cleaned ECG - Wavelet Filter {LOWCUT_HZ}-{HIGHCUT_HZ} Hz + Z-score - {lead_name}"
    )
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude z-score")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_png_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# MAIN
# ============================================================

def filter_raw_npz():
    files = sorted(RAW_DIR.glob("*.npz"))

    print(f"Found {len(files)} raw NPZ files in:")
    print(RAW_DIR)

    if len(files) == 0:
        raise RuntimeError(f"No NPZ files found in {RAW_DIR}")

    errors = []
    example_saved = False

    for i, path in enumerate(files, start=1):
        base = path.stem

        print(f"[{i}/{len(files)}] Filtering: {base}", flush=True)

        try:
            raw_mat, fs, lead_names = load_npz_leads(path)

            clean_mat = clean_ecg_wavelet(
                raw_mat,
                fs=fs,
                wavelet=WAVELET_NAME,
                lowcut_hz=LOWCUT_HZ,
                highcut_hz=HIGHCUT_HZ
            )

            out_clean = CLEAN_DIR / f"{base}.npz"

            save_npz_leads(
                out_clean,
                clean_mat,
                fs,
                lead_names
            )

            if not example_saved:
                out_png = OUTPUTS_DIR / f"{base}_before_after_filter.png"

                save_before_after_plot(
                    out_png,
                    raw_mat,
                    clean_mat,
                    fs,
                    lead_names,
                    PLOT_LEAD_INDEX,
                    PLOT_SECONDS
                )

                print(f"Saved example plot to: {out_png}", flush=True)

                example_saved = True

        except Exception as e:
            errors.append({
                "file": path.name,
                "error": str(e)
            })

            print(f"FAILED: {e}", flush=True)

    if errors:
        err_path = OUTPUTS_DIR / "errors_cleaning.csv"
        pd.DataFrame(errors).to_csv(err_path, index=False)

        print("\nSome files failed.")
        print(f"Errors saved to: {err_path}")

    print("\nDone filtering.")
    print(f"Clean files saved to: {CLEAN_DIR}")
    print(f"Outputs saved to: {OUTPUTS_DIR}")


if __name__ == "__main__":
    filter_raw_npz()
