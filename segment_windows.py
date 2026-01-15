import os
import glob
import numpy as np
import pandas as pd


# ----------------------------
# Configuration
# ----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(BASE_DIR, "cleaned_signals_full")
OUTPUT_DIR = os.path.join(BASE_DIR, "segmented_10s")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INDEX_CSV = os.path.join(OUTPUT_DIR, "segments_index.csv")
FEATURES_CSV = os.path.join(OUTPUT_DIR, "segments_features.csv")

WINDOW_SEC = 10
OVERLAP = 0.0  # set to 0.5 for 50% overlap


# ----------------------------
# NPZ loading (fs + per-lead arrays)
# ----------------------------

def load_npz_leads(npz_path):
    """
    Load ECG from NPZ where:
      - 'fs' exists
      - each lead is stored as its own 1D array (e.g., I, II, III)
    Returns:
      mat: (T, C)
      fs: float
      lead_names: list[str]
    """
    d = np.load(npz_path, allow_pickle=True)

    if "fs" not in d.files:
        raise ValueError("Missing 'fs' key in NPZ.")

    fs = float(np.array(d["fs"]).reshape(-1)[0])

    lead_keys = [k for k in d.files if k != "fs"]
    lead_keys = [
        k for k in lead_keys
        if isinstance(d[k], np.ndarray) and d[k].ndim == 1 and d[k].size > 30
    ]
    if not lead_keys:
        raise ValueError("No valid lead arrays found.")

    lead_keys = sorted(lead_keys)

    # Align lengths (truncate to minimum)
    lengths = [d[k].shape[0] for k in lead_keys]
    T = min(lengths)

    mat = np.stack([d[k][:T].astype(np.float32) for k in lead_keys], axis=1)
    return mat, fs, lead_keys


# ----------------------------
# Segmentation
# ----------------------------

def segment_signal(mat, fs, window_sec=10, overlap=0.0):
    """
    Split signal into fixed windows of window_sec seconds.
    Returns list of (start_sample, segment_matrix).
    """
    win = int(window_sec * fs)
    if win <= 0:
        raise ValueError("Invalid window size.")

    step = int(win * (1.0 - overlap))
    if step <= 0:
        raise ValueError("Overlap too high (step becomes zero).")

    if mat.shape[0] < win:
        # No segments possible for this file
        return []

    segments = []
    for start in range(0, mat.shape[0] - win + 1, step):
        segments.append((start, mat[start:start + win, :]))
    return segments


def save_segment_npz(out_path, segment_mat, fs, lead_names, pid, segment_id, start_sample):
    """
    Save one segment in NPZ format:
      - fs
      - pid
      - segment_id
      - start_sample
      - per-lead arrays
    """
    payload = {
        "fs": np.array([fs], dtype=np.float32),
        "PID": np.array([pid]),  # make it consistent with CSV (PID uppercase)
        "segment_id": np.array([segment_id]),
        "start_sample": np.array([start_sample], dtype=np.int64),
    }
    for c, name in enumerate(lead_names):
        payload[name] = segment_mat[:, c].astype(np.float32)

    np.savez(out_path, **payload)


# ----------------------------
# Features
# ----------------------------

def dominant_frequency_hz(x, fs):
    """Dominant frequency via FFT peak (ignoring DC)."""
    x = x - np.mean(x)
    n = len(x)
    if n < 4:
        return 0.0

    spec = np.fft.rfft(x)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    if mag.size <= 1:
        return 0.0

    idx = int(np.argmax(mag[1:]) + 1)
    return float(freqs[idx])


def compute_segment_features(seg_mat, fs, lead_names):
    """
    Compute features per lead + mean across leads:
      - rms, std, energy, dominant frequency
    """
    feats = {}
    rms_list, std_list, energy_list, domf_list = [], [], [], []

    for c, name in enumerate(lead_names):
        x = seg_mat[:, c]

        rms = float(np.sqrt(np.mean(x**2)))
        std = float(np.std(x))
        energy = float(np.mean(x**2))
        domf = dominant_frequency_hz(x, fs)

        feats[f"{name}_rms"] = rms
        feats[f"{name}_std"] = std
        feats[f"{name}_energy"] = energy
        feats[f"{name}_domfreq_hz"] = domf

        rms_list.append(rms)
        std_list.append(std)
        energy_list.append(energy)
        domf_list.append(domf)

    feats["rms_mean"] = float(np.mean(rms_list))
    feats["std_mean"] = float(np.mean(std_list))
    feats["energy_mean"] = float(np.mean(energy_list))
    feats["domfreq_mean_hz"] = float(np.mean(domf_list))

    return feats


# ----------------------------
# Main
# ----------------------------

def main():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.npz")))
    print(f"Found {len(files)} cleaned files in: {INPUT_DIR}")

    index_rows = []
    feat_rows = []
    total_segments = 0
    errors = []
    no_segment_files = 0

    for i, path in enumerate(files, start=1):
        base = os.path.splitext(os.path.basename(path))[0]
        pid = base.split("_")[0]

        print(f"[{i}/{len(files)}] Segmenting: {base}")

        file_out_dir = os.path.join(OUTPUT_DIR, base)
        os.makedirs(file_out_dir, exist_ok=True)

        try:
            mat, fs, lead_names = load_npz_leads(path)
            segs = segment_signal(mat, fs, WINDOW_SEC, OVERLAP)

            if not segs:
                no_segment_files += 1
                print("  -> no segments (file shorter than one window)")
                continue

            for j, (start_sample, seg_mat) in enumerate(segs):
                segment_id = f"{base}_seg{j:04d}"
                out_path = os.path.join(file_out_dir, f"{segment_id}.npz")

                save_segment_npz(
                    out_path=out_path,
                    segment_mat=seg_mat,
                    fs=fs,
                    lead_names=lead_names,
                    pid=pid,
                    segment_id=segment_id,
                    start_sample=start_sample
                )

                start_sec = start_sample / fs

                # Index row
                index_rows.append({
                    "file_base": base,
                    "PID": pid,
                    "segment_id": segment_id,
                    "fs": fs,
                    "start_sample": start_sample,
                    "start_sec": start_sec,
                    "end_sec": start_sec + WINDOW_SEC,
                    "window_sec": WINDOW_SEC,
                    "num_leads": seg_mat.shape[1],
                    "num_samples": seg_mat.shape[0],
                    "segment_path": out_path,
                })

                # Features row
                feats = compute_segment_features(seg_mat, fs, lead_names)
                feats.update({
                    "file_base": base,
                    "PID": pid,
                    "segment_id": segment_id,
                    "start_sec": start_sec,
                    "end_sec": start_sec + WINDOW_SEC,
                    "segment_path": out_path,  # useful link back to the segment
                })
                feat_rows.append(feats)

            total_segments += len(segs)
            print(f"  -> created {len(segs)} segments")

        except Exception as e:
            errors.append({"file": os.path.basename(path), "error": str(e)})
            print(f"  -> FAILED: {e}")

    # Save outputs (always)
    df_index = pd.DataFrame(index_rows)
    df_feat = pd.DataFrame(feat_rows)

    df_index.to_csv(INDEX_CSV, index=False)
    df_feat.to_csv(FEATURES_CSV, index=False)

    print("\nDone.")
    print(f"Total segments: {total_segments}")
    print(f"Files with no segments: {no_segment_files}")
    print(f"Index rows: {len(df_index)} | Feature rows: {len(df_feat)}")
    print(f"Index saved to: {INDEX_CSV}")
    print(f"Features saved to: {FEATURES_CSV}")

    if errors:
        err_path = os.path.join(OUTPUT_DIR, "errors_segmentation.csv")
        pd.DataFrame(errors).to_csv(err_path, index=False)
        print(f"Some files failed. Errors saved to: {err_path}")


if __name__ == "__main__":
    main()
