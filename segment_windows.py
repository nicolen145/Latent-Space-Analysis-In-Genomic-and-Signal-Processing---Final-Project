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
    overlap: 0.0 -> no overlap, 0.5 -> 50% overlap, etc.
    Returns list of (start_sample, segment_matrix).
    """
    win = int(window_sec * fs)
    if win <= 0:
        raise ValueError("Invalid window size.")

    step = int(win * (1.0 - overlap))
    if step <= 0:
        raise ValueError("Overlap too high (step becomes zero).")

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
        "pid": np.array([pid]),
        "segment_id": np.array([segment_id]),
        "start_sample": np.array([start_sample], dtype=np.int64),
    }
    for c, name in enumerate(lead_names):
        payload[name] = segment_mat[:, c].astype(np.float32)

    np.savez(out_path, **payload)


# ----------------------------
# Main
# ----------------------------

def main():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.npz")))
    print(f"Found {len(files)} cleaned files in: {INPUT_DIR}")

    index_rows = []
    total_segments = 0
    errors = []

    for i, path in enumerate(files, start=1):
        base = os.path.splitext(os.path.basename(path))[0]  # e.g., "1002688_6025_0_0"
        pid = base.split("_")[0]

        print(f"[{i}/{len(files)}] Segmenting: {base}")

        # Create a subfolder per original file
        file_out_dir = os.path.join(OUTPUT_DIR, base)
        os.makedirs(file_out_dir, exist_ok=True)

        try:
            mat, fs, lead_names = load_npz_leads(path)
            segs = segment_signal(mat, fs, WINDOW_SEC, OVERLAP)

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

                index_rows.append({
                    "file_base": base,
                    "PID": pid,
                    "segment_id": segment_id,
                    "fs": fs,
                    "start_sample": start_sample,
                    "start_sec": start_sample / fs,
                    "end_sec": (start_sample / fs) + WINDOW_SEC,
                    "window_sec": WINDOW_SEC,
                    "num_leads": seg_mat.shape[1],
                    "num_samples": seg_mat.shape[0],
                    "segment_path": out_path,
                })

            total_segments += len(segs)
            print(f"  -> created {len(segs)} segments")

        except Exception as e:
            errors.append({"file": os.path.basename(path), "error": str(e)})
            print(f"  -> FAILED: {e}")

    pd.DataFrame(index_rows).to_csv(INDEX_CSV, index=False)
    print(f"\nDone. Total segments: {total_segments}")
    print(f"Index saved to: {INDEX_CSV}")

    if errors:
        err_path = os.path.join(OUTPUT_DIR, "errors_segmentation.csv")
        pd.DataFrame(errors).to_csv(err_path, index=False)
        print(f"Some files failed. Errors saved to: {err_path}")


if __name__ == "__main__":
    main()
