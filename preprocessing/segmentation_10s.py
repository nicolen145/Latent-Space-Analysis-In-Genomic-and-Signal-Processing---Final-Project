# ============================================================
# SEGMENT CLEANED ECG SIGNALS + FEATURES + TREND MATCH
# CLUSTER VERSION
# ============================================================

from pathlib import Path
import time
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path("/sise/nadav-group/nadavrap-group/Final_Project-Nicole_Gal_Lilach")

DATA_DIR = BASE_DIR / "data"

INPUT_DIR = DATA_DIR / "cleaned_signals_full"
OUTPUT_DIR = DATA_DIR / "segmented_10s"

TREND_CSV = DATA_DIR / "trend_timeseries_all.csv"

INDEX_CSV = OUTPUT_DIR / "segments_index.csv"
FEATURES_CSV = OUTPUT_DIR / "segments_features.csv"
SEGMENTS_WITH_TREND_CSV = OUTPUT_DIR / "segments_with_trend.csv"
ERRORS_CSV = OUTPUT_DIR / "errors_segmentation.csv"

WINDOW_SEC = 10
OVERLAP = 0.0

SKIP_EXISTING_SEGMENTS = True

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD NPZ
# ============================================================

def load_npz_leads(npz_path):
    d = np.load(npz_path, allow_pickle=True)

    if "fs" not in d.files:
        raise ValueError("Missing 'fs' key in NPZ.")

    fs = float(np.array(d["fs"]).reshape(-1)[0])

    metadata_keys = {
        "fs", "PID", "pid", "segment_id",
        "start_sample", "start_sec", "end_sec"
    }

    lead_keys = [
        k for k in d.files
        if k not in metadata_keys
        and isinstance(d[k], np.ndarray)
        and d[k].ndim == 1
        and d[k].size > 30
    ]

    if not lead_keys:
        raise ValueError("No valid lead arrays found.")

    lead_keys = sorted(lead_keys)

    T = min(d[k].shape[0] for k in lead_keys)

    mat = np.stack(
        [d[k][:T].astype(np.float32) for k in lead_keys],
        axis=1
    )

    return mat, fs, lead_keys


# ============================================================
# SEGMENTATION
# ============================================================

def segment_signal(mat, fs, window_sec=10, overlap=0.0):
    win = int(window_sec * fs)

    if win <= 0:
        raise ValueError("Invalid window size.")

    step = int(win * (1.0 - overlap))

    if step <= 0:
        raise ValueError("Overlap too high.")

    if mat.shape[0] < win:
        return []

    segments = []

    for start in range(0, mat.shape[0] - win + 1, step):
        segments.append((start, mat[start:start + win, :]))

    return segments


def save_segment_npz(out_path, segment_mat, fs, lead_names, pid, segment_id, start_sample):
    payload = {
        "fs": np.array([fs], dtype=np.float32),
        "PID": np.array([pid]),
        "segment_id": np.array([segment_id]),
        "start_sample": np.array([start_sample], dtype=np.int64),
    }

    for c, name in enumerate(lead_names):
        payload[name] = segment_mat[:, c].astype(np.float32)

    np.savez_compressed(out_path, **payload)


# ============================================================
# FEATURES
# ============================================================

def dominant_frequency_hz(x, fs):
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
    feats = {}

    rms_list = []
    std_list = []
    energy_list = []
    domf_list = []

    for c, name in enumerate(lead_names):
        x = seg_mat[:, c]

        rms = float(np.sqrt(np.mean(x ** 2)))
        std = float(np.std(x))
        energy = float(np.mean(x ** 2))
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


# ============================================================
# TREND DATA MATCHING
# ============================================================

def load_trend_lookup():
    if not TREND_CSV.exists():
        print(f"Trend CSV not found, skipping trend matching: {TREND_CSV}")
        return None

    trend_df = pd.read_csv(TREND_CSV, dtype={"file_base": str})

    trend_df["file_base"] = trend_df["file_base"].astype(str).str.strip()
    trend_df["time_sec"] = pd.to_numeric(trend_df["time_sec"], errors="coerce")

    trend_df = trend_df.dropna(subset=["file_base", "time_sec"])

    trend_by_file = {
        file_base: group.sort_values("time_sec").reset_index(drop=True)
        for file_base, group in trend_df.groupby("file_base")
    }

    print(f"Loaded TrendData for {len(trend_by_file)} files")

    return trend_by_file


def get_nearest_trend(trend_by_file, file_base, start_sec, end_sec):
    empty = {
        "phase": None,
        "speed_rpm": np.nan,
        "load_W": np.nan,
        "mets": np.nan,
        "heart_rate": np.nan,
        "trend_time_sec": np.nan,
        "trend_time_diff_sec": np.nan,
    }

    if trend_by_file is None:
        return empty

    if file_base not in trend_by_file:
        return empty

    sub = trend_by_file[file_base]

    if sub.empty:
        return empty

    mid_sec = (start_sec + end_sec) / 2.0

    idx = (sub["time_sec"] - mid_sec).abs().idxmin()
    row = sub.loc[idx]

    return {
        "phase": row.get("phase"),
        "speed_rpm": row.get("speed_rpm"),
        "load_W": row.get("load_W"),
        "mets": row.get("mets"),
        "heart_rate": row.get("heart_rate"),
        "trend_time_sec": row.get("time_sec"),
        "trend_time_diff_sec": abs(row.get("time_sec") - mid_sec),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    files = sorted(INPUT_DIR.glob("*.npz"))

    print("Input dir:", INPUT_DIR)
    print("Output dir:", OUTPUT_DIR)
    print(f"Found cleaned NPZ files: {len(files)}")

    if len(files) == 0:
        raise RuntimeError(f"No NPZ files found in {INPUT_DIR}")

    trend_by_file = load_trend_lookup()

    index_rows = []
    feature_rows = []
    trend_rows = []
    errors = []

    total_segments = 0
    no_segment_files = 0

    start_time = time.time()

    for i, path in enumerate(files, start=1):
        base = path.stem
        pid = base.split("_")[0]

        print(f"[{i}/{len(files)}] Segmenting: {base}", flush=True)

        file_out_dir = OUTPUT_DIR / base
        file_out_dir.mkdir(parents=True, exist_ok=True)

        try:
            mat, fs, lead_names = load_npz_leads(path)

            segments = segment_signal(
                mat,
                fs,
                window_sec=WINDOW_SEC,
                overlap=OVERLAP
            )

            if not segments:
                no_segment_files += 1
                print("  -> no segments", flush=True)
                continue

            for j, (start_sample, seg_mat) in enumerate(segments):
                segment_id = f"{base}_seg{j:04d}"
                out_path = file_out_dir / f"{segment_id}.npz"

                if not (SKIP_EXISTING_SEGMENTS and out_path.exists()):
                    save_segment_npz(
                        out_path,
                        seg_mat,
                        fs,
                        lead_names,
                        pid,
                        segment_id,
                        start_sample
                    )

                start_sec = start_sample / fs
                end_sec = start_sec + WINDOW_SEC

                index_row = {
                    "file_base": base,
                    "PID": pid,
                    "segment_id": segment_id,
                    "fs": fs,
                    "start_sample": start_sample,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "window_sec": WINDOW_SEC,
                    "num_leads": seg_mat.shape[1],
                    "num_samples": seg_mat.shape[0],
                    "segment_path": str(out_path),
                }

                index_rows.append(index_row)

                feats = compute_segment_features(seg_mat, fs, lead_names)

                feats.update({
                    "file_base": base,
                    "PID": pid,
                    "segment_id": segment_id,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "segment_path": str(out_path),
                })

                feature_rows.append(feats)

                trend_info = get_nearest_trend(
                    trend_by_file,
                    base,
                    start_sec,
                    end_sec
                )

                trend_row = index_row.copy()
                trend_row.update(trend_info)

                trend_rows.append(trend_row)

            total_segments += len(segments)

            print(f"  -> created/found {len(segments)} segments", flush=True)

        except Exception as e:
            errors.append({
                "file": path.name,
                "error": str(e)
            })

            print(f"  -> FAILED: {e}", flush=True)

        if i % 50 == 0 or i == len(files):
            elapsed = time.time() - start_time
            progress = 100 * i / len(files)
            avg_per_file = elapsed / i
            eta_sec = (len(files) - i) * avg_per_file

            print(
                f"Progress: {i}/{len(files)} "
                f"({progress:.1f}%) | "
                f"Elapsed: {elapsed / 60:.1f} min | "
                f"ETA: {eta_sec / 60:.1f} min | "
                f"Errors: {len(errors)}",
                flush=True
            )

    df_index = pd.DataFrame(index_rows)
    df_features = pd.DataFrame(feature_rows)
    df_trend = pd.DataFrame(trend_rows)

    df_index.to_csv(INDEX_CSV, index=False)
    df_features.to_csv(FEATURES_CSV, index=False)
    df_trend.to_csv(SEGMENTS_WITH_TREND_CSV, index=False)

    print("\nDone.")
    print(f"Total segments: {total_segments}")
    print(f"Files with no segments: {no_segment_files}")
    print(f"Index rows: {len(df_index)}")
    print(f"Feature rows: {len(df_features)}")
    print(f"Segments with Trend rows: {len(df_trend)}")

    print("\nSaved:")
    print(INDEX_CSV)
    print(FEATURES_CSV)
    print(SEGMENTS_WITH_TREND_CSV)

    if errors:
        pd.DataFrame(errors).to_csv(ERRORS_CSV, index=False)
        print(f"Errors saved to: {ERRORS_CSV}")


if __name__ == "__main__":
    main()
