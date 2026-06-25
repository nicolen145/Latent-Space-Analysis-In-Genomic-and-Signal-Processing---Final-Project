# ============================================================
# EXTRACT XML FROM ZIPs TO NPZ + TREND DATA - CLUSTER VERSION
# Incremental version: appends CSVs and avoids duplicates
# ============================================================

from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd


BASE_DIR = Path("/sise/nadav-group/nadavrap-group/Final_Project-Nicole_Gal_Lilach")

ZIP_DIR = BASE_DIR / "data/Fitness_full"
OUT_DATA_DIR = BASE_DIR / "data"

full_dir = OUT_DATA_DIR / "parsed_signals_full"
strip_dir = OUT_DATA_DIR / "parsed_signals_strip"

out_trend_csv = OUT_DATA_DIR / "trend_timeseries_all.csv"
out_metadata_csv = OUT_DATA_DIR / "metadata_all_with_signals.csv"
out_errors_csv = OUT_DATA_DIR / "extract_errors.csv"

full_dir.mkdir(parents=True, exist_ok=True)
strip_dir.mkdir(parents=True, exist_ok=True)

# Put zip names here if you want to run only specific files.
# Example: ["fitness_11.zip", "fitness_12.zip"]
# If empty, it will run on all fitness_*.zip files.
ZIP_NAMES_TO_PROCESS = []

SKIP_EXISTING_NPZ = True


LEAD_MAP = {
    "1": "I", "2": "II", "3": "III",
    "4": "aVR", "5": "aVL", "6": "aVF",
    "I": "I", "II": "II", "III": "III",
    "aVR": "aVR", "aVL": "aVL", "aVF": "aVF",
    "V1": "V1", "V2": "V2", "V3": "V3",
    "V4": "V4", "V5": "V5", "V6": "V6",
}


def norm_lead(x):
    x = (x or "").strip()
    return LEAD_MAP.get(x, x)


def find_first(elem, tag):
    return elem.find(f".//{tag}")


def text_of(elem):
    if elem is None or elem.text is None:
        return None
    t = elem.text.strip()
    return t if t else None


def safe_float(elem):
    text = text_of(elem)
    try:
        return float(text) if text is not None else None
    except ValueError:
        return None


def extract_trend_timeseries(root, file_name, source_zip):
    file_base = Path(file_name).stem
    rows = []

    trend = root.find(".//TrendData")
    if trend is None:
        return rows

    for entry in trend.findall(".//TrendEntry"):
        t_min = safe_float(entry.find(".//EntryTime/Minute")) or 0
        t_sec = safe_float(entry.find(".//EntryTime/Second")) or 0

        rows.append({
            "file_base": file_base,
            "source_zip": source_zip,
            "time_sec": t_min * 60 + t_sec,
            "phase": text_of(entry.find(".//PhaseName")),
            "speed_rpm": safe_float(entry.find(".//Grade")),
            "load_W": safe_float(entry.find(".//Load")),
            "mets": safe_float(entry.find(".//Mets")),
            "heart_rate": safe_float(entry.find(".//HeartRate")),
        })

    return rows


def parse_multichannel_waveform(n_channels, leads, data_str, resolution_uV_per_lsb):
    arr = np.fromstring(data_str, sep=",", dtype=np.int32)

    rem = arr.size % n_channels
    if rem != 0:
        arr = arr[: arr.size - rem]

    if arr.size == 0:
        raise ValueError("Waveform data parsed to empty array")

    sig = arr.reshape(-1, n_channels).astype(np.float32)

    if resolution_uV_per_lsb is not None:
        sig = (sig * float(resolution_uV_per_lsb)) / 1000.0

    signals = {}
    for ch in range(n_channels):
        lead_name = leads[ch] if ch < len(leads) else f"CH{ch + 1}"
        signals[lead_name] = sig[:, ch]

    return signals


def parse_full_disclosure(root):
    fd = find_first(root, "FullDisclosure")
    if fd is None:
        return None

    n_channels = int(text_of(find_first(fd, "NumberOfChannels")) or "0")
    fs_hz = float(text_of(find_first(fd, "SampleRate")) or "0")
    lead_order = text_of(find_first(fd, "LeadOrder")) or ""
    data_str = text_of(find_first(fd, "FullDisclosureData")) or ""
    res_str = text_of(find_first(fd, "Resolution"))

    resolution_uV_per_lsb = float(res_str) if res_str is not None else None

    if n_channels <= 0 or fs_hz <= 0 or not lead_order or not data_str:
        raise ValueError("FullDisclosure exists but missing required fields")

    leads = [norm_lead(x) for x in lead_order.split(",") if x.strip()]

    signals = parse_multichannel_waveform(
        n_channels=n_channels,
        leads=leads,
        data_str=data_str,
        resolution_uV_per_lsb=resolution_uV_per_lsb,
    )

    return fs_hz, leads, signals


def parse_strip(root):
    strip_data = find_first(root, "StripData")
    if strip_data is None:
        return None

    data_str = text_of(strip_data)
    if not data_str:
        return None

    container = find_first(root, "Strip") or root

    n_channels_str = text_of(find_first(container, "NumberOfChannels"))
    fs_str = text_of(find_first(container, "SampleRate"))
    res_str = text_of(find_first(container, "Resolution"))
    lead_order_str = text_of(find_first(container, "LeadOrder"))

    n_channels = int(n_channels_str) if n_channels_str else 3
    fs_hz = float(fs_str) if fs_str else 0.0
    resolution_uV_per_lsb = float(res_str) if res_str else None

    leads = (
        [norm_lead(x) for x in lead_order_str.split(",")]
        if lead_order_str
        else ["I", "II", "III"][:n_channels]
    )

    leads = [x.strip() for x in leads if x.strip()]

    signals = parse_multichannel_waveform(
        n_channels=n_channels,
        leads=leads,
        data_str=data_str,
        resolution_uV_per_lsb=resolution_uV_per_lsb,
    )

    return fs_hz, leads, signals


def save_npz(out_path, fs_hz, signals):
    np.savez_compressed(
        out_path,
        fs=np.array([fs_hz], dtype=np.float32),
        **signals
    )


def append_without_duplicates(new_df, out_csv, subset_cols):
    if new_df.empty:
        return pd.DataFrame()

    if out_csv.exists():
        old_df = pd.read_csv(out_csv, dtype=str)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df.copy()

    combined = combined.drop_duplicates(
        subset=subset_cols,
        keep="last"
    )

    combined.to_csv(out_csv, index=False)
    return combined


def get_zip_files():
    if ZIP_NAMES_TO_PROCESS:
        zip_files = [ZIP_DIR / name for name in ZIP_NAMES_TO_PROCESS]
        missing = [str(p) for p in zip_files if not p.exists()]
        if missing:
            raise RuntimeError(f"Missing ZIP files: {missing}")
        return zip_files

    return sorted(ZIP_DIR.glob("fitness_*.zip"))


def extract_all_zips():
    zip_files = get_zip_files()

    if not zip_files:
        raise RuntimeError(f"No ZIP files found in: {ZIP_DIR}")

    print(f"ZIP files to process: {len(zip_files)}")

    metadata_rows = []
    trend_rows = []
    errors = []

    for zip_i, zip_path in enumerate(zip_files, start=1):
        source_zip = zip_path.stem

        print("\n" + "=" * 70)
        print(f"[ZIP {zip_i}/{len(zip_files)}] {zip_path.name}")
        print("=" * 70)

        with zipfile.ZipFile(zip_path, "r") as z:
            xml_files = [
                f for f in z.namelist()
                if f.lower().endswith(".xml")
            ]

            print(f"XML files found: {len(xml_files)}")

            for i, xml_in_zip in enumerate(xml_files, start=1):
                file_name = Path(xml_in_zip).name
                test_id = Path(file_name).stem
                pid = test_id.split("_")[0]

                if i % 100 == 0:
                    print(f"[{i}/{len(xml_files)}] Processing...", flush=True)

                full_signal_path = ""
                strip_signal_path = ""

                try:
                    with z.open(xml_in_zip) as f:
                        root = ET.parse(f).getroot()
                except Exception as e:
                    errors.append({
                        "source_zip": source_zip,
                        "file": file_name,
                        "stage": "xml_parse",
                        "error": str(e)
                    })
                    continue

                try:
                    rows = extract_trend_timeseries(root, file_name, source_zip)
                    trend_rows.extend(rows)
                except Exception as e:
                    errors.append({
                        "source_zip": source_zip,
                        "file": file_name,
                        "stage": "trend",
                        "error": str(e)
                    })

                try:
                    full_out_path = full_dir / f"{test_id}.npz"

                    if SKIP_EXISTING_NPZ and full_out_path.exists():
                        full_signal_path = str(full_out_path)
                    else:
                        fd = parse_full_disclosure(root)

                        if fd is not None:
                            fs_hz, leads, signals = fd
                            save_npz(full_out_path, fs_hz, signals)
                            full_signal_path = str(full_out_path)

                except Exception as e:
                    errors.append({
                        "source_zip": source_zip,
                        "file": file_name,
                        "stage": "full_disclosure",
                        "error": str(e)
                    })

                try:
                    strip_out_path = strip_dir / f"{test_id}.npz"

                    if SKIP_EXISTING_NPZ and strip_out_path.exists():
                        strip_signal_path = str(strip_out_path)
                    else:
                        st = parse_strip(root)

                        if st is not None:
                            fs_hz, leads, signals = st
                            save_npz(strip_out_path, fs_hz, signals)
                            strip_signal_path = str(strip_out_path)

                except Exception as e:
                    errors.append({
                        "source_zip": source_zip,
                        "file": file_name,
                        "stage": "strip",
                        "error": str(e)
                    })

                metadata_rows.append({
                    "PID": pid,
                    "file_base": test_id,
                    "source_zip": source_zip,
                    "full_signal_path": full_signal_path,
                    "strip_signal_path": strip_signal_path
                })

    metadata_df = pd.DataFrame(metadata_rows)
    trend_df = pd.DataFrame(trend_rows)
    errors_df = pd.DataFrame(errors)

    final_metadata = append_without_duplicates(
        metadata_df,
        out_metadata_csv,
        subset_cols=["file_base"]
    )

    final_trend = append_without_duplicates(
        trend_df,
        out_trend_csv,
        subset_cols=["file_base", "time_sec"]
    )

    if len(errors_df) > 0:
        if out_errors_csv.exists():
            old_errors = pd.read_csv(out_errors_csv, dtype=str)
            errors_df = pd.concat([old_errors, errors_df], ignore_index=True)

        errors_df.to_csv(out_errors_csv, index=False)

    print("\nDone.")
    print(f"Metadata saved to: {out_metadata_csv}")
    print(f"TrendData saved to: {out_trend_csv}")
    print(f"Full signals saved to: {full_dir}")
    print(f"Strip signals saved to: {strip_dir}")
    print(f"New metadata rows: {len(metadata_df)}")
    print(f"Total metadata rows: {len(final_metadata)}")
    print(f"New TrendData rows: {len(trend_df)}")
    print(f"Total TrendData rows: {len(final_trend)}")
    print(f"New errors: {len(errors_df)}")

    if len(final_trend) > 0 and "phase" in final_trend.columns:
        print("\nPhase distribution:")
        print(final_trend["phase"].value_counts(dropna=False))


if __name__ == "__main__":
    extract_all_zips()
