from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd


# LeadOrder in your XML can be like: "I,2,3" meaning I,II,III
LEAD_MAP = {
    "1": "I",
    "2": "II",
    "3": "III",
    "4": "aVR",
    "5": "aVL",
    "6": "aVF",
    "I": "I",
    "II": "II",
    "III": "III",
    "aVR": "aVR",
    "aVL": "aVL",
    "aVF": "aVF",
    "V1": "V1",
    "V2": "V2",
    "V3": "V3",
    "V4": "V4",
    "V5": "V5",
    "V6": "V6",
}


def norm_lead(x: str) -> str:
    x = (x or "").strip()
    return LEAD_MAP.get(x, x)


def find_first(elem: ET.Element, tag: str) -> ET.Element | None:
    """Find first occurrence of tag anywhere under elem (handles nested XML)."""
    return elem.find(f".//{tag}")


def text_of(elem: ET.Element | None) -> str | None:
    if elem is None or elem.text is None:
        return None
    t = elem.text.strip()
    return t if t else None


def parse_multichannel_waveform(
    *,
    n_channels: int,
    leads: list[str],
    data_str: str,
    resolution_uV_per_lsb: float | None,
) -> dict[str, np.ndarray]:
    """
    Parse comma-separated integers into (samples, channels),
    then return dict lead->array. Convert to mV if resolution is provided (uV/LSB).
    """
    arr = np.fromstring(data_str, sep=",", dtype=np.int32)

    rem = arr.size % n_channels
    if rem != 0:
        arr = arr[: arr.size - rem]

    if arr.size == 0:
        raise ValueError("Waveform data parsed to empty array")

    sig = arr.reshape(-1, n_channels).astype(np.float32)

    if resolution_uV_per_lsb is not None:
        sig = (sig * float(resolution_uV_per_lsb)) / 1000.0  # mV

    signals = {}
    for ch in range(n_channels):
        lead_name = leads[ch] if ch < len(leads) else f"CH{ch+1}"
        signals[lead_name] = sig[:, ch]
    return signals


def parse_full_disclosure(root: ET.Element):
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


def parse_strip(root: ET.Element):
    """
    Best-effort strip parsing (XMLs vary).
    We look for <StripData>. If not found, return None.
    """
    strip_data = find_first(root, "StripData")
    if strip_data is None:
        return None

    data_str = text_of(strip_data)
    if not data_str:
        return None

    # Try to read strip metadata (may or may not exist)
    container = find_first(root, "Strip") or root

    n_channels_str = text_of(find_first(container, "NumberOfChannels"))
    fs_str = text_of(find_first(container, "SampleRate"))
    res_str = text_of(find_first(container, "Resolution"))
    lead_order_str = text_of(find_first(container, "LeadOrder"))

    n_channels = int(n_channels_str) if n_channels_str else 3
    fs_hz = float(fs_str) if fs_str else 0.0
    resolution_uV_per_lsb = float(res_str) if res_str else None

    leads = [norm_lead(x) for x in lead_order_str.split(",")] if lead_order_str else ["I", "II", "III"][:n_channels]
    leads = [x.strip() for x in leads if x.strip()]

    signals = parse_multichannel_waveform(
        n_channels=n_channels,
        leads=leads,
        data_str=data_str,
        resolution_uV_per_lsb=resolution_uV_per_lsb,
    )
    return fs_hz, leads, signals


def save_npz(out_path: Path, fs_hz: float, signals: dict[str, np.ndarray]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, fs=np.array([fs_hz], dtype=np.float32), **signals)


def main():

    xml_dir = Path("Fitness")
    full_dir = Path("parsed_signals_full")
    strip_dir = Path("parsed_signals_strip")


    meta_csv = Path("metadata_all.csv") if Path("metadata_all.csv").exists() else Path("metadata_all")

    if not meta_csv.exists():
        raise SystemExit("Cannot find metadata CSV. Expected 'metadata_all.csv' in repo root.")

    # Read metadata (PID must be string to avoid losing leading zeros)
    meta_df = pd.read_csv(meta_csv, dtype={"PID": str})

    # Ensure columns exist 
    if "full_signal_path" not in meta_df.columns:
        meta_df["full_signal_path"] = ""
    if "strip_signal_path" not in meta_df.columns:
        meta_df["strip_signal_path"] = ""

    # Build mapping PID -> best path(s)
    full_map: dict[str, str] = {}
    strip_map: dict[str, str] = {}

    xml_files = sorted(xml_dir.glob("*.xml"))
    if not xml_files:
        raise SystemExit(f"No XML files found in: {xml_dir.resolve()}")

    for xml_path in xml_files:
        test_id = xml_path.stem
        pid = test_id.split("_")[0]

        # Parse XML root
        try:
            root = ET.parse(xml_path).getroot()
        except Exception as e:
            print(f"[FAIL] XML parse {xml_path.name}: {e}")
            continue

        # ---- Full Disclosure
        try:
            fd = parse_full_disclosure(root)
            if fd is not None:
                fs_hz, leads, signals = fd
                out_path = full_dir / f"{test_id}.npz"
                save_npz(out_path, fs_hz, signals)
                full_map[pid] = str(out_path.as_posix())
                print(f"[FULL OK] {xml_path.name} -> {out_path} leads={list(signals.keys())}")
            else:
                print(f"[FULL SKIP] {xml_path.name}: no FullDisclosure")
        except Exception as e:
            print(f"[FULL FAIL] {xml_path.name}: {e}")

        # ---- Strip
        try:
            st = parse_strip(root)
            if st is not None:
                fs_hz, leads, signals = st
                out_path = strip_dir / f"{test_id}.npz"
                save_npz(out_path, fs_hz, signals)
                strip_map[pid] = str(out_path.as_posix())
                print(f"[STRIP OK] {xml_path.name} -> {out_path} leads={list(signals.keys())}")
            else:
                print(f"[STRIP SKIP] {xml_path.name}: no StripData")
        except Exception as e:
            print(f"[STRIP FAIL] {xml_path.name}: {e}")

    # Update metadata rows by PID
    meta_df["full_signal_path"] = meta_df["PID"].map(full_map).fillna(meta_df["full_signal_path"])
    meta_df["strip_signal_path"] = meta_df["PID"].map(strip_map).fillna(meta_df["strip_signal_path"])

    # Save updated CSV next to original (safer), and also overwrite if you want later
    out_csv = Path("metadata_all_with_signals.csv")
    meta_df.to_csv(out_csv, index=False)
    print(f"\nSaved updated metadata to: {out_csv.resolve()}")




if __name__ == "__main__":
    main()
