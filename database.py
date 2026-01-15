def extract_metadata(root, file_name):
    """
    Extract ONLY the required metadata:
    - PID (from file name)
    - Age
    - Gender
    - Height

    All other fields are intentionally ignored.
    """

    # PID is the first part of the file name
    pid = file_name.split("_")[0]

    meta = {
        "PID": pid,
        "Age": None,
        "Gender": None,
        "Height": None,
    }

    patient_info = root.find("PatientInfo")
    if patient_info is None:
        return meta

    for child in patient_info:
        if child.text is None:
            continue

        tag = child.tag.lower()
        value = child.text.strip()

        if tag == "age":
            meta["Age"] = int(value)

        elif tag == "gender":
            meta["Gender"] = value

        elif tag == "height":
            meta["Height"] = int(float(value))

    return meta
import os
import glob
import base64
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

# ----------------------------
# Configuration
# ----------------------------

# Folder containing your ECG XML files (e.g., 60 files)
# Put your XML files inside a folder named "Fitness" next to this script.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "Fitness")

# Output files/folders (created next to this script)
OUT_METADATA_CSV = os.path.join(BASE_DIR, "metadata_all.csv")
OUT_SIGNALS_DIR = os.path.join(BASE_DIR, "signals_parquet")
OUT_ERRORS_CSV = os.path.join(BASE_DIR, "errors.csv")

os.makedirs(OUT_SIGNALS_DIR, exist_ok=True)


# ----------------------------
# Helpers (XML parsing)
# ----------------------------

def safe_text(elem):
    """Return stripped text if exists, else None."""
    if elem is None or elem.text is None:
        return None
    t = elem.text.strip()
    return t if t != "" else None


def extract_metadata(root, file_name):
    """
    Extract ONLY the required metadata:
    - PID (from file name)
    - Age
    - Gender
    - Height

    Units and all other fields are intentionally ignored.
    """
    # PID is the first part of the file name: PID_xxx_x_x.xml
    pid = file_name.split("_")[0]

    meta = {
        "PID": pid,
        "Age": None,
        "Gender": None,
        "Height": None,
    }

    patient_info = root.find("PatientInfo")
    if patient_info is None:
        return meta

    for child in patient_info:
        tag = child.tag.lower()
        value = safe_text(child)
        if value is None:
            continue

        if tag == "age":
            # Age is typically an integer (e.g., "45")
            try:
                meta["Age"] = int(float(value))
            except Exception:
                meta["Age"] = None

        elif tag == "gender":
            meta["Gender"] = value

        elif tag == "height":
            # Height often comes as number (e.g., "175")
            try:
                meta["Height"] = int(float(value))
            except Exception:
                meta["Height"] = None

    return meta


def decode_fulldisclosure_to_table(root):
    """
    Decode FullDisclosureData (base64) into a tabular DataFrame:
    rows = samples, columns = leads, plus time_sec.

    Assumptions (fits your sample file):
    - FullDisclosure contains:
        NumberOfChannels, SampleRate, Resolution, LeadOrder, FullDisclosureData
    - Raw samples are interleaved int16 little-endian:
        [ch1_s0, ch2_s0, ..., chN_s0, ch1_s1, ch2_s1, ..., chN_s1, ...]
    - If Resolution is like '5 uVperLsb', values are converted to microvolts.
    """
    fd = root.find(".//FullDisclosure")
    if fd is None:
        raise ValueError("Missing <FullDisclosure> section")

    n_ch_txt = safe_text(fd.find("NumberOfChannels"))
    fs_txt = safe_text(fd.find("SampleRate"))
    res_txt = safe_text(fd.find("Resolution")) or ""
    lead_order_txt = safe_text(fd.find("LeadOrder"))

    if not n_ch_txt or not fs_txt:
        raise ValueError("Missing NumberOfChannels or SampleRate")

    n_ch = int(float(n_ch_txt))
    fs = float(fs_txt)

    data_elem = fd.find("FullDisclosureData")
    b64 = safe_text(data_elem)
    if not b64:
        raise ValueError("Missing or empty <FullDisclosureData>")

    # Decode base64 -> bytes -> int16 array
    raw_bytes = base64.b64decode(b64)
    samples = np.frombuffer(raw_bytes, dtype="<i2")  # little-endian int16

    if n_ch <= 0:
        raise ValueError("Invalid NumberOfChannels")

    # Ensure divisible by number of channels (truncate if needed)
    if samples.size % n_ch != 0:
        samples = samples[: (samples.size // n_ch) * n_ch]

    # Reshape into (num_samples, num_channels)
    mat = samples.reshape(-1, n_ch)

    # Build lead names from LeadOrder if possible
    if lead_order_txt:
        leads = [x.strip() for x in lead_order_txt.split(",") if x.strip()]
        if len(leads) != n_ch:
            leads = [f"ch{i+1}" for i in range(n_ch)]
    else:
        leads = [f"ch{i+1}" for i in range(n_ch)]

    df = pd.DataFrame(mat, columns=leads)

    # Add time axis in seconds
    df.insert(0, "time_sec", np.arange(len(df)) / fs)

    # Convert to microvolts if Resolution starts with a numeric factor
    # Example: "5 uVperLsb" => multiply signal columns by 5
    try:
        factor = float(res_txt.split()[0])
        df[leads] = df[leads].astype(np.float32) * factor
        df.attrs["units"] = "microvolts"
    except Exception:
        df.attrs["units"] = "raw_adc"

    return df


# ----------------------------
# Main (batch processing)
# ----------------------------

def main():
    if not os.path.isdir(INPUT_DIR):
        raise FileNotFoundError(
            f"Input folder not found: {INPUT_DIR}\n"
            f"Create it and put your XML files inside it (e.g., 60 files)."
        )

    xml_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.xml")))
    print(f"Found {len(xml_files)} XML files in: {INPUT_DIR}")

    all_meta_rows = []
    errors = []

    for i, path in enumerate(xml_files, start=1):
        file_name = os.path.basename(path)
        print(f"[{i}/{len(xml_files)}] Processing: {file_name}")

        try:
            tree = ET.parse(path)
            root = tree.getroot()

            # 1) Extract minimal metadata (PID, Age, Gender, Height)
            meta = extract_metadata(root, file_name)
            all_meta_rows.append(meta)

            # 2) Extract raw ECG signal table (time_sec + leads)
            signals_df = decode_fulldisclosure_to_table(root)

            # Save per-file signal table as Parquet (efficient for many files)
            out_path = os.path.join(OUT_SIGNALS_DIR, file_name.replace(".xml", ".parquet"))
            signals_df.to_parquet(out_path, index=False)

        except Exception as e:
            errors.append({"file_name": file_name, "error": str(e)})

    # Save all metadata in one CSV
    metadata_df = pd.DataFrame(all_meta_rows)
    metadata_df.to_csv(OUT_METADATA_CSV, index=False)
    print(f"\nMetadata saved to: {OUT_METADATA_CSV}")

    # Save errors (if any)
    if errors:
        pd.DataFrame(errors).to_csv(OUT_ERRORS_CSV, index=False)
        print(f"Some files failed. Errors saved to: {OUT_ERRORS_CSV}")
    else:
        print("No errors found.")

    print(f"Signals saved to folder: {OUT_SIGNALS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
