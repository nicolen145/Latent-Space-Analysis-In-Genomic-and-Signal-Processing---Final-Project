import pandas as pd
from pathlib import Path


def main():
    in_csv = Path("metadata_all_with_signals.csv")
    out_csv = Path("metedata_all_full_only.csv")

    if not in_csv.exists():
        raise SystemExit(f"Input CSV not found: {in_csv.resolve()}")

    # Read CSV
    df = pd.read_csv(in_csv, dtype={"PID": str})

    # 1) Drop strip_signal_path column if exists
    if "strip_signal_path" in df.columns:
        df = df.drop(columns=["strip_signal_path"])

    # 2) Keep only rows with non-empty full_signal_path
    df = df[df["full_signal_path"].notna()]
    df = df[df["full_signal_path"].astype(str).str.strip() != ""]

    # Optional: reset index
    df = df.reset_index(drop=True)

    # Save cleaned CSV
    df.to_csv(out_csv, index=False)

    print(f"Saved cleaned metadata to: {out_csv.resolve()}")
    print(f"Rows kept: {len(df)}")


if __name__ == "__main__":
    main()
