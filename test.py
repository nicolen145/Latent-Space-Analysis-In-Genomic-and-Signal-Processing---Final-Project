import pandas as pd

df = pd.read_csv("fft_features/fft_features_windows.csv", dtype={"PID": str})
counts = df.groupby("PID")["segment_id"].nunique().sort_values()
print(counts.head(10))
print(counts.describe())
feat_cols = [c for c in df.columns if c.startswith("bp_")]
print(df[feat_cols].describe(percentiles=[0.01,0.99]).T.sort_values("99%").tail(10))
