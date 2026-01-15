import pandas as pd

df = pd.read_csv("fft_features/fft_features_windows.csv")

# compute thresholds per lead
thresholds = {}
for lead in df["lead"].unique():
    sub = df[df["lead"] == lead]
    thresholds[lead] = {
        "bp_0.5_4": sub["bp_0.5_4"].quantile(0.99),
        "bp_4_8": sub["bp_4_8"].quantile(0.99),
    }

def is_noisy(row):
    th = thresholds[row["lead"]]
    return (
        row["bp_0.5_4"] > th["bp_0.5_4"]
        or row["bp_4_8"] > th["bp_4_8"]
    )

df["is_noisy"] = df.apply(is_noisy, axis=1)

print(df["is_noisy"].value_counts(normalize=True))
df.to_csv("fft_features/fft_features_windows_with_noise_flag.csv", index=False)

# embedding

clean_df = df[~df["is_noisy"]].copy()

feat_cols = [c for c in clean_df.columns if c.startswith("bp_")]

pid_df = (
    clean_df
    .groupby(["PID", "lead"])[feat_cols]
    .median()
    .reset_index()
)

pid_pivot = pid_df.pivot(index="PID", columns="lead", values=feat_cols)
pid_pivot.columns = [f"{feat}_{lead}" for feat, lead in pid_pivot.columns]
pid_pivot = pid_pivot.reset_index()

pid_pivot.to_csv("fft_features/fft_features_pid_clean.csv", index=False)
print("Saved fft_features_pid_clean.csv")

