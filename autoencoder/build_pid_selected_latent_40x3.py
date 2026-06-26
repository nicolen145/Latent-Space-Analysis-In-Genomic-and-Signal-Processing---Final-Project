import pandas as pd
import numpy as np

# =========================================================
# CONFIG
# =========================================================

IN_CSV = "/sise/nadav-group/nadavrap-group/Final_Project-Nicole_Gal_Lilach/Final_Project/Final_AE/all_segments_latent_40.csv"

OUT_CSV = "/sise/nadav-group/nadavrap-group/Final_Project-Nicole_Gal_Lilach/Final_Project/Final_AE/pid_selected_latent_120.csv"

Z_COLS = [f"z{i}" for i in range(1, 41)]


# =========================================================
# LOAD
# =========================================================

print("Loading CSV...")

df = pd.read_csv(
    IN_CSV,
    dtype={"PID": str},
    low_memory=False
)

print("Shape:", df.shape)

df["phase"] = df["phase"].astype(str)

df["start_sec"] = pd.to_numeric(df["start_sec"], errors="coerce")
df["end_sec"] = pd.to_numeric(df["end_sec"], errors="coerce")


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def select_pretest(g):

    pre = g[g["phase"].str.lower() == "pretest"].copy()

    if pre.empty:
        return None

    pre = pre.sort_values("start_sec")

    # take middle row
    return pre.iloc[len(pre) // 2]


def select_exercise_30s_before_end(g):

    ex = g[g["phase"].str.lower() == "exercise"].copy()

    if ex.empty:
        return None

    ex = ex.sort_values("start_sec")

    exercise_end = ex["end_sec"].max()

    target_time = exercise_end - 30

    ex["dist_to_target"] = (
        ex["start_sec"] - target_time
    ).abs()

    return ex.loc[ex["dist_to_target"].idxmin()]


def select_rest_middle(g):

    rest = g[g["phase"].str.lower() == "rest"].copy()

    if rest.empty:
        return None

    rest = rest.sort_values("start_sec")

    rest_start = rest["start_sec"].min()
    rest_end = rest["end_sec"].max()

    target_time = (rest_start + rest_end) / 2

    rest["dist_to_target"] = (
        rest["start_sec"] - target_time
    ).abs()

    return rest.loc[rest["dist_to_target"].idxmin()]


# =========================================================
# BUILD PID ROW
# =========================================================

def build_pid_row(pid, g):

    row = {"PID": pid}

    selections = {
        "pretest": select_pretest(g),
        "exercise": select_exercise_30s_before_end(g),
        "rest": select_rest_middle(g)
    }

    for prefix, selected in selections.items():

        if selected is None:

            for z in Z_COLS:
                row[f"{prefix}_{z}"] = np.nan

        else:

            for z in Z_COLS:
                row[f"{prefix}_{z}"] = selected[z]

    return row


# =========================================================
# PROCESS
# =========================================================

print("\nBuilding PID table...")

rows = []

total_pids = df["PID"].nunique()

for i, (pid, g) in enumerate(df.groupby("PID")):

    if i % 100 == 0:
        print(f"Processed {i}/{total_pids} PIDs")

    rows.append(build_pid_row(pid, g))

out_df = pd.DataFrame(rows)

print("\nFinal shape:", out_df.shape)


# =========================================================
# SAVE
# =========================================================

print("\nSaving CSV...")

out_df.to_csv(OUT_CSV, index=False)

print("\nSaved to:")
print(OUT_CSV)

print("\nDONE")