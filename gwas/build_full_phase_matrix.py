import os
import numpy as np
import pandas as pd

# ==================================================
# PATHS
# ==================================================

GWAS_DIR = (
    "/sise/nadav-group/nadavrap-group/"
    "Final_Project-Nicole_Gal_Lilach/"
    "Final_Project/GWAS"
)

RESULTS_DIR = os.path.join(GWAS_DIR, "GWAS_results")

OUT_DIR = os.path.join(GWAS_DIR, "GWAS_full_phase_matrix")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_MATRIX = os.path.join(
    OUT_DIR,
    "gwas_all_chr_all_z_all_phases_matrix.csv.gz"
)

OUT_SUMMARY = os.path.join(
    OUT_DIR,
    "gwas_full_phase_matrix_summary_by_chr_z.csv"
)

OUT_INTERESTING = os.path.join(
    OUT_DIR,
    "gwas_interesting_phase_changes.csv"
)

# Optional folder with smaller files per z and chromosome
PER_CHR_Z_DIR = os.path.join(OUT_DIR, "by_chr_z")
os.makedirs(PER_CHR_Z_DIR, exist_ok=True)

# ==================================================
# SETTINGS
# ==================================================

PHASES = ["pretest", "exercise", "rest"]
Z_RANGE = range(1, 41)
CHROM_RANGE = range(1, 23)

GW_SIG = 5e-8
SUGGESTIVE = 1e-5

# Difference in -log10(P) that we consider interesting
DELTA_LOGP_THRESHOLD = 3

# Save also smaller files for each z+chromosome
SAVE_PER_CHR_Z = True


# ==================================================
# FUNCTIONS
# ==================================================

def get_file_path(phase, z, chrom):
    """
    Example:
    GWAS_results/rest/gwas_chr22_rest.rest_z21.glm.linear
    """
    phenotype = f"{phase}_z{z}"

    return os.path.join(
        RESULTS_DIR,
        phase,
        f"gwas_chr{chrom}_{phase}.{phenotype}.glm.linear"
    )


def read_glm_file(file_path, phase):
    """
    Reads one PLINK .glm.linear file.
    Keeps all original PLINK columns and adds phase suffix.
    Shared columns CHROM, POS, ID, SNP_KEY stay without suffix.
    """

    df = pd.read_csv(file_path, sep="\t", low_memory=False)

    if "#CHROM" in df.columns:
        df = df.rename(columns={"#CHROM": "CHROM"})

    # Keep only additive test if TEST column exists
    if "TEST" in df.columns:
        df = df[df["TEST"] == "ADD"].copy()

    required = ["CHROM", "POS", "ID"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns {missing} in file: {file_path}")

    df["CHROM"] = df["CHROM"].astype(str)
    df["POS"] = pd.to_numeric(df["POS"], errors="coerce")

    df = df.dropna(subset=["CHROM", "POS", "ID"]).copy()

    df["SNP_KEY"] = (
        df["CHROM"].astype(str)
        + ":"
        + df["POS"].astype(int).astype(str)
        + ":"
        + df["ID"].astype(str)
    )

    # Convert known numeric columns if they exist
    numeric_cols = [
        "A1_FREQ",
        "OBS_CT",
        "BETA",
        "SE",
        "T_STAT",
        "P"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep CHROM/POS/ID/SNP_KEY shared.
    shared_cols = ["SNP_KEY", "CHROM", "POS", "ID"]

    cols_to_suffix = [c for c in df.columns if c not in shared_cols]

    rename_dict = {
        c: f"{c}_{phase}"
        for c in cols_to_suffix
    }

    df = df.rename(columns=rename_dict)

    return df


def safe_logp(p_series):
    """
    Compute -log10(P), safely handling missing/zero values.
    """
    p = pd.to_numeric(p_series, errors="coerce")
    p = p.where(p > 0, np.nan)
    return -np.log10(p)


def add_phase_comparison_columns(df):
    """
    Adds logP, significance flags, phase differences, gain/loss columns.
    """

    # logP and significance
    for phase in PHASES:
        p_col = f"P_{phase}"

        if p_col not in df.columns:
            df[f"logP_{phase}"] = np.nan
            df[f"significant_{phase}"] = False
            df[f"suggestive_{phase}"] = False
            continue

        df[f"logP_{phase}"] = safe_logp(df[p_col])
        df[f"significant_{phase}"] = df[p_col] < GW_SIG
        df[f"suggestive_{phase}"] = df[p_col] < SUGGESTIVE

    # Significance pattern
    def sig_pattern(row):
        sig_phases = [
            phase for phase in PHASES
            if bool(row.get(f"significant_{phase}", False))
        ]

        if len(sig_phases) == 0:
            return "none"
        if len(sig_phases) == 3:
            return "all_phases"

        return "_and_".join(sig_phases)

    df["sig_pattern"] = df.apply(sig_pattern, axis=1)

    comparisons = [
        ("exercise", "pretest"),
        ("rest", "pretest"),
        ("rest", "exercise"),
    ]

    for phase_a, phase_b in comparisons:
        label = f"{phase_a}_vs_{phase_b}"

        # Raw P difference
        df[f"P_diff_{label}"] = df[f"P_{phase_a}"] - df[f"P_{phase_b}"]

        # More useful P comparison:
        # positive means phase_a has stronger significance
        df[f"delta_logP_{label}"] = (
            df[f"logP_{phase_a}"] - df[f"logP_{phase_b}"]
        )

        # BETA differences
        df[f"BETA_diff_{label}"] = (
            df[f"BETA_{phase_a}"] - df[f"BETA_{phase_b}"]
        )

        df[f"abs_BETA_diff_{label}"] = df[f"BETA_diff_{label}"].abs()

        # Direction flip in effect
        df[f"BETA_direction_flip_{label}"] = (
            np.sign(df[f"BETA_{phase_a}"]) != np.sign(df[f"BETA_{phase_b}"])
        )

        # Significant gain/loss
        df[f"sig_gain_{label}"] = (
            (~df[f"significant_{phase_b}"]) & (df[f"significant_{phase_a}"])
        )

        df[f"sig_loss_{label}"] = (
            (df[f"significant_{phase_b}"]) & (~df[f"significant_{phase_a}"])
        )

    return df


def reorder_columns(df):
    """
    Put the most important columns first.
    All other original PLINK columns remain after them.
    """

    first_cols = [
        "z",
        "CHROM",
        "POS",
        "ID",
        "SNP_KEY",

        # Main PLINK outputs by phase
        "A1_pretest",
        "A1_FREQ_pretest",
        "OBS_CT_pretest",
        "BETA_pretest",
        "SE_pretest",
        "T_STAT_pretest",
        "P_pretest",
        "ERRCODE_pretest",

        "A1_exercise",
        "A1_FREQ_exercise",
        "OBS_CT_exercise",
        "BETA_exercise",
        "SE_exercise",
        "T_STAT_exercise",
        "P_exercise",
        "ERRCODE_exercise",

        "A1_rest",
        "A1_FREQ_rest",
        "OBS_CT_rest",
        "BETA_rest",
        "SE_rest",
        "T_STAT_rest",
        "P_rest",
        "ERRCODE_rest",

        # Derived columns
        "logP_pretest",
        "logP_exercise",
        "logP_rest",

        "significant_pretest",
        "significant_exercise",
        "significant_rest",

        "suggestive_pretest",
        "suggestive_exercise",
        "suggestive_rest",

        "sig_pattern",

        "delta_logP_exercise_vs_pretest",
        "delta_logP_rest_vs_pretest",
        "delta_logP_rest_vs_exercise",

        "BETA_diff_exercise_vs_pretest",
        "BETA_diff_rest_vs_pretest",
        "BETA_diff_rest_vs_exercise",

        "abs_BETA_diff_exercise_vs_pretest",
        "abs_BETA_diff_rest_vs_pretest",
        "abs_BETA_diff_rest_vs_exercise",

        "BETA_direction_flip_exercise_vs_pretest",
        "BETA_direction_flip_rest_vs_pretest",
        "BETA_direction_flip_rest_vs_exercise",

        "sig_gain_exercise_vs_pretest",
        "sig_loss_exercise_vs_pretest",
        "sig_gain_rest_vs_pretest",
        "sig_loss_rest_vs_pretest",
        "sig_gain_rest_vs_exercise",
        "sig_loss_rest_vs_exercise",
    ]

    first_cols = [c for c in first_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in first_cols]

    return df[first_cols + other_cols]


# ==================================================
# MAIN
# ==================================================

if os.path.exists(OUT_MATRIX):
    print(f"Removing previous output file: {OUT_MATRIX}")
    os.remove(OUT_MATRIX)

summary_rows = []
interesting_rows = []
first_write = True

for z in Z_RANGE:

    z_out_dir = os.path.join(PER_CHR_Z_DIR, f"z{z}")
    os.makedirs(z_out_dir, exist_ok=True)

    for chrom in CHROM_RANGE:

        print("\n======================================")
        print(f"Building matrix for z{z}, chromosome {chrom}")
        print("======================================")

        phase_dfs = {}
        missing_any = False

        for phase in PHASES:
            file_path = get_file_path(phase, z, chrom)

            if not os.path.exists(file_path):
                print(f"Missing file: {file_path}")
                missing_any = True
                break

            print(f"Reading {phase}: {file_path}")
            phase_dfs[phase] = read_glm_file(file_path, phase)

        if missing_any:
            print(f"Skipping z{z}, chromosome {chrom} due to missing files.")
            continue

        # Merge the 3 phases side by side
        mat = phase_dfs["pretest"]

        mat = mat.merge(
            phase_dfs["exercise"],
            on=["SNP_KEY", "CHROM", "POS", "ID"],
            how="inner"
        )

        mat = mat.merge(
            phase_dfs["rest"],
            on=["SNP_KEY", "CHROM", "POS", "ID"],
            how="inner"
        )

        mat.insert(0, "z", z)

        mat = add_phase_comparison_columns(mat)
        mat = reorder_columns(mat)

        print(f"Merged rows: {len(mat):,}")

        # Save smaller per z+chromosome file
        if SAVE_PER_CHR_Z:
            per_file = os.path.join(
                z_out_dir,
                f"chr{chrom}_z{z}_phase_matrix.csv.gz"
            )

            mat.to_csv(
                per_file,
                index=False,
                compression="gzip"
            )

            print(f"Saved per chr/z file: {per_file}")

        # Append to one big matrix
        mat.to_csv(
            OUT_MATRIX,
            index=False,
            compression="gzip",
            mode="a",
            header=first_write
        )

        first_write = False

        # Summary row
        summary_rows.append({
            "z": z,
            "chromosome": chrom,
            "n_snps": len(mat),

            "n_sig_pretest": int(mat["significant_pretest"].sum()),
            "n_sig_exercise": int(mat["significant_exercise"].sum()),
            "n_sig_rest": int(mat["significant_rest"].sum()),

            "n_sig_any_phase": int(
                mat[[
                    "significant_pretest",
                    "significant_exercise",
                    "significant_rest"
                ]].any(axis=1).sum()
            ),

            "n_sig_all_phases": int(
                mat[[
                    "significant_pretest",
                    "significant_exercise",
                    "significant_rest"
                ]].all(axis=1).sum()
            ),

            "n_sig_pretest_only": int((mat["sig_pattern"] == "pretest").sum()),
            "n_sig_exercise_only": int((mat["sig_pattern"] == "exercise").sum()),
            "n_sig_rest_only": int((mat["sig_pattern"] == "rest").sum()),

            "max_logP_pretest": mat["logP_pretest"].max(),
            "max_logP_exercise": mat["logP_exercise"].max(),
            "max_logP_rest": mat["logP_rest"].max(),

            "max_abs_delta_logP_exercise_vs_pretest": mat["delta_logP_exercise_vs_pretest"].abs().max(),
            "max_abs_delta_logP_rest_vs_pretest": mat["delta_logP_rest_vs_pretest"].abs().max(),
            "max_abs_delta_logP_rest_vs_exercise": mat["delta_logP_rest_vs_exercise"].abs().max(),
        })

        # Interesting rows:
        # 1. significant in at least one phase
        # 2. significant gain/loss
        # 3. large delta logP
        # 4. beta direction flip
        interesting_mask = (
            mat[[
                "significant_pretest",
                "significant_exercise",
                "significant_rest"
            ]].any(axis=1)
            |
            mat[[
                "sig_gain_exercise_vs_pretest",
                "sig_loss_exercise_vs_pretest",
                "sig_gain_rest_vs_pretest",
                "sig_loss_rest_vs_pretest",
                "sig_gain_rest_vs_exercise",
                "sig_loss_rest_vs_exercise",
            ]].any(axis=1)
            |
            (mat["delta_logP_exercise_vs_pretest"].abs() >= DELTA_LOGP_THRESHOLD)
            |
            (mat["delta_logP_rest_vs_pretest"].abs() >= DELTA_LOGP_THRESHOLD)
            |
            (mat["delta_logP_rest_vs_exercise"].abs() >= DELTA_LOGP_THRESHOLD)
            |
            (mat["BETA_direction_flip_exercise_vs_pretest"])
            |
            (mat["BETA_direction_flip_rest_vs_pretest"])
            |
            (mat["BETA_direction_flip_rest_vs_exercise"])
        )

        interesting = mat[interesting_mask].copy()

        if not interesting.empty:
            interesting["source_z"] = z
            interesting["source_chromosome"] = chrom
            interesting_rows.append(interesting)

# ==================================================
# SAVE SUMMARY FILES
# ==================================================

print("\n======================================")
print("Saving summary files")
print("======================================")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_SUMMARY, index=False)

print("Saved summary:")
print(OUT_SUMMARY)

if interesting_rows:

    interesting_df = pd.concat(interesting_rows, ignore_index=True)

    interesting_df["max_logP_any_phase"] = interesting_df[
        ["logP_pretest", "logP_exercise", "logP_rest"]
    ].max(axis=1)

    interesting_df["max_abs_delta_logP"] = interesting_df[
        [
            "delta_logP_exercise_vs_pretest",
            "delta_logP_rest_vs_pretest",
            "delta_logP_rest_vs_exercise",
        ]
    ].abs().max(axis=1)

    interesting_df = interesting_df.sort_values(
        ["max_logP_any_phase", "max_abs_delta_logP"],
        ascending=[False, False]
    )

    interesting_df.to_csv(OUT_INTERESTING, index=False)

    print("Saved interesting phase changes:")
    print(OUT_INTERESTING)

    print("\nTop 30 interesting rows:")
    show_cols = [
        "z",
        "CHROM",
        "POS",
        "ID",
        "sig_pattern",

        "P_pretest",
        "P_exercise",
        "P_rest",

        "logP_pretest",
        "logP_exercise",
        "logP_rest",

        "delta_logP_exercise_vs_pretest",
        "delta_logP_rest_vs_pretest",
        "delta_logP_rest_vs_exercise",

        "BETA_pretest",
        "BETA_exercise",
        "BETA_rest",
    ]

    show_cols = [c for c in show_cols if c in interesting_df.columns]

    print(interesting_df[show_cols].head(30).to_string(index=False))

else:
    print("No interesting rows found.")

print("\nBig matrix saved to:")
print(OUT_MATRIX)

print("\nDone.")
