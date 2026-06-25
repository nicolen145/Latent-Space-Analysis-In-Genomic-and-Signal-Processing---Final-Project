import os
import sys
import subprocess

# ==================================================
# INPUT PHASE
# ==================================================

if len(sys.argv) != 2:
    raise ValueError("Usage: python run_gwas_one_phase.py <pretest/exercise/rest>")

phase = sys.argv[1]

if phase not in ["pretest", "exercise", "rest"]:
    raise ValueError("Phase must be one of: pretest, exercise, rest")

# ==================================================
# PATHS
# ==================================================

GEN_DIR = "/sise/nadav-group/nadavrap-group/UKBB/GEN"

GWAS_DIR = (
    "/sise/nadav-group/nadavrap-group/"
    "Final_Project-Nicole_Gal_Lilach/"
    "Final_Project/GWAS"
)

PHENO_FILE = os.path.join(GWAS_DIR, "latent_pheno_for_plink.tsv")
COVAR_FILE = os.path.join(GWAS_DIR, "covars_ready_for_plink.cov")


COVAR_NAMES_FILE = os.path.join(GWAS_DIR, "covar_names_for_plink_no_age2.txt")

OUTPUT_DIR = os.path.join(GWAS_DIR, "GWAS_results", phase)
os.makedirs(OUTPUT_DIR, exist_ok=True)

PLINK = "plink2"

# ==================================================
# LOAD COVAR NAMES
# ==================================================

with open(COVAR_NAMES_FILE, "r") as f:
    covar_names = f.read().strip()

print("======================================")
print(f"Running GWAS phase: {phase}")
print("Covariates:")
print(covar_names)
print("======================================")

# ==================================================
# PHENOTYPE NAMES
# ==================================================

pheno_names = ",".join([f"{phase}_z{i}" for i in range(1, 41)])

print("\nPhenotypes:")
print(pheno_names)

# ==================================================
# RUN CHROMOSOMES 1-22
# ==================================================

for chrom in range(1, 23):

    bfile_path = os.path.join(GEN_DIR, f"ukb{chrom}")
    output_file = os.path.join(OUTPUT_DIR, f"gwas_chr{chrom}_{phase}")

    cmd = [
        PLINK,
        "--bfile", bfile_path,
        "--pheno", PHENO_FILE,
        "--pheno-name", pheno_names,
        "--covar", COVAR_FILE,
        "--covar-name", covar_names,
        "--covar-variance-standardize",
        "--glm", "hide-covar",
        "--no-input-missing-phenotype",
        "--out", output_file
    ]

    print("\n======================================")
    print(f"Running chromosome {chrom} for phase {phase}")
    print("Command:")
    print(" ".join(cmd))
    print("======================================")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    print("\nSTDOUT:")
    print(result.stdout)

    print("\nSTDERR:")
    print(result.stderr)

    if result.returncode != 0:
        print(f"\nGWAS failed for phase {phase}, chromosome {chrom}")
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            output=result.stdout,
            stderr=result.stderr
        )

    print(f"\nFinished chromosome {chrom} for phase {phase}")

print("\n======================================")
print(f"GWAS completed successfully for phase: {phase}")
print("======================================")