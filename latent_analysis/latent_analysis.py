# =========================================================
# FULL LATENT SPACE ANALYSIS FOR ECG AE
# PCA, LDA, t-SNE, UMAP, Reconstruction Error, Clustering
# =========================================================

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import umap


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = "/sise/nadav-group/nadavrap-group/Final_Project-Nicole_Gal_Lilach"

INPUT_CSV = os.path.join(
    BASE_DIR,
    "Final_Project/Final_AE/all_segments_latent_40.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Final_Project/Final_AE/full_latent_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_SIZE = 200000      # PCA / LDA / correlations / boxplots
TSNE_SAMPLE = 20000       # t-SNE only
UMAP_SAMPLE = 50000       # UMAP only

RANDOM_SEED = 42


# =========================================================
# LOAD DATA
# =========================================================

print("Loading latent CSV:")
print(INPUT_CSV)

df_all = pd.read_csv(INPUT_CSV)

z_cols = [c for c in df_all.columns if c.startswith("z")]

required_cols = ["phase", "reconstruction_error"]

for c in required_cols:
    if c not in df_all.columns:
        raise ValueError(f"Missing required column: {c}")

df_all = df_all.dropna(subset=["phase"] + z_cols).reset_index(drop=True)

print("Total rows after cleaning:", len(df_all))
print("Latent dimensions:", len(z_cols))


# =========================================================
# MAIN SAMPLE FOR PCA / LDA / BASIC ANALYSIS
# =========================================================

df = df_all.sample(
    n=min(SAMPLE_SIZE, len(df_all)),
    random_state=RANDOM_SEED
).reset_index(drop=True)

print("Main analysis sample:", len(df))

X = df[z_cols].values
y = df["phase"].astype(str).values

X_scaled = StandardScaler().fit_transform(X)


# =========================================================
# HELPER FUNCTION
# =========================================================

def save_scatter(df_plot, x_col, y_col, color_col, title, out_path):
    plt.figure(figsize=(8, 6))

    if df_plot[color_col].dtype == object:
        for val in sorted(df_plot[color_col].dropna().unique()):
            sub = df_plot[df_plot[color_col] == val]

            plt.scatter(
                sub[x_col],
                sub[y_col],
                s=5,
                alpha=0.5,
                label=str(val)
            )

        plt.legend(markerscale=3)

    else:
        sc = plt.scatter(
            df_plot[x_col],
            df_plot[y_col],
            c=df_plot[color_col],
            s=5,
            alpha=0.5
        )

        plt.colorbar(sc, label=color_col)

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)
    
    
def set_balanced_limits(Z_2d, ax):
    x_min, x_max = np.percentile(Z_2d[:, 0], [2, 98])
    y_min, y_max = np.percentile(Z_2d[:, 1], [2, 98])

    dx = (x_max - x_min) * 0.25
    dy = (y_max - y_min) * 0.25

    ax.set_xlim(x_min - dx, x_max + dx)
    ax.set_ylim(y_min - dy, y_max + dy)


# =========================================================
# PCA
# =========================================================

print("\nRunning PCA...")

pca = PCA(n_components=2, random_state=RANDOM_SEED)
X_pca = pca.fit_transform(X_scaled)

df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

save_scatter(
    df,
    "PCA1",
    "PCA2",
    "phase",
    f"PCA of Latent Space by Phase (sample={len(df):,})",
    os.path.join(OUTPUT_DIR, "01_pca_by_phase.png")
)


# =========================================================
# LDA
# =========================================================

print("\nRunning LDA...")

n_classes = len(np.unique(y))
n_components = min(2, n_classes - 1)

if n_components < 1:
    raise ValueError("LDA needs at least 2 phase classes.")

lda = LinearDiscriminantAnalysis(n_components=n_components)
X_lda = lda.fit_transform(X_scaled, y)

df["LDA1"] = X_lda[:, 0]

if n_components == 2:
    df["LDA2"] = X_lda[:, 1]
else:
    df["LDA2"] = 0.0

save_scatter(
    df,
    "LDA1",
    "LDA2",
    "phase",
    f"LDA of Latent Space by Phase (sample={len(df):,})",
    os.path.join(OUTPUT_DIR, "02_lda_by_phase.png")
)


# =========================================================
# LDA COLORED BY LOAD
# =========================================================

if "load_W" in df.columns:
    print("\nPlotting LDA colored by load...")

    temp = df.copy()
    temp["load_W"] = pd.to_numeric(temp["load_W"], errors="coerce")
    temp = temp.dropna(subset=["load_W", "LDA1", "LDA2"])

    plt.figure(figsize=(8, 6))

    sc = plt.scatter(
        temp["LDA1"],
        temp["LDA2"],
        c=temp["load_W"],
        s=5,
        alpha=0.5
    )

    plt.colorbar(sc, label="Load (Watts)")
    plt.xlabel("LDA 1")
    plt.ylabel("LDA 2")
    plt.title(f"Exercise Load in Latent Space - LDA (sample={len(temp):,})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "lda_colored_by_load.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print("Saved:", path)
    
    


# =========================================================
# t-SNE
# =========================================================

print("\nRunning t-SNE...")

tsne_df = df_all.sample(
    n=min(TSNE_SAMPLE, len(df_all)),
    random_state=RANDOM_SEED
).reset_index(drop=True)

X_tsne_scaled = StandardScaler().fit_transform(tsne_df[z_cols].values)

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=RANDOM_SEED,
    init="pca",
    learning_rate="auto"
)

X_tsne = tsne.fit_transform(X_tsne_scaled)

tsne_df["TSNE1"] = X_tsne[:, 0]
tsne_df["TSNE2"] = X_tsne[:, 1]

save_scatter(
    tsne_df,
    "TSNE1",
    "TSNE2",
    "phase",
    f"t-SNE of Latent Space by Phase (sample={len(tsne_df):,})",
    os.path.join(OUTPUT_DIR, "03_tsne_by_phase.png")
)


# =========================================================
# UMAP
# =========================================================

print("\nRunning UMAP...")

umap_df = df_all.sample(
    n=min(UMAP_SAMPLE, len(df_all)),
    random_state=RANDOM_SEED
).reset_index(drop=True)

X_umap_scaled = StandardScaler().fit_transform(umap_df[z_cols].values)

umap_model = umap.UMAP(
    n_components=2,
    random_state=RANDOM_SEED
)

X_umap = umap_model.fit_transform(X_umap_scaled)

umap_df["UMAP1"] = X_umap[:, 0]
umap_df["UMAP2"] = X_umap[:, 1]

save_scatter(
    umap_df,
    "UMAP1",
    "UMAP2",
    "phase",
    f"UMAP of Latent Space by Phase (sample={len(umap_df):,})",
    os.path.join(OUTPUT_DIR, "04_umap_by_phase.png")
)


# =========================================================
# UMAP COLORED BY CONTINUOUS VARIABLES
# =========================================================

continuous_cols = [
    "heart_rate",
    "load_W",
    "mets",
    "speed_rpm"
]

for col in continuous_cols:
    if col not in umap_df.columns:
        continue

    temp = umap_df.copy()
    temp[col] = pd.to_numeric(temp[col], errors="coerce")
    temp = temp.dropna(subset=[col])

    if len(temp) == 0:
        continue

    save_scatter(
        temp,
        "UMAP1",
        "UMAP2",
        col,
        f"UMAP Colored by {col} (sample={len(temp):,})",
        os.path.join(OUTPUT_DIR, f"05_umap_by_{col}.png")
    )


# =========================================================
# RECONSTRUCTION ERROR BY PHASE
# =========================================================

print("\nPlotting reconstruction error by phase...")

plt.figure(figsize=(8, 5))

df.boxplot(
    column="reconstruction_error",
    by="phase"
)

plt.title(f"Reconstruction Error by Phase (sample={len(df):,})")
plt.suptitle("")
plt.xlabel("Phase")
plt.ylabel("Reconstruction Error")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "06_reconstruction_error_by_phase.png")
plt.savefig(path, dpi=300)
plt.close()

print("Saved:", path)


# =========================================================
# RECONSTRUCTION ERROR DISTRIBUTION
# =========================================================

print("\nPlotting reconstruction error distribution...")

recon = pd.to_numeric(df["reconstruction_error"], errors="coerce").dropna().values
x_max = np.percentile(recon, 99)

plt.figure(figsize=(8, 5))
plt.hist(
    recon[recon <= x_max],
    bins=80,
    edgecolor="white"
)

plt.xlabel("Reconstruction Error (MSE)")
plt.ylabel("Count")
plt.title(f"Reconstruction Error Distribution (sample={len(recon):,}, zoomed to 99th percentile)")
plt.grid(True, alpha=0.3)
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "reconstruction_error_distribution.png")
plt.savefig(path, dpi=300)
plt.close()

print("Saved:", path)


# =========================================================
# KMEANS CLUSTERING ON LATENT SPACE
# =========================================================

print("\nRunning KMeans clustering...")

kmeans = KMeans(
    n_clusters=3,
    random_state=RANDOM_SEED,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

df["cluster"] = clusters.astype(str)

save_scatter(
    df,
    "PCA1",
    "PCA2",
    "cluster",
    f"KMeans Clusters on PCA Space (sample={len(df):,})",
    os.path.join(OUTPUT_DIR, "07_kmeans_clusters_pca.png")
)


# =========================================================
# TEMPORAL TRAJECTORIES ON UMAP
# =========================================================

if "time_sec" in umap_df.columns and "PID" in umap_df.columns:

    print("\nPlotting temporal trajectories...")

    umap_df["time_sec"] = pd.to_numeric(
        umap_df["time_sec"],
        errors="coerce"
    )

    sampled_pids = umap_df["PID"].dropna().unique()[:10]

    plt.figure(figsize=(10, 8))

    for pid in sampled_pids:
        sub = (
            umap_df[umap_df["PID"] == pid]
            .dropna(subset=["time_sec"])
            .sort_values("time_sec")
        )

        if len(sub) < 5:
            continue

        plt.plot(
            sub["UMAP1"],
            sub["UMAP2"],
            alpha=0.8,
            linewidth=1,
            label=str(pid)
        )

    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title(f"Temporal Trajectories in Latent Space (UMAP sample={len(umap_df):,})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "08_temporal_trajectories_umap.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print("Saved:", path)


# =========================================================
# CORRELATION ANALYSIS
# =========================================================

print("\nComputing correlations...")

corr_targets = [
    "heart_rate",
    "load_W",
    "mets",
    "speed_rpm"
]

corr_rows = []

for target in corr_targets:
    if target not in df.columns:
        continue

    df[target] = pd.to_numeric(df[target], errors="coerce")
    temp = df.dropna(subset=[target])

    if len(temp) == 0:
        continue

    for z in z_cols:
        corr = temp[z].corr(temp[target])

        corr_rows.append({
            "target": target,
            "latent_dim": z,
            "correlation": corr
        })

corr_df = pd.DataFrame(corr_rows)

corr_csv = os.path.join(OUTPUT_DIR, "latent_correlations.csv")
corr_df.to_csv(corr_csv, index=False)

print("Saved:", corr_csv)


# =========================================================
# SILHOUETTE SCORES
# =========================================================

print("\nComputing silhouette scores...")

silhouette_results = {}

try:
    silhouette_results["PCA"] = silhouette_score(X_pca, y)
except Exception as e:
    silhouette_results["PCA"] = f"Failed: {e}"

try:
    silhouette_results["LDA"] = silhouette_score(X_lda, y)
except Exception as e:
    silhouette_results["LDA"] = f"Failed: {e}"

try:
    silhouette_results["UMAP"] = silhouette_score(X_umap, umap_df["phase"].astype(str).values)
except Exception as e:
    silhouette_results["UMAP"] = f"Failed: {e}"

try:
    silhouette_results["TSNE"] = silhouette_score(X_tsne, tsne_df["phase"].astype(str).values)
except Exception as e:
    silhouette_results["TSNE"] = f"Failed: {e}"


# =========================================================
# SAVE SUMMARY
# =========================================================

summary_path = os.path.join(OUTPUT_DIR, "analysis_summary.txt")

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("FULL LATENT ANALYSIS SUMMARY\n")
    f.write("=" * 50 + "\n\n")

    f.write(f"Input CSV: {INPUT_CSV}\n")
    f.write(f"Total rows after cleaning: {len(df_all)}\n")
    f.write(f"Main sample size: {len(df)}\n")
    f.write(f"t-SNE sample size: {len(tsne_df)}\n")
    f.write(f"UMAP sample size: {len(umap_df)}\n")
    f.write(f"Latent dimensions: {len(z_cols)}\n\n")

    f.write("PCA Explained Variance Ratio:\n")
    f.write(str(pca.explained_variance_ratio_))
    f.write("\n\n")

    f.write("Silhouette Scores:\n")
    for k, v in silhouette_results.items():
        f.write(f"{k}: {v}\n")

print("Saved:", summary_path)


# =========================================================
# SAVE ANALYSIS CSVs
# =========================================================

main_csv = os.path.join(OUTPUT_DIR, "main_pca_lda_analysis.csv")
tsne_csv = os.path.join(OUTPUT_DIR, "tsne_coordinates.csv")
umap_csv = os.path.join(OUTPUT_DIR, "umap_coordinates.csv")

df.to_csv(main_csv, index=False)
tsne_df.to_csv(tsne_csv, index=False)
umap_df.to_csv(umap_csv, index=False)

print("Saved:", main_csv)
print("Saved:", tsne_csv)
print("Saved:", umap_csv)

# =========================================================
# SUMMARY TABLE PNG
# =========================================================

summary = pd.DataFrame({
    "Metric": [
        "Mean Recon Error",
        "Std Recon Error",
        "Max Recon Error",
        "Min Recon Error",
        "Samples Used",
        "Latent Dim"
    ],
    "Value": [
        round(np.nanmean(recon), 6),
        round(np.nanstd(recon), 6),
        round(np.nanmax(recon), 6),
        round(np.nanmin(recon), 6),
        len(df),
        len(z_cols)
    ]
})

summary_csv = os.path.join(OUTPUT_DIR, "summary_table.csv")
summary.to_csv(summary_csv, index=False)

fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")

tbl = ax.table(
    cellText=summary.values,
    colLabels=summary.columns,
    cellLoc="center",
    loc="center"
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.8)

plt.title("AE Latent Analysis Summary")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "summary_table.png")
plt.savefig(path, dpi=300, bbox_inches="tight")
plt.close()

print("Saved:", summary_csv)
print("Saved:", path)

print("\nDONE.")
