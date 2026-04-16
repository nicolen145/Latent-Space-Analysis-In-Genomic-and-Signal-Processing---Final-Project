import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances


# =========================================================
# CONFIG
# =========================================================
VAE_OUTPUT_DIR = "vae_multi_latent_outputs_2"
ORIGINAL_INPUT_PATH = "fft_features/fft_features_pid_clean.csv"

# אם המודל הטוב ביותר שלך הוא latent 32:
LATENT_FILE = os.path.join(VAE_OUTPUT_DIR, "pid_vae_latent_best_32.csv")
RECON_FILE = os.path.join(VAE_OUTPUT_DIR, "vae_reconstructed_dataset.csv")
ERROR_FILE = os.path.join(VAE_OUTPUT_DIR, "pid_vae_reconstruction_error.csv")

ANALYSIS_DIR = os.path.join(VAE_OUTPUT_DIR, "analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)


# =========================================================
# LOAD FILES
# =========================================================
print("Loading files...")

original_df = pd.read_csv(ORIGINAL_INPUT_PATH, dtype={"PID": str})
latent_df = pd.read_csv(LATENT_FILE, dtype={"PID": str})
recon_df = pd.read_csv(RECON_FILE, dtype={"PID": str})

if os.path.exists(ERROR_FILE):
    error_df = pd.read_csv(ERROR_FILE, dtype={"PID": str})
else:
    error_df = latent_df[["PID", "reconstruction_error"]].copy()

print("Original shape:", original_df.shape)
print("Latent shape:", latent_df.shape)
print("Recon shape:", recon_df.shape)
print("Error shape:", error_df.shape)


# =========================================================
# ALIGN DATA BY PID
# =========================================================
# keep only common PIDs
common_pids = sorted(set(original_df["PID"]) & set(latent_df["PID"]) & set(recon_df["PID"]))
if len(common_pids) == 0:
    raise ValueError("No common PIDs found between original, latent, and reconstructed files.")

original_df = original_df[original_df["PID"].isin(common_pids)].copy()
latent_df = latent_df[latent_df["PID"].isin(common_pids)].copy()
recon_df = recon_df[recon_df["PID"].isin(common_pids)].copy()
error_df = error_df[error_df["PID"].isin(common_pids)].copy()

original_df = original_df.sort_values("PID").reset_index(drop=True)
latent_df = latent_df.sort_values("PID").reset_index(drop=True)
recon_df = recon_df.sort_values("PID").reset_index(drop=True)
error_df = error_df.sort_values("PID").reset_index(drop=True)

# make sure order is identical
assert list(original_df["PID"]) == list(latent_df["PID"]) == list(recon_df["PID"])

print("Common PIDs:", len(common_pids))


# =========================================================
# COLUMN DETECTION
# =========================================================
feature_cols = [c for c in original_df.columns if c != "PID"]

# reconstructed dataset may include reconstruction_error column
recon_feature_cols = [c for c in recon_df.columns if c not in ["PID", "reconstruction_error"]]

if set(feature_cols) != set(recon_feature_cols):
    raise ValueError(
        "Feature columns in original and reconstructed datasets do not match.\n"
        f"Original only: {sorted(set(feature_cols) - set(recon_feature_cols))}\n"
        f"Recon only: {sorted(set(recon_feature_cols) - set(feature_cols))}"
    )

# reorder reconstructed features to match original order
recon_feature_cols = feature_cols.copy()

latent_cols = [c for c in latent_df.columns if c.startswith("z")]
if len(latent_cols) == 0:
    raise ValueError("No latent z columns found in latent file.")

print("Num features:", len(feature_cols))
print("Num latent dims:", len(latent_cols))


# =========================================================
# NUMPY MATRICES
# =========================================================
X_orig = original_df[feature_cols].values.astype(float)
X_recon = recon_df[recon_feature_cols].values.astype(float)
Z = latent_df[latent_cols].values.astype(float)
reconstruction_error = error_df["reconstruction_error"].values.astype(float)
pids = original_df["PID"].values


# =========================================================
# 1. TRUE FEATURE RECONSTRUCTION ERROR
# =========================================================
feature_mse = np.mean((X_orig - X_recon) ** 2, axis=0)

feature_error_df = pd.DataFrame({
    "feature": feature_cols,
    "mse": feature_mse
}).sort_values("mse", ascending=False)

feature_error_path = os.path.join(ANALYSIS_DIR, "feature_reconstruction_error.csv")
feature_error_df.to_csv(feature_error_path, index=False)

plt.figure(figsize=(11, 6))
top_n = min(12, len(feature_error_df))
plot_df = feature_error_df.head(top_n).iloc[::-1]
plt.barh(plot_df["feature"], plot_df["mse"])
plt.xlabel("Mean Squared Error")
plt.ylabel("Feature")
plt.title("Top Features with Highest Reconstruction Error")
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "feature_reconstruction_error_bar.png"), dpi=300)
plt.show()


# =========================================================
# 2. PCA ON LATENT SPACE
# =========================================================
pca = PCA(n_components=2, random_state=42)
Z_2d = pca.fit_transform(Z)

pca_df = pd.DataFrame({
    "PID": pids,
    "PC1": Z_2d[:, 0],
    "PC2": Z_2d[:, 1],
    "reconstruction_error": reconstruction_error
})
pca_df.to_csv(os.path.join(ANALYSIS_DIR, "latent_pca_coordinates.csv"), index=False)

plt.figure(figsize=(7, 6))
plt.scatter(Z_2d[:, 0], Z_2d[:, 1], s=35)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Latent Space PCA")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "latent_pca.png"), dpi=300)
plt.show()


# =========================================================
# 3. PCA COLORED BY RECONSTRUCTION ERROR
# =========================================================
plt.figure(figsize=(7, 6))
sc = plt.scatter(Z_2d[:, 0], Z_2d[:, 1], c=reconstruction_error, s=45)
plt.colorbar(sc, label="Reconstruction Error")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Latent Space Colored by Reconstruction Error")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "latent_colored_error.png"), dpi=300)
plt.show()


# =========================================================
# 4. RECONSTRUCTION ERROR DISTRIBUTION
# =========================================================
plt.figure(figsize=(8, 5))
plt.hist(reconstruction_error, bins=20)
plt.xlabel("Reconstruction Error")
plt.ylabel("Count")
plt.title("Reconstruction Error Distribution")
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "reconstruction_error_hist.png"), dpi=300)
plt.show()


# =========================================================
# 5. TOP ANOMALIES
# =========================================================
top_k = min(10, len(error_df))
top_anomalies_df = error_df.sort_values("reconstruction_error", ascending=False).head(top_k).copy()
top_anomalies_path = os.path.join(ANALYSIS_DIR, "top_anomalies.csv")
top_anomalies_df.to_csv(top_anomalies_path, index=False)

plt.figure(figsize=(10, 5))
plt.bar(top_anomalies_df["PID"].astype(str), top_anomalies_df["reconstruction_error"])
plt.xticks(rotation=90)
plt.xlabel("PID")
plt.ylabel("Reconstruction Error")
plt.title(f"Top {top_k} Anomalies")
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "top_anomalies_bar.png"), dpi=300)
plt.show()


# =========================================================
# 6. ORIGINAL VS RECONSTRUCTED FOR TOP ANOMALIES
# =========================================================
compare_dir = os.path.join(ANALYSIS_DIR, "patient_feature_comparisons")
os.makedirs(compare_dir, exist_ok=True)

for pid in top_anomalies_df["PID"].astype(str).tolist()[:5]:
    idx = np.where(pids == pid)[0][0]
    orig = X_orig[idx]
    recon = X_recon[idx]

    plt.figure(figsize=(14, 5))
    plt.plot(feature_cols, orig, marker="o", label="Original")
    plt.plot(feature_cols, recon, marker="x", label="Reconstructed")
    plt.xticks(rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Value")
    plt.title(f"PID {pid} - Original vs Reconstructed Features")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, f"{pid}_original_vs_reconstructed.png"), dpi=300)
    plt.show()


# =========================================================
# 7. HEATMAP-LIKE IMAGE OF ORIGINAL VS RECON FOR A FEW PATIENTS
# =========================================================
heatmap_dir = os.path.join(ANALYSIS_DIR, "heatmaps")
os.makedirs(heatmap_dir, exist_ok=True)

selected_pids = top_anomalies_df["PID"].astype(str).tolist()[:3]

for pid in selected_pids:
    idx = np.where(pids == pid)[0][0]

    mat = np.vstack([X_orig[idx], X_recon[idx]])

    plt.figure(figsize=(14, 3))
    plt.imshow(mat, aspect="auto")
    plt.yticks([0, 1], ["Original", "Reconstructed"])
    plt.xticks(range(len(feature_cols)), feature_cols, rotation=90)
    plt.colorbar(label="Feature Value")
    plt.title(f"PID {pid} - Original vs Reconstructed Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(heatmap_dir, f"{pid}_heatmap.png"), dpi=300)
    plt.show()


# =========================================================
# 8. PER-PATIENT TRUE RECONSTRUCTION ERROR FROM MATRICES
# =========================================================
true_error = np.mean((X_orig - X_recon) ** 2, axis=1)

true_error_df = pd.DataFrame({
    "PID": pids,
    "true_reconstruction_mse": true_error
}).sort_values("true_reconstruction_mse", ascending=False)

true_error_df.to_csv(os.path.join(ANALYSIS_DIR, "true_reconstruction_error_by_pid.csv"), index=False)


# =========================================================
# 9. NEAREST PATIENTS IN LATENT SPACE
# =========================================================
dist_matrix = euclidean_distances(Z)

nearest_rows = []
for i, pid in enumerate(pids):
    dists = dist_matrix[i]
    nearest_idx = np.argsort(dists)[1:4]  # skip itself
    nearest_pids = [pids[j] for j in nearest_idx]
    nearest_distances = [float(dists[j]) for j in nearest_idx]

    nearest_rows.append({
        "PID": pid,
        "nearest_1": nearest_pids[0],
        "dist_1": nearest_distances[0],
        "nearest_2": nearest_pids[1],
        "dist_2": nearest_distances[1],
        "nearest_3": nearest_pids[2],
        "dist_3": nearest_distances[2],
    })

nearest_df = pd.DataFrame(nearest_rows)
nearest_df.to_csv(os.path.join(ANALYSIS_DIR, "nearest_patients.csv"), index=False)


# =========================================================
# 10. SUMMARY TABLE
# =========================================================
summary_df = pd.DataFrame({
    "metric": [
        "num_patients",
        "num_features",
        "num_latent_dims",
        "mean_reconstruction_error",
        "median_reconstruction_error",
        "max_reconstruction_error",
        "min_reconstruction_error",
        "mean_true_feature_mse"
    ],
    "value": [
        len(pids),
        len(feature_cols),
        len(latent_cols),
        float(np.mean(reconstruction_error)),
        float(np.median(reconstruction_error)),
        float(np.max(reconstruction_error)),
        float(np.min(reconstruction_error)),
        float(np.mean(true_error))
    ]
})
summary_df.to_csv(os.path.join(ANALYSIS_DIR, "analysis_summary.csv"), index=False)


# =========================================================
# 11. TOP FEATURES FOR TOP ANOMALY
# =========================================================
top_pid = top_anomalies_df.iloc[0]["PID"]
top_idx = np.where(pids == top_pid)[0][0]

feature_abs_error = np.abs(X_orig[top_idx] - X_recon[top_idx])
feature_abs_error_df = pd.DataFrame({
    "feature": feature_cols,
    "abs_error": feature_abs_error
}).sort_values("abs_error", ascending=False)

feature_abs_error_df.to_csv(
    os.path.join(ANALYSIS_DIR, f"top_anomaly_{top_pid}_feature_abs_error.csv"),
    index=False
)

plt.figure(figsize=(11, 6))
plot_df = feature_abs_error_df.head(min(12, len(feature_abs_error_df))).iloc[::-1]
plt.barh(plot_df["feature"], plot_df["abs_error"])
plt.xlabel("Absolute Error")
plt.ylabel("Feature")
plt.title(f"Top Feature Errors for Most Anomalous PID {top_pid}")
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, f"top_anomaly_{top_pid}_feature_abs_error_bar.png"), dpi=300)
plt.show()


# =========================================================
# DONE
# =========================================================
print("\nAnalysis complete.")
print("Analysis output folder:", ANALYSIS_DIR)
print("Saved files:")
print("-", feature_error_path)
print("-", top_anomalies_path)
print("-", os.path.join(ANALYSIS_DIR, "nearest_patients.csv"))
print("-", os.path.join(ANALYSIS_DIR, "analysis_summary.csv"))