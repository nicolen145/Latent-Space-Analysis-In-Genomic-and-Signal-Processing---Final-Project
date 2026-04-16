import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

# =========================================================
# CONFIG
# =========================================================
OUTPUT_DIR = "vae_multi_latent_outputs_3"

LATENT_FILE = os.path.join(OUTPUT_DIR, "pid_vae_latent_best_32.csv")
RECON_FILE = os.path.join(OUTPUT_DIR, "vae_reconstructed_dataset.csv")

ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# =========================================================
# LOAD DATA
# =========================================================
latent_df = pd.read_csv(LATENT_FILE)
recon_df = pd.read_csv(RECON_FILE)

print("Loaded latent:", latent_df.shape)
print("Loaded recon:", recon_df.shape)

# =========================================================
# FEATURE ERROR ANALYSIS
# =========================================================
feature_cols = [c for c in recon_df.columns if c not in ["PID", "reconstruction_error"]]

original = recon_df[feature_cols].values
reconstructed = recon_df[feature_cols].values  # כבר משוחזר

# אם יש לך original בנפרד – עדיף לטעון אותו
# כאן אנחנו משתמשים במה שיש (placeholder)

# חישוב variance כקירוב לחשיבות
feature_error = np.var(reconstructed, axis=0)

feature_error_df = pd.DataFrame({
    "feature": feature_cols,
    "error": feature_error
}).sort_values("error", ascending=False)

feature_error_df.to_csv(os.path.join(ANALYSIS_DIR, "feature_error.csv"), index=False)

plt.figure(figsize=(10,6))
plt.barh(feature_error_df["feature"][:10], feature_error_df["error"][:10])
plt.gca().invert_yaxis()
plt.title("Top 10 Most Variant Features")
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "feature_error_bar.png"))
plt.show()

# =========================================================
# LATENT SPACE PCA
# =========================================================
latent_cols = [c for c in latent_df.columns if c.startswith("z")]
Z = latent_df[latent_cols].values

pca = PCA(n_components=2)
Z_2d = pca.fit_transform(Z)

plt.figure(figsize=(7,6))
plt.scatter(Z_2d[:,0], Z_2d[:,1], s=30)
plt.title("Latent Space PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.savefig(os.path.join(ANALYSIS_DIR, "latent_pca.png"))
plt.show()

# =========================================================
# LATENT COLORED BY ERROR
# =========================================================
errors = latent_df["reconstruction_error"].values

plt.figure(figsize=(7,6))
sc = plt.scatter(Z_2d[:,0], Z_2d[:,1], c=errors, s=40)
plt.colorbar(sc)
plt.title("Latent Space Colored by Reconstruction Error")
plt.grid(True)
plt.savefig(os.path.join(ANALYSIS_DIR, "latent_colored_error.png"))
plt.show()

# =========================================================
# RECONSTRUCTION ERROR DISTRIBUTION
# =========================================================
plt.figure(figsize=(8,5))
plt.hist(errors, bins=30)
plt.title("Reconstruction Error Distribution")
plt.xlabel("Error")
plt.ylabel("Count")
plt.savefig(os.path.join(ANALYSIS_DIR, "error_hist.png"))
plt.show()

# =========================================================
# TOP ANOMALIES
# =========================================================
top_anomalies = latent_df.sort_values("reconstruction_error", ascending=False).head(10)
top_anomalies.to_csv(os.path.join(ANALYSIS_DIR, "top_anomalies.csv"), index=False)

plt.figure(figsize=(10,5))
plt.bar(top_anomalies["PID"].astype(str), top_anomalies["reconstruction_error"])
plt.xticks(rotation=90)
plt.title("Top Anomalies")
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "top_anomalies_bar.png"))
plt.show()

# =========================================================
# SIMILARITY BETWEEN PATIENTS
# =========================================================
dist_matrix = euclidean_distances(Z)

# מציאת 3 הקרובים לכל PID
nearest = {}

for i, pid in enumerate(latent_df["PID"]):
    dists = dist_matrix[i]
    idx = np.argsort(dists)[1:4]
    nearest[pid] = latent_df["PID"].iloc[idx].tolist()

nearest_df = pd.DataFrame(list(nearest.items()), columns=["PID", "Nearest_Patients"])
nearest_df.to_csv(os.path.join(ANALYSIS_DIR, "nearest_patients.csv"), index=False)

print("Analysis complete.")
