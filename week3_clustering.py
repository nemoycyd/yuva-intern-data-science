# Week 3: Unsupervised Learning and Clustering (Dataset: Palmer Penguins)
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv")  # or data/penguins.csv
print("Raw shape:", df.shape, "| missing:", int(df.isnull().sum().sum()))
df = df.dropna().reset_index(drop=True)
features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
X = StandardScaler().fit_transform(df[features])           # clustering is distance-based -> scale first
print("Clean shape:", df.shape)

# Choose k: elbow + silhouette
ks = range(2, 9); inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertia.append(km.inertia_); sil.append(silhouette_score(X, km.labels_))
    print(f"k={k}  inertia={km.inertia_:.1f}  silhouette={sil[-1]:.3f}")

K = 3
km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X)
df["kmeans"] = km.labels_
hc = AgglomerativeClustering(n_clusters=K, linkage="ward").fit(X)
df["hier"] = hc.labels_
print("\nSilhouette  KMeans: %.3f | Hierarchical: %.3f" % (silhouette_score(X, df.kmeans), silhouette_score(X, df.hier)))
print("Agreement KMeans vs Hierarchical (ARI): %.3f" % adjusted_rand_score(df.kmeans, df.hier))
print("Agreement KMeans vs true species (ARI): %.3f" % adjusted_rand_score(df.species, df.kmeans))   # only for validation
print("\nCluster profiles (mean of each feature):\n", df.groupby("kmeans")[features].mean().round(1))
print("\nCluster size:\n", df.kmeans.value_counts().sort_index())
print("\nSpecies inside each cluster:\n", pd.crosstab(df.kmeans, df.species))

pca = PCA(n_components=2).fit(X); P = pca.transform(X)
print("\nPCA variance explained:", pca.explained_variance_ratio_.round(3))

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].plot(list(ks), inertia, "o-"); ax[0].set_title("Elbow method"); ax[0].set_xlabel("k"); ax[0].set_ylabel("Inertia")
ax[1].plot(list(ks), sil, "o-", color="green"); ax[1].set_title("Silhouette score"); ax[1].set_xlabel("k")
sns.scatterplot(x=P[:, 0], y=P[:, 1], hue=df.kmeans, palette="Set1", ax=ax[2]); ax[2].set_title("K-Means clusters (PCA 2D)")
ax[2].set_xlabel("PC1"); ax[2].set_ylabel("PC2")
plt.tight_layout(); plt.savefig("figs/w3_kmeans.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
dendrogram(linkage(X, "ward"), truncate_mode="lastp", p=20, ax=ax[0]); ax[0].set_title("Hierarchical clustering dendrogram (Ward)")
sns.scatterplot(x=P[:, 0], y=P[:, 1], hue=df.species, palette="Set2", ax=ax[1]); ax[1].set_title("Same data coloured by true species (validation only)")
plt.tight_layout(); plt.savefig("figs/w3_hier.png", dpi=130); plt.close()
