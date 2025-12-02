import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.decomposition import PCA
import umap.umap_ as umap
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')



"""
    Loading dataset + basic analysis
"""
df = pd.read_csv("../clustering_dataset/merged_dataset_clean.csv")

print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Columns: {df.columns.tolist()}")



#dbscan_df = [
#    "trap_index",
#    "boombap_index",
#    "cloud_rap_index",
#    "drill_index"]
#
#df = df[dbscan_df].copy()
#
#print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")



"""
    Normalizing data:
    - StandardScaler -> use if we have normally distributed data
    - MinMaxScaler -> use if there are many outliers or want range [0,1]
    - RobustScaler -> use if we have many outliers
"""
scaler = StandardScaler()
X = scaler.fit_transform(df)



"""
    Analyzing data distribution, since is an issue for DBSCAN.
"""
def dataset_distribution(X):
    # Calcola le distanze dai k vicini più prossimi
    k = 10  # numero di vicini
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    # Distanza media per ogni punto
    mean_distances = distances.mean(axis=1)

    # Visualizza la distribuzione
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.hist(mean_distances, bins=50, edgecolor='black')
    plt.xlabel('Distanza media dai k-vicini')
    plt.ylabel('Frequenza')
    plt.title('Distribuzione delle distanze')

    plt.subplot(1, 3, 2)
    plt.boxplot(mean_distances)
    plt.ylabel('Distanza media')
    plt.title('Boxplot delle distanze')

    plt.subplot(1, 3, 3)
    sorted_distances = np.sort(mean_distances)
    plt.plot(sorted_distances)
    plt.xlabel('Punti (ordinati)')
    plt.ylabel('Distanza media')
    plt.title('Distanze ordinate')
    plt.tight_layout()
    plt.show()

    # Statistiche
    print(f"Media: {mean_distances.mean():.4f}")
    print(f"Std Dev: {mean_distances.std():.4f}")
    print(f"Coefficiente di variazione: {mean_distances.std() / mean_distances.mean():.4f}")
    print(f"Min: {mean_distances.min():.4f}, Max: {mean_distances.max():.4f}")
    print(f"Rapporto Max/Min: {mean_distances.max() / mean_distances.min():.2f}")
    return

dataset_distribution(X)



"""
    Since distribution is not a problem, we try to reduce
    dimensionality using by PCA:
    - we try different numbers of components and find the best
    tradeoff
"""
for n in [4, 5, 6, 7, 8]:
    pca = PCA(n_components=n)
    X_pca = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"{n} componenti: {var_explained:.1%} varianza")


pca = PCA(n_components=6)
X_pca = pca.fit_transform(X)

print(f"Dataset shape after PCA: {X.shape[0]} rows x {X.shape[1]} columns")



"""
    DBSCAN parameter tuning:
    > fixing MinPts k (parameter)
        - 2 * n_features -> conservative
        - n_features -> balanced
        - fixed values e.g. 5, 10, 15 for high dim datasets
    > calculating k-distances and sorting them from k-nearest
    > plotting k-distance to get estimation of epsilon
"""
def plot_k_distance(X_pca, k):
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(X_pca)
    distances, indices = neighbors.kneighbors(X_pca)

    # sorting distances from k-nearest
    k_distances = np.sort(distances[:, k-1])

    # find the knee of the curve using KneeLocator
    #kneeloc = KneeLocator(range(len(k_distances)), k_distances, curve=curve, direction="increasing")
    #knee_index = kneeloc.knee
    #knee_value = kneeloc.knee_y

    #print(knee_value)

    eps_min = np.percentile(k_distances, 75)
    eps_max = np.percentile(k_distances, 95)

    plt.figure(figsize=(8,4))
    plt.plot(k_distances)
    plt.axhline(y=eps_min, color='red', linestyle='--', label=f"eps_min = {eps_min:.2f}")
    plt.axhline(y=eps_max, color='orange', linestyle='--', label=f"eps_max = {eps_max:.2f}")
    plt.title(f"K-distance plot (k={k})")
    plt.xlabel("Points sorted by distance")
    plt.ylabel("Distance to k-th nearest neighbor")
    plt.grid(True)
    plt.show()

    return eps_min, eps_max

eps_min, eps_max = plot_k_distance(X_pca, k=12)



"""
    DBSCAN clustering:
    > grid search
        - setting range for epsilon after having observed the k-distance plot
        - min_samples = MinPts
    > run dbscan with best parameters from grid search
"""
eps_values = np.linspace(eps_min, eps_max, 30)
min_samples=12

def grid_search(eps_values):
    best = []
    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        clusters = dbscan.fit_predict(X_pca)

        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = (clusters == -1).sum()

        unique, counts = np.unique(clusters, return_counts=True)

        # silhouette only if >1 cluster
        sil = -1
        if n_clusters > 1:
            try:
                sil = silhouette_score(X_pca, clusters)
            except:
                sil = -1

        best.append((eps, n_clusters, n_noise, sil, list(zip(unique, counts))))

    best_sorted = sorted(best, key=lambda t: (-t[1], t[2], -t[3]))
    for item in best_sorted[:20]:
        print(item)
    return

grid_search(eps_values)

dbscan = DBSCAN(
    eps=1,
    min_samples=10,
    metric='euclidean'
)
clusters = dbscan.fit_predict(X_pca)



"""
    Cluster validity:
    > clustering statistics:
        - number of clusters found
        - number of noise points over dataset
    > clusters dimension distribution
    - silhouette score
"""
# Clustering statistics
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise = list(clusters).count(-1)

print(f"Number of clusters found: {n_clusters}")
print(f"Points classified as noise: {n_noise} ({n_noise/len(clusters)*100:.1f}%)")
print(f"Points belonging to clusters: {len(clusters) - n_noise}")

# Clusters dimensions
unique, counts = np.unique(clusters, return_counts=True)
print(f"Clusters dimension:")
for cluster_id, count in zip(unique, counts):
    print(f"    Cluster {cluster_id}: {count} punti")

# Silhouette score (only if >1 cluster and without noise)
if n_clusters > 1:
    sil = silhouette_score(X, clusters)
    print("Silhouette score:", sil)
else:
    print("Silhouette score could not be quantified.")



"""
    Clustering visualization:
    - PCA method
    - UMAP method
    - TODO t-SNE
    - TODO heatmap for feature's mean per cluster
"""
# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap="tab10", s=40)
plt.title("DBSCAN Clustering (PCA 2D projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# UMAP
reducer = umap.UMAP(
    n_neighbors=40,
    min_dist=0.0,
    metric='euclidean',
    n_jobs=1
)

X_umap = reducer.fit_transform(X)

df_vis = pd.DataFrame({
    "UMAP1": X_umap[:,0],
    "UMAP2": X_umap[:,1],
    "cluster": clusters
})

plt.figure(figsize=(10,8))

sns.scatterplot(data=df_vis, x="UMAP1", y="UMAP2", hue="cluster",
                palette="tab10", s=20, linewidth=0, alpha=0.9)

plt.title("DBSCAN clusters visualized with UMAP", fontsize=16)
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")

plt.legend(
    title="Cluster",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    borderaxespad=0.
)

plt.tight_layout()
plt.show()

# t-SNE
#tsne = TSNE(n_components=2, 
#            random_state=42, 
#            perplexity=70)  # <-- AGGIUSTA IN BASE ALLE DIMENSIONI DEL DATASET
#X_tsne = tsne.fit_transform(X_scaled)
#
#plt.figure(figsize=(12, 8))
#scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
#                     c=clusters, cmap='tab10', 
#                     s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
#plt.colorbar(scatter, label='Cluster ID')
#plt.xlabel('t-SNE Dimensione 1', fontsize=12)
#plt.ylabel('t-SNE Dimensione 2', fontsize=12)
#plt.title('Clustering DBSCAN - Visualizzazione t-SNE', fontsize=14, fontweight='bold')
#plt.grid(True, alpha=0.3)
#plt.tight_layout()
#plt.show()

# Heatmap of mean features per cluster
#df_with_clusters = df.copy()
#df_with_clusters['cluster'] = clusters
#cluster_means = df_with_clusters[df_with_clusters['cluster'] != -1].groupby('cluster').mean()
#
#n_features_to_show = min(8, len(df.columns))  # <-- MODIFICA IL NUMERO DI FEATURES DA MOSTRARE
#plt.figure(figsize=(12, 8))
#sns.heatmap(cluster_means.iloc[:, :n_features_to_show].T, annot=True, fmt='.2f', 
#            cmap='RdYlBu_r', center=0, linewidths=0.5)
#plt.title(f'Profilo Medio Features per Cluster (prime {n_features_to_show} features)', 
#          fontsize=14, fontweight='bold')
#plt.xlabel('Cluster ID', fontsize=12)
#plt.ylabel('Features', fontsize=12)
#plt.tight_layout()
#plt.show()






