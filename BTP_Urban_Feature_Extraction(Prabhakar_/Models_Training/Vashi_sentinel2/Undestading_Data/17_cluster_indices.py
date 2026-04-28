import rasterio
import numpy as np
from sklearn.cluster import KMeans

with rasterio.open("160_feature_stack_indices.tif") as src:
    feats = src.read()
    profile = src.profile

bands, H, W = feats.shape

valid_mask = np.any(feats[:4] > 0.01, axis=0)
X = feats[:, valid_mask].T
X = np.nan_to_num(X)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

cluster_map = np.full((H, W), -1, dtype=np.int16)
cluster_map[valid_mask] = labels

profile.update(count=1, dtype="int16")

with rasterio.open("170_clusters_indices.tif", "w", **profile) as dst:
    dst.write(cluster_map, 1)

print("Saved: 170_clusters_indices.tif")
