import rasterio
import numpy as np
from sklearn.cluster import KMeans

with rasterio.open("70_feature_stack_final.tif") as src:
    features = src.read()
    profile = src.profile

bands, H, W = features.shape
X = features.reshape(bands, -1).T
X = np.nan_to_num(X)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X[np.random.choice(X.shape[0], 50000, replace=False)])

labels = kmeans.predict(X).reshape(H, W).astype(np.uint8)

profile.update(count=1, dtype="uint8")

with rasterio.open("80_clusters.tif", "w", **profile) as dst:
    dst.write(labels, 1)

print("Saved: 80_clusters.tif")
