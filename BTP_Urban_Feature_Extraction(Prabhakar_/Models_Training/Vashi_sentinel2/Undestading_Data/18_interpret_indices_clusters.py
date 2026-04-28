import rasterio
import numpy as np

with rasterio.open("170_clusters_indices.tif") as src:
    clusters = src.read(1)

with rasterio.open("160_feature_stack_indices.tif") as src:
    feats = src.read()

valid_mask = clusters != -1
cluster_ids = np.unique(clusters[valid_mask])

print("Cluster interpretation (spectral-index based):")

for cid in cluster_ids:
    mask = clusters == cid

    mean_ndvi = feats[5][mask].mean()
    mean_ndwi = feats[6][mask].mean()
    mean_built = feats[7][mask].mean()
    mean_var = feats[4][mask].mean()

    print(f"\nCluster {cid}:")
    print(f"  Mean NDVI   : {mean_ndvi:.4f}")
    print(f"  Mean NDWI   : {mean_ndwi:.4f}")
    print(f"  Mean Built : {mean_built:.4f}")
    print(f"  Mean Var   : {mean_var:.6f}")
