import rasterio
import numpy as np

with rasterio.open("130_clusters_masked.tif") as src:
    clusters = src.read(1)

with rasterio.open("110_feature_stack_combined.tif") as src:
    features = src.read()

# valid pixels only (exclude background = -1)
valid_mask = clusters != -1
cluster_ids = np.unique(clusters[valid_mask])

print("Masked Cluster Interpretation:")
for cid in cluster_ids:
    mask = clusters == cid

    mean_band1 = features[0][mask].mean()
    mean_var   = features[4][mask].mean()
    mean_ndvi  = features[-1][mask].mean()

    print(f"\nCluster {cid}:")
    print(f"  Mean Band1   : {mean_band1:.4f}")
    print(f"  Mean Variance: {mean_var:.6f}")
    print(f"  Mean NDVI    : {mean_ndvi:.4f}")
