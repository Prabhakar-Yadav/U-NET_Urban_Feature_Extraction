import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Load clusters
with rasterio.open("170_clusters_indices.tif") as src:
    clusters = src.read(1)

# Load RGB for reference
with rasterio.open("20_VASHI_2020_norm.tif") as src:
    rgb = src.read([3, 2, 1])  # RGB

valid_mask = clusters != -1

# Show RGB
plt.figure(figsize=(6,6))
plt.imshow(np.transpose(rgb, (1,2,0)))
plt.title("Original Image (RGB)")
plt.axis("off")
plt.show()

# Show clusters
plt.figure(figsize=(6,6))
cluster_vis = clusters.astype(float)
cluster_vis[~valid_mask] = np.nan
plt.imshow(cluster_vis, cmap="tab10")
plt.title("Clusters (Spectral Indices Based)")
plt.colorbar(label="Cluster ID")
plt.axis("off")
plt.show()

# # Show each cluster separately
# for cid in np.unique(clusters[valid_mask]):
#     plt.figure(figsize=(5,5))
#     plt.imshow(clusters == cid, cmap="tab20")  # categorical colormap
#     plt.title(f"Cluster {cid}")
#     plt.axis("off")
#     plt.show()

colormaps = ["Reds", "Blues", "Greens", "Purples", "Oranges"]
for idx, cid in enumerate(np.unique(clusters[valid_mask])):
    plt.figure(figsize=(5,5))
    plt.imshow(clusters == cid, cmap=colormaps[idx % len(colormaps)])
    plt.title(f"Cluster {cid}")
    plt.axis("off")
    plt.show()

    