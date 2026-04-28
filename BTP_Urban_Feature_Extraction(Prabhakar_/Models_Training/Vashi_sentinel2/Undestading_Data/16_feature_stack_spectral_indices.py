import rasterio
import numpy as np

with rasterio.open("30_features_band.tif") as src:
    profile = src.profile
    bands = src.read()   # B1–B4

with rasterio.open("40_feature_variance_b1.tif") as src:
    var = src.read(1)

with rasterio.open("90_feature_ndvi.tif") as src:
    ndvi = src.read(1)

with rasterio.open("140_feature_ndwi.tif") as src:
    ndwi = src.read(1)

with rasterio.open("150_feature_builtup.tif") as src:
    built = src.read(1)

stack = np.vstack([
    bands,
    var[np.newaxis, :, :],
    ndvi[np.newaxis, :, :],
    ndwi[np.newaxis, :, :],
    built[np.newaxis, :, :]
])

profile.update(count=stack.shape[0], dtype="float32")

with rasterio.open("160_feature_stack_indices.tif", "w", **profile) as dst:
    dst.write(stack)

print("Saved: 160_feature_stack_indices.tif")
