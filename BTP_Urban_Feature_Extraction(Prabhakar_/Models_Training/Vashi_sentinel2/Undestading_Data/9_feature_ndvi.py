import rasterio
import numpy as np

with rasterio.open("20_VASHI_2020_norm.tif") as src:
    profile = src.profile
    red = src.read(3)   # Band 3
    nir = src.read(4)   # Band 4 (assumed NIR)

ndvi = (nir - red) / (nir + red + 1e-6)

profile.update(count=1, dtype="float32")

with rasterio.open("90_feature_ndvi.tif", "w", **profile) as dst:
    dst.write(ndvi, 1)

print("Saved: 90_feature_ndvi.tif")
