import rasterio
import numpy as np

with rasterio.open("20_VASHI_2020_norm.tif") as src:
    profile = src.profile
    green = src.read(2)   # Band 2
    nir   = src.read(4)   # Band 4

ndwi = (green - nir) / (green + nir + 1e-6)

profile.update(count=1, dtype="float32")

with rasterio.open("140_feature_ndwi.tif", "w", **profile) as dst:
    dst.write(ndwi, 1)

print("Saved: 140_feature_ndwi.tif")
