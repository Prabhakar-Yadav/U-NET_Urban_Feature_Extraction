import rasterio
import numpy as np

with rasterio.open("20_VASHI_2020_norm.tif") as src:
    profile = src.profile
    red = src.read(3)
    nir = src.read(4)

# Built-up proxy: concrete reflects more in red than NIR
built_up = red / (nir + 1e-6)

profile.update(count=1, dtype="float32")

with rasterio.open("150_feature_builtup.tif", "w", **profile) as dst:
    dst.write(built_up, 1)

print("Saved: 150_feature_builtup.tif")
