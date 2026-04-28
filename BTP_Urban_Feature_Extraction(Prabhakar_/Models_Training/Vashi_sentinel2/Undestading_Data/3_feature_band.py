import rasterio

with rasterio.open("20_VASHI_2020_norm.tif") as src:
    profile = src.profile
    bands = src.read()

with rasterio.open("30_features_band.tif", "w", **profile) as dst:
    dst.write(bands)

print("Saved: features_band.tif")
