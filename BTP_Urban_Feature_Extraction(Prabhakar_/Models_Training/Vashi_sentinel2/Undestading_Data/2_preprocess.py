import rasterio
import numpy as np

IMAGE_PATH = r"C:\Users\PRABHAKAR\Downloads\OneDrive_1_15-12-2025\VASHI_2020.tif"
OUT_PATH = "30_VASHI_2020_norm.tif"

with rasterio.open(IMAGE_PATH) as src:
    profile = src.profile
    img = src.read().astype(np.float32)

img_norm = (img - img.min()) / (img.max() - img.min())

profile.update(dtype="float32")

with rasterio.open(OUT_PATH, "w", **profile) as dst:
    dst.write(img_norm)

print("Saved:", OUT_PATH)

