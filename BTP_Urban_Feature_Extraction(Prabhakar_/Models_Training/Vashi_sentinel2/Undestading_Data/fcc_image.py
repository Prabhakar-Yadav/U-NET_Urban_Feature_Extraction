import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Input image path
IMAGE_PATH = r"C:\Users\PRABHAKAR\Downloads\OneDrive_1_15-12-2025\VASHI_2020.tif"

# Output FCC path
OUT_FCC = "FCC_VASHI_2020.tif"

with rasterio.open(IMAGE_PATH) as src:
    profile = src.profile
    nir   = src.read(4).astype(np.float32)   # Band 4 = NIR
    red   = src.read(3).astype(np.float32)   # Band 3 = Red
    green = src.read(2).astype(np.float32)   # Band 2 = Green

# Stack FCC bands (R=NIR, G=Red, B=Green)
fcc = np.stack([nir, red, green], axis=0)

# Contrast stretching (2–98%)
for i in range(3):
    p2, p98 = np.percentile(fcc[i], (2, 98))
    fcc[i] = np.clip((fcc[i] - p2) / (p98 - p2), 0, 1)

# Update profile for 3-band float image
profile.update(
    count=3,
    dtype="float32"
)

# Save FCC GeoTIFF
with rasterio.open(OUT_FCC, "w", **profile) as dst:
    dst.write(fcc)

print("FCC image saved as:", OUT_FCC)

# Optional: display FCC
fcc_disp = np.transpose(fcc, (1, 2, 0))
plt.figure(figsize=(6,6))
plt.imshow(fcc_disp)
plt.title("False Color Composite (FCC)")
plt.axis("off")
plt.show()
