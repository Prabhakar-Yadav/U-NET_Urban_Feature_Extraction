import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Path to original image
IMAGE_PATH = r"C:\Users\PRABHAKAR\Downloads\OneDrive_1_15-12-2025\VASHI_2020.tif"

with rasterio.open(IMAGE_PATH) as src:
    nir   = src.read(4).astype(float)  # NIR
    red   = src.read(3).astype(float)  # Red
    green = src.read(2).astype(float)  # Green

# Stack as FCC (R=NIR, G=Red, B=Green)
fcc = np.stack([nir, red, green], axis=-1)

# Contrast stretching (very important)
def stretch(img, pmin=2, pmax=98):
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(3):
        vmin, vmax = np.percentile(img[:, :, i], (pmin, pmax))
        out[:, :, i] = np.clip((img[:, :, i] - vmin) / (vmax - vmin), 0, 1)
    return out

fcc_stretched = stretch(fcc)

# Show FCC
plt.figure(figsize=(6, 6))
plt.imshow(fcc_stretched)
plt.title("False Color Composite (FCC)")
plt.axis("off")
plt.show()
