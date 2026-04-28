import rasterio
import numpy as np

IMAGE_PATH = r"C:\Users\PRABHAKAR\Downloads\OneDrive_1_15-12-2025\VASHI_2020.tif"

with rasterio.open(IMAGE_PATH) as src:
    print("Width:", src.width)
    print("Height:", src.height)
    print("Number of bands:", src.count)
    print("Data type:", src.dtypes)
    print("CRS:", src.crs)

    for i in range(1, src.count + 1):
        band = src.read(i)
        print(f"\nBand {i}:")
        print(" Min:", band.min())
        print(" Max:", band.max())
        print(" Mean:", band.mean())
