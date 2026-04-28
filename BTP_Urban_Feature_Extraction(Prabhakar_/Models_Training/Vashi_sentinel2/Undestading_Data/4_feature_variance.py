import rasterio
import numpy as np
import rasterio
import numpy as np
from scipy.ndimage import generic_filter

def local_variance(x):
    return np.var(x)

with rasterio.open("20_VASHI_2020_norm.tif") as src:
    profile = src.profile
    band1 = src.read(1)

variance = generic_filter(band1, local_variance, size=5)

profile.update(count=1, dtype="float32")

with rasterio.open("40_feature_variance_b1.tif", "w", **profile) as dst:
    dst.write(variance, 1)

print("Saved: 40_feature_variance_b1.tif")
