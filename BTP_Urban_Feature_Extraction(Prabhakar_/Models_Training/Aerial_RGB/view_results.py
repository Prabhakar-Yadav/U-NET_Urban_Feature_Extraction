"""Quick script to visualise extraction.ipynb outputs — run from venv310"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

CLASS_NAMES  = ['Ignore','Water','Buildings','Roads','Vegetation','Open Ground','Other Land']
CLASS_COLORS = ['black','deepskyblue','red','gray','green','yellow','orange']

with rasterio.open('GB_NM_normalized.tif') as s:  norm = s.read()
with rasterio.open('GB_NM_pseudolabels.tif') as s: lbls = s.read(1)
with rasterio.open('GB_NM_clusters.tif') as s:     clus = s.read(1)

rgb = np.clip(norm[:3].transpose(1,2,0), 0, 1)

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
axes[0].imshow(rgb);  axes[0].set_title('RGB (normalized)'); axes[0].axis('off')
axes[1].imshow(clus, cmap='tab10'); axes[1].set_title('KMeans Clusters (raw)'); axes[1].axis('off')
axes[2].imshow(lbls, cmap='tab10', vmin=0, vmax=9)
axes[2].set_title('Pseudo-labels (mapped to classes)'); axes[2].axis('off')

legend = [mpatches.Patch(color=c, label=n)
          for c, n in zip(CLASS_COLORS, CLASS_NAMES)]
axes[2].legend(handles=legend, bbox_to_anchor=(1.02,1), loc='upper left', fontsize=9)

plt.suptitle('GB_NM extraction.ipynb outputs', fontsize=14)
plt.tight_layout()
plt.savefig('results_preview.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: results_preview.png')
