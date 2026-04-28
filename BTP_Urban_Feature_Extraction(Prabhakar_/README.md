# GeoVision - Urban Feature Extraction from Satellite Imagery

A deep learning system for extracting urban and environmental features from multi-resolution satellite imagery, deployed as an interactive web platform.

## Models

| Model | Architecture | Input | Classes | Accuracy |
|-------|-------------|-------|---------|----------|
| U-Net GBNM | U-Net (GroupNorm) | 3-band aerial RGB (1.6 m) | 6 classes | 69.65% OA |
| ResNet-18 Cartosat | ResNet-18 (frozen backbone) | 1-band panchromatic (1 m) | 6 classes | 94.74% OA |
| U-Net Vashi | U-Net (GroupNorm) | 4-band Sentinel-2 (10 m) | 7 classes | 75.71% OA |
| Mask R-CNN | ResNet-50 + FPN | 3-band RGB | Building detection | 87.9% recall |

## Project Structure

```
├── WebPlatform/              # Streamlit web application
│   ├── app.py                # Main app entry point
│   ├── model_utils.py        # Model architectures and loaders
│   ├── inference_utils.py    # Sliding-window inference pipelines
│   ├── viz_utils.py          # Visualization utilities
│   ├── export_utils.py       # GeoTIFF, Shapefile, GeoJSON export
│   ├── labeling_utils.py     # KMeans pseudo-label generation
│   ├── metrics_utils.py      # Accuracy metrics
│   ├── requirements.txt      # Python dependencies
│   └── run.bat               # Windows launcher
├── Aerial_RGB/               # GBNM training notebooks and model
│   ├── CLass_formation.ipynb # Pseudo-label generation
│   ├── train_model.ipynb     # U-Net training
│   └── models/               # Trained checkpoints
├── Cartosat/                 # CARTOSAT training notebooks and model
│   ├── Model.ipynb           # ResNet-18 training
│   └── models/               # Trained checkpoints
├── Vashi_sentinel2/          # Vashi training notebooks and models
│   ├── Model_train.ipynb     # U-Net + Mask R-CNN training
│   └── models/               # Trained checkpoints
└── Test_Model/               # Test evaluation
    ├── test_all_models.ipynb # Evaluation notebook
    ├── geotiff/              # Test images (GeoTIFF)
    └── results/              # Test results and figures
```

## Setup Instructions

### Prerequisites

- Python 3.10
- NVIDIA GPU with CUDA (recommended, not required)
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/Prabhakar-Yadav/Urban-Feature-Ext.git
cd Urban-Feature-Ext
```

### 2. Create Virtual Environment

```bash
python -m venv venv310
```

Activate it:

- **Windows (Command Prompt):**
  ```bash
  venv310\Scripts\activate
  ```
- **Windows (PowerShell):**
  ```bash
  venv310\Scripts\Activate.ps1
  ```
- **Linux / macOS:**
  ```bash
  source venv310/bin/activate
  ```

### 3. Install Dependencies

```bash
pip install -r WebPlatform/requirements.txt
```

For GPU support, install PyTorch with CUDA separately first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r WebPlatform/requirements.txt
```

### 4. Download Model Checkpoints

Model checkpoint files (`.pth`) are tracked with Git LFS. After cloning, pull them:

```bash
git lfs pull
```

If Git LFS is not installed, download the checkpoints manually from the GitHub releases or LFS storage and place them in:

```
Aerial_RGB/models/unet_gbnm_checkpoint.pth
Cartosat/models/cartosat_classifier_checkpoint.pth
Vashi_sentinel2/models/unet_vashi_checkpoint.pth
Vashi_sentinel2/models/maskrcnn_checkpoint.pth
Vashi_sentinel2/models/patchcnn_checkpoint.pth
```

### 5. Run the Web Platform

```bash
cd WebPlatform
streamlit run app.py --server.port 8501
```

Or use the batch launcher on Windows:

```bash
WebPlatform\run.bat
```

The platform opens at `http://localhost:8501`.

## Using the Platform

1. **Upload** a satellite image (GeoTIFF, PNG, JPEG, or NPY)
2. The platform **auto-selects** the best model based on band count:
   - 4+ bands → U-Net Vashi (Sentinel-2)
   - 1 band → ResNet-18 Cartosat (panchromatic)
   - 3 bands → U-Net GBNM (aerial RGB)
3. Click **Run Inference** to generate the classified map
4. View results: segmentation overlay, confidence heatmap, per-class statistics
5. **Export** as GeoTIFF, ESRI Shapefile, or GeoJSON

## Training Notebooks

Each model has a training notebook in its respective directory:

- `Aerial_RGB/CLass_formation.ipynb` — Pseudo-label generation using spectral-index KMeans clustering
- `Aerial_RGB/train_model.ipynb` — U-Net training on aerial RGB imagery
- `Cartosat/Model.ipynb` — ResNet-18 deep-feature clustering and patch classifier training
- `Vashi_sentinel2/Model_train.ipynb` — U-Net, PatchCNN, and Mask R-CNN training on Sentinel-2 data

## Testing

```bash
# Run the test evaluation notebook
jupyter notebook Test_Model/test_all_models.ipynb
```

## Authors

- Prabhakar Yadav (23B0653) — Department of Civil Engineering, IIT Bombay

## Supervisor

- Prof. Pennan Chinnasamy — Department of Civil Engineering, IIT Bombay
