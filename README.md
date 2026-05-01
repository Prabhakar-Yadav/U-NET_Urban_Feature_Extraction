# GeoVision — Urban Feature Extraction Platform

A unified web platform for satellite image classification and segmentation, combining multiple deep learning models trained on panchromatic and multispectral imagery of Indian urban regions.

## Models

| Model | Type | Input | Classes | Dataset |
|-------|------|-------|---------|---------|
| JNPA V2 Patch Classifier | Random Forest | 1-band PAN | 5 | JNPA 2.5m |
| CARTOSAT V3 4-Class Patch Classifier | Random Forest | 1-band PAN | 4 | Cartosat |
| JNPA Dense U-Net V128 | U-Net (dense) | 1-band PAN | 5 | JNPA 2.5m |
| CARTOSAT Dense U-Net V128 | U-Net (dense) | 1-band PAN | 5 | Cartosat |
| JNPA U-Net V1 | U-Net | 1-band PAN | 5 | JNPA 2.5m |
| Vashi U-Net (4-Band) | U-Net | 4-band Sentinel-2 | 7 | Vashi |
| GBNM U-Net (3-Band RGB) | U-Net | 3-band RGB | 7 | Greater Bhiwandi-Nijampur-Mira |
| Cartosat ResNet-18 Classifier | ResNet-18 | 3-band RGB | 6 | Cartosat |
| Vashi Mask R-CNN | Mask R-CNN | 3-band RGB | 2 | Vashi (building detection) |

## Project Structure

```
Feature-Ext(Prabhakar)/
├── web_platform/                    # Unified Flask web platform
│   ├── app.py                       # Entry point
│   ├── config/
│   │   ├── model_registry.json      # All model configurations
│   │   └── prabhakar_models/        # Summary JSONs for multi-band models
│   └── platform_app/
│       ├── inference.py             # Core inference service
│       ├── prabhakar_inference.py   # Multi-band model inference
│       ├── prabhakar_models.py      # U-Net, ResNet-18, Mask R-CNN architectures
│       ├── model_registry.py        # Model loading and metadata
│       ├── image_utils.py           # Image loading and tiling
│       ├── unet_model.py            # Panchromatic U-Net architecture
│       ├── routes.py                # Flask API endpoints
│       ├── config.py                # Platform settings
│       ├── static/                  # CSS and JS
│       └── templates/               # HTML templates
│
├── BTP_Urban_Feature_Extraction(Prabhakar_)/
│   └── Models_Training/
│       ├── Aerial_RGB/              # GBNM RGB U-Net training
│       ├── Cartosat/                # Cartosat ResNet-18 training
│       └── Vashi_sentinel2/         # Vashi U-Net + Mask R-CNN training
│
├── BTP_Urban_Feature_Extraction(Vansh)/
│   ├── trained_models/              # Panchromatic model checkpoints
│   ├── training_code/               # Training scripts
│   ├── notebooks/                   # Training notebooks
│   └── final_results/               # Evaluation results
│
└── Test_Model/                      # Test satellite imagery
```

## Setup

```bash
git clone https://github.com/Prabhakar-Yadav/U-NET_Urban_Feature_Extraction.git
cd U-NET_Urban_Feature_Extraction

# Pull model checkpoints (requires Git LFS)
git lfs pull

# Create virtual environment
python -m venv venv310
venv310\Scripts\activate        # Windows
# source venv310/bin/activate   # Linux/Mac

# Install dependencies
pip install flask torch torchvision rasterio geopandas shapely joblib scikit-learn opencv-python-headless pillow pandas

# Run the platform
cd web_platform
python app.py
```

Open **http://localhost:5050** in your browser.

## Usage

1. Upload a satellite image (PNG, JPEG, GeoTIFF up to 500 MB)
2. Select **Auto** mode for automatic model selection, or **Manual** to pick a specific model
3. View segmentation results, class statistics, and download outputs (GeoTIFF, Shapefiles, CSV)

## Features

- Automatic model selection based on domain similarity scoring
- Supports panchromatic (1-band), RGB (3-band), and multispectral (4-band) imagery
- Streaming mode for large GeoTIFFs (40M+ pixels)
- Sliding-window inference with Hann blending for smooth U-Net predictions
- Instance-level building detection via Mask R-CNN
- GeoTIFF and Shapefile export with CRS preservation
- Per-class area statistics and confidence metrics

## Authors

- Prabhakar Yadav — Multi-band models (U-Net, ResNet-18, Mask R-CNN), platform integration
- Vansh — Panchromatic models (Patch Classifier, Dense U-Net), web platform UI, scoring system
