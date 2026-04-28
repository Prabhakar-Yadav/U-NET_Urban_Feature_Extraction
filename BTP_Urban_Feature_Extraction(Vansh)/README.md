# BTP Panchromatic Urban Feature Extraction

This folder contains the complete B.Tech project package for urban feature extraction from high-resolution panchromatic satellite imagery using deep learning.

The project covers:

- JNPA grayscale GeoTIFF segmentation with 5 classes
- CARTOSAT panchromatic segmentation with refined 4-class labeling
- patch-classifier baselines and dense U-Net models
- a deployable web platform for non-technical users
- final results, documentation, report, and presentation assets

## Folder Guide

```text
BTP_Panchromatic_Urban_Feature_Extraction/
├── notebooks/                 # Main project notebooks and version comparison
├── training_code/             # Python training and evaluation scripts
├── trained_models/            # Final trained weights and model summaries
├── final_results/             # Final maps, summary boards, and test-data evaluation outputs
├── report_and_presentation/   # Final LaTeX report, PDF, PPT, references, and figure assets
├── web_platform/              # Deployable Flask-based inference platform
└── docs/                      # Short text summaries from the project iterations
```

## Included Final Assets

- Final report PDF:
  `report_and_presentation/Urban_Feature_Extraction_Dissertation_Report_Submittable_LaTeX_v2.pdf`
- Final presentation:
  `report_and_presentation/Urban_Feature_Extraction_Final_Presentation_2026_03_25_v5.pptx`
- Final production platform:
  `web_platform/`
- Final production models:
  `trained_models/jnpa_dense_unet_v128/`
  `trained_models/cartosat_dense_unet_v128/`

## Raw Data Note

The original satellite datasets and bulky runtime outputs are not committed here because they are large and environment-specific.

Expected dataset names in the original workspace were:

- `JNPA/JNPA_2_5.tif`
- `Monocromatic/CARTOSAT_1M_PAN.tif`
- `Test_Data/`

See:

- `web_platform/data/README.md`

## Best Final Models

| Dataset | Selected model | Labeling strategy | Validation result |
| --- | --- | --- | --- |
| JNPA | Dense U-Net V128 | Soft/proportion pseudo-labels | 91.07% pixel accuracy, 78.73% mean IoU |
| CARTOSAT | Dense U-Net V128 | Hard dense pseudo-masks | 93.97% pixel accuracy, 77.41% mean IoU |

## Running the Platform

```bash
cd BTP_Panchromatic_Urban_Feature_Extraction/web_platform
pip install -r requirements.txt
python3 app.py
```

Then open:

```text
http://127.0.0.1:5050
```

## Notes

- The platform is registry-driven, so retrained models can be swapped in by updating `web_platform/config/model_registry.json`.
- Cached domain profiles are included for the current published model set so auto-selection works without recomputing them immediately.
- The predictions are based on pseudo-label supervised training, not manually annotated ground-truth segmentation masks.
