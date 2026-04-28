# Urban Feature Platform

Interactive web platform for running the trained urban feature extraction models on uploaded imagery.

This app is kept separate from the training pipeline and lives fully inside:

- `web_platform/`

It is designed for non-technical users:

- upload an image
- let the platform pick the best trained model, or choose one manually
- review the marked result image, segmentation map, and class percentages
- download the generated outputs

## What the platform uses

The current production platform uses the trained project models committed in this project package:

- `../trained_models/jnpa_dense_unet_v128`
- `../trained_models/cartosat_dense_unet_v128`
- `../trained_models/cartosat_patch_classifier_v3_4class`

No outside dataset is required for day-to-day inference.

## Main capabilities

- Accepts `png`, `jpg`, `jpeg`, `tif`, `tiff`, `bmp`, `webp`, `gif`
- Supports large satellite uploads up to roughly `500 MB` with extra headroom in server config
- Converts uploaded imagery into the grayscale representation expected by the trained models
- Supports `Auto` mode and `Manual` model selection
- Returns:
  - marked preview image
  - segmentation preview image
  - class percentage breakdown
  - candidate model ranking
  - downloadable result files
- Exports georeferenced label GeoTIFF when the upload is a raster with spatial metadata
- Keeps outputs in distinct timestamped result folders
- Uses a registry-based backend so retrained models or new datasets can be added without changing UI code
- Uses streaming raster inference for large TIFF scenes so they do not need to be fully expanded in memory

## Auto selection behavior

`Auto` mode does not simply choose the model with the highest historical accuracy.

It compares enabled models using:

- normalized prediction confidence
- prediction certainty
- cluster separation
- stored validation accuracy
- domain similarity between the uploaded scene texture profile and each model's original training imagery

This makes selection more robust when multiple trained models are available.

The first request for a model may take a little longer because the app computes and caches a domain profile from that model's source training image.

## Run locally

```bash
cd web_platform
python3 app.py
```

Open:

```text
http://127.0.0.1:5050
```

## Production run

```bash
cd web_platform
gunicorn --config gunicorn.conf.py wsgi:app
```

## Result files

Every prediction creates a distinct folder under:

- `runtime/results/<timestamp>_<id>/`

Typical contents:

- `input_preview.png`
- `marked_preview.png`
- `segmentation_preview.png`
- `marked_full.png` when the image is small enough for safe full-size rendering
- `segmentation_full.png` when the image is small enough for safe full-size rendering
- `segmentation_labels.npy`
- `segmentation_labels_georef.tif` for geospatial raster inputs
- `class_shapefiles_georef.zip` for geospatial raster inputs, containing one georeferenced ESRI Shapefile per detected class
- `class_shapefiles/` with the extracted class-wise vector layers
- `patch_predictions.csv`
- `class_percentages.json`
- `prediction_summary.json`
- `RESULTS_SIMPLE.txt`

For very large scenes, the platform always generates preview outputs, georeferenced labels, and class-wise vector shapefiles when raster georeferencing is available, while skipping unsafe full-size PNG rendering when needed.

## Folder structure

- `app.py` - local dev entry point
- `wsgi.py` - production entry point
- `gunicorn.conf.py` - production process settings
- `config/model_registry.json` - model registration and thresholds
- `platform_app/` - backend service, inference logic, routes, templates, static assets
- `runtime/uploads/` - uploaded files
- `runtime/results/` - prediction outputs
- `runtime/cache/domain_profiles/` - cached model domain signatures used by auto selection
- `docs/` - deployment and model-management documentation

## How retraining and new datasets work

You do not need to rewrite the web app when a model is retrained.

The app is registry-driven:

1. Train a model and save the normal V2 output bundle.
2. Add or update one entry in `config/model_registry.json`.
3. Restart the app.

See:

- `docs/MODEL_MANAGEMENT.md`
- `docs/DEPLOYMENT.md`
- `docs/API_USAGE.md`

## Important note on accuracy claims

The current models are trained from cluster-derived pseudo-labels.
So platform predictions are legitimate outputs from the trained project models,
but accuracy values reflect the pseudo-label setup, not manually annotated ground truth.
