# Model Management

## Goal

The platform is designed so you can:

- retrain an existing model
- add a new dataset model
- enable or disable models
- keep the same UI and API

without rewriting frontend code.

## Model contract

The platform currently supports two model types:

- `patch_classifier`
- `unet`
- `dense_unet`

### Patch classifier contract

Each production patch-classifier entry should provide the same artifact structure that comes from the V2/V3 pipeline:

1. `run_summary_v2.json`
2. `patch_classifier_model_v2.joblib`

These are the outputs produced by:

- `urban_feature_patch_classifier_v2.py`
- `urban_feature_patch_classifier_v3.py`

### U-Net contract

Each U-Net comparison entry should provide:

1. `run_summary.json`
2. `best_model.pt`

These are the outputs produced by:

- `urban_feature_pipeline.py`

### Dense U-Net contract

Each dense-segmentation production entry should provide:

1. `run_summary_dense_v128.json`
2. `hard_unet_best_model.pt` or `soft_unet_best_model.pt`

These are the outputs produced by:

- `urban_feature_dense_segmentation_v128.py`

## Registry-driven setup

The app reads enabled models from:

- `web_platform/config/model_registry.json`

Each entry should contain:

- `id`
- `display_name`
- `model_type`
- `auto_select`
- `summary_path`
- `model_path`
- `patch_size`
- `std_threshold`
- `valid_fraction_threshold`
- `dark_water_mean_threshold`
- `enabled`
- `priority`

## Current example

```json
{
  "id": "jnpa_v2_prod",
  "display_name": "JNPA V2 Patch Classifier",
  "model_type": "patch_classifier",
  "auto_select": true,
  "summary_path": "../../trained_models/jnpa_patch_classifier_v2/run_summary_v2.json",
  "model_path": "../../trained_models/jnpa_patch_classifier_v2/patch_classifier_model_v2.joblib",
  "patch_size": 256,
  "std_threshold": 0.045,
  "valid_fraction_threshold": 0.85,
  "dark_water_mean_threshold": 0.32,
  "enabled": true,
  "priority": 100
}
```

## How to replace a retrained model

Example:

1. Train a fresh model into a distinct output folder, for example:
   `../../trained_models/jnpa_patch_classifier_v3`
2. Confirm that folder contains:
   - `run_summary_v2.json`
   - `patch_classifier_model_v2.joblib`
3. Open:
   `web_platform/config/model_registry.json`
4. Update the `summary_path` and `model_path` for the existing model entry
5. If thresholds changed during retraining, also update:
   - `patch_size`
   - `std_threshold`
   - `valid_fraction_threshold`
   - `dark_water_mean_threshold`
6. Keep `auto_select: true` only for models you want considered by automatic model selection
7. Restart the Flask or Gunicorn app

## How to add a new dataset model

1. Train the dataset and save its output bundle in a new folder under `../../trained_models/`
2. Add a new entry in `config/model_registry.json`
3. Restart the app

Example:

```json
{
  "id": "new_dataset_v1",
  "display_name": "New Dataset Patch Classifier",
  "model_type": "patch_classifier",
  "auto_select": true,
  "summary_path": "../../trained_models/new_dataset_patch_classifier_v1/run_summary_v2.json",
  "model_path": "../../trained_models/new_dataset_patch_classifier_v1/patch_classifier_model_v2.joblib",
  "patch_size": 256,
  "std_threshold": 0.05,
  "valid_fraction_threshold": 0.9,
  "dark_water_mean_threshold": 0.28,
  "enabled": true,
  "priority": 80
}
```

Dense-segmentation example:

```json
{
  "id": "jnpa_dense_v128_prod",
  "display_name": "JNPA Dense U-Net V128 (Soft Labels)",
  "model_type": "dense_unet",
  "auto_select": true,
  "summary_path": "../../trained_models/jnpa_dense_unet_v128/run_summary_dense_v128.json",
  "model_path": "../../trained_models/jnpa_dense_unet_v128/soft_unet_best_model.pt",
  "patch_size": 128,
  "std_threshold": 0.02,
  "valid_fraction_threshold": 0.85,
  "dark_water_mean_threshold": 0.32,
  "enabled": true,
  "priority": 120
}
```

## What happens automatically

Once a model is registered, the platform automatically:

- loads the model bundle lazily
- exposes it in the web UI
- makes it available to the API
- considers it in `Auto` mode if `auto_select` is `true`
- computes and caches a domain signature in:
  `runtime/cache/domain_profiles/`

No template edits or route edits are required.

## Auto-mode selection logic

Auto mode combines:

- filename dataset hints when uploads already contain `JNPA` or `CARTOSAT`
- normalized model confidence
- prediction certainty
- cluster separation for patch-classifier models
- stored validation accuracy
- domain similarity between the upload and the model's own training-image texture profile

U-Net comparison models can still be used in manual mode even when they are excluded from Auto mode.
Dense U-Net production models return true per-pixel segmentation maps instead of one uniform class per patch.

This keeps the selector extendable even when you add more dataset-specific models later.

## Recommended naming

Use distinct output folders and keep versioned model IDs.

Good examples:

- `jnpa_v2_prod`
- `jnpa_v3_experiment`
- `cartosat_v2_prod`
- `coastal_dataset_v1_prod`

## Important note for reports and presentations

The current production models are trained on pseudo-labels derived from clustering.

So:

- the app predictions are real outputs from the trained project models
- the stored validation scores are legitimate for that pseudo-label setup
- those scores should not be presented as manual ground-truth segmentation accuracy unless you later add annotated masks
