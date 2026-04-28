# API Usage

## Base routes

- `GET /`
- `GET /health`
- `GET /api/models`
- `POST /api/predict`
- `GET /results/<result_id>/<filename>`

## Health check

```text
GET /health
```

Example response:

```json
{
  "status": "ok",
  "enabled_models": 2
}
```

## List models

```text
GET /api/models
```

Returns the enabled models shown in the UI.

## Run prediction

```text
POST /api/predict
Content-Type: multipart/form-data
```

Form fields:

- `image` - uploaded file
- `model_mode` - `auto` or `manual`
- `model_id` - required only when `model_mode=manual`

Example with `curl`:

```bash
curl -X POST http://127.0.0.1:5050/api/predict \
  -F "image=@/path/to/input.tif" \
  -F "model_mode=auto"
```

Manual example:

```bash
curl -X POST http://127.0.0.1:5050/api/predict \
  -F "image=@/path/to/input.png" \
  -F "model_mode=manual" \
  -F "model_id=jnpa_v2_prod"
```

## Response structure

Example high-level response:

```json
{
  "result_id": "20260323_000000_ab12cd34",
  "image": {
    "loader": "rasterio",
    "width": 768,
    "height": 768
  },
  "selection_mode": "auto",
  "selected_model": {
    "id": "jnpa_v2_prod",
    "display_name": "JNPA V2 Patch Classifier",
    "dataset_name": "JNPA 2.5m PAN Patch Classifier V2"
  },
  "score_breakdown": {
    "final_score": 0.61,
    "domain_similarity": 0.68
  },
  "class_percentages": [
    {
      "class_id": 4,
      "class_name": "Urban Built-up",
      "percentage": 52.18
    }
  ],
  "candidate_rankings": [],
  "legend": [],
  "artifacts": {
    "marked_preview_url": "/results/.../marked_preview.png",
    "segmentation_preview_url": "/results/.../segmentation_preview.png",
    "class_shapefiles_zip_url": "/results/.../class_shapefiles_georef.zip"
  },
  "note": "..."
}
```

## Artifact URLs

The API returns relative URLs for generated files, for example:

- `marked_preview_url`
- `segmentation_preview_url`
- `marked_full_url`
- `segmentation_full_url`
- `labels_npy_url`
- `geotiff_url`
- `class_shapefiles_zip_url`
- `summary_json_url`
- `summary_text_url`
- `patch_predictions_csv_url`
- `class_percentages_url`

## Error behavior

Common `400` cases:

- missing image
- unsupported extension
- invalid mode
- manual mode without `model_id`
- unknown model id

Common `500` case:

- runtime inference failure

## Supported file types

- `.png`
- `.jpg`
- `.jpeg`
- `.tif`
- `.tiff`
- `.bmp`
- `.webp`
- `.gif`

## Large upload note

The platform is configured for large satellite uploads up to roughly `500 MB`.

For large GeoTIFF inputs:

- the file is saved to disk first
- inference runs in streaming patch batches
- preview artifacts are always returned
- full-size PNG artifacts may be skipped automatically if the scene is too large
