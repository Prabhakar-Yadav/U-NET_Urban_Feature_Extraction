from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import rasterio
import torch
from PIL import Image

from .image_utils import LoadedImage, robust_normalize
from .model_registry import ModelRecord
from .prabhakar_models import (
    load_prabhakar_maskrcnn,
    load_prabhakar_resnet18,
    load_prabhakar_unet,
)

if TYPE_CHECKING:
    from .inference import ModelPrediction

NO_DATA_LABEL = 255


def _make_prediction(**kwargs) -> ModelPrediction:
    from .inference import ModelPrediction
    return ModelPrediction(**kwargs)


def _class_percentage_table(
    class_pixel_counts: np.ndarray,
    class_names: dict[int, str],
) -> list[dict[str, Any]]:
    total_pixels = int(class_pixel_counts.sum())
    rows = []
    for class_id, class_name in class_names.items():
        class_pixels = int(class_pixel_counts[class_id]) if class_id < len(class_pixel_counts) else 0
        percent = (100.0 * class_pixels / total_pixels) if total_pixels else 0.0
        rows.append({
            "class_id": int(class_id),
            "class_name": class_name,
            "pixel_count": class_pixels,
            "percentage": round(percent, 2),
        })
    rows.sort(key=lambda item: item["percentage"], reverse=True)
    return rows


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _read_multiband_raster(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        band_count = src.count
        height, width = src.height, src.width
        nodata = src.nodata

        if band_count >= 4:
            bands = src.read([1, 2, 3, 4]).astype(np.float32)
        elif band_count >= 3:
            bands = src.read([1, 2, 3]).astype(np.float32)
        else:
            bands = src.read(1).astype(np.float32)[np.newaxis, ...]

        valid_mask = np.ones((height, width), dtype=bool)
        if nodata is not None:
            valid_mask &= np.all(np.not_equal(bands, nodata), axis=0)

        normalized_bands = []
        for i in range(bands.shape[0]):
            norm, _ = robust_normalize(bands[i], valid_mask)
            normalized_bands.append(norm)

    return np.stack(normalized_bands, axis=0), valid_mask


def _read_multiband_pillow(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(path).convert("RGB")
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    bands = rgb.transpose(2, 0, 1)
    valid_mask = np.ones(bands.shape[1:], dtype=bool)
    return bands, valid_mask


def load_multiband_image(path: Path) -> tuple[np.ndarray, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        return _read_multiband_raster(path)
    return _read_multiband_pillow(path)


def _hann_window_2d(size: int) -> np.ndarray:
    h = np.hanning(size)
    return np.outer(h, h).astype(np.float32)


def _build_score_breakdown(mean_conf: float, val_acc: float, score: float, **extra: Any) -> dict[str, Any]:
    bd = {
        "confidence_mean": round(mean_conf, 4),
        "adjusted_confidence_mean": round(mean_conf, 4),
        "certainty": round(mean_conf, 4),
        "confidence_p90": round(mean_conf, 4),
        "adjusted_confidence_p90": round(mean_conf, 4),
        "cluster_separation": 0.0,
        "validation_accuracy": round(val_acc, 4),
        "domain_similarity": 0.5,
        "final_score": round(score, 4),
    }
    bd.update(extra)
    return bd


def _compute_patch_stats(
    pred_map: np.ndarray,
    confidence_map: np.ndarray,
    valid_mask: np.ndarray,
    height: int,
    width: int,
    patch_size: int,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = int(np.ceil(height / patch_size))
    cols = int(np.ceil(width / patch_size))
    patch_labels = np.zeros(rows * cols, dtype=np.uint8)
    patch_confidences = np.zeros(rows * cols, dtype=np.float32)
    patch_coverages = np.zeros(rows * cols, dtype=np.int64)

    for r in range(rows):
        for c_idx in range(cols):
            y0, x0 = r * patch_size, c_idx * patch_size
            y1, x1 = min(y0 + patch_size, height), min(x0 + patch_size, width)
            flat_idx = r * cols + c_idx
            region_valid = valid_mask[y0:y1, x0:x1]
            patch_coverages[flat_idx] = int(region_valid.sum())
            valid_pixels = pred_map[y0:y1, x0:x1][region_valid]
            if valid_pixels.size:
                patch_labels[flat_idx] = int(np.bincount(valid_pixels, minlength=num_classes).argmax())
                patch_confidences[flat_idx] = float(confidence_map[y0:y1, x0:x1][region_valid].mean())

    grid = patch_labels.reshape(rows, cols)
    return grid, patch_labels, patch_confidences, patch_coverages


def predict_prabhakar_unet(
    loaded_image: LoadedImage,
    record: ModelRecord,
    batch_size: int = 8,
) -> ModelPrediction:
    device = _resolve_device()
    model, ckpt = load_prabhakar_unet(str(record.model_path), device)
    in_channels = ckpt.get("in_channels", 4)
    num_classes = ckpt.get("num_classes", 7)

    bands, valid_mask = load_multiband_image(loaded_image.source_path)
    actual_channels = bands.shape[0]
    height, width = bands.shape[1], bands.shape[2]

    if actual_channels < in_channels:
        bands = np.concatenate([bands, np.zeros((in_channels - actual_channels, height, width), dtype=np.float32)], axis=0)
    elif actual_channels > in_channels:
        bands = bands[:in_channels]

    patch_size = record.patch_size
    stride = patch_size // 2
    hann = _hann_window_2d(patch_size)

    pad_h = (patch_size - height % patch_size) % patch_size
    pad_w = (patch_size - width % patch_size) % patch_size
    if pad_h or pad_w:
        bands = np.pad(bands, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
        valid_mask = np.pad(valid_mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=False)

    padded_h, padded_w = bands.shape[1], bands.shape[2]
    accumulator = np.zeros((num_classes, padded_h, padded_w), dtype=np.float32)
    weight_map = np.zeros((padded_h, padded_w), dtype=np.float32)

    positions = [
        (y0, x0)
        for y0 in range(0, padded_h - patch_size + 1, stride)
        for x0 in range(0, padded_w - patch_size + 1, stride)
    ]

    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            end = min(start + batch_size, len(positions))
            batch_patches = [bands[:, y0:y0 + patch_size, x0:x0 + patch_size] for y0, x0 in positions[start:end]]
            logits = model(torch.from_numpy(np.stack(batch_patches)).float().to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            for idx, (y0, x0) in enumerate(positions[start:end]):
                for c in range(num_classes):
                    accumulator[c, y0:y0 + patch_size, x0:x0 + patch_size] += probs[idx, c] * hann
                weight_map[y0:y0 + patch_size, x0:x0 + patch_size] += hann

    weight_map = np.maximum(weight_map, 1e-8)
    for c in range(num_classes):
        accumulator[c] /= weight_map

    accumulator = accumulator[:, :height, :width]
    valid_mask = valid_mask[:height, :width]

    pred_map = accumulator.argmax(axis=0).astype(np.uint8)
    confidence_map = accumulator.max(axis=0).astype(np.float32)
    pred_map[~valid_mask] = NO_DATA_LABEL

    class_pixel_counts = np.zeros(num_classes, dtype=np.int64)
    for c in range(num_classes):
        class_pixel_counts[c] = int(np.count_nonzero((pred_map == c) & valid_mask))

    class_percentages = _class_percentage_table(class_pixel_counts, record.class_names)
    dominant_class_name = class_percentages[0]["class_name"] if class_percentages else "Unknown"

    grid, patch_labels, patch_confidences, patch_coverages = _compute_patch_stats(
        pred_map, confidence_map, valid_mask, height, width, patch_size, num_classes,
    )

    mean_conf = float(patch_confidences[patch_coverages > 0].mean()) if np.any(patch_coverages > 0) else 0.5
    val_acc = float(record.model_metrics.get("val_accuracy", 0.0) or 0.0)
    score = 0.6 * mean_conf + 0.4 * val_acc

    return _make_prediction(
        record=record,
        patch_grid=grid,
        patch_confidences=patch_confidences,
        patch_labels=patch_labels,
        patch_coverages=patch_coverages,
        class_percentages=class_percentages,
        selected_score=score,
        score_breakdown=_build_score_breakdown(mean_conf, val_acc, score),
        heuristic_water_overrides=0,
        dominant_class_name=dominant_class_name,
        full_label_map=pred_map,
    )


def predict_prabhakar_resnet18(
    loaded_image: LoadedImage,
    record: ModelRecord,
    batch_size: int = 32,
) -> ModelPrediction:
    device = _resolve_device()
    model, ckpt = load_prabhakar_resnet18(str(record.model_path), device)

    crop_size = ckpt.get("crop_size", 256)
    imagenet_mean = ckpt.get("imagenet_mean", [0.485, 0.456, 0.406])
    imagenet_std = ckpt.get("imagenet_std", [0.229, 0.224, 0.225])
    num_classes = len(record.class_names)

    bands, valid_mask = load_multiband_image(loaded_image.source_path)
    if bands.shape[0] == 1:
        bands = np.repeat(bands, 3, axis=0)
    elif bands.shape[0] > 3:
        bands = bands[:3]
    height, width = bands.shape[1], bands.shape[2]

    patch_size = record.patch_size
    rows = int(np.ceil(height / patch_size))
    cols = int(np.ceil(width / patch_size))
    total = rows * cols

    patch_labels = np.zeros(total, dtype=np.uint8)
    patch_confidences = np.zeros(total, dtype=np.float32)
    patch_coverages = np.zeros(total, dtype=np.int64)

    mean_t = torch.tensor(imagenet_mean, dtype=torch.float32).view(3, 1, 1).to(device)
    std_t = torch.tensor(imagenet_std, dtype=torch.float32).view(3, 1, 1).to(device)

    batch_patches: list[np.ndarray] = []
    batch_indices: list[int] = []

    def flush():
        if not batch_patches:
            return
        tensor = torch.from_numpy(np.stack(batch_patches)).float().to(device)
        tensor = (tensor - mean_t) / std_t
        resized = torch.nn.functional.interpolate(tensor, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
        with torch.no_grad():
            logits = model(resized)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        for local_idx, flat_idx in enumerate(batch_indices):
            patch_labels[flat_idx] = probs[local_idx].argmax()
            patch_confidences[flat_idx] = probs[local_idx].max()
        batch_patches.clear()
        batch_indices.clear()

    for r in range(rows):
        for c in range(cols):
            y0, x0 = r * patch_size, c * patch_size
            y1, x1 = min(y0 + patch_size, height), min(x0 + patch_size, width)
            patch = bands[:, y0:y1, x0:x1]
            flat_idx = r * cols + c
            patch_coverages[flat_idx] = int(valid_mask[y0:y1, x0:x1].sum())

            if patch.shape[1] < patch_size or patch.shape[2] < patch_size:
                padded = np.zeros((3, patch_size, patch_size), dtype=np.float32)
                padded[:, :patch.shape[1], :patch.shape[2]] = patch
                patch = padded

            batch_patches.append(patch)
            batch_indices.append(flat_idx)
            if len(batch_patches) >= batch_size:
                flush()
    flush()

    class_pixel_counts = np.zeros(num_classes, dtype=np.int64)
    for flat_idx in range(total):
        class_pixel_counts[patch_labels[flat_idx]] += patch_coverages[flat_idx]

    class_percentages = _class_percentage_table(class_pixel_counts, record.class_names)
    dominant_class_name = class_percentages[0]["class_name"] if class_percentages else "Unknown"

    mean_conf = float(patch_confidences.mean())
    val_acc = float(record.model_metrics.get("val_accuracy", 0.0) or 0.0)
    score = 0.6 * mean_conf + 0.4 * val_acc

    return _make_prediction(
        record=record,
        patch_grid=patch_labels.reshape(rows, cols),
        patch_confidences=patch_confidences,
        patch_labels=patch_labels,
        patch_coverages=patch_coverages,
        class_percentages=class_percentages,
        selected_score=score,
        score_breakdown=_build_score_breakdown(mean_conf, val_acc, score,
            confidence_p90=round(float(np.percentile(patch_confidences, 90)), 4),
            adjusted_confidence_p90=round(float(np.percentile(patch_confidences, 90)), 4),
        ),
        heuristic_water_overrides=0,
        dominant_class_name=dominant_class_name,
    )


def predict_prabhakar_maskrcnn(
    loaded_image: LoadedImage,
    record: ModelRecord,
) -> ModelPrediction:
    device = _resolve_device()
    model, ckpt = load_prabhakar_maskrcnn(str(record.model_path), device)
    num_classes = len(record.class_names)

    bands, valid_mask = load_multiband_image(loaded_image.source_path)
    if bands.shape[0] == 1:
        bands = np.repeat(bands, 3, axis=0)
    elif bands.shape[0] > 3:
        bands = bands[:3]
    height, width = bands.shape[1], bands.shape[2]

    score_threshold = 0.5
    pred_map = np.zeros((height, width), dtype=np.uint8)
    confidence_map = np.zeros((height, width), dtype=np.float32)
    num_detections = 0

    tile_size = 1024
    overlap = 128

    with torch.no_grad():
        for y0 in range(0, height, tile_size - overlap):
            for x0 in range(0, width, tile_size - overlap):
                y1 = min(y0 + tile_size, height)
                x1 = min(x0 + tile_size, width)
                tile = bands[:, y0:y1, x0:x1]
                tile_tensor = torch.from_numpy(tile).float().to(device)
                outputs = model([tile_tensor])[0]

                if len(outputs["scores"]) == 0:
                    continue

                keep = outputs["scores"] >= score_threshold
                if not keep.any():
                    continue

                masks = outputs["masks"][keep].cpu().numpy()
                labels = outputs["labels"][keep].cpu().numpy()
                scores = outputs["scores"][keep].cpu().numpy()
                num_detections += int(keep.sum())

                for idx in np.argsort(scores):
                    mask = masks[idx, 0] > 0.5
                    th, tw = mask.shape
                    pred_map[y0:y0 + th, x0:x0 + tw][mask] = labels[idx]
                    confidence_map[y0:y0 + th, x0:x0 + tw][mask] = scores[idx]

    pred_map[~valid_mask] = NO_DATA_LABEL

    class_pixel_counts = np.zeros(num_classes, dtype=np.int64)
    for c in range(num_classes):
        class_pixel_counts[c] = int(np.count_nonzero((pred_map == c) & valid_mask))

    class_percentages = _class_percentage_table(class_pixel_counts, record.class_names)
    dominant_class_name = class_percentages[0]["class_name"] if class_percentages else "Unknown"

    patch_size = record.patch_size
    grid, patch_labels, patch_confidences, patch_coverages = _compute_patch_stats(
        pred_map, confidence_map, valid_mask, height, width, patch_size, num_classes,
    )

    mean_conf = float(patch_confidences[patch_coverages > 0].mean()) if np.any(patch_coverages > 0) else 0.5
    val_acc = float(record.model_metrics.get("val_accuracy", 0.0) or 0.0)
    score = 0.6 * mean_conf + 0.4 * val_acc

    return _make_prediction(
        record=record,
        patch_grid=grid,
        patch_confidences=patch_confidences,
        patch_labels=patch_labels,
        patch_coverages=patch_coverages,
        class_percentages=class_percentages,
        selected_score=score,
        score_breakdown=_build_score_breakdown(mean_conf, val_acc, score, num_detections=num_detections),
        heuristic_water_overrides=0,
        dominant_class_name=dominant_class_name,
        full_label_map=pred_map,
    )
