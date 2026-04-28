from __future__ import annotations

import csv
import json
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio import features as rasterio_features
from shapely.geometry import shape as shapely_shape
import torch
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

from .config import Settings
from .image_utils import (
    LoadedImage,
    TiledImage,
    build_full_label_map,
    ensure_dir,
    load_uploaded_image,
    read_raster_valid_mask,
    read_normalized_raster_patch,
    robust_normalize,
    tile_image,
    to_uint8,
)
from .model_registry import ModelRecord, ModelRegistry
from .unet_model import UNet


NO_DATA_LABEL = 255


@dataclass
class ModelPrediction:
    record: ModelRecord
    patch_grid: np.ndarray
    patch_confidences: np.ndarray
    patch_labels: np.ndarray
    patch_coverages: np.ndarray
    class_percentages: list[dict[str, Any]]
    selected_score: float
    score_breakdown: dict[str, float]
    heuristic_water_overrides: int
    dominant_class_name: str
    full_label_map: np.ndarray | None = None


def compute_patch_features(
    patches: np.ndarray,
    resize_for_features: int,
    canny_low: int,
    canny_high: int,
) -> np.ndarray:
    features = []
    for patch in patches:
        small = cv2.resize(patch, (resize_for_features, resize_for_features), interpolation=cv2.INTER_AREA)
        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        patch_u8 = np.clip(patch * 255.0, 0, 255).astype(np.uint8)
        edges = cv2.Canny(patch_u8, canny_low, canny_high)
        stats = np.array(
            [
                patch.mean(),
                patch.std(),
                np.percentile(patch, 10),
                np.percentile(patch, 90),
                grad.mean(),
                grad.std(),
                edges.mean() / 255.0,
            ],
            dtype=np.float32,
        )
        features.append(np.concatenate([small.flatten(), stats]))
    return np.stack(features).astype(np.float32)


def colorize_labels(label_image: np.ndarray, record: ModelRecord) -> np.ndarray:
    color = np.full((*label_image.shape, 3), 232, dtype=np.uint8)
    for class_id, class_name in record.class_names.items():
        rgb = record.display_class_colors.get(class_id, (200, 200, 200))
        color[label_image == class_id] = rgb
    color[label_image == NO_DATA_LABEL] = (220, 220, 220)
    return color


def resize_label_preview(patch_grid: np.ndarray, width: int, height: int) -> np.ndarray:
    preview = Image.fromarray(patch_grid.astype(np.uint8), mode="L")
    preview = preview.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(preview, dtype=np.uint8)


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-8, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    max_entropy = np.log(probabilities.shape[1]) if probabilities.shape[1] > 1 else 1.0
    return entropy / max(max_entropy, 1e-8)


def cluster_separation_values(reduced_features: np.ndarray, kmeans: Any) -> np.ndarray:
    distances = kmeans.transform(reduced_features)
    if distances.shape[1] < 2:
        return np.full((reduced_features.shape[0],), 0.5, dtype=np.float32)
    sorted_distances = np.sort(distances, axis=1)
    first = sorted_distances[:, 0]
    second = sorted_distances[:, 1]
    separation = np.clip((second - first) / np.maximum(second, 1e-6), 0.0, 1.0)
    return separation.astype(np.float32)


def cluster_separation_score(reduced_features: np.ndarray, kmeans: Any) -> float:
    return float(np.mean(cluster_separation_values(reduced_features, kmeans)))


def class_percentage_table(
    labels: np.ndarray,
    coverages: np.ndarray,
    class_names: dict[int, str],
) -> list[dict[str, Any]]:
    total_pixels = int(coverages.sum())
    rows = []
    for class_id, class_name in class_names.items():
        class_pixels = int(coverages[labels == class_id].sum())
        percent = (100.0 * class_pixels / total_pixels) if total_pixels else 0.0
        rows.append(
            {
                "class_id": int(class_id),
                "class_name": class_name,
                "pixel_count": class_pixels,
                "percentage": round(percent, 2),
            }
        )
    rows.sort(key=lambda item: item["percentage"], reverse=True)
    return rows


def class_percentage_table_from_counts(
    class_pixel_counts: np.ndarray,
    class_names: dict[int, str],
) -> list[dict[str, Any]]:
    total_pixels = int(class_pixel_counts.sum())
    rows = []
    for class_id, class_name in class_names.items():
        class_pixels = int(class_pixel_counts[class_id]) if class_id < len(class_pixel_counts) else 0
        percent = (100.0 * class_pixels / total_pixels) if total_pixels else 0.0
        rows.append(
            {
                "class_id": int(class_id),
                "class_name": class_name,
                "pixel_count": class_pixels,
                "percentage": round(percent, 2),
            }
        )
    rows.sort(key=lambda item: item["percentage"], reverse=True)
    return rows


def normalized_entropy_map(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-8, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    max_entropy = np.log(probabilities.shape[1]) if probabilities.shape[1] > 1 else 1.0
    return entropy / max(max_entropy, 1e-8)


def resolve_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class InferenceService:
    def __init__(self, settings: Settings, registry: ModelRegistry):
        self.settings = settings
        self.registry = registry
        self._domain_profiles: dict[str, dict[str, Any]] = {}
        self._torch_models: dict[str, tuple[UNet, torch.device]] = {}
        self._dense_auto_selector: dict[str, Any] | None = None

    def _domain_profile_cache_path(self, record: ModelRecord) -> Path:
        return self.settings.runtime_dir / "cache" / "domain_profiles" / f"{record.id}.json"

    def _get_domain_profile(self, record: ModelRecord) -> dict[str, Any]:
        if record.id in self._domain_profiles:
            return self._domain_profiles[record.id]

        cache_path = self._domain_profile_cache_path(record)
        ensure_dir(cache_path.parent)
        if cache_path.exists():
            profile = json.loads(cache_path.read_text())
            self._domain_profiles[record.id] = profile
            return profile

        project_root = self.settings.root_dir.parent
        dataset = record.summary["dataset"]
        image_rel_path = dataset.get("image_path") or dataset.get("spec", {}).get("image_path")
        if not image_rel_path:
            raise ValueError(f"Training image path is missing for model {record.id}")
        image_path = (project_root / image_rel_path).resolve()
        with rasterio.open(image_path) as src:
            band = src.read(1).astype(np.float32)
            nodata = src.nodata
            valid_mask = np.ones_like(band, dtype=bool)
            if nodata is not None:
                valid_mask &= np.not_equal(band, nodata)
        normalized, _ = robust_normalize(band, valid_mask)
        tiled = tile_image(normalized, valid_mask, patch_size=record.patch_size)

        required_coverage = int(np.ceil(record.valid_fraction_threshold_value * (record.patch_size ** 2)))
        keep_mask = (tiled.patch_coverages >= required_coverage) & (tiled.patch_stds >= record.std_threshold_value)
        training_patches = tiled.patches[keep_mask]
        if len(training_patches) == 0:
            training_patches = tiled.patches

        if len(training_patches) > 256:
            sample_indices = np.linspace(0, len(training_patches) - 1, 256, dtype=int)
            training_patches = training_patches[sample_indices]

        config = record.model_config
        feature_matrix = compute_patch_features(
            training_patches,
            resize_for_features=int(config.get("resize_for_features", 32)),
            canny_low=int(config.get("canny_low", 60)),
            canny_high=int(config.get("canny_high", 160)),
        )
        handcrafted = feature_matrix[:, -7:]
        profile = {
            "image_path": str(image_path),
            "sample_count": int(len(training_patches)),
            "handcrafted_mean": np.mean(handcrafted, axis=0).round(6).tolist(),
            "handcrafted_std": np.maximum(np.std(handcrafted, axis=0), 1e-4).round(6).tolist(),
        }
        cache_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
        self._domain_profiles[record.id] = profile
        return profile

    def _domain_similarity(self, record: ModelRecord, handcrafted_features: np.ndarray) -> float:
        profile = self._get_domain_profile(record)
        reference_mean = np.asarray(profile["handcrafted_mean"], dtype=np.float32)
        reference_std = np.asarray(profile["handcrafted_std"], dtype=np.float32)
        upload_mean = np.mean(handcrafted_features, axis=0)
        upload_std = np.std(handcrafted_features, axis=0)
        mean_distance = np.mean(np.abs(upload_mean - reference_mean) / np.maximum(reference_std, 1e-4))
        std_distance = np.mean(np.abs(upload_std - reference_std) / np.maximum(reference_std, 1e-4))
        combined_distance = 0.65 * mean_distance + 0.35 * std_distance
        return float(1.0 / (1.0 + combined_distance))

    def _dense_auto_selector_cache_path(self, records: list[ModelRecord]) -> Path:
        selector_key = "_".join(sorted(record.id for record in records))
        return self.settings.runtime_dir / "cache" / "dense_auto_selector" / f"{selector_key}_v2.joblib"

    def _get_dense_auto_selector(self, records: list[ModelRecord]) -> dict[str, Any]:
        if self._dense_auto_selector is not None:
            return self._dense_auto_selector

        cache_path = self._dense_auto_selector_cache_path(records)
        ensure_dir(cache_path.parent)
        if cache_path.exists():
            self._dense_auto_selector = json.loads(cache_path.read_text())
            self._dense_auto_selector["classifier"] = joblib.load(str(cache_path.with_suffix(".clf.joblib")))
            return self._dense_auto_selector

        patch_size = records[0].patch_size
        feature_blocks = []
        label_blocks: list[str] = []
        for record in records:
            sample_patches: np.ndarray
            dense_metadata_path = record.summary.get("artifacts", {}).get("dense_metadata_path")
            if dense_metadata_path and Path(dense_metadata_path).exists():
                dense_metadata = pd.read_csv(dense_metadata_path)
                sample_frames = []
                for _, group in dense_metadata.groupby("dominant_class"):
                    if len(group) > 320:
                        sample_frames.append(group.sample(n=320, random_state=42))
                    else:
                        sample_frames.append(group)
                sampled = pd.concat(sample_frames, ignore_index=True)
                image_patches = []
                for _, row in sampled.iterrows():
                    patch = np.asarray(Image.open(row["image_path"]).convert("L"), dtype=np.float32) / 255.0
                    image_patches.append(patch)
                sample_patches = np.stack(image_patches).astype(np.float32)
            else:
                project_root = self.settings.root_dir.parent
                dataset = record.summary["dataset"]
                image_rel_path = dataset.get("image_path") or dataset.get("spec", {}).get("image_path")
                image_path = (project_root / image_rel_path).resolve()
                with rasterio.open(image_path) as src:
                    band = src.read(1).astype(np.float32)
                    nodata = src.nodata
                    valid_mask = np.ones_like(band, dtype=bool)
                    if nodata is not None:
                        valid_mask &= np.not_equal(band, nodata)
                normalized, _ = robust_normalize(band, valid_mask)
                tiled = tile_image(normalized, valid_mask, patch_size=patch_size)
                keep_mask = tiled.patch_coverages >= int(np.ceil(record.valid_fraction_threshold_value * (patch_size ** 2)))
                sample_patches = tiled.patches[keep_mask]
                if len(sample_patches) == 0:
                    sample_patches = tiled.patches
                if len(sample_patches) > 256:
                    sample_indices = np.linspace(0, len(sample_patches) - 1, 256, dtype=int)
                    sample_patches = sample_patches[sample_indices]
            feature_matrix = compute_patch_features(
                sample_patches,
                resize_for_features=int(record.model_config.get("resize_for_features", 32)),
                canny_low=int(record.model_config.get("canny_low", 60)),
                canny_high=int(record.model_config.get("canny_high", 160)),
            )
            feature_blocks.append(feature_matrix)
            label_blocks.extend([record.id] * len(feature_matrix))

        x = np.vstack(feature_blocks).astype(np.float32)
        y = np.asarray(label_blocks)
        classifier = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
        classifier.fit(x, y)

        meta = {"model_ids": sorted(record.id for record in records), "patch_size": patch_size}
        cache_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        joblib.dump(classifier, str(cache_path.with_suffix(".clf.joblib")))
        meta["classifier"] = classifier
        self._dense_auto_selector = meta
        return meta

    def _sample_selector_patches(self, loaded_image: LoadedImage, patch_size: int) -> np.ndarray:
        target_count = 96
        if loaded_image.processing_mode != "streaming_raster":
            assert loaded_image.grayscale is not None
            assert loaded_image.valid_mask is not None
            tiled = tile_image(loaded_image.grayscale, loaded_image.valid_mask, patch_size=patch_size)
            sample_patches = tiled.patches
            if len(sample_patches) > target_count:
                sample_indices = np.linspace(0, len(sample_patches) - 1, target_count, dtype=int)
                sample_patches = sample_patches[sample_indices]
            return sample_patches.astype(np.float32)

        normalization_low = float(loaded_image.normalization_low or 0.0)
        normalization_high = float(loaded_image.normalization_high or 1.0)
        row_positions = np.linspace(0, max(loaded_image.height - patch_size, 0), 8, dtype=int)
        col_positions = np.linspace(0, max(loaded_image.width - patch_size, 0), 8, dtype=int)
        patches = []
        with rasterio.open(loaded_image.source_path) as src:
            for y0 in row_positions:
                for x0 in col_positions:
                    patch, _ = read_normalized_raster_patch(
                        src,
                        x0=int(x0),
                        y0=int(y0),
                        patch_size=patch_size,
                        normalization_low=normalization_low,
                        normalization_high=normalization_high,
                    )
                    patches.append(patch)
        return np.stack(patches).astype(np.float32)

    def _choose_record_from_filename_hint(self, loaded_image: LoadedImage, records: list[ModelRecord]) -> ModelRecord | None:
        filename = loaded_image.source_path.name.lower()
        dataset_tokens = {
            "jnpa": ("jnpa",),
            "cartosat": ("cartosat",),
        }

        for dataset_key, tokens in dataset_tokens.items():
            if not any(token in filename for token in tokens):
                continue
            matched = [
                record
                for record in records
                if dataset_key in record.id.lower()
                or dataset_key in record.display_name.lower()
                or dataset_key in record.dataset_name.lower()
            ]
            if matched:
                return sorted(matched, key=lambda record: (-record.priority, record.display_name))[0]
        return None

    def _choose_dense_auto_record(self, loaded_image: LoadedImage, records: list[ModelRecord]) -> ModelRecord:
        if not records:
            raise ValueError("At least one dense model record is required.")

        hinted_record = self._choose_record_from_filename_hint(loaded_image, records)
        if hinted_record is not None:
            return hinted_record

        sampled_by_patch_size: dict[int, np.ndarray] = {}
        best_record = records[0]
        best_score = float("-inf")

        for record in records:
            sample_patches = sampled_by_patch_size.get(record.patch_size)
            if sample_patches is None:
                sample_patches = self._sample_selector_patches(loaded_image, record.patch_size)
                sampled_by_patch_size[record.patch_size] = sample_patches
            if len(sample_patches) == 0:
                continue

            config = record.model_config
            resize_for_features = int(config.get("resize_for_features", 32))
            canny_low = int(config.get("canny_low", 60))
            canny_high = int(config.get("canny_high", 160))
            feature_matrix = compute_patch_features(
                sample_patches,
                resize_for_features=resize_for_features,
                canny_low=canny_low,
                canny_high=canny_high,
            )
            handcrafted = feature_matrix[:, -7:]

            model, device = self._get_unet_model(record)
            confidence_batches: list[np.ndarray] = []
            entropy_batches: list[np.ndarray] = []
            batch_size = min(max(1, int(self.settings.prediction_batch_patches)), 48)

            with torch.no_grad():
                for start in range(0, len(sample_patches), batch_size):
                    end = min(start + batch_size, len(sample_patches))
                    batch = torch.from_numpy(sample_patches[start:end]).unsqueeze(1).float().to(device)
                    logits = model(batch)
                    probabilities = torch.softmax(logits, dim=1).cpu().numpy()
                    confidence_maps = probabilities.max(axis=1)
                    entropy_maps = normalized_entropy_map(probabilities)
                    confidence_batches.append(confidence_maps.reshape(confidence_maps.shape[0], -1).mean(axis=1).astype(np.float32))
                    entropy_batches.append(entropy_maps.reshape(entropy_maps.shape[0], -1).mean(axis=1).astype(np.float32))

            score, _ = self._score_unet_fit_from_metrics(
                record,
                max_confidences=np.concatenate(confidence_batches),
                normalized_entropies=np.concatenate(entropy_batches),
                handcrafted_features=handcrafted,
                class_count=len(record.class_names),
            )
            if score > best_score:
                best_score = score
                best_record = record

        return best_record

    def _score_model_fit_from_metrics(
        self,
        record: ModelRecord,
        *,
        max_confidences: np.ndarray,
        normalized_entropies: np.ndarray,
        separations: np.ndarray,
        handcrafted_features: np.ndarray,
        class_count: int,
    ) -> tuple[float, dict[str, float]]:
        class_count = max(1, class_count)
        chance_level = 1.0 / class_count
        confidence_mean = float(np.mean(max_confidences))
        confidence_p90 = float(np.percentile(max_confidences, 90))
        adjusted_confidence_mean = float((confidence_mean - chance_level) / max(1.0 - chance_level, 1e-6))
        adjusted_confidence_p90 = float((confidence_p90 - chance_level) / max(1.0 - chance_level, 1e-6))
        adjusted_confidence_mean = float(np.clip(adjusted_confidence_mean, 0.0, 1.0))
        adjusted_confidence_p90 = float(np.clip(adjusted_confidence_p90, 0.0, 1.0))
        certainty = float(1.0 - np.mean(normalized_entropies))
        separation = float(np.mean(separations))
        validation_accuracy = float(record.model_metrics.get("val_accuracy", 0.0) or 0.0)
        domain_similarity = self._domain_similarity(record, handcrafted_features)
        score = (
            0.18 * adjusted_confidence_mean
            + 0.14 * certainty
            + 0.13 * adjusted_confidence_p90
            + 0.10 * separation
            + 0.10 * validation_accuracy
            + 0.35 * domain_similarity
        )
        breakdown = {
            "confidence_mean": round(confidence_mean, 4),
            "adjusted_confidence_mean": round(adjusted_confidence_mean, 4),
            "certainty": round(certainty, 4),
            "confidence_p90": round(confidence_p90, 4),
            "adjusted_confidence_p90": round(adjusted_confidence_p90, 4),
            "cluster_separation": round(separation, 4),
            "validation_accuracy": round(validation_accuracy, 4),
            "domain_similarity": round(domain_similarity, 4),
            "final_score": round(float(score), 4),
        }
        return float(score), breakdown

    def _score_model_fit(
        self,
        record: ModelRecord,
        probabilities: np.ndarray,
        reduced_features: np.ndarray,
        kmeans: Any,
        handcrafted_features: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        return self._score_model_fit_from_metrics(
            record,
            max_confidences=probabilities.max(axis=1),
            normalized_entropies=normalized_entropy(probabilities),
            separations=cluster_separation_values(reduced_features, kmeans),
            handcrafted_features=handcrafted_features,
            class_count=probabilities.shape[1],
        )

    def _score_unet_fit_from_metrics(
        self,
        record: ModelRecord,
        *,
        max_confidences: np.ndarray,
        normalized_entropies: np.ndarray,
        handcrafted_features: np.ndarray,
        class_count: int,
    ) -> tuple[float, dict[str, float]]:
        class_count = max(1, class_count)
        chance_level = 1.0 / class_count
        confidence_mean = float(np.mean(max_confidences))
        confidence_p90 = float(np.percentile(max_confidences, 90))
        adjusted_confidence_mean = float((confidence_mean - chance_level) / max(1.0 - chance_level, 1e-6))
        adjusted_confidence_p90 = float((confidence_p90 - chance_level) / max(1.0 - chance_level, 1e-6))
        adjusted_confidence_mean = float(np.clip(adjusted_confidence_mean, 0.0, 1.0))
        adjusted_confidence_p90 = float(np.clip(adjusted_confidence_p90, 0.0, 1.0))
        certainty = float(1.0 - np.mean(normalized_entropies))
        validation_accuracy = float(record.model_metrics.get("val_accuracy", 0.0) or 0.0)
        domain_similarity = self._domain_similarity(record, handcrafted_features)
        score = (
            0.22 * adjusted_confidence_mean
            + 0.18 * certainty
            + 0.15 * adjusted_confidence_p90
            + 0.15 * validation_accuracy
            + 0.30 * domain_similarity
        )
        breakdown = {
            "confidence_mean": round(confidence_mean, 4),
            "adjusted_confidence_mean": round(adjusted_confidence_mean, 4),
            "certainty": round(certainty, 4),
            "confidence_p90": round(confidence_p90, 4),
            "adjusted_confidence_p90": round(adjusted_confidence_p90, 4),
            "cluster_separation": 0.0,
            "validation_accuracy": round(validation_accuracy, 4),
            "domain_similarity": round(domain_similarity, 4),
            "final_score": round(float(score), 4),
        }
        return float(score), breakdown

    def _get_unet_model(self, record: ModelRecord) -> tuple[UNet, torch.device]:
        cached = self._torch_models.get(record.id)
        if cached is not None:
            return cached

        checkpoint = record.load_bundle()
        state_dict = checkpoint.get("model", checkpoint)
        base_channels = int(state_dict["down1.block.0.weight"].shape[0])
        model = UNet(in_channels=1, num_classes=len(record.class_names), base_channels=base_channels)
        model.load_state_dict(state_dict)
        device = resolve_torch_device()
        model = model.to(device)
        model.eval()
        self._torch_models[record.id] = (model, device)
        return model, device

    def _predict_with_patch_classifier(
        self,
        loaded_image: LoadedImage,
        tiled: TiledImage,
        record: ModelRecord,
    ) -> ModelPrediction:
        bundle = record.load_bundle()
        config = bundle.get("config", {})
        resize_for_features = int(config.get("resize_for_features", 32))
        canny_low = int(config.get("canny_low", 60))
        canny_high = int(config.get("canny_high", 160))

        feature_matrix = compute_patch_features(
            tiled.patches,
            resize_for_features=resize_for_features,
            canny_low=canny_low,
            canny_high=canny_high,
        )
        handcrafted = feature_matrix[:, -7:]

        reduced = bundle["pca"].transform(feature_matrix)
        classifier = bundle["classifier"]
        probabilities = classifier.predict_proba(reduced)
        predicted_indices = probabilities.argmax(axis=1)
        predicted_labels = classifier.classes_[predicted_indices].astype(np.uint8)
        patch_confidences = probabilities[np.arange(len(predicted_indices)), predicted_indices].astype(np.float32)

        overrides = 0
        for idx in range(len(predicted_labels)):
            if (
                tiled.patch_stds[idx] < record.std_threshold_value
                and tiled.patch_means[idx] < record.dark_water_mean_threshold_value
                and 0 in record.class_names
            ):
                if predicted_labels[idx] != 0:
                    overrides += 1
                predicted_labels[idx] = 0
                patch_confidences[idx] = max(float(patch_confidences[idx]), 0.85)

        score, score_breakdown = self._score_model_fit(
            record,
            probabilities,
            reduced,
            bundle["kmeans"],
            handcrafted,
        )
        class_percentages = class_percentage_table(predicted_labels, tiled.patch_coverages, record.class_names)
        dominant_class_name = class_percentages[0]["class_name"] if class_percentages else "Unknown"

        return ModelPrediction(
            record=record,
            patch_grid=predicted_labels.reshape(tiled.rows, tiled.cols),
            patch_confidences=patch_confidences,
            patch_labels=predicted_labels,
            patch_coverages=tiled.patch_coverages,
            class_percentages=class_percentages,
            selected_score=score,
            score_breakdown=score_breakdown,
            heuristic_water_overrides=overrides,
            dominant_class_name=dominant_class_name,
        )

    def _predict_with_unet(
        self,
        loaded_image: LoadedImage,
        tiled: TiledImage,
        record: ModelRecord,
    ) -> ModelPrediction:
        model, device = self._get_unet_model(record)
        config = record.model_config
        resize_for_features = int(config.get("resize_for_features", 32))
        canny_low = int(config.get("canny_low", 60))
        canny_high = int(config.get("canny_high", 160))

        feature_matrix = compute_patch_features(
            tiled.patches,
            resize_for_features=resize_for_features,
            canny_low=canny_low,
            canny_high=canny_high,
        )
        handcrafted = feature_matrix[:, -7:]

        batch_size = max(1, int(self.settings.prediction_batch_patches))
        patch_labels = np.zeros(len(tiled.patches), dtype=np.uint8)
        patch_confidences = np.zeros(len(tiled.patches), dtype=np.float32)
        max_confidences: list[np.ndarray] = []
        entropies: list[np.ndarray] = []
        overrides = 0

        with torch.no_grad():
            for start in range(0, len(tiled.patches), batch_size):
                end = min(start + batch_size, len(tiled.patches))
                batch = torch.from_numpy(tiled.patches[start:end]).unsqueeze(1).float().to(device)
                logits = model(batch)
                patch_logits = logits.mean(dim=(2, 3))
                probabilities = torch.softmax(patch_logits, dim=1).cpu().numpy()
                predicted_indices = probabilities.argmax(axis=1)
                predicted = predicted_indices.astype(np.uint8)
                confidences = probabilities[np.arange(len(predicted_indices)), predicted_indices].astype(np.float32)

                for local_idx, global_idx in enumerate(range(start, end)):
                    if (
                        tiled.patch_stds[global_idx] < record.std_threshold_value
                        and tiled.patch_means[global_idx] < record.dark_water_mean_threshold_value
                        and 0 in record.class_names
                    ):
                        if predicted[local_idx] != 0:
                            overrides += 1
                        predicted[local_idx] = 0
                        confidences[local_idx] = max(float(confidences[local_idx]), 0.85)

                patch_labels[start:end] = predicted
                patch_confidences[start:end] = confidences
                max_confidences.append(confidences)
                entropies.append(normalized_entropy(probabilities).astype(np.float32))

        score, score_breakdown = self._score_unet_fit_from_metrics(
            record,
            max_confidences=np.concatenate(max_confidences),
            normalized_entropies=np.concatenate(entropies),
            handcrafted_features=handcrafted,
            class_count=len(record.class_names),
        )
        class_percentages = class_percentage_table(patch_labels, tiled.patch_coverages, record.class_names)
        dominant_class_name = class_percentages[0]["class_name"] if class_percentages else "Unknown"

        return ModelPrediction(
            record=record,
            patch_grid=patch_labels.reshape(tiled.rows, tiled.cols),
            patch_confidences=patch_confidences,
            patch_labels=patch_labels,
            patch_coverages=tiled.patch_coverages,
            class_percentages=class_percentages,
            selected_score=score,
            score_breakdown=score_breakdown,
            heuristic_water_overrides=overrides,
            dominant_class_name=dominant_class_name,
        )

    def _predict_with_dense_unet(
        self,
        loaded_image: LoadedImage,
        tiled: TiledImage,
        record: ModelRecord,
    ) -> ModelPrediction:
        model, device = self._get_unet_model(record)
        config = record.model_config
        resize_for_features = int(config.get("resize_for_features", 32))
        canny_low = int(config.get("canny_low", 60))
        canny_high = int(config.get("canny_high", 160))

        feature_matrix = compute_patch_features(
            tiled.patches,
            resize_for_features=resize_for_features,
            canny_low=canny_low,
            canny_high=canny_high,
        )
        handcrafted = feature_matrix[:, -7:]

        patch_size = record.patch_size
        full_label_map = np.full((tiled.rows * patch_size, tiled.cols * patch_size), NO_DATA_LABEL, dtype=np.uint8)
        valid_crop = np.zeros(full_label_map.shape, dtype=bool)
        assert loaded_image.valid_mask is not None
        valid_h, valid_w = loaded_image.valid_mask.shape
        valid_crop[:valid_h, :valid_w] = loaded_image.valid_mask
        patch_labels = np.zeros(len(tiled.patches), dtype=np.uint8)
        patch_confidences = np.zeros(len(tiled.patches), dtype=np.float32)
        patch_coverages = tiled.patch_coverages.copy()
        patch_entropies = np.zeros(len(tiled.patches), dtype=np.float32)
        class_pixel_counts = np.zeros(len(record.class_names), dtype=np.int64)
        overrides = 0

        batch_size = max(1, int(self.settings.prediction_batch_patches))
        with torch.no_grad():
            for start in range(0, len(tiled.patches), batch_size):
                end = min(start + batch_size, len(tiled.patches))
                batch = torch.from_numpy(tiled.patches[start:end]).unsqueeze(1).float().to(device)
                logits = model(batch)
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()
                predicted_masks = probabilities.argmax(axis=1).astype(np.uint8)
                confidence_maps = probabilities.max(axis=1).astype(np.float32)
                entropy_maps = normalized_entropy_map(probabilities).astype(np.float32)

                for local_idx, global_idx in enumerate(range(start, end)):
                    row = global_idx // tiled.cols
                    col = global_idx % tiled.cols
                    y0 = row * patch_size
                    x0 = col * patch_size
                    y1 = y0 + patch_size
                    x1 = x0 + patch_size
                    pred_mask = predicted_masks[local_idx]
                    conf_map = confidence_maps[local_idx]
                    entropy_map = entropy_maps[local_idx]

                    if (
                        tiled.patch_stds[global_idx] < record.std_threshold_value
                        and tiled.patch_means[global_idx] < record.dark_water_mean_threshold_value
                        and 0 in record.class_names
                    ):
                        if np.any(pred_mask != 0):
                            overrides += 1
                        pred_mask = np.zeros_like(pred_mask, dtype=np.uint8)
                        conf_map = np.maximum(conf_map, 0.85)
                        entropy_map = np.minimum(entropy_map, 0.15)

                    valid_slice = valid_crop[y0:y1, x0:x1]
                    patch_valid_pixels = pred_mask[valid_slice]
                    if patch_valid_pixels.size:
                        patch_labels[global_idx] = int(np.bincount(patch_valid_pixels.reshape(-1), minlength=len(record.class_names)).argmax())
                        patch_confidences[global_idx] = float(conf_map[valid_slice].mean())
                        patch_entropies[global_idx] = float(entropy_map[valid_slice].mean())
                        class_pixel_counts += np.bincount(
                            patch_valid_pixels.reshape(-1),
                            minlength=len(record.class_names),
                        ).astype(np.int64)
                    else:
                        patch_labels[global_idx] = 0
                        patch_confidences[global_idx] = 0.0
                        patch_entropies[global_idx] = 1.0

                    target = full_label_map[y0:y1, x0:x1]
                    target[:] = pred_mask
                    target[~valid_slice] = NO_DATA_LABEL

        score, score_breakdown = self._score_unet_fit_from_metrics(
            record,
            max_confidences=patch_confidences,
            normalized_entropies=patch_entropies,
            handcrafted_features=handcrafted,
            class_count=len(record.class_names),
        )
        class_percentages = class_percentage_table_from_counts(class_pixel_counts, record.class_names)
        dominant_class_name = class_percentages[0]["class_name"] if class_percentages else "Unknown"

        return ModelPrediction(
            record=record,
            patch_grid=patch_labels.reshape(tiled.rows, tiled.cols),
            patch_confidences=patch_confidences,
            patch_labels=patch_labels,
            patch_coverages=patch_coverages,
            class_percentages=class_percentages,
            selected_score=score,
            score_breakdown=score_breakdown,
            heuristic_water_overrides=overrides,
            dominant_class_name=dominant_class_name,
            full_label_map=full_label_map,
        )

    def _predict_with_patch_classifier_streaming(self, loaded_image: LoadedImage, record: ModelRecord) -> ModelPrediction:
        bundle = record.load_bundle()
        config = bundle.get("config", {})
        resize_for_features = int(config.get("resize_for_features", 32))
        canny_low = int(config.get("canny_low", 60))
        canny_high = int(config.get("canny_high", 160))
        classifier = bundle["classifier"]
        pca = bundle["pca"]
        kmeans = bundle["kmeans"]

        patch_size = record.patch_size
        rows = int(np.ceil(loaded_image.height / patch_size))
        cols = int(np.ceil(loaded_image.width / patch_size))
        total_patches = rows * cols

        patch_labels = np.zeros(total_patches, dtype=np.uint8)
        patch_confidences = np.zeros(total_patches, dtype=np.float32)
        patch_coverages = np.zeros(total_patches, dtype=np.int64)
        max_confidences: list[np.ndarray] = []
        entropies: list[np.ndarray] = []
        separations: list[np.ndarray] = []
        handcrafted_batches: list[np.ndarray] = []
        overrides = 0

        batch_size = max(1, int(self.settings.prediction_batch_patches))
        normalization_low = float(loaded_image.normalization_low or 0.0)
        normalization_high = float(loaded_image.normalization_high or 1.0)

        with rasterio.open(loaded_image.source_path) as src:
            batch_patches: list[np.ndarray] = []
            batch_means: list[float] = []
            batch_stds: list[float] = []
            batch_coverages: list[int] = []
            batch_indices: list[int] = []

            def flush_batch() -> None:
                nonlocal overrides
                if not batch_patches:
                    return

                patches = np.stack(batch_patches).astype(np.float32)
                feature_matrix = compute_patch_features(
                    patches,
                    resize_for_features=resize_for_features,
                    canny_low=canny_low,
                    canny_high=canny_high,
                )
                handcrafted = feature_matrix[:, -7:]
                reduced = pca.transform(feature_matrix)
                probabilities = classifier.predict_proba(reduced)
                predicted_indices = probabilities.argmax(axis=1)
                predicted = classifier.classes_[predicted_indices].astype(np.uint8)
                confidences = probabilities[np.arange(len(predicted_indices)), predicted_indices].astype(np.float32)

                for local_idx, flat_idx in enumerate(batch_indices):
                    if (
                        batch_stds[local_idx] < record.std_threshold_value
                        and batch_means[local_idx] < record.dark_water_mean_threshold_value
                        and 0 in record.class_names
                    ):
                        if predicted[local_idx] != 0:
                            overrides += 1
                        predicted[local_idx] = 0
                        confidences[local_idx] = max(float(confidences[local_idx]), 0.85)

                    patch_labels[flat_idx] = predicted[local_idx]
                    patch_confidences[flat_idx] = confidences[local_idx]
                    patch_coverages[flat_idx] = batch_coverages[local_idx]

                max_confidences.append(confidences)
                entropies.append(normalized_entropy(probabilities).astype(np.float32))
                separations.append(cluster_separation_values(reduced, kmeans).astype(np.float32))
                handcrafted_batches.append(handcrafted.astype(np.float32))

                batch_patches.clear()
                batch_means.clear()
                batch_stds.clear()
                batch_coverages.clear()
                batch_indices.clear()

            for row in range(rows):
                for col in range(cols):
                    y0 = row * patch_size
                    x0 = col * patch_size
                    patch, valid_mask = read_normalized_raster_patch(
                        src,
                        x0=x0,
                        y0=y0,
                        patch_size=patch_size,
                        normalization_low=normalization_low,
                        normalization_high=normalization_high,
                    )
                    batch_patches.append(patch)
                    batch_means.append(float(patch.mean()))
                    batch_stds.append(float(patch.std()))
                    batch_coverages.append(int(valid_mask.sum()))
                    batch_indices.append(row * cols + col)
                    if len(batch_patches) >= batch_size:
                        flush_batch()

            flush_batch()

        handcrafted_features = np.vstack(handcrafted_batches)
        score, score_breakdown = self._score_model_fit_from_metrics(
            record,
            max_confidences=np.concatenate(max_confidences),
            normalized_entropies=np.concatenate(entropies),
            separations=np.concatenate(separations),
            handcrafted_features=handcrafted_features,
            class_count=len(classifier.classes_),
        )

        class_percentages = class_percentage_table(patch_labels, patch_coverages, record.class_names)
        dominant_class_name = class_percentages[0]["class_name"] if class_percentages else "Unknown"

        return ModelPrediction(
            record=record,
            patch_grid=patch_labels.reshape(rows, cols),
            patch_confidences=patch_confidences,
            patch_labels=patch_labels,
            patch_coverages=patch_coverages,
            class_percentages=class_percentages,
            selected_score=score,
            score_breakdown=score_breakdown,
            heuristic_water_overrides=overrides,
            dominant_class_name=dominant_class_name,
        )

    def _predict_with_dense_unet_streaming(self, loaded_image: LoadedImage, record: ModelRecord) -> ModelPrediction:
        model, device = self._get_unet_model(record)
        config = record.model_config
        resize_for_features = int(config.get("resize_for_features", 32))
        canny_low = int(config.get("canny_low", 60))
        canny_high = int(config.get("canny_high", 160))

        patch_size = record.patch_size
        rows = int(np.ceil(loaded_image.height / patch_size))
        cols = int(np.ceil(loaded_image.width / patch_size))
        total_patches = rows * cols

        patch_labels = np.zeros(total_patches, dtype=np.uint8)
        patch_confidences = np.zeros(total_patches, dtype=np.float32)
        patch_coverages = np.zeros(total_patches, dtype=np.int64)
        patch_entropies = np.zeros(total_patches, dtype=np.float32)
        class_pixel_counts = np.zeros(len(record.class_names), dtype=np.int64)
        handcrafted_batches: list[np.ndarray] = []
        overrides = 0

        full_label_map = np.full((loaded_image.height, loaded_image.width), NO_DATA_LABEL, dtype=np.uint8)
        batch_size = max(1, int(self.settings.prediction_batch_patches))
        normalization_low = float(loaded_image.normalization_low or 0.0)
        normalization_high = float(loaded_image.normalization_high or 1.0)

        with rasterio.open(loaded_image.source_path) as src:
            batch_patches: list[np.ndarray] = []
            batch_means: list[float] = []
            batch_stds: list[float] = []
            batch_coverages: list[int] = []
            batch_indices: list[int] = []
            batch_valid_masks: list[np.ndarray] = []

            def flush_batch() -> None:
                nonlocal overrides, class_pixel_counts
                if not batch_patches:
                    return

                patches = np.stack(batch_patches).astype(np.float32)
                feature_matrix = compute_patch_features(
                    patches,
                    resize_for_features=resize_for_features,
                    canny_low=canny_low,
                    canny_high=canny_high,
                )
                handcrafted_batches.append(feature_matrix[:, -7:].astype(np.float32))

                with torch.no_grad():
                    batch = torch.from_numpy(patches).unsqueeze(1).float().to(device)
                    logits = model(batch)
                    probabilities = torch.softmax(logits, dim=1).cpu().numpy()

                predicted_masks = probabilities.argmax(axis=1).astype(np.uint8)
                confidence_maps = probabilities.max(axis=1).astype(np.float32)
                entropy_maps = normalized_entropy_map(probabilities).astype(np.float32)

                for local_idx, flat_idx in enumerate(batch_indices):
                    row = flat_idx // cols
                    col = flat_idx % cols
                    y0 = row * patch_size
                    x0 = col * patch_size
                    y1 = min(y0 + patch_size, loaded_image.height)
                    x1 = min(x0 + patch_size, loaded_image.width)
                    valid_slice = batch_valid_masks[local_idx][: y1 - y0, : x1 - x0]
                    pred_mask = predicted_masks[local_idx][: y1 - y0, : x1 - x0]
                    conf_map = confidence_maps[local_idx][: y1 - y0, : x1 - x0]
                    entropy_map = entropy_maps[local_idx][: y1 - y0, : x1 - x0]

                    if (
                        batch_stds[local_idx] < record.std_threshold_value
                        and batch_means[local_idx] < record.dark_water_mean_threshold_value
                        and 0 in record.class_names
                    ):
                        if np.any(pred_mask != 0):
                            overrides += 1
                        pred_mask = np.zeros_like(pred_mask, dtype=np.uint8)
                        conf_map = np.maximum(conf_map, 0.85)
                        entropy_map = np.minimum(entropy_map, 0.15)

                    full_label_map[y0:y1, x0:x1] = pred_mask
                    full_label_map[y0:y1, x0:x1][~valid_slice] = NO_DATA_LABEL

                    valid_pixels = pred_mask[valid_slice]
                    patch_coverages[flat_idx] = int(valid_slice.sum())
                    if valid_pixels.size:
                        patch_labels[flat_idx] = int(np.bincount(valid_pixels.reshape(-1), minlength=len(record.class_names)).argmax())
                        patch_confidences[flat_idx] = float(conf_map[valid_slice].mean())
                        patch_entropies[flat_idx] = float(entropy_map[valid_slice].mean())
                        class_pixel_counts += np.bincount(valid_pixels.reshape(-1), minlength=len(record.class_names)).astype(np.int64)
                    else:
                        patch_labels[flat_idx] = 0
                        patch_confidences[flat_idx] = 0.0
                        patch_entropies[flat_idx] = 1.0

                batch_patches.clear()
                batch_means.clear()
                batch_stds.clear()
                batch_coverages.clear()
                batch_indices.clear()
                batch_valid_masks.clear()

            for row in range(rows):
                for col in range(cols):
                    y0 = row * patch_size
                    x0 = col * patch_size
                    patch, patch_valid_mask = read_normalized_raster_patch(
                        src,
                        x0=x0,
                        y0=y0,
                        patch_size=patch_size,
                        normalization_low=normalization_low,
                        normalization_high=normalization_high,
                    )
                    batch_patches.append(patch)
                    batch_means.append(float(patch.mean()))
                    batch_stds.append(float(patch.std()))
                    batch_coverages.append(int(patch_valid_mask.sum()))
                    batch_indices.append(row * cols + col)
                    batch_valid_masks.append(patch_valid_mask)
                    if len(batch_patches) >= batch_size:
                        flush_batch()

            flush_batch()

        handcrafted_features = np.vstack(handcrafted_batches)
        score, score_breakdown = self._score_unet_fit_from_metrics(
            record,
            max_confidences=patch_confidences,
            normalized_entropies=patch_entropies,
            handcrafted_features=handcrafted_features,
            class_count=len(record.class_names),
        )
        class_percentages = class_percentage_table_from_counts(class_pixel_counts, record.class_names)
        dominant_class_name = class_percentages[0]["class_name"] if class_percentages else "Unknown"

        return ModelPrediction(
            record=record,
            patch_grid=patch_labels.reshape(rows, cols),
            patch_confidences=patch_confidences,
            patch_labels=patch_labels,
            patch_coverages=patch_coverages,
            class_percentages=class_percentages,
            selected_score=score,
            score_breakdown=score_breakdown,
            heuristic_water_overrides=overrides,
            dominant_class_name=dominant_class_name,
            full_label_map=full_label_map,
        )

    def _predict_with_unet_streaming(self, loaded_image: LoadedImage, record: ModelRecord) -> ModelPrediction:
        model, device = self._get_unet_model(record)
        config = record.model_config
        resize_for_features = int(config.get("resize_for_features", 32))
        canny_low = int(config.get("canny_low", 60))
        canny_high = int(config.get("canny_high", 160))

        patch_size = record.patch_size
        rows = int(np.ceil(loaded_image.height / patch_size))
        cols = int(np.ceil(loaded_image.width / patch_size))
        total_patches = rows * cols

        patch_labels = np.zeros(total_patches, dtype=np.uint8)
        patch_confidences = np.zeros(total_patches, dtype=np.float32)
        patch_coverages = np.zeros(total_patches, dtype=np.int64)
        max_confidences: list[np.ndarray] = []
        entropies: list[np.ndarray] = []
        handcrafted_batches: list[np.ndarray] = []
        overrides = 0

        batch_size = max(1, int(self.settings.prediction_batch_patches))
        normalization_low = float(loaded_image.normalization_low or 0.0)
        normalization_high = float(loaded_image.normalization_high or 1.0)

        with rasterio.open(loaded_image.source_path) as src:
            batch_patches: list[np.ndarray] = []
            batch_means: list[float] = []
            batch_stds: list[float] = []
            batch_coverages: list[int] = []
            batch_indices: list[int] = []

            def flush_batch() -> None:
                nonlocal overrides
                if not batch_patches:
                    return

                patches = np.stack(batch_patches).astype(np.float32)
                feature_matrix = compute_patch_features(
                    patches,
                    resize_for_features=resize_for_features,
                    canny_low=canny_low,
                    canny_high=canny_high,
                )
                handcrafted = feature_matrix[:, -7:]
                handcrafted_batches.append(handcrafted.astype(np.float32))

                with torch.no_grad():
                    batch = torch.from_numpy(patches).unsqueeze(1).float().to(device)
                    logits = model(batch)
                    patch_logits = logits.mean(dim=(2, 3))
                    probabilities = torch.softmax(patch_logits, dim=1).cpu().numpy()

                predicted_indices = probabilities.argmax(axis=1)
                predicted = predicted_indices.astype(np.uint8)
                confidences = probabilities[np.arange(len(predicted_indices)), predicted_indices].astype(np.float32)

                for local_idx, flat_idx in enumerate(batch_indices):
                    if (
                        batch_stds[local_idx] < record.std_threshold_value
                        and batch_means[local_idx] < record.dark_water_mean_threshold_value
                        and 0 in record.class_names
                    ):
                        if predicted[local_idx] != 0:
                            overrides += 1
                        predicted[local_idx] = 0
                        confidences[local_idx] = max(float(confidences[local_idx]), 0.85)

                    patch_labels[flat_idx] = predicted[local_idx]
                    patch_confidences[flat_idx] = confidences[local_idx]
                    patch_coverages[flat_idx] = batch_coverages[local_idx]

                max_confidences.append(confidences)
                entropies.append(normalized_entropy(probabilities).astype(np.float32))

                batch_patches.clear()
                batch_means.clear()
                batch_stds.clear()
                batch_coverages.clear()
                batch_indices.clear()

            for row in range(rows):
                for col in range(cols):
                    y0 = row * patch_size
                    x0 = col * patch_size
                    patch, valid_mask = read_normalized_raster_patch(
                        src,
                        x0=x0,
                        y0=y0,
                        patch_size=patch_size,
                        normalization_low=normalization_low,
                        normalization_high=normalization_high,
                    )
                    batch_patches.append(patch)
                    batch_means.append(float(patch.mean()))
                    batch_stds.append(float(patch.std()))
                    batch_coverages.append(int(valid_mask.sum()))
                    batch_indices.append(row * cols + col)
                    if len(batch_patches) >= batch_size:
                        flush_batch()

            flush_batch()

        handcrafted_features = np.vstack(handcrafted_batches)
        score, score_breakdown = self._score_unet_fit_from_metrics(
            record,
            max_confidences=np.concatenate(max_confidences),
            normalized_entropies=np.concatenate(entropies),
            handcrafted_features=handcrafted_features,
            class_count=len(record.class_names),
        )
        class_percentages = class_percentage_table(patch_labels, patch_coverages, record.class_names)
        dominant_class_name = class_percentages[0]["class_name"] if class_percentages else "Unknown"

        return ModelPrediction(
            record=record,
            patch_grid=patch_labels.reshape(rows, cols),
            patch_confidences=patch_confidences,
            patch_labels=patch_labels,
            patch_coverages=patch_coverages,
            class_percentages=class_percentages,
            selected_score=score,
            score_breakdown=score_breakdown,
            heuristic_water_overrides=overrides,
            dominant_class_name=dominant_class_name,
        )

    def _write_geotiff_labels(
        self,
        destination: Path,
        patch_grid: np.ndarray,
        patch_size: int,
        loaded_image: LoadedImage,
        full_label_map: np.ndarray | None = None,
    ) -> None:
        if loaded_image.raster_profile is None:
            return

        profile = loaded_image.raster_profile.copy()
        profile.update(
            count=1,
            dtype=rasterio.uint8,
            nodata=NO_DATA_LABEL,
            compress="lzw",
            tiled=True,
        )

        height = loaded_image.height
        width = loaded_image.width

        with rasterio.open(destination, "w", **profile) as dst:
            if full_label_map is not None:
                label_crop = full_label_map[:height, :width].astype(np.uint8)
                dst.write(label_crop, 1)
                return
            src = rasterio.open(loaded_image.source_path)
            try:
                for row_index in range(patch_grid.shape[0]):
                    y0 = row_index * patch_size
                    y1 = min((row_index + 1) * patch_size, height)
                    row_block = np.repeat(patch_grid[row_index : row_index + 1, :], patch_size, axis=0)
                    row_block = np.repeat(row_block, patch_size, axis=1)
                    row_block = row_block[: y1 - y0, :width].astype(np.uint8)
                    if loaded_image.valid_mask is not None:
                        mask_slice = loaded_image.valid_mask[y0:y1, :width]
                    else:
                        mask_slice = read_raster_valid_mask(
                            src,
                            x0=0,
                            y0=y0,
                            width=width,
                            height=y1 - y0,
                        )
                    row_block = row_block.copy()
                    row_block[~mask_slice] = NO_DATA_LABEL
                    window = rasterio.windows.Window(col_off=0, row_off=y0, width=width, height=y1 - y0)
                    dst.write(row_block, 1, window=window)
            finally:
                src.close()

    def _write_class_shapefiles(
        self,
        geotiff_path: Path,
        destination_dir: Path,
        record: ModelRecord,
    ) -> tuple[Path | None, list[dict[str, Any]]]:
        ensure_dir(destination_dir)

        with rasterio.open(geotiff_path) as src:
            if src.crs is None:
                return None, []

            label_image = src.read(1)
            valid_mask = label_image != NO_DATA_LABEL
            if not np.any(valid_mask):
                return None, []

            transform = src.transform
            crs = src.crs
            pixel_area = abs(transform.a * transform.e)

            class_features: dict[int, list[dict[str, Any]]] = {
                int(class_id): [] for class_id in record.class_names
            }
            for geometry, value in rasterio_features.shapes(
                label_image,
                mask=valid_mask,
                transform=transform,
            ):
                class_id = int(value)
                if class_id == NO_DATA_LABEL or class_id not in record.class_names:
                    continue
                polygon = shapely_shape(geometry)
                if polygon.is_empty:
                    continue
                class_features[class_id].append(
                    {
                        "cls_id": class_id,
                        "cls_name": record.class_names[class_id],
                        "area_est": float(polygon.area),
                        "geometry": polygon,
                    }
                )

        written_layers: list[dict[str, Any]] = []
        class_summary_path = destination_dir / "vector_class_summary.csv"
        with class_summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["class_id", "class_name", "feature_count", "pixel_count", "area_estimate"])

            for class_id, rows in class_features.items():
                class_name = record.class_names[class_id]
                pixel_count = int(np.count_nonzero(label_image == class_id))
                area_estimate = float(pixel_count * pixel_area)
                writer.writerow([class_id, class_name, len(rows), pixel_count, round(area_estimate, 4)])

                if not rows:
                    continue

                class_slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in class_name).strip("_")
                while "__" in class_slug:
                    class_slug = class_slug.replace("__", "_")
                layer_dir = ensure_dir(destination_dir / f"class_{class_id:02d}_{class_slug}")
                shapefile_path = layer_dir / f"class_{class_id:02d}_{class_slug}.shp"

                gdf = gpd.GeoDataFrame(rows, crs=crs)
                gdf["cls_pix"] = pixel_count
                gdf["cls_area"] = area_estimate
                gdf.to_file(shapefile_path, driver="ESRI Shapefile")

                written_layers.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "feature_count": len(rows),
                        "pixel_count": pixel_count,
                        "area_estimate": round(area_estimate, 4),
                        "shapefile": str(shapefile_path.relative_to(destination_dir)),
                    }
                )

        zip_path = destination_dir.parent / "class_shapefiles_georef.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file_path in sorted(destination_dir.rglob("*")):
                if file_path.is_file():
                    archive.write(file_path, file_path.relative_to(destination_dir.parent))

        return zip_path, written_layers

    def _save_outputs(
        self,
        result_dir: Path,
        loaded_image: LoadedImage,
        prediction: ModelPrediction,
        mode: str,
        candidate_rankings: list[dict[str, Any]],
        requested_model_id: str | None,
    ) -> dict[str, Any]:
        ensure_dir(result_dir)

        input_preview_path = result_dir / "input_preview.png"
        Image.fromarray(loaded_image.preview_rgb, mode="RGB").save(input_preview_path)

        preview_height, preview_width = loaded_image.preview_rgb.shape[:2]
        if prediction.full_label_map is not None:
            label_preview = np.asarray(
                Image.fromarray(prediction.full_label_map.astype(np.uint8), mode="L").resize(
                    (preview_width, preview_height),
                    Image.Resampling.NEAREST,
                ),
                dtype=np.uint8,
            )
        else:
            label_preview = resize_label_preview(prediction.patch_grid, preview_width, preview_height)
        color_preview = colorize_labels(label_preview, prediction.record)
        overlay_preview = cv2.addWeighted(loaded_image.preview_rgb, 0.48, color_preview, 0.52, 0)

        segmentation_preview_path = result_dir / "segmentation_preview.png"
        marked_preview_path = result_dir / "marked_preview.png"
        Image.fromarray(color_preview, mode="RGB").save(segmentation_preview_path)
        Image.fromarray(overlay_preview, mode="RGB").save(marked_preview_path)

        patch_prediction_csv = result_dir / "patch_predictions.csv"
        with patch_prediction_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["row", "col", "class_id", "class_name", "confidence", "coverage_pixels"])
            for row_idx in range(prediction.patch_grid.shape[0]):
                for col_idx in range(prediction.patch_grid.shape[1]):
                    flat_idx = row_idx * prediction.patch_grid.shape[1] + col_idx
                    class_id = int(prediction.patch_labels[flat_idx])
                    writer.writerow(
                        [
                            row_idx,
                            col_idx,
                            class_id,
                            prediction.record.class_names[class_id],
                            round(float(prediction.patch_confidences[flat_idx]), 4),
                            int(prediction.patch_coverages[flat_idx]),
                        ]
                    )

        percentages_json_path = result_dir / "class_percentages.json"
        percentages_json_path.write_text(json.dumps(prediction.class_percentages, indent=2), encoding="utf-8")

        labels_npy_path = result_dir / "segmentation_labels.npy"
        np.save(labels_npy_path, (prediction.full_label_map if prediction.full_label_map is not None else prediction.patch_grid).astype(np.uint8))

        geotiff_path = None
        vector_zip_path = None
        vector_layers: list[dict[str, Any]] = []
        if loaded_image.raster_profile is not None:
            geotiff_path = result_dir / "segmentation_labels_georef.tif"
            self._write_geotiff_labels(
                geotiff_path,
                patch_grid=prediction.patch_grid,
                patch_size=prediction.record.patch_size,
                loaded_image=loaded_image,
                full_label_map=prediction.full_label_map,
            )
            vector_zip_path, vector_layers = self._write_class_shapefiles(
                geotiff_path=geotiff_path,
                destination_dir=result_dir / "class_shapefiles",
                record=prediction.record,
            )

        full_outputs_created = False
        segmentation_full_path = None
        marked_full_path = None
        if prediction.full_label_map is not None:
            full_labels = prediction.full_label_map[: loaded_image.height, : loaded_image.width]
            full_color = colorize_labels(full_labels, prediction.record)
            if (
                loaded_image.grayscale is not None
                and loaded_image.width * loaded_image.height <= self.settings.full_visual_pixel_limit
            ):
                gray_u8 = to_uint8(loaded_image.grayscale[: full_labels.shape[0], : full_labels.shape[1]])
                base_rgb = np.dstack([gray_u8] * 3)
                full_overlay = cv2.addWeighted(base_rgb, 0.48, full_color, 0.52, 0)

                segmentation_full_path = result_dir / "segmentation_full.png"
                marked_full_path = result_dir / "marked_full.png"
                Image.fromarray(full_color, mode="RGB").save(segmentation_full_path)
                Image.fromarray(full_overlay, mode="RGB").save(marked_full_path)
                full_outputs_created = True
        elif (
            loaded_image.grayscale is not None
            and loaded_image.valid_mask is not None
            and loaded_image.width * loaded_image.height <= self.settings.full_visual_pixel_limit
        ):
            full_labels = build_full_label_map(
                prediction.patch_grid,
                patch_size=prediction.record.patch_size,
                height=loaded_image.height,
                width=loaded_image.width,
            )
            full_labels[~loaded_image.valid_mask] = NO_DATA_LABEL
            full_color = colorize_labels(full_labels, prediction.record)
            gray_u8 = to_uint8(loaded_image.grayscale)
            base_rgb = np.dstack([gray_u8] * 3)
            full_overlay = cv2.addWeighted(base_rgb, 0.48, full_color, 0.52, 0)

            segmentation_full_path = result_dir / "segmentation_full.png"
            marked_full_path = result_dir / "marked_full.png"
            Image.fromarray(full_color, mode="RGB").save(segmentation_full_path)
            Image.fromarray(full_overlay, mode="RGB").save(marked_full_path)
            full_outputs_created = True

        summary_text_path = result_dir / "RESULTS_SIMPLE.txt"
        summary_lines = [
            "Urban Feature Platform Result",
            "",
            f"Selected model: {prediction.record.display_name}",
            f"Dataset family: {prediction.record.dataset_name}",
            f"Model type: {prediction.record.model_type}",
            f"Prediction mode: {mode}",
            f"Image processing mode: {loaded_image.processing_mode}",
            f"Requested model id: {requested_model_id or 'Auto'}",
            f"Top detected class: {prediction.dominant_class_name}",
            f"Model fit score: {prediction.selected_score:.4f}",
            f"{prediction.record.primary_metric_label} stored for this model: {float(prediction.record.primary_metric_value or 0.0):.4f}",
            "",
            "Class percentages:",
        ]
        for item in prediction.class_percentages:
            summary_lines.append(f"- {item['class_name']}: {item['percentage']:.2f}%")
        summary_lines.extend(
            [
                "",
                "Files generated:",
                f"- {input_preview_path.name}",
                f"- {marked_preview_path.name}",
                f"- {segmentation_preview_path.name}",
                f"- {patch_prediction_csv.name}",
                f"- {percentages_json_path.name}",
                f"- {labels_npy_path.name}",
            ]
        )
        if geotiff_path is not None:
            summary_lines.append(f"- {geotiff_path.name}")
        if vector_zip_path is not None:
            summary_lines.append(f"- {vector_zip_path.name}")
            summary_lines.append("- class_shapefiles/ folder with one georeferenced shapefile per detected class")
        if full_outputs_created:
            summary_lines.append(f"- {marked_full_path.name}")
            summary_lines.append(f"- {segmentation_full_path.name}")
        else:
            summary_lines.append("- Full-size PNG visuals were skipped to keep memory usage safe for this image size.")
        summary_text_path.write_text("\n".join(summary_lines), encoding="utf-8")

        summary_json = {
            "result_generated_at": datetime.now().isoformat(timespec="seconds"),
            "image": loaded_image.source_metadata,
            "selected_model": {
                "id": prediction.record.id,
                "display_name": prediction.record.display_name,
                "dataset_name": prediction.record.dataset_name,
                "model_type": prediction.record.model_type,
                "primary_metric_label": prediction.record.primary_metric_label,
                "secondary_metric_label": prediction.record.secondary_metric_label,
                "validation_accuracy": prediction.record.model_metrics.get("val_accuracy"),
                "validation_macro_f1": prediction.record.secondary_metric_value,
            },
            "selection_mode": mode,
            "requested_model_id": requested_model_id,
            "model_fit_score": round(prediction.selected_score, 4),
            "score_breakdown": prediction.score_breakdown,
            "dominant_class": prediction.dominant_class_name,
            "heuristic_water_overrides": prediction.heuristic_water_overrides,
            "class_percentages": prediction.class_percentages,
            "vector_shapefiles": vector_layers,
            "candidate_rankings": candidate_rankings,
            "full_outputs_created": full_outputs_created,
            "note": "These project models were trained on cluster-derived pseudo-labels, so the stored validation scores describe fit to that pseudo-label setup rather than fully manual ground-truth annotation.",
        }

        summary_json_path = result_dir / "prediction_summary.json"
        summary_json_path.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

        def file_url(path: Path | None) -> str | None:
            if path is None:
                return None
            return f"/results/{result_dir.name}/{path.name}"

        legend = [
            {
                "class_id": int(class_id),
                "class_name": class_name,
                "color_rgb": list(prediction.record.display_class_colors.get(class_id, (200, 200, 200))),
            }
            for class_id, class_name in sorted(prediction.record.class_names.items())
        ]

        return {
            "result_id": result_dir.name,
            "image": loaded_image.source_metadata,
            "selection_mode": mode,
            "selected_model": summary_json["selected_model"],
            "score_breakdown": prediction.score_breakdown,
            "class_percentages": prediction.class_percentages,
            "dominant_class": prediction.dominant_class_name,
            "candidate_rankings": candidate_rankings,
            "legend": legend,
            "artifacts": {
                "input_preview_url": file_url(input_preview_path),
                "marked_preview_url": file_url(marked_preview_path),
                "segmentation_preview_url": file_url(segmentation_preview_path),
                "marked_full_url": file_url(marked_full_path),
                "segmentation_full_url": file_url(segmentation_full_path),
                "labels_npy_url": file_url(labels_npy_path),
                "geotiff_url": file_url(geotiff_path),
                "class_shapefiles_zip_url": file_url(vector_zip_path),
                "summary_json_url": file_url(summary_json_path),
                "summary_text_url": file_url(summary_text_path),
                "patch_predictions_csv_url": file_url(patch_prediction_csv),
                "class_percentages_url": file_url(percentages_json_path),
            },
            "note": summary_json["note"],
        }

    def predict_file(self, file_path: Path, mode: str = "auto", model_id: str | None = None) -> dict[str, Any]:
        loaded_image = load_uploaded_image(
            file_path,
            preview_max_dimension=self.settings.preview_max_dimension,
            raster_stream_pixel_limit=self.settings.raster_stream_pixel_limit,
            raster_stream_file_size_mb=self.settings.raster_stream_file_size_mb,
            raster_stats_sample_max_dimension=self.settings.raster_stats_sample_max_dimension,
        )
        records: list[ModelRecord]
        if mode == "manual":
            if not model_id:
                raise ValueError("A model must be selected in manual mode.")
            records = [self.registry.get(model_id)]
        else:
            records = [self.registry.get(model["id"]) for model in self.registry.list_models(auto_only=True)]
            dense_auto_records = [record for record in records if record.model_type == "dense_unet"]
            if dense_auto_records and len(dense_auto_records) == len(records):
                records = [self._choose_dense_auto_record(loaded_image, dense_auto_records)]

        if not records:
            raise RuntimeError("No enabled models are available.")

        best_prediction: ModelPrediction | None = None
        candidate_rankings: list[dict[str, Any]] = []
        predictions_by_id: dict[str, ModelPrediction] = {}
        tile_cache: dict[int, TiledImage] = {}

        for record in records:
            if record.model_type in {"prabhakar_unet", "prabhakar_resnet18", "prabhakar_maskrcnn"}:
                from .prabhakar_inference import predict_prabhakar_unet, predict_prabhakar_resnet18, predict_prabhakar_maskrcnn
                if record.model_type == "prabhakar_unet":
                    prediction = predict_prabhakar_unet(loaded_image, record, batch_size=4)
                elif record.model_type == "prabhakar_resnet18":
                    prediction = predict_prabhakar_resnet18(loaded_image, record, batch_size=16)
                else:
                    prediction = predict_prabhakar_maskrcnn(loaded_image, record)
            elif loaded_image.processing_mode == "streaming_raster":
                if record.model_type == "patch_classifier":
                    prediction = self._predict_with_patch_classifier_streaming(loaded_image, record)
                elif record.model_type == "unet":
                    prediction = self._predict_with_unet_streaming(loaded_image, record)
                elif record.model_type == "dense_unet":
                    prediction = self._predict_with_dense_unet_streaming(loaded_image, record)
                else:
                    raise ValueError(f"Unsupported model type: {record.model_type}")
            else:
                tiled = tile_cache.get(record.patch_size)
                if tiled is None:
                    assert loaded_image.grayscale is not None
                    assert loaded_image.valid_mask is not None
                    tiled = tile_image(loaded_image.grayscale, loaded_image.valid_mask, patch_size=record.patch_size)
                    tile_cache[record.patch_size] = tiled
                if record.model_type == "patch_classifier":
                    prediction = self._predict_with_patch_classifier(loaded_image, tiled, record)
                elif record.model_type == "unet":
                    prediction = self._predict_with_unet(loaded_image, tiled, record)
                elif record.model_type == "dense_unet":
                    prediction = self._predict_with_dense_unet(loaded_image, tiled, record)
                else:
                    raise ValueError(f"Unsupported model type: {record.model_type}")
            candidate_entry = {
                "model_id": record.id,
                "display_name": record.display_name,
                "dataset_name": record.dataset_name,
                "model_type": record.model_type,
                "score": round(prediction.selected_score, 4),
                "dominant_class": prediction.dominant_class_name,
                "primary_metric_label": record.primary_metric_label,
                "validation_accuracy": round(float(record.model_metrics.get("val_accuracy", 0.0) or 0.0), 4),
                "secondary_metric_label": record.secondary_metric_label,
                "secondary_metric_value": round(float(record.secondary_metric_value or 0.0), 4)
                if record.secondary_metric_value is not None
                else None,
                "score_breakdown": prediction.score_breakdown,
            }
            candidate_rankings.append(candidate_entry)
            predictions_by_id[record.id] = prediction

            if best_prediction is None or prediction.selected_score > best_prediction.selected_score:
                best_prediction = prediction

        assert best_prediction is not None
        if mode == "auto" and len(candidate_rankings) > 1:
            ranked_by_domain = sorted(
                candidate_rankings,
                key=lambda item: item["score_breakdown"]["domain_similarity"],
                reverse=True,
            )
            domain_gap = ranked_by_domain[0]["score_breakdown"]["domain_similarity"] - ranked_by_domain[1]["score_breakdown"]["domain_similarity"]
            if domain_gap >= 0.04:
                allowed_ids = {
                    item["model_id"]
                    for item in candidate_rankings
                    if item["score_breakdown"]["domain_similarity"] >= ranked_by_domain[0]["score_breakdown"]["domain_similarity"] - 0.02
                }
                filtered_rankings = [item for item in candidate_rankings if item["model_id"] in allowed_ids]
                if filtered_rankings:
                    candidate_rankings = filtered_rankings
                    best_prediction = max(
                        (predictions_by_id[item["model_id"]] for item in candidate_rankings),
                        key=lambda pred: pred.selected_score,
                    )

        candidate_rankings.sort(key=lambda item: item["score"], reverse=True)
        if mode == "auto" and candidate_rankings:
            best_prediction = predictions_by_id[candidate_rankings[0]["model_id"]]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
        result_dir = ensure_dir(self.settings.result_dir / result_id)

        return self._save_outputs(
            result_dir=result_dir,
            loaded_image=loaded_image,
            prediction=best_prediction,
            mode=mode,
            candidate_rankings=candidate_rankings,
            requested_model_id=model_id,
        )
