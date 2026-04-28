from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import torch

from .config import Settings


@dataclass
class ModelRecord:
    id: str
    display_name: str
    summary_path: Path
    model_path: Path
    enabled: bool
    priority: int
    configured_model_type: str
    configured_auto_select: bool
    configured_patch_size: int
    configured_std_threshold: float
    configured_valid_fraction_threshold: float
    configured_dark_water_mean_threshold: float
    summary: dict[str, Any]
    bundle: dict[str, Any] | None = None

    @property
    def class_names(self) -> dict[int, str]:
        dataset = self.summary["dataset"]
        raw = dataset.get("class_names") or dataset.get("spec", {}).get("class_names", {})
        return {int(k): v for k, v in raw.items()}

    @property
    def model_metrics(self) -> dict[str, Any]:
        if "metrics" in self.summary:
            return self.summary["metrics"]

        if self.model_type == "dense_unet":
            model_name = self.model_path.name
            if "hard_unet" in model_name:
                metrics = self.summary.get("hard_unet", {}).get("final_val_metrics", {})
            elif "soft_unet" in model_name:
                metrics = self.summary.get("soft_unet", {}).get("final_val_metrics", {})
            else:
                metrics = self.summary.get("best_variant_metrics", {})
            return {
                "val_accuracy": metrics.get("pixel_accuracy"),
                "val_macro_f1": None,
                "mean_iou": metrics.get("mean_iou"),
            }

        final_val = self.summary.get("final_val_metrics", {})
        return {
            "val_accuracy": final_val.get("pixel_accuracy"),
            "val_macro_f1": None,
            "mean_iou": final_val.get("mean_iou"),
        }

    @property
    def model_type(self) -> str:
        return self.configured_model_type

    @property
    def auto_select(self) -> bool:
        return self.configured_auto_select

    @property
    def model_config(self) -> dict[str, Any]:
        return self.summary.get("config", {})

    @property
    def patch_size(self) -> int:
        return int(self.configured_patch_size)

    @property
    def std_threshold_value(self) -> float:
        return float(self.configured_std_threshold)

    @property
    def dark_water_mean_threshold_value(self) -> float:
        return float(self.configured_dark_water_mean_threshold)

    @property
    def dataset_name(self) -> str:
        dataset = self.summary["dataset"]
        return dataset.get("name") or dataset.get("spec", {}).get("name", "Unknown Dataset")

    @property
    def valid_fraction_threshold_value(self) -> float:
        return float(self.configured_valid_fraction_threshold)

    @property
    def primary_metric_label(self) -> str:
        if self.model_type in {"patch_classifier", "prabhakar_resnet18"}:
            return "Validation accuracy"
        return "Pixel accuracy"

    @property
    def primary_metric_value(self) -> float | None:
        value = self.model_metrics.get("val_accuracy")
        return None if value is None else float(value)

    @property
    def secondary_metric_label(self) -> str:
        if self.model_type in {"patch_classifier", "prabhakar_resnet18"}:
            return "Macro F1"
        return "Mean IoU"

    @property
    def secondary_metric_value(self) -> float | None:
        if self.model_type in {"patch_classifier", "prabhakar_resnet18"}:
            value = self.model_metrics.get("val_macro_f1")
        else:
            value = self.model_metrics.get("mean_iou")
        return None if value is None else float(value)

    @property
    def display_class_colors(self) -> dict[int, tuple[int, int, int]]:
        return {
            0: (31, 119, 180),
            1: (214, 39, 40),
            2: (140, 86, 75),
            3: (44, 160, 44),
            4: (255, 215, 0),
            5: (148, 103, 189),
            6: (255, 127, 14),
        }

    def load_bundle(self) -> dict[str, Any]:
        if self.bundle is None:
            if self.model_type == "patch_classifier":
                self.bundle = joblib.load(self.model_path)
            elif self.model_type in {"unet", "dense_unet", "prabhakar_unet", "prabhakar_resnet18", "prabhakar_maskrcnn"}:
                self.bundle = torch.load(self.model_path, map_location="cpu", weights_only=False)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        return self.bundle


class ModelRegistry:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self._records: dict[str, ModelRecord] | None = None

    def _load(self) -> dict[str, ModelRecord]:
        raw_entries = json.loads(self.settings.registry_path.read_text())
        records: dict[str, ModelRecord] = {}
        for entry in raw_entries:
            if not entry.get("enabled", True):
                continue
            summary_path = (self.settings.root_dir / entry["summary_path"]).resolve()
            model_path = (self.settings.root_dir / entry["model_path"]).resolve()
            summary = json.loads(summary_path.read_text())
            record = ModelRecord(
                id=entry["id"],
                display_name=entry["display_name"],
                summary_path=summary_path,
                model_path=model_path,
                enabled=bool(entry.get("enabled", True)),
                priority=int(entry.get("priority", 0)),
                configured_model_type=str(entry.get("model_type", "patch_classifier")),
                configured_auto_select=bool(entry.get("auto_select", True)),
                configured_patch_size=int(entry.get("patch_size", 256)),
                configured_std_threshold=float(entry.get("std_threshold", 0.05)),
                configured_valid_fraction_threshold=float(entry.get("valid_fraction_threshold", 0.9)),
                configured_dark_water_mean_threshold=float(entry.get("dark_water_mean_threshold", 0.3)),
                summary=summary,
            )
            records[record.id] = record
        return records

    def get_records(self) -> dict[str, ModelRecord]:
        with self._lock:
            if self._records is None:
                self._records = self._load()
            return self._records

    def list_models(self, *, auto_only: bool = False) -> list[dict[str, Any]]:
        models = []
        for record in sorted(self.get_records().values(), key=lambda r: (-r.priority, r.display_name)):
            if auto_only and not record.auto_select:
                continue
            models.append(
                {
                    "id": record.id,
                    "display_name": record.display_name,
                    "dataset_name": record.dataset_name,
                    "class_names": record.class_names,
                    "model_type": record.model_type,
                    "available_in_auto": record.auto_select,
                    "patch_size": record.patch_size,
                    "validation_accuracy": record.primary_metric_value,
                    "validation_macro_f1": record.secondary_metric_value,
                    "primary_metric_label": record.primary_metric_label,
                    "primary_metric_value": record.primary_metric_value,
                    "secondary_metric_label": record.secondary_metric_label,
                    "secondary_metric_value": record.secondary_metric_value,
                }
            )
        return models

    def get(self, model_id: str) -> ModelRecord:
        records = self.get_records()
        if model_id not in records:
            raise KeyError(f"Unknown model: {model_id}")
        return records[model_id]
