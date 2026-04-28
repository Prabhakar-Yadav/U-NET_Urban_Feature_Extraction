from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


CLASS_COLORS = {
    0: (31, 119, 180),
    1: (214, 39, 40),
    2: (140, 86, 75),
    3: (44, 160, 44),
    4: (255, 215, 0),
}


@dataclass(frozen=True)
class DenseSegmentationSpec:
    key: str
    name: str
    image_path: str
    teacher_summary_path: str
    teacher_model_path: str
    class_names: dict[int, str]
    teacher_patch_size: int
    student_patch_size: int
    valid_fraction_threshold: float
    teacher_seed_min_confidence: float
    teacher_max_patches_per_class: int
    pixel_samples_per_patch: int
    dense_context_pad: int
    dense_std_threshold: float
    dark_water_mean_threshold: float


SPECS = {
    "jnpa_dense_v128": DenseSegmentationSpec(
        key="jnpa_dense_v128",
        name="JNPA Dense Segmentation V128",
        image_path="JNPA/JNPA_2_5.tif",
        teacher_summary_path="outputs/jnpa_patch_classifier_v2/run_summary_v2.json",
        teacher_model_path="outputs/jnpa_patch_classifier_v2/patch_classifier_model_v2.joblib",
        class_names={
            0: "Water Bodies",
            1: "Industrial / Port Infrastructure",
            2: "Bare Land / Soil",
            3: "Vegetation / Mangroves",
            4: "Urban Built-up",
        },
        teacher_patch_size=256,
        student_patch_size=128,
        valid_fraction_threshold=0.85,
        teacher_seed_min_confidence=0.72,
        teacher_max_patches_per_class=220,
        pixel_samples_per_patch=96,
        dense_context_pad=16,
        dense_std_threshold=0.02,
        dark_water_mean_threshold=0.32,
    ),
    "cartosat_dense_v128": DenseSegmentationSpec(
        key="cartosat_dense_v128",
        name="CARTOSAT Dense Segmentation V128",
        image_path="Monocromatic/CARTOSAT_1M_PAN.tif",
        teacher_summary_path="outputs/cartosat_patch_classifier_v3_4class/run_summary_v3.json",
        teacher_model_path="outputs/cartosat_patch_classifier_v3_4class/patch_classifier_model_v3.joblib",
        class_names={
            0: "Water",
            1: "Dense Urban Built-up",
            2: "Port / Waterfront Infrastructure",
            3: "Terrain / Open Ground",
        },
        teacher_patch_size=256,
        student_patch_size=128,
        valid_fraction_threshold=0.95,
        teacher_seed_min_confidence=0.70,
        teacher_max_patches_per_class=260,
        pixel_samples_per_patch=96,
        dense_context_pad=16,
        dense_std_threshold=0.018,
        dark_water_mean_threshold=0.26,
    ),
}


@dataclass
class DenseSegmentationConfig:
    dataset_key: str
    output_dir: str
    seed: int = 42
    robust_percentiles: tuple[float, float] = (1.0, 99.0)
    resize_for_teacher_features: int = 32
    teacher_canny_low: int = 60
    teacher_canny_high: int = 160
    pixel_mlp_hidden: int = 64
    pixel_mlp_epochs: int = 18
    pixel_mlp_batch_size: int = 8192
    pixel_mlp_learning_rate: float = 1e-3
    pixel_val_fraction: float = 0.2
    unet_epochs: int = 8
    unet_batch_size: int = 24
    unet_learning_rate: float = 8e-4
    unet_weight_decay: float = 1e-4
    unet_patience: int = 3
    train_fraction: float = 0.8
    num_workers: int = 0
    base_channels: int = 16
    preview_count: int = 8
    compare_margin_miou: float = 0.03
    max_train_patches: int = 1800
    max_val_patches: int = 480

    def output_path(self) -> Path:
        return Path(self.output_dir)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_ready(obj: Any):
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_raster(spec: DenseSegmentationSpec, robust_percentiles: tuple[float, float]):
    with rasterio.open(spec.image_path) as src:
        image = src.read(1).astype(np.float32)
        nodata = src.nodata
        valid_mask = np.ones_like(image, dtype=bool)
        if nodata is not None:
            valid_mask &= np.not_equal(image, nodata)
        valid_pixels = image[valid_mask]
        low, high = np.percentile(valid_pixels, robust_percentiles)
        image = np.clip((image - low) / max(high - low, 1e-6), 0.0, 1.0)
        image[~valid_mask] = 0.0
        metadata = {
            "path": spec.image_path,
            "width": int(src.width),
            "height": int(src.height),
            "dtype": str(src.dtypes[0]),
            "crs": str(src.crs),
            "transform": list(src.transform),
            "nodata": float(nodata) if nodata is not None else None,
            "normalization_low": float(low),
            "normalization_high": float(high),
        }
    return image, valid_mask, metadata


def compute_teacher_patch_features(
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


def build_patch_table(
    image: np.ndarray,
    valid_mask: np.ndarray,
    patch_size: int,
    valid_fraction_threshold: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows = image.shape[0] // patch_size
    cols = image.shape[1] // patch_size
    patches = []
    records = []
    for row in range(rows):
        for col in range(cols):
            y0 = row * patch_size
            x0 = col * patch_size
            patch = image[y0 : y0 + patch_size, x0 : x0 + patch_size]
            patch_valid = valid_mask[y0 : y0 + patch_size, x0 : x0 + patch_size]
            valid_fraction = float(patch_valid.mean())
            patch_std = float(patch.std())
            patches.append(patch)
            records.append(
                {
                    "grid_row": row,
                    "grid_col": col,
                    "y0": y0,
                    "x0": x0,
                    "valid_fraction": valid_fraction,
                    "std": patch_std,
                    "keep": valid_fraction >= valid_fraction_threshold,
                }
            )
    frame = pd.DataFrame.from_records(records)
    return frame, np.stack(patches).astype(np.float32)


def load_teacher(spec: DenseSegmentationSpec):
    summary = json.loads(Path(spec.teacher_summary_path).read_text())
    bundle = joblib.load(spec.teacher_model_path)
    return summary, bundle


def predict_teacher_patches(
    spec: DenseSegmentationSpec,
    config: DenseSegmentationConfig,
    image: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    teacher_summary, bundle = load_teacher(spec)
    patch_table, patches = build_patch_table(
        image=image,
        valid_mask=valid_mask,
        patch_size=spec.teacher_patch_size,
        valid_fraction_threshold=spec.valid_fraction_threshold,
    )
    feature_matrix = compute_teacher_patch_features(
        patches,
        resize_for_features=int(teacher_summary["config"].get("resize_for_features", config.resize_for_teacher_features)),
        canny_low=int(teacher_summary["config"].get("canny_low", config.teacher_canny_low)),
        canny_high=int(teacher_summary["config"].get("canny_high", config.teacher_canny_high)),
    )
    reduced = bundle["pca"].transform(feature_matrix)
    classifier = bundle["classifier"]
    probabilities = classifier.predict_proba(reduced)
    predicted_indices = probabilities.argmax(axis=1)
    predicted_labels = classifier.classes_[predicted_indices].astype(np.uint8)
    confidences = probabilities[np.arange(len(predicted_indices)), predicted_indices].astype(np.float32)

    for idx, row in patch_table.iterrows():
        if row["std"] < spec.dense_std_threshold and patches[idx].mean() < spec.dark_water_mean_threshold and 0 in spec.class_names:
            predicted_labels[idx] = 0
            confidences[idx] = max(confidences[idx], 0.85)

    patch_table = patch_table.copy()
    patch_table["teacher_label"] = predicted_labels
    patch_table["teacher_confidence"] = confidences
    patch_table["teacher_patch_id"] = np.arange(len(patch_table))
    return patch_table, patches


def select_seed_patches(
    patch_table: pd.DataFrame,
    spec: DenseSegmentationSpec,
) -> pd.DataFrame:
    selected = []
    usable = patch_table[patch_table["keep"]].copy()
    usable = usable.sort_values("teacher_confidence", ascending=False)
    for class_id in sorted(spec.class_names):
        class_rows = usable[usable["teacher_label"] == class_id].copy()
        class_rows = class_rows[class_rows["teacher_confidence"] >= spec.teacher_seed_min_confidence]
        if class_rows.empty:
            class_rows = usable[usable["teacher_label"] == class_id].head(spec.teacher_max_patches_per_class)
        else:
            class_rows = class_rows.head(spec.teacher_max_patches_per_class)
        selected.append(class_rows)
    result = pd.concat(selected, ignore_index=True).drop_duplicates(subset=["teacher_patch_id"])
    return result.sort_values(["teacher_label", "teacher_confidence"], ascending=[True, False]).reset_index(drop=True)


def to_u8(image: np.ndarray) -> np.ndarray:
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


class PixelFeatureExtractor:
    def __init__(self, pad: int):
        self.pad = pad

    def build_context(self, padded_image: np.ndarray, y0: int, x0: int, patch_size: int) -> np.ndarray:
        size = patch_size + 2 * self.pad
        return padded_image[y0 : y0 + size, x0 : x0 + size]

    def compute_patch_features(self, context: np.ndarray, patch_size: int) -> np.ndarray:
        center = context[self.pad : self.pad + patch_size, self.pad : self.pad + patch_size]
        mean5 = cv2.blur(context, (5, 5))
        sqmean5 = cv2.blur(context * context, (5, 5))
        std5 = np.sqrt(np.clip(sqmean5 - mean5 * mean5, 0.0, None))
        mean15 = cv2.blur(context, (15, 15))
        sqmean15 = cv2.blur(context * context, (15, 15))
        std15 = np.sqrt(np.clip(sqmean15 - mean15 * mean15, 0.0, None))
        gx = cv2.Sobel(context, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(context, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        lap = np.abs(cv2.Laplacian(context, cv2.CV_32F, ksize=3))
        edges = cv2.Canny(to_u8(context), 40, 140).astype(np.float32) / 255.0

        slices = (slice(self.pad, self.pad + patch_size), slice(self.pad, self.pad + patch_size))
        features = np.stack(
            [
                center,
                mean5[slices],
                std5[slices],
                mean15[slices],
                std15[slices],
                grad[slices],
                lap[slices],
                edges[slices],
            ],
            axis=-1,
        )
        return features.astype(np.float32)


class PixelMLP(nn.Module):
    def __init__(self, in_features: int, num_classes: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PixelDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, weights: np.ndarray):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()
        self.weights = torch.from_numpy(weights).float()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index], self.weights[index]


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, base_channels: int):
        super().__init__()
        self.down1 = DoubleConv(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.bridge = DoubleConv(base_channels * 4, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)
        self.head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        bridge = self.bridge(self.pool3(d3))
        u3 = self.up3(bridge)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.dec3(u3)
        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.dec2(u2)
        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.dec1(u1)
        return self.head(u1)


def infer_unet_base_channels(state_dict: dict[str, torch.Tensor]) -> int:
    return int(state_dict["down1.block.0.weight"].shape[0])


def load_saved_unet(model_path: Path, num_classes: int) -> tuple[UNet, dict[str, Any]]:
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    base_channels = infer_unet_base_channels(state_dict)
    model = UNet(in_channels=1, num_classes=num_classes, base_channels=base_channels)
    model.load_state_dict(state_dict)
    return model, checkpoint


def soft_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    return -(target_probs * log_probs).sum(dim=1).mean()


def dice_loss_hard(logits: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    target_1h = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = torch.sum(probs * target_1h, dim=dims)
    cardinality = torch.sum(probs + target_1h, dim=dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


def dice_loss_soft(logits: torch.Tensor, target_probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    dims = (0, 2, 3)
    intersection = torch.sum(probs * target_probs, dim=dims)
    cardinality = torch.sum(probs + target_probs, dim=dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


def confusion_to_metrics(confusion: np.ndarray) -> dict[str, Any]:
    total = confusion.sum()
    correct = np.trace(confusion)
    pixel_accuracy = float(correct / max(total, 1))
    class_total = confusion.sum(axis=1)
    class_correct = np.diag(confusion)
    class_accuracy = class_correct / np.maximum(class_total, 1)
    union = class_total + confusion.sum(axis=0) - class_correct
    iou = class_correct / np.maximum(union, 1)
    return {
        "pixel_accuracy": pixel_accuracy,
        "class_accuracy": class_accuracy.tolist(),
        "class_iou": iou.tolist(),
        "mean_iou": float(np.nanmean(iou)),
    }


def colorize_mask(mask: np.ndarray, class_names: dict[int, str]) -> np.ndarray:
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id in class_names:
        color[mask == class_id] = CLASS_COLORS.get(class_id, (200, 200, 200))
    return color


def save_map_outputs(image: np.ndarray, label_map: np.ndarray, output_dir: Path, prefix: str, class_names: dict[int, str]) -> dict[str, str]:
    pred_dir = ensure_dir(output_dir / "predictions")
    color = colorize_mask(label_map, class_names)
    base = np.dstack([to_u8(image)] * 3)
    overlay = cv2.addWeighted(base, 0.45, color, 0.55, 0)
    label_path = pred_dir / f"{prefix}_labels.npy"
    color_path = pred_dir / f"{prefix}_color.png"
    overlay_path = pred_dir / f"{prefix}_overlay.png"
    np.save(label_path, label_map.astype(np.uint8))
    Image.fromarray(color, mode="RGB").save(color_path)
    Image.fromarray(overlay, mode="RGB").save(overlay_path)
    return {
        "label_path": str(label_path),
        "color_path": str(color_path),
        "overlay_path": str(overlay_path),
    }


def collect_pixel_training_data(
    image: np.ndarray,
    valid_mask: np.ndarray,
    seed_patches: pd.DataFrame,
    spec: DenseSegmentationSpec,
    feature_extractor: PixelFeatureExtractor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pad = feature_extractor.pad
    padded_image = np.pad(image, pad_width=pad, mode="reflect")
    padded_valid = np.pad(valid_mask.astype(np.uint8), pad_width=pad, mode="edge").astype(bool)

    feature_blocks = []
    label_blocks = []
    weight_blocks = []
    for _, row in seed_patches.iterrows():
        y0 = int(row["y0"])
        x0 = int(row["x0"])
        context = feature_extractor.build_context(padded_image, y0, x0, spec.teacher_patch_size)
        features = feature_extractor.compute_patch_features(context, spec.teacher_patch_size)
        valid_patch = padded_valid[y0 + pad : y0 + pad + spec.teacher_patch_size, x0 + pad : x0 + pad + spec.teacher_patch_size]
        valid_idx = np.argwhere(valid_patch)
        if len(valid_idx) == 0:
            continue
        sample_count = min(spec.pixel_samples_per_patch, len(valid_idx))
        choose = np.random.choice(len(valid_idx), size=sample_count, replace=False)
        coords = valid_idx[choose]
        samples = features[coords[:, 0], coords[:, 1]]
        labels = np.full(sample_count, int(row["teacher_label"]), dtype=np.int64)
        weights = np.full(sample_count, float(row["teacher_confidence"]), dtype=np.float32)
        feature_blocks.append(samples.astype(np.float32))
        label_blocks.append(labels)
        weight_blocks.append(weights)

    x = np.concatenate(feature_blocks, axis=0)
    y = np.concatenate(label_blocks, axis=0)
    w = np.concatenate(weight_blocks, axis=0)
    return x, y, w


def train_pixel_mlp(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    spec: DenseSegmentationSpec,
    config: DenseSegmentationConfig,
    output_dir: Path,
) -> tuple[PixelMLP, dict[str, Any], np.ndarray, np.ndarray]:
    train_idx, val_idx = train_test_split(
        np.arange(len(y)),
        test_size=config.pixel_val_fraction,
        random_state=config.seed,
        stratify=y,
    )
    x_train = x[train_idx]
    x_val = x[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    w_train = w[train_idx]
    w_val = w[val_idx]

    mean = x_train.mean(axis=0, keepdims=True)
    std = np.maximum(x_train.std(axis=0, keepdims=True), 1e-4)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    train_loader = DataLoader(
        PixelDataset(x_train.astype(np.float32), y_train.astype(np.int64), w_train.astype(np.float32)),
        batch_size=config.pixel_mlp_batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        PixelDataset(x_val.astype(np.float32), y_val.astype(np.int64), w_val.astype(np.float32)),
        batch_size=config.pixel_mlp_batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = resolve_device()
    model = PixelMLP(in_features=x.shape[1], num_classes=len(spec.class_names), hidden=config.pixel_mlp_hidden).to(device)
    class_counts = np.bincount(y_train, minlength=len(spec.class_names)).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    class_weights_t = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.pixel_mlp_learning_rate)

    history = []
    best_state = None
    best_acc = -1.0
    for epoch in range(1, config.pixel_mlp_epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch_x, batch_y, batch_w in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)
            logits = model(batch_x)
            per_sample = F.cross_entropy(logits, batch_y, weight=class_weights_t, reduction="none")
            loss = (per_sample * batch_w).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            train_count += batch_x.size(0)

        model.eval()
        val_preds = []
        val_targets = []
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch_x, batch_y, batch_w in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_w = batch_w.to(device)
                logits = model(batch_x)
                per_sample = F.cross_entropy(logits, batch_y, weight=class_weights_t, reduction="none")
                loss = (per_sample * batch_w).mean()
                val_loss += loss.item() * batch_x.size(0)
                val_count += batch_x.size(0)
                val_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                val_targets.append(batch_y.cpu().numpy())
        val_pred = np.concatenate(val_preds)
        val_target = np.concatenate(val_targets)
        val_acc = float((val_pred == val_target).mean())
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / max(train_count, 1),
                "val_loss": val_loss / max(val_count, 1),
                "val_accuracy": val_acc,
            }
        )
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()

    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save(
        {
            "model": model.state_dict(),
            "feature_mean": mean.astype(np.float32),
            "feature_std": std.astype(np.float32),
            "num_classes": len(spec.class_names),
        },
        output_dir / "pixel_teacher_mlp.pt",
    )
    pd.DataFrame(history).to_csv(output_dir / "pixel_teacher_history.csv", index=False)

    with torch.no_grad():
        x_val_t = torch.from_numpy(x_val.astype(np.float32)).to(device)
        logits = model(x_val_t)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        val_pred = probabilities.argmax(axis=1)
    confusion = confusion_matrix(y_val, val_pred, labels=np.arange(len(spec.class_names)))
    metrics = confusion_to_metrics(confusion)
    metrics["val_accuracy"] = metrics["pixel_accuracy"]
    metrics["feature_mean"] = mean.squeeze(0).tolist()
    metrics["feature_std"] = std.squeeze(0).tolist()
    return model, metrics, mean.astype(np.float32), std.astype(np.float32)


def predict_pixel_probabilities(
    model: PixelMLP,
    feature_block: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    normalized = (feature_block - mean) / std
    flat = normalized.reshape(-1, normalized.shape[-1]).astype(np.float32)
    with torch.no_grad():
        tensor = torch.from_numpy(flat).to(device)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs.reshape(feature_block.shape[0], feature_block.shape[1], -1).astype(np.float32)


def build_dense_pseudo_dataset(
    image: np.ndarray,
    valid_mask: np.ndarray,
    teacher_patch_table: pd.DataFrame,
    pixel_model: PixelMLP,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    spec: DenseSegmentationSpec,
    config: DenseSegmentationConfig,
    output_dir: Path,
) -> pd.DataFrame:
    dataset_dir = ensure_dir(output_dir / "dense_dataset")
    image_dir = ensure_dir(dataset_dir / "images")
    hard_mask_dir = ensure_dir(dataset_dir / "hard_masks")
    soft_target_dir = ensure_dir(dataset_dir / "soft_targets")

    feature_extractor = PixelFeatureExtractor(spec.dense_context_pad)
    pad = feature_extractor.pad
    padded_image = np.pad(image, pad_width=pad, mode="reflect")
    padded_valid = np.pad(valid_mask.astype(np.uint8), pad_width=pad, mode="edge").astype(bool)
    device = resolve_device()
    pixel_model = pixel_model.to(device)
    pixel_model.eval()

    student_table, student_patches = build_patch_table(
        image=image,
        valid_mask=valid_mask,
        patch_size=spec.student_patch_size,
        valid_fraction_threshold=spec.valid_fraction_threshold,
    )
    teacher_rows = int(teacher_patch_table["grid_row"].max()) + 1
    teacher_cols = int(teacher_patch_table["grid_col"].max()) + 1
    records = []
    soft_entropy_values = []
    for idx, row in student_table.iterrows():
        if not bool(row["keep"]):
            continue
        y0 = int(row["y0"])
        x0 = int(row["x0"])
        context = feature_extractor.build_context(padded_image, y0, x0, spec.student_patch_size)
        feature_block = feature_extractor.compute_patch_features(context, spec.student_patch_size)
        probs = predict_pixel_probabilities(pixel_model, feature_block, feature_mean, feature_std, device)
        valid_patch = padded_valid[y0 + pad : y0 + pad + spec.student_patch_size, x0 + pad : x0 + pad + spec.student_patch_size]
        hard_mask = probs.argmax(axis=-1).astype(np.uint8)
        hard_mask[~valid_patch] = 0
        probs[~valid_patch] = 0.0
        probs_sum = probs.sum(axis=-1, keepdims=True)
        probs_sum = np.where(probs_sum > 0, probs_sum, 1.0)
        probs = probs / probs_sum

        patch_image_path = image_dir / f"patch_{idx:05d}.png"
        hard_mask_path = hard_mask_dir / f"mask_{idx:05d}.png"
        soft_target_path = soft_target_dir / f"soft_{idx:05d}.npy"
        Image.fromarray(to_u8(student_patches[idx]), mode="L").save(patch_image_path)
        Image.fromarray(hard_mask, mode="L").save(hard_mask_path)
        np.save(soft_target_path, probs.astype(np.float16))

        class_counts = np.bincount(hard_mask.reshape(-1), minlength=len(spec.class_names))
        dominant_class = int(class_counts.argmax())
        entropy = float(-(np.clip(probs, 1e-6, 1.0) * np.log(np.clip(probs, 1e-6, 1.0))).sum(axis=-1).mean())
        soft_entropy_values.append(entropy)
        teacher_row = min(int(y0 // spec.teacher_patch_size), teacher_rows - 1)
        teacher_col = min(int(x0 // spec.teacher_patch_size), teacher_cols - 1)
        teacher_id = teacher_row * teacher_cols + teacher_col
        teacher_info = teacher_patch_table.iloc[teacher_id]
        records.append(
            {
                "patch_id": idx,
                "grid_row": int(row["grid_row"]),
                "grid_col": int(row["grid_col"]),
                "y0": y0,
                "x0": x0,
                "valid_fraction": float(row["valid_fraction"]),
                "std": float(row["std"]),
                "dominant_class": dominant_class,
                "image_path": str(patch_image_path),
                "hard_mask_path": str(hard_mask_path),
                "soft_target_path": str(soft_target_path),
                "teacher_label": int(teacher_info["teacher_label"]),
                "teacher_confidence": float(teacher_info["teacher_confidence"]),
                "mean_soft_entropy": entropy,
            }
        )

    metadata = pd.DataFrame.from_records(records).sort_values(["grid_row", "grid_col"]).reset_index(drop=True)
    metadata.to_csv(dataset_dir / "metadata.csv", index=False)
    with open(dataset_dir / "dense_label_summary.json", "w", encoding="utf-8") as fh:
        json.dump(
            json_ready(
                {
                    "kept_student_patches": int(len(metadata)),
                    "mean_soft_entropy": float(np.mean(soft_entropy_values)) if soft_entropy_values else 0.0,
                    "dominant_class_counts": metadata["dominant_class"].value_counts().sort_index().to_dict(),
                }
            ),
            fh,
            indent=2,
        )
    return metadata


def random_augment_triplet(image: np.ndarray, hard_mask: np.ndarray | None, soft_target: np.ndarray | None):
    if random.random() < 0.5:
        image = np.flip(image, axis=1).copy()
        if hard_mask is not None:
            hard_mask = np.flip(hard_mask, axis=1).copy()
        if soft_target is not None:
            soft_target = np.flip(soft_target, axis=2).copy()
    if random.random() < 0.5:
        image = np.flip(image, axis=0).copy()
        if hard_mask is not None:
            hard_mask = np.flip(hard_mask, axis=0).copy()
        if soft_target is not None:
            soft_target = np.flip(soft_target, axis=1).copy()
    k = random.randint(0, 3)
    if k:
        image = np.rot90(image, k, axes=(0, 1)).copy()
        if hard_mask is not None:
            hard_mask = np.rot90(hard_mask, k, axes=(0, 1)).copy()
        if soft_target is not None:
            soft_target = np.rot90(soft_target, k, axes=(1, 2)).copy()
    return image, hard_mask, soft_target


class HardMaskDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, augment: bool):
        self.frame = frame.reset_index(drop=True)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = np.asarray(Image.open(row["image_path"]).convert("L"), dtype=np.float32) / 255.0
        hard_mask = np.asarray(Image.open(row["hard_mask_path"]).convert("L"), dtype=np.int64)
        if self.augment:
            image, hard_mask, _ = random_augment_triplet(image, hard_mask, None)
        return torch.from_numpy(image).unsqueeze(0).float(), torch.from_numpy(hard_mask).long()


class SoftMaskDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, augment: bool):
        self.frame = frame.reset_index(drop=True)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = np.asarray(Image.open(row["image_path"]).convert("L"), dtype=np.float32) / 255.0
        soft_target = np.load(row["soft_target_path"]).astype(np.float32)
        hard_mask = soft_target.argmax(axis=-1).astype(np.int64)
        soft_target = np.transpose(soft_target, (2, 0, 1))
        if self.augment:
            image, hard_mask, soft_target = random_augment_triplet(image, hard_mask, soft_target)
        return (
            torch.from_numpy(image).unsqueeze(0).float(),
            torch.from_numpy(soft_target).float(),
            torch.from_numpy(hard_mask).long(),
        )


def build_segmentation_splits(metadata: pd.DataFrame, config: DenseSegmentationConfig):
    train_frame, val_frame = train_test_split(
        metadata,
        test_size=1.0 - config.train_fraction,
        random_state=config.seed,
        stratify=metadata["dominant_class"],
    )

    if len(train_frame) > config.max_train_patches:
        keep_fraction = config.max_train_patches / len(train_frame)
        train_frame, _ = train_test_split(
            train_frame,
            train_size=keep_fraction,
            random_state=config.seed,
            stratify=train_frame["dominant_class"],
        )
    if len(val_frame) > config.max_val_patches:
        keep_fraction = config.max_val_patches / len(val_frame)
        val_frame, _ = train_test_split(
            val_frame,
            train_size=keep_fraction,
            random_state=config.seed,
            stratify=val_frame["dominant_class"],
        )

    train_frame = train_frame.copy().sort_values(["grid_row", "grid_col"])
    val_frame = val_frame.copy().sort_values(["grid_row", "grid_col"])
    train_frame["split"] = "train"
    val_frame["split"] = "val"
    return train_frame, val_frame


def build_hard_dataloaders(train_frame: pd.DataFrame, val_frame: pd.DataFrame, config: DenseSegmentationConfig):
    train_loader = DataLoader(
        HardMaskDataset(train_frame, augment=True),
        batch_size=config.unet_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        HardMaskDataset(val_frame, augment=False),
        batch_size=config.unet_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader


def build_soft_dataloaders(train_frame: pd.DataFrame, val_frame: pd.DataFrame, config: DenseSegmentationConfig):
    train_loader = DataLoader(
        SoftMaskDataset(train_frame, augment=True),
        batch_size=config.unet_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        SoftMaskDataset(val_frame, augment=False),
        batch_size=config.unet_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader


def train_hard_unet(
    train_loader: DataLoader,
    val_loader: DataLoader,
    spec: DenseSegmentationSpec,
    config: DenseSegmentationConfig,
    output_dir: Path,
) -> tuple[UNet, pd.DataFrame, dict[str, Any]]:
    device = resolve_device()
    model = UNet(in_channels=1, num_classes=len(spec.class_names), base_channels=config.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.unet_learning_rate, weight_decay=config.unet_weight_decay)
    class_counts = np.zeros(len(spec.class_names), dtype=np.float32)
    for _, masks in train_loader:
        class_counts += np.bincount(masks.reshape(-1).numpy(), minlength=len(spec.class_names)).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    class_weights_t = torch.as_tensor(class_weights, dtype=torch.float32, device=device)

    best_state = None
    best_miou = -1.0
    patience = 0
    history = []
    for epoch in range(1, config.unet_epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, masks, weight=class_weights_t) + dice_loss_hard(logits, masks, len(spec.class_names))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            train_count += images.size(0)

        metrics = evaluate_hard_unet(model, val_loader, len(spec.class_names), device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / max(train_count, 1),
                "val_pixel_accuracy": metrics["pixel_accuracy"],
                "val_mean_iou": metrics["mean_iou"],
            }
        )
        if metrics["mean_iou"] > best_miou:
            best_miou = metrics["mean_iou"]
            best_state = {
                "model": model.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
            }
            patience = 0
        else:
            patience += 1
            if patience >= config.unet_patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state["model"])
    torch.save(best_state, output_dir / "hard_unet_best_model.pt")
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "hard_unet_history.csv", index=False)
    return model, history_frame, {"best_epoch": int(best_state["epoch"]), "best_val_metrics": best_state["metrics"]}


def evaluate_hard_unet(model: UNet, loader: DataLoader, num_classes: int, device: torch.device) -> dict[str, Any]:
    model = model.to(device)
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for images, masks in loader:
            logits = model(images.to(device))
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            target = masks.numpy()
            confusion += confusion_matrix(target.reshape(-1), preds.reshape(-1), labels=np.arange(num_classes))
    return confusion_to_metrics(confusion)


def train_soft_unet(
    train_loader: DataLoader,
    val_loader: DataLoader,
    spec: DenseSegmentationSpec,
    config: DenseSegmentationConfig,
    output_dir: Path,
) -> tuple[UNet, pd.DataFrame, dict[str, Any]]:
    device = resolve_device()
    model = UNet(in_channels=1, num_classes=len(spec.class_names), base_channels=config.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.unet_learning_rate, weight_decay=config.unet_weight_decay)

    best_state = None
    best_miou = -1.0
    patience = 0
    history = []
    for epoch in range(1, config.unet_epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for images, soft_targets, _ in train_loader:
            images = images.to(device)
            soft_targets = soft_targets.to(device)
            logits = model(images)
            loss = soft_cross_entropy(logits, soft_targets) + dice_loss_soft(logits, soft_targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            train_count += images.size(0)

        metrics = evaluate_soft_unet(model, val_loader, len(spec.class_names), device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / max(train_count, 1),
                "val_pixel_accuracy": metrics["pixel_accuracy"],
                "val_mean_iou": metrics["mean_iou"],
                "val_soft_ce": metrics["soft_cross_entropy"],
            }
        )
        if metrics["mean_iou"] > best_miou:
            best_miou = metrics["mean_iou"]
            best_state = {
                "model": model.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
            }
            patience = 0
        else:
            patience += 1
            if patience >= config.unet_patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state["model"])
    torch.save(best_state, output_dir / "soft_unet_best_model.pt")
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "soft_unet_history.csv", index=False)
    return model, history_frame, {"best_epoch": int(best_state["epoch"]), "best_val_metrics": best_state["metrics"]}


def evaluate_soft_unet(model: UNet, loader: DataLoader, num_classes: int, device: torch.device) -> dict[str, Any]:
    model = model.to(device)
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    soft_ce_values = []
    with torch.no_grad():
        for images, soft_targets, hard_masks in loader:
            images = images.to(device)
            soft_targets = soft_targets.to(device)
            logits = model(images)
            soft_ce_values.append(float(soft_cross_entropy(logits, soft_targets).item()))
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            target = hard_masks.numpy()
            confusion += confusion_matrix(target.reshape(-1), preds.reshape(-1), labels=np.arange(num_classes))
    metrics = confusion_to_metrics(confusion)
    metrics["soft_cross_entropy"] = float(np.mean(soft_ce_values)) if soft_ce_values else 0.0
    return metrics


def evaluate_teacher_baseline(
    val_frame: pd.DataFrame,
    teacher_patch_table: pd.DataFrame,
    spec: DenseSegmentationSpec,
) -> dict[str, Any]:
    confusion = np.zeros((len(spec.class_names), len(spec.class_names)), dtype=np.int64)
    teacher_rows = int(teacher_patch_table["grid_row"].max()) + 1
    cols_per_teacher_row = int(teacher_patch_table["grid_col"].max()) + 1
    for _, row in val_frame.iterrows():
        y0 = int(row["y0"])
        x0 = int(row["x0"])
        teacher_row = min(y0 // spec.teacher_patch_size, teacher_rows - 1)
        teacher_col = min(x0 // spec.teacher_patch_size, cols_per_teacher_row - 1)
        teacher_id = teacher_row * cols_per_teacher_row + teacher_col
        teacher_label = int(teacher_patch_table.iloc[teacher_id]["teacher_label"])
        hard_mask = np.asarray(Image.open(row["hard_mask_path"]).convert("L"), dtype=np.uint8)
        pred = np.full_like(hard_mask, teacher_label)
        confusion += confusion_matrix(hard_mask.reshape(-1), pred.reshape(-1), labels=np.arange(len(spec.class_names)))
    return confusion_to_metrics(confusion)


def predict_full_map_dense(
    image: np.ndarray,
    valid_mask: np.ndarray,
    model: UNet,
    spec: DenseSegmentationSpec,
    config: DenseSegmentationConfig,
    output_dir: Path,
    prefix: str,
) -> dict[str, str]:
    device = resolve_device()
    model = model.to(device)
    model.eval()
    rows = image.shape[0] // spec.student_patch_size
    cols = image.shape[1] // spec.student_patch_size
    label_map = np.zeros((rows * spec.student_patch_size, cols * spec.student_patch_size), dtype=np.uint8)
    with torch.no_grad():
        for row in range(rows):
            batch_patches = []
            positions = []
            for col in range(cols):
                y0 = row * spec.student_patch_size
                x0 = col * spec.student_patch_size
                patch = image[y0 : y0 + spec.student_patch_size, x0 : x0 + spec.student_patch_size]
                batch_patches.append(patch)
                positions.append((y0, x0))
            batch = torch.from_numpy(np.stack(batch_patches)).unsqueeze(1).float().to(device)
            logits = model(batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
            for pred, (y0, x0) in zip(preds, positions):
                label_map[y0 : y0 + spec.student_patch_size, x0 : x0 + spec.student_patch_size] = pred
    label_map[~valid_mask[: label_map.shape[0], : label_map.shape[1]]] = 0
    image_crop = image[: label_map.shape[0], : label_map.shape[1]]
    return save_map_outputs(image_crop, label_map, output_dir, prefix, spec.class_names)


def predict_teacher_baseline_map(
    image: np.ndarray,
    valid_mask: np.ndarray,
    teacher_patch_table: pd.DataFrame,
    spec: DenseSegmentationSpec,
    output_dir: Path,
) -> dict[str, str]:
    rows = image.shape[0] // spec.teacher_patch_size
    cols = image.shape[1] // spec.teacher_patch_size
    patch_grid = teacher_patch_table["teacher_label"].to_numpy().reshape(rows, cols)
    label_map = np.repeat(np.repeat(patch_grid, spec.teacher_patch_size, axis=0), spec.teacher_patch_size, axis=1)
    label_map = label_map[: rows * spec.teacher_patch_size, : cols * spec.teacher_patch_size].astype(np.uint8)
    label_map[~valid_mask[: label_map.shape[0], : label_map.shape[1]]] = 0
    image_crop = image[: label_map.shape[0], : label_map.shape[1]]
    return save_map_outputs(image_crop, label_map, output_dir, "teacher_baseline_v256", spec.class_names)


def create_comparison_figure(
    spec: DenseSegmentationSpec,
    output_dir: Path,
    teacher_metrics: dict[str, Any],
    hard_metrics: dict[str, Any],
    soft_metrics: dict[str, Any],
) -> str:
    width = 1500
    height = 820
    canvas = Image.new("RGB", (width, height), (247, 244, 237))
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default()
    body_font = ImageFont.load_default()

    draw.text((40, 28), f"{spec.name} - 128x128 Dense Label Comparison", fill=(26, 30, 33), font=title_font)
    draw.text((40, 70), "Baseline is the current production patch-classifier map expanded to pixel space.", fill=(90, 95, 100), font=body_font)

    columns = [
        ("Current Baseline", teacher_metrics),
        ("Hard Per-Pixel U-Net", hard_metrics),
        ("Soft Label U-Net", soft_metrics),
    ]
    x_positions = [40, 520, 1000]
    for (label, metrics), x0 in zip(columns, x_positions):
        draw.rounded_rectangle((x0, 120, x0 + 420, 470), radius=26, fill=(255, 255, 255), outline=(219, 212, 199))
        draw.text((x0 + 24, 150), label, fill=(26, 30, 33), font=title_font)
        draw.text((x0 + 24, 200), f"Pixel Accuracy: {metrics['pixel_accuracy'] * 100:.2f}%", fill=(40, 48, 52), font=body_font)
        draw.text((x0 + 24, 235), f"Mean IoU: {metrics['mean_iou'] * 100:.2f}%", fill=(40, 48, 52), font=body_font)
        if "soft_cross_entropy" in metrics:
            draw.text((x0 + 24, 270), f"Soft CE: {metrics['soft_cross_entropy']:.4f}", fill=(40, 48, 52), font=body_font)
        y = 320
        draw.text((x0 + 24, y), "Class IoU:", fill=(60, 68, 72), font=body_font)
        y += 26
        for class_id, class_name in spec.class_names.items():
            iou = metrics["class_iou"][class_id] if class_id < len(metrics["class_iou"]) else 0.0
            draw.text((x0 + 24, y), f"- {class_name}: {iou * 100:.2f}%", fill=(80, 86, 92), font=body_font)
            y += 24

    file_path = output_dir / "comparison_summary_board.png"
    canvas.save(file_path)
    return str(file_path)


def save_preview_panel(
    output_dir: Path,
    spec: DenseSegmentationSpec,
    train_frame: pd.DataFrame,
) -> str:
    subset = train_frame.head(6)
    patch_size = spec.student_patch_size
    canvas = Image.new("RGB", (patch_size * 3, patch_size * max(len(subset), 1)), (255, 255, 255))
    for idx, (_, row) in enumerate(subset.iterrows()):
        image = Image.open(row["image_path"]).convert("L").convert("RGB")
        hard = colorize_mask(np.asarray(Image.open(row["hard_mask_path"]).convert("L"), dtype=np.uint8), spec.class_names)
        soft = np.load(row["soft_target_path"]).astype(np.float32)
        soft_argmax = colorize_mask(soft.argmax(axis=-1).astype(np.uint8), spec.class_names)
        for col, tile in enumerate([np.asarray(image), hard, soft_argmax]):
            canvas.paste(Image.fromarray(tile), (col * patch_size, idx * patch_size))
    file_path = output_dir / "dense_label_preview_panel.png"
    canvas.save(file_path)
    return str(file_path)


def run_dense_segmentation_pipeline(config: DenseSegmentationConfig) -> dict[str, Any]:
    set_seed(config.seed)
    spec = SPECS[config.dataset_key]
    output_dir = ensure_dir(config.output_path())
    image, valid_mask, raster_metadata = load_raster(spec, config.robust_percentiles)

    teacher_csv_path = output_dir / "teacher_patch_predictions.csv"
    seed_csv_path = output_dir / "teacher_seed_patches.csv"
    metadata_path = output_dir / "dense_dataset" / "metadata.csv"
    split_path = output_dir / "dense_dataset" / "metadata_with_split.csv"

    if teacher_csv_path.exists():
        teacher_patch_table = pd.read_csv(teacher_csv_path)
    else:
        teacher_patch_table, _ = predict_teacher_patches(spec, config, image, valid_mask)
        teacher_patch_table.to_csv(teacher_csv_path, index=False)

    if seed_csv_path.exists():
        seed_patches = pd.read_csv(seed_csv_path)
    else:
        seed_patches = select_seed_patches(teacher_patch_table, spec)
        seed_patches.to_csv(seed_csv_path, index=False)

    pixel_metrics: dict[str, Any]
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        pixel_metrics = {"reused_from_existing_dense_dataset": True}
    else:
        feature_extractor = PixelFeatureExtractor(spec.dense_context_pad)
        x, y, w = collect_pixel_training_data(image, valid_mask, seed_patches, spec, feature_extractor)
        pixel_model, pixel_metrics, feature_mean, feature_std = train_pixel_mlp(x, y, w, spec, config, output_dir)

        metadata = build_dense_pseudo_dataset(
            image=image,
            valid_mask=valid_mask,
            teacher_patch_table=teacher_patch_table,
            pixel_model=pixel_model,
            feature_mean=feature_mean,
            feature_std=feature_std,
            spec=spec,
            config=config,
            output_dir=output_dir,
        )

    if split_path.exists():
        split_frame = pd.read_csv(split_path)
        train_frame = split_frame[split_frame["split"] == "train"].copy()
        val_frame = split_frame[split_frame["split"] == "val"].copy()
        if len(train_frame) > config.max_train_patches or len(val_frame) > config.max_val_patches:
            train_frame, val_frame = build_segmentation_splits(metadata, config)
            split_frame = pd.concat([train_frame, val_frame], ignore_index=True)
            split_frame.to_csv(split_path, index=False)
    else:
        train_frame, val_frame = build_segmentation_splits(metadata, config)
        split_frame = pd.concat([train_frame, val_frame], ignore_index=True)
        split_frame.to_csv(split_path, index=False)

    hard_train_loader, hard_val_loader = build_hard_dataloaders(train_frame, val_frame, config)
    soft_train_loader, soft_val_loader = build_soft_dataloaders(train_frame, val_frame, config)

    hard_model_path = output_dir / "hard_unet_best_model.pt"
    if hard_model_path.exists():
        hard_model, hard_checkpoint = load_saved_unet(hard_model_path, len(spec.class_names))
        hard_summary = {
            "best_epoch": int(hard_checkpoint.get("epoch", 0)),
            "best_val_metrics": hard_checkpoint.get("metrics", {}),
        }
        hard_history = pd.read_csv(output_dir / "hard_unet_history.csv") if (output_dir / "hard_unet_history.csv").exists() else pd.DataFrame()
    else:
        hard_model, hard_history, hard_summary = train_hard_unet(hard_train_loader, hard_val_loader, spec, config, output_dir)

    soft_model_path = output_dir / "soft_unet_best_model.pt"
    if soft_model_path.exists():
        soft_model, soft_checkpoint = load_saved_unet(soft_model_path, len(spec.class_names))
        soft_summary = {
            "best_epoch": int(soft_checkpoint.get("epoch", 0)),
            "best_val_metrics": soft_checkpoint.get("metrics", {}),
        }
        soft_history = pd.read_csv(output_dir / "soft_unet_history.csv") if (output_dir / "soft_unet_history.csv").exists() else pd.DataFrame()
    else:
        soft_model, soft_history, soft_summary = train_soft_unet(soft_train_loader, soft_val_loader, spec, config, output_dir)

    device = resolve_device()
    hard_final_metrics = evaluate_hard_unet(hard_model, hard_val_loader, len(spec.class_names), device)
    soft_final_metrics = evaluate_soft_unet(soft_model, soft_val_loader, len(spec.class_names), device)
    teacher_baseline_metrics = evaluate_teacher_baseline(val_frame, teacher_patch_table, spec)

    teacher_map = predict_teacher_baseline_map(image, valid_mask, teacher_patch_table, spec, output_dir)
    hard_map = predict_full_map_dense(image, valid_mask, hard_model, spec, config, output_dir, "hard_unet_v128")
    soft_map = predict_full_map_dense(image, valid_mask, soft_model, spec, config, output_dir, "soft_unet_v128")

    best_variant = "hard_unet_v128"
    best_metrics = hard_final_metrics
    best_map = hard_map
    if soft_final_metrics["mean_iou"] > hard_final_metrics["mean_iou"]:
        best_variant = "soft_unet_v128"
        best_metrics = soft_final_metrics
        best_map = soft_map

    improvement = best_metrics["mean_iou"] - teacher_baseline_metrics["mean_iou"]
    should_promote = improvement >= config.compare_margin_miou

    preview_panel = save_preview_panel(output_dir, spec, train_frame)
    comparison_board = create_comparison_figure(spec, output_dir, teacher_baseline_metrics, hard_final_metrics, soft_final_metrics)

    result_summary = {
        "config": asdict(config),
        "dataset": {
            "name": spec.name,
            "image_path": spec.image_path,
            "class_names": spec.class_names,
            "raster_metadata": raster_metadata,
            "teacher_patch_size": spec.teacher_patch_size,
            "student_patch_size": spec.student_patch_size,
            "teacher_seed_patch_count": int(len(seed_patches)),
            "dense_dataset_patch_count": int(len(metadata)),
            "train_patch_count": int(len(train_frame)),
            "val_patch_count": int(len(val_frame)),
        },
        "pixel_teacher_metrics": pixel_metrics,
        "teacher_baseline_metrics": teacher_baseline_metrics,
        "hard_unet": {
            "training": hard_summary,
            "final_val_metrics": hard_final_metrics,
            "history_path": str(output_dir / "hard_unet_history.csv"),
            "model_path": str(output_dir / "hard_unet_best_model.pt"),
            "prediction_maps": hard_map,
        },
        "soft_unet": {
            "training": soft_summary,
            "final_val_metrics": soft_final_metrics,
            "history_path": str(output_dir / "soft_unet_history.csv"),
            "model_path": str(output_dir / "soft_unet_best_model.pt"),
            "prediction_maps": soft_map,
        },
        "best_variant": best_variant,
        "best_variant_metrics": best_metrics,
        "teacher_baseline_maps": teacher_map,
        "comparison": {
            "baseline_mean_iou": teacher_baseline_metrics["mean_iou"],
            "best_mean_iou": best_metrics["mean_iou"],
            "mean_iou_improvement": improvement,
            "should_promote_to_platform": should_promote,
            "promotion_rule": f"Promote only if mean IoU improves by at least {config.compare_margin_miou:.2f}.",
        },
        "artifacts": {
            "dense_preview_panel": preview_panel,
            "comparison_board": comparison_board,
            "dense_metadata_path": str(output_dir / "dense_dataset" / "metadata.csv"),
        },
    }
    summary_path = output_dir / "run_summary_dense_v128.json"
    summary_path.write_text(json.dumps(json_ready(result_summary), indent=2), encoding="utf-8")

    lines = [
        f"{spec.name} Dense Segmentation V128",
        "",
        f"Teacher baseline pixel accuracy: {teacher_baseline_metrics['pixel_accuracy'] * 100:.2f}%",
        f"Teacher baseline mean IoU: {teacher_baseline_metrics['mean_iou'] * 100:.2f}%",
        "",
        f"Hard U-Net pixel accuracy: {hard_final_metrics['pixel_accuracy'] * 100:.2f}%",
        f"Hard U-Net mean IoU: {hard_final_metrics['mean_iou'] * 100:.2f}%",
        "",
        f"Soft U-Net pixel accuracy: {soft_final_metrics['pixel_accuracy'] * 100:.2f}%",
        f"Soft U-Net mean IoU: {soft_final_metrics['mean_iou'] * 100:.2f}%",
    ]
    if "soft_cross_entropy" in soft_final_metrics:
        lines.append(f"Soft U-Net soft CE: {soft_final_metrics['soft_cross_entropy']:.4f}")
    lines.extend(
        [
            "",
            f"Best variant: {best_variant}",
            f"Best mean IoU improvement over current baseline: {improvement * 100:.2f} percentage points",
            f"Promote to platform: {'YES' if should_promote else 'NO'}",
        ]
    )
    (output_dir / "RESULTS_SUMMARY_DENSE_V128.txt").write_text("\n".join(lines), encoding="utf-8")
    return result_summary


def parse_args() -> DenseSegmentationConfig:
    parser = argparse.ArgumentParser(description="Train 128x128 dense hard-mask and soft-label segmentation models.")
    parser.add_argument("--dataset", choices=["jnpa_dense_v128", "cartosat_dense_v128"], required=True)
    parser.add_argument("--output-dir", type=str, required=False, help="Output directory override.")
    args = parser.parse_args()

    default_output = {
        "jnpa_dense_v128": "outputs/jnpa_dense_segmentation_v128",
        "cartosat_dense_v128": "outputs/cartosat_dense_segmentation_v128",
    }[args.dataset]
    return DenseSegmentationConfig(dataset_key=args.dataset, output_dir=args.output_dir or default_output)


if __name__ == "__main__":
    cfg = parse_args()
    result = run_dense_segmentation_pipeline(cfg)
    print(json.dumps(json_ready(result["comparison"]), indent=2))
