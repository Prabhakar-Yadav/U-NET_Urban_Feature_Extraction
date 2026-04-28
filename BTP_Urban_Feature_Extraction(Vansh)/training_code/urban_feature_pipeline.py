from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


CLASS_COLORS = {
    0: (31, 119, 180),
    1: (214, 39, 40),
    2: (140, 86, 75),
    3: (44, 160, 44),
    4: (255, 215, 0),
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    image_path: str
    patch_size: int
    std_threshold: float
    valid_fraction: float
    preview_clusters: int
    cluster_to_class: Dict[int, int]
    class_names: Dict[int, str]
    cluster_descriptions: Dict[int, str]
    notes: str


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "jnpa": DatasetSpec(
        name="JNPA 2.5m PAN",
        image_path="JNPA/JNPA_2_5.tif",
        patch_size=256,
        std_threshold=0.045,
        valid_fraction=0.85,
        preview_clusters=7,
        cluster_to_class={
            0: 1,
            1: 2,
            2: 3,
            3: 1,
            4: 0,
            5: 4,
            6: 4,
        },
        class_names={
            0: "Water Bodies",
            1: "Industrial / Port Infrastructure",
            2: "Bare Land / Soil",
            3: "Vegetation / Mangroves",
            4: "Urban Built-up",
        },
        cluster_descriptions={
            0: "Small bright outlier group with tanks, paved yards, and port-side structures.",
            1: "Dry terrain, exposed soil, quarries, sparse roads, and mixed barren ground.",
            2: "Dark textured vegetation, mangroves, wetlands, and vegetated coastal edges.",
            3: "Dense industrial estates, tank farms, container yards, and port infrastructure.",
            4: "Open water, harbor water, rivers, and large homogeneous dark water bodies.",
            5: "Mixed built-up urban fabric with roads, settlements, and developed land near water.",
            6: "Transport corridors and medium-density built-up urban areas.",
        },
        notes=(
            "The JNPA fine-cluster mapping was assigned manually after inspecting cluster "
            "contact sheets generated from the raw panchromatic patches."
        ),
    ),
    "cartosat": DatasetSpec(
        name="CARTOSAT 1m PAN",
        image_path="Monocromatic/CARTOSAT_1M_PAN.tif",
        patch_size=256,
        std_threshold=0.055,
        valid_fraction=0.95,
        preview_clusters=5,
        cluster_to_class={
            0: 1,
            1: 2,
            2: 0,
            3: 3,
            4: 1,
        },
        class_names={
            0: "Water",
            1: "Urban",
            2: "Terrain",
            3: "Vegetation",
        },
        cluster_descriptions={
            0: "Dense man-made areas with roads and built-up structure.",
            1: "Mixed exposed terrain and less-developed land.",
            2: "Water and very dark homogeneous areas.",
            3: "Vegetated textured areas.",
            4: "Bright urban and transport corridors.",
        },
        notes=(
            "The CARTOSAT mapping is provided as a ready baseline and should be refined "
            "after reviewing the notebook cluster previews."
        ),
    ),
}


@dataclass
class PipelineConfig:
    dataset_key: str = "jnpa"
    output_dir: str = "outputs/jnpa_run"
    seed: int = 42
    robust_percentiles: Tuple[float, float] = (1.0, 99.0)
    resize_for_features: int = 32
    pca_components: int = 25
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 6
    train_fraction: float = 0.8
    num_workers: int = 0
    canny_low: int = 60
    canny_high: int = 160
    val_preview_count: int = 6
    contact_sheet_limit: int = 16

    def output_path(self) -> Path:
        return Path(self.output_dir)


@dataclass
class RasterBundle:
    image: np.ndarray
    valid_mask: np.ndarray
    metadata: Dict[str, object]


@dataclass
class PatchTable:
    patches: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    valid_fraction: np.ndarray
    std: np.ndarray
    keep_mask: np.ndarray

    def filtered(self) -> "PatchTable":
        return PatchTable(
            patches=self.patches[self.keep_mask],
            rows=self.rows[self.keep_mask],
            cols=self.cols[self.keep_mask],
            valid_fraction=self.valid_fraction[self.keep_mask],
            std=self.std[self.keep_mask],
            keep_mask=np.ones(int(self.keep_mask.sum()), dtype=bool),
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def load_raster(spec: DatasetSpec, robust_percentiles: Tuple[float, float]) -> RasterBundle:
    image_path = Path(spec.image_path)
    with rasterio.open(image_path) as src:
        image = src.read(1).astype(np.float32)
        nodata = src.nodata
        valid_mask = np.ones_like(image, dtype=bool)
        if nodata is not None:
            valid_mask &= np.not_equal(image, nodata)
        valid_pixels = image[valid_mask]
        lo, hi = np.percentile(valid_pixels, robust_percentiles)
        image = np.clip(image, lo, hi)
        image = (image - lo) / max(hi - lo, 1e-6)
        image[~valid_mask] = 0.0
        metadata = {
            "path": str(image_path),
            "width": src.width,
            "height": src.height,
            "dtype": str(src.dtypes[0]),
            "crs": str(src.crs),
            "transform": list(src.transform)[:6],
            "nodata": nodata,
            "normalization_low": float(lo),
            "normalization_high": float(hi),
        }
    return RasterBundle(image=image, valid_mask=valid_mask, metadata=metadata)


def extract_patch_table(
    image: np.ndarray,
    valid_mask: np.ndarray,
    patch_size: int,
    std_threshold: float,
    valid_fraction_threshold: float,
) -> PatchTable:
    rows = image.shape[0] // patch_size
    cols = image.shape[1] // patch_size
    patches: List[np.ndarray] = []
    patch_rows: List[int] = []
    patch_cols: List[int] = []
    patch_valid_fraction: List[float] = []
    patch_std: List[float] = []
    keep_mask: List[bool] = []

    for row in range(rows):
        for col in range(cols):
            y0 = row * patch_size
            x0 = col * patch_size
            patch = image[y0 : y0 + patch_size, x0 : x0 + patch_size]
            patch_valid = valid_mask[y0 : y0 + patch_size, x0 : x0 + patch_size]
            valid_fraction = float(patch_valid.mean())
            std = float(patch.std())
            keep = valid_fraction >= valid_fraction_threshold and std >= std_threshold
            patches.append(patch)
            patch_rows.append(row)
            patch_cols.append(col)
            patch_valid_fraction.append(valid_fraction)
            patch_std.append(std)
            keep_mask.append(keep)

    return PatchTable(
        patches=np.stack(patches),
        rows=np.asarray(patch_rows, dtype=np.int32),
        cols=np.asarray(patch_cols, dtype=np.int32),
        valid_fraction=np.asarray(patch_valid_fraction, dtype=np.float32),
        std=np.asarray(patch_std, dtype=np.float32),
        keep_mask=np.asarray(keep_mask, dtype=bool),
    )


def compute_patch_features(
    patches: np.ndarray,
    resize_for_features: int,
    canny_low: int,
    canny_high: int,
) -> Tuple[np.ndarray, np.ndarray]:
    features: List[np.ndarray] = []
    handcrafted: List[np.ndarray] = []

    for patch in patches:
        small = cv2.resize(
            patch,
            (resize_for_features, resize_for_features),
            interpolation=cv2.INTER_AREA,
        )
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
        handcrafted.append(stats)

    return np.stack(features).astype(np.float32), np.stack(handcrafted).astype(np.float32)


def cluster_patch_features(
    feature_matrix: np.ndarray,
    num_clusters: int,
    pca_components: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, PCA, MiniBatchKMeans]:
    n_components = min(pca_components, feature_matrix.shape[0], feature_matrix.shape[1])
    pca = PCA(n_components=n_components, random_state=seed, svd_solver="randomized")
    reduced = pca.fit_transform(feature_matrix)
    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters,
        random_state=seed,
        batch_size=min(256, feature_matrix.shape[0]),
        n_init=20,
    )
    clusters = kmeans.fit_predict(reduced)
    return clusters, reduced, pca, kmeans


def write_grayscale(path: Path, image: np.ndarray) -> None:
    image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(image_u8, mode="L").save(path)


def write_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(mask.astype(np.uint8), mode="L").save(path)


def make_contact_sheet(patches: np.ndarray, patch_indices: Iterable[int], patch_size: int) -> np.ndarray:
    indices = list(patch_indices)
    if not indices:
        return np.zeros((patch_size, patch_size), dtype=np.uint8)
    side = math.ceil(math.sqrt(len(indices)))
    sheet = np.full((side * patch_size, side * patch_size), 255, dtype=np.uint8)
    for n, patch_idx in enumerate(indices):
        row, col = divmod(n, side)
        tile = np.clip(patches[patch_idx] * 255.0, 0, 255).astype(np.uint8)
        y0 = row * patch_size
        x0 = col * patch_size
        sheet[y0 : y0 + patch_size, x0 : x0 + patch_size] = tile
    return sheet


def save_cluster_previews(
    output_dir: Path,
    patches: np.ndarray,
    cluster_ids: np.ndarray,
    spec: DatasetSpec,
    contact_sheet_limit: int,
) -> pd.DataFrame:
    preview_dir = ensure_dir(output_dir / "cluster_previews")
    records = []

    for cluster_id in sorted(np.unique(cluster_ids)):
        indices = np.where(cluster_ids == cluster_id)[0]
        sample = indices[: min(len(indices), contact_sheet_limit)]
        sheet = make_contact_sheet(patches, sample, spec.patch_size)
        preview_path = preview_dir / f"cluster_{cluster_id}.png"
        Image.fromarray(sheet, mode="L").save(preview_path)
        records.append(
            {
                "cluster_id": int(cluster_id),
                "count": int(len(indices)),
                "class_id": int(spec.cluster_to_class[int(cluster_id)]),
                "class_name": spec.class_names[int(spec.cluster_to_class[int(cluster_id)])],
                "description": spec.cluster_descriptions.get(int(cluster_id), ""),
                "preview_path": str(preview_path),
            }
        )

    cluster_table = pd.DataFrame.from_records(records).sort_values("cluster_id")
    cluster_table.to_csv(output_dir / "cluster_summary.csv", index=False)
    return cluster_table


def save_dataset(
    output_dir: Path,
    patch_table: PatchTable,
    clusters: np.ndarray,
    class_ids: np.ndarray,
    spec: DatasetSpec,
) -> pd.DataFrame:
    dataset_dir = ensure_dir(output_dir / "dataset")
    image_dir = ensure_dir(dataset_dir / "images")
    mask_dir = ensure_dir(dataset_dir / "masks")
    filtered = patch_table.filtered()

    records = []
    for idx, patch in enumerate(filtered.patches):
        image_path = image_dir / f"patch_{idx:04d}.png"
        mask_path = mask_dir / f"mask_{idx:04d}.png"
        write_grayscale(image_path, patch)
        write_mask(mask_path, np.full((spec.patch_size, spec.patch_size), class_ids[idx], dtype=np.uint8))
        records.append(
            {
                "patch_id": idx,
                "grid_row": int(filtered.rows[idx]),
                "grid_col": int(filtered.cols[idx]),
                "cluster_id": int(clusters[idx]),
                "class_id": int(class_ids[idx]),
                "class_name": spec.class_names[int(class_ids[idx])],
                "std": float(filtered.std[idx]),
                "valid_fraction": float(filtered.valid_fraction[idx]),
                "image_path": str(image_path),
                "mask_path": str(mask_path),
            }
        )

    metadata = pd.DataFrame.from_records(records).sort_values(["grid_row", "grid_col"])
    metadata.to_csv(dataset_dir / "metadata.csv", index=False)
    metadata["class_name"].value_counts().rename_axis("class_name").reset_index(name="count").to_csv(
        dataset_dir / "class_distribution.csv", index=False
    )
    return metadata


class PatchDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, augment: bool = False):
        self.frame = frame.reset_index(drop=True)
        self.augment = augment
        self.transform = (
            A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.RandomBrightnessContrast(p=0.3),
                    A.GaussNoise(p=0.2),
                ]
            )
            if augment
            else None
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = np.asarray(Image.open(row["image_path"]).convert("L"), dtype=np.float32) / 255.0
        mask = np.asarray(Image.open(row["mask_path"]).convert("L"), dtype=np.int64)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = torch.from_numpy(image).unsqueeze(0).float()
        mask = torch.from_numpy(mask).long()
        return image, mask


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
    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 32):
        super().__init__()
        self.down1 = DoubleConv(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.bridge = DoubleConv(base_channels * 4, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
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


def dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    target_1h = torch.nn.functional.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = torch.sum(probs * target_1h, dim=dims)
    cardinality = torch.sum(probs + target_1h, dim=dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        class_weights: torch.Tensor | None = None,
        patch_loss_weight: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.patch_loss_weight = patch_loss_weight
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.patch_ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        patch_logits = logits.mean(dim=(2, 3))
        patch_target = target[:, 0, 0]
        return (
            self.ce(logits, target)
            + dice_loss(logits, target, self.num_classes)
            + self.patch_loss_weight * self.patch_ce(patch_logits, patch_target)
        )


def compute_class_weights(class_ids: Iterable[int], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(np.asarray(list(class_ids), dtype=np.int64), minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def build_dataloaders(
    metadata: pd.DataFrame,
    num_classes: int,
    config: PipelineConfig,
) -> Tuple[DataLoader, DataLoader, pd.DataFrame, pd.DataFrame]:
    train_frame, val_frame = train_test_split(
        metadata,
        test_size=1.0 - config.train_fraction,
        random_state=config.seed,
        stratify=metadata["class_id"],
    )
    train_frame = train_frame.copy().sort_values(["grid_row", "grid_col"])
    val_frame = val_frame.copy().sort_values(["grid_row", "grid_col"])
    train_frame["split"] = "train"
    val_frame["split"] = "val"

    train_loader = DataLoader(
        PatchDataset(train_frame, augment=True),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        PatchDataset(val_frame, augment=False),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader, train_frame, val_frame


def batch_confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> np.ndarray:
    pred = pred.reshape(-1).detach().cpu().numpy()
    target = target.reshape(-1).detach().cpu().numpy()
    mask = (target >= 0) & (target < num_classes)
    hist = np.bincount(
        num_classes * target[mask].astype(np.int64) + pred[mask].astype(np.int64),
        minlength=num_classes ** 2,
    )
    return hist.reshape(num_classes, num_classes)


def confusion_to_metrics(confusion: np.ndarray) -> Dict[str, object]:
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


def logits_to_patch_predictions(logits: torch.Tensor) -> torch.Tensor:
    patch_pred = torch.argmax(logits.mean(dim=(2, 3)), dim=1)
    height = logits.shape[2]
    width = logits.shape[3]
    return patch_pred[:, None, None].expand(-1, height, width)


def heuristic_patch_override(patch: np.ndarray, patch_std: float, std_threshold: float) -> int | None:
    if patch_std < std_threshold and float(patch.mean()) < 0.32:
        return 0
    return None


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    loss_fn: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Dict[str, object]:
    training = optimizer is not None
    model.train(training)
    epoch_loss = 0.0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = loss_fn(logits, masks)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item() * images.size(0)
        preds = logits_to_patch_predictions(logits)
        confusion += batch_confusion_matrix(preds, masks, num_classes)

    metrics = confusion_to_metrics(confusion)
    metrics["loss"] = epoch_loss / max(len(loader.dataset), 1)
    return metrics


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    class_weights: torch.Tensor,
    config: PipelineConfig,
    output_dir: Path,
) -> Tuple[nn.Module, pd.DataFrame, Dict[str, object]]:
    device = resolve_device()
    model = UNet(in_channels=1, num_classes=num_classes).to(device)
    loss_fn = CombinedLoss(num_classes=num_classes, class_weights=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    history: List[Dict[str, object]] = []
    best_state = None
    best_metric = -float("inf")
    patience_counter = 0

    for epoch in tqdm(range(1, config.epochs + 1), desc="Training"):
        train_metrics = run_epoch(model, train_loader, optimizer, loss_fn, device, num_classes)
        val_metrics = run_epoch(model, val_loader, None, loss_fn, device, num_classes)
        scheduler.step(val_metrics["mean_iou"])

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_pixel_accuracy": train_metrics["pixel_accuracy"],
            "train_mean_iou": train_metrics["mean_iou"],
            "val_loss": val_metrics["loss"],
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
            "val_mean_iou": val_metrics["mean_iou"],
        }
        history.append(row)

        if val_metrics["mean_iou"] > best_metric:
            best_metric = val_metrics["mean_iou"]
            best_state = {
                "model": model.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("Training finished without producing a valid checkpoint.")

    torch.save(best_state, output_dir / "best_model.pt")
    model.load_state_dict(best_state["model"])
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "training_history.csv", index=False)

    summary = {
        "device": str(device),
        "best_epoch": int(best_state["epoch"]),
        "best_val_metrics": best_state["val_metrics"],
    }
    return model, history_frame, summary


def collect_metrics(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> Dict[str, object]:
    loss_fn = CombinedLoss(num_classes=num_classes)
    metrics = run_epoch(model, loader, None, loss_fn, device, num_classes)
    return metrics


def colorize_mask(mask: np.ndarray, class_colors: Dict[int, Tuple[int, int, int]]) -> np.ndarray:
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, rgb in class_colors.items():
        color[mask == class_id] = rgb
    return color


def save_label_map_outputs(
    label_grid: np.ndarray,
    base_image: np.ndarray,
    output_dir: Path,
    prefix: str,
) -> Dict[str, str]:
    color_map = colorize_mask(label_grid, CLASS_COLORS)
    base = np.dstack([base_image * 255.0] * 3).astype(np.uint8)
    overlay = cv2.addWeighted(base, 0.45, color_map, 0.55, 0)

    pred_dir = ensure_dir(output_dir / "predictions")
    label_path = pred_dir / f"{prefix}_labels.npy"
    color_path = pred_dir / f"{prefix}_color.png"
    overlay_path = pred_dir / f"{prefix}_overlay.png"
    np.save(label_path, label_grid)
    Image.fromarray(color_map).save(color_path)
    Image.fromarray(overlay).save(overlay_path)
    return {
        "label_path": str(label_path),
        "color_path": str(color_path),
        "overlay_path": str(overlay_path),
    }


def save_training_preview(
    model: nn.Module,
    val_frame: pd.DataFrame,
    spec: DatasetSpec,
    device: torch.device,
    output_dir: Path,
    limit: int,
) -> None:
    preview_dir = ensure_dir(output_dir / "validation_previews")
    subset = val_frame.head(limit)
    model.eval()
    with torch.no_grad():
        for _, row in subset.iterrows():
            image = np.asarray(Image.open(row["image_path"]).convert("L"), dtype=np.float32) / 255.0
            tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
            logits = model(tensor)
            pred = logits_to_patch_predictions(logits).squeeze(0).cpu().numpy().astype(np.uint8)
            color = colorize_mask(pred, CLASS_COLORS)
            overlay = np.dstack([image * 255.0] * 3).astype(np.uint8)
            overlay = cv2.addWeighted(overlay, 0.5, color, 0.5, 0)
            Image.fromarray(overlay).save(preview_dir / f"patch_{int(row['patch_id']):04d}_overlay.png")
            Image.fromarray(color).save(preview_dir / f"patch_{int(row['patch_id']):04d}_color.png")


def predict_full_map(
    model: nn.Module,
    patch_table: PatchTable,
    spec: DatasetSpec,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, str]:
    model.eval()
    rows = int(patch_table.rows.max()) + 1
    cols = int(patch_table.cols.max()) + 1
    label_grid = np.zeros((rows * spec.patch_size, cols * spec.patch_size), dtype=np.uint8)

    with torch.no_grad():
        for patch, row, col, patch_std in tqdm(
            zip(patch_table.patches, patch_table.rows, patch_table.cols, patch_table.std),
            total=len(patch_table.patches),
            desc="Reconstructing",
        ):
            override = heuristic_patch_override(patch, float(patch_std), spec.std_threshold)
            if override is not None:
                pred = np.full((spec.patch_size, spec.patch_size), override, dtype=np.uint8)
            else:
                tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
                logits = model(tensor)
                pred = logits_to_patch_predictions(logits).squeeze(0).cpu().numpy().astype(np.uint8)
            y0 = int(row) * spec.patch_size
            x0 = int(col) * spec.patch_size
            label_grid[y0 : y0 + spec.patch_size, x0 : x0 + spec.patch_size] = pred

    raw_image = np.clip(patch_table.patches.reshape(rows, cols, spec.patch_size, spec.patch_size), 0, 1)
    raw_image = raw_image.transpose(0, 2, 1, 3).reshape(rows * spec.patch_size, cols * spec.patch_size)
    return save_label_map_outputs(label_grid, raw_image, output_dir, prefix="full_prediction")


def predict_clustering_baseline(
    patch_table: PatchTable,
    spec: DatasetSpec,
    config: PipelineConfig,
    pca: PCA,
    kmeans: MiniBatchKMeans,
    output_dir: Path,
) -> Dict[str, str]:
    rows = int(patch_table.rows.max()) + 1
    cols = int(patch_table.cols.max()) + 1
    feature_matrix, _ = compute_patch_features(
        patch_table.patches,
        resize_for_features=config.resize_for_features,
        canny_low=config.canny_low,
        canny_high=config.canny_high,
    )
    reduced = pca.transform(feature_matrix)
    clusters = kmeans.predict(reduced)
    class_ids = np.asarray([spec.cluster_to_class[int(cluster_id)] for cluster_id in clusters], dtype=np.uint8)
    label_grid = np.zeros((rows * spec.patch_size, cols * spec.patch_size), dtype=np.uint8)
    for patch, row, col, patch_std, class_id in zip(
        patch_table.patches,
        patch_table.rows,
        patch_table.cols,
        patch_table.std,
        class_ids,
    ):
        override = heuristic_patch_override(patch, float(patch_std), spec.std_threshold)
        final_class = override if override is not None else int(class_id)
        y0 = int(row) * spec.patch_size
        x0 = int(col) * spec.patch_size
        label_grid[y0 : y0 + spec.patch_size, x0 : x0 + spec.patch_size] = final_class
    raw_image = np.clip(patch_table.patches.reshape(rows, cols, spec.patch_size, spec.patch_size), 0, 1)
    raw_image = raw_image.transpose(0, 2, 1, 3).reshape(rows * spec.patch_size, cols * spec.patch_size)
    return save_label_map_outputs(label_grid, raw_image, output_dir, prefix="clustering_baseline")


def save_preview_images(bundle: RasterBundle, output_dir: Path) -> Dict[str, str]:
    preview_dir = ensure_dir(output_dir / "previews")
    image = bundle.image
    preview_width = 1400
    preview_height = int(image.shape[0] * preview_width / image.shape[1])
    preview = cv2.resize(image, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
    preview_path = preview_dir / "full_scene_preview.png"
    write_grayscale(preview_path, preview)
    return {"full_scene_preview": str(preview_path)}


def run_pipeline(config: PipelineConfig) -> Dict[str, object]:
    set_seed(config.seed)
    spec = DATASET_SPECS[config.dataset_key]
    output_dir = ensure_dir(config.output_path())
    bundle = load_raster(spec, config.robust_percentiles)
    previews = save_preview_images(bundle, output_dir)
    patch_table = extract_patch_table(
        image=bundle.image,
        valid_mask=bundle.valid_mask,
        patch_size=spec.patch_size,
        std_threshold=spec.std_threshold,
        valid_fraction_threshold=spec.valid_fraction,
    )
    filtered_table = patch_table.filtered()
    feature_matrix, handcrafted = compute_patch_features(
        filtered_table.patches,
        resize_for_features=config.resize_for_features,
        canny_low=config.canny_low,
        canny_high=config.canny_high,
    )
    clusters, _, pca, kmeans = cluster_patch_features(
        feature_matrix,
        num_clusters=spec.preview_clusters,
        pca_components=config.pca_components,
        seed=config.seed,
    )
    class_ids = np.asarray([spec.cluster_to_class[int(cluster_id)] for cluster_id in clusters], dtype=np.uint8)
    cluster_table = save_cluster_previews(output_dir, filtered_table.patches, clusters, spec, config.contact_sheet_limit)
    metadata = save_dataset(output_dir, patch_table, clusters, class_ids, spec)
    train_loader, val_loader, train_frame, val_frame = build_dataloaders(
        metadata=metadata,
        num_classes=len(spec.class_names),
        config=config,
    )
    device = resolve_device()
    class_weights = compute_class_weights(train_frame["class_id"], len(spec.class_names), device)
    model, history_frame, train_summary = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=len(spec.class_names),
        class_weights=class_weights,
        config=config,
        output_dir=output_dir,
    )
    final_train_metrics = collect_metrics(model, train_loader, len(spec.class_names), device)
    final_val_metrics = collect_metrics(model, val_loader, len(spec.class_names), device)
    save_training_preview(model, val_frame, spec, device, output_dir, config.val_preview_count)
    reconstruction = predict_full_map(model, patch_table, spec, device, output_dir)
    clustering_baseline = predict_clustering_baseline(
        patch_table=patch_table,
        spec=spec,
        config=config,
        pca=pca,
        kmeans=kmeans,
        output_dir=output_dir,
    )

    summary = {
        "config": asdict(config),
        "dataset": {
            "key": config.dataset_key,
            "spec": {
                "name": spec.name,
                "image_path": spec.image_path,
                "patch_size": spec.patch_size,
                "std_threshold": spec.std_threshold,
                "valid_fraction": spec.valid_fraction,
                "notes": spec.notes,
                "class_names": spec.class_names,
                "cluster_to_class": spec.cluster_to_class,
                "cluster_descriptions": spec.cluster_descriptions,
            },
            "raster_metadata": bundle.metadata,
            "total_grid_patches": int(len(patch_table.patches)),
            "kept_training_patches": int(len(filtered_table.patches)),
        },
        "previews": previews,
        "training": train_summary,
        "final_train_metrics": final_train_metrics,
        "final_val_metrics": final_val_metrics,
        "reconstruction": reconstruction,
        "clustering_baseline": clustering_baseline,
        "cluster_summary_path": str(output_dir / "cluster_summary.csv"),
        "metadata_path": str(output_dir / "dataset" / "metadata.csv"),
        "training_history_path": str(output_dir / "training_history.csv"),
        "feature_summary": {
            "mean": np.round(handcrafted.mean(axis=0), 6).tolist(),
            "std": np.round(handcrafted.std(axis=0), 6).tolist(),
        },
        "pca_explained_variance": np.round(pca.explained_variance_ratio_, 6).tolist(),
        "kmeans_inertia": float(kmeans.inertia_),
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as fh:
        json.dump(json_ready(summary), fh, indent=2)

    split_frame = pd.concat([train_frame, val_frame], ignore_index=True).sort_values(["grid_row", "grid_col"])
    split_frame.to_csv(output_dir / "dataset" / "metadata_with_split.csv", index=False)
    cluster_table.to_csv(output_dir / "cluster_summary.csv", index=False)
    history_frame.to_csv(output_dir / "training_history.csv", index=False)
    return summary


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Urban feature extraction from PAN imagery.")
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="jnpa")
    parser.add_argument("--output-dir", default="outputs/jnpa_run")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return PipelineConfig(
        dataset_key=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )


if __name__ == "__main__":
    cfg = parse_args()
    result = run_pipeline(cfg)
    print(json.dumps(json_ready(result), indent=2))
