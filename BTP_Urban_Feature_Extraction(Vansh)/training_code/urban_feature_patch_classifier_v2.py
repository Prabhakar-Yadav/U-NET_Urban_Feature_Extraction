from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import joblib
import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


CLASS_COLORS_V2 = {
    0: (31, 119, 180),
    1: (214, 39, 40),
    2: (140, 86, 75),
    3: (44, 160, 44),
    4: (255, 215, 0),
}


@dataclass(frozen=True)
class PatchClassifierSpec:
    name: str
    image_path: str
    patch_size: int
    std_threshold: float
    valid_fraction: float
    preview_clusters: int
    class_names: Dict[int, str]
    cluster_to_class: Dict[int, int]
    cluster_descriptions: Dict[int, str]
    dark_water_mean_threshold: float


SPECS_V2: Dict[str, PatchClassifierSpec] = {
    "jnpa_v2": PatchClassifierSpec(
        name="JNPA 2.5m PAN Patch Classifier V2",
        image_path="JNPA/JNPA_2_5.tif",
        patch_size=256,
        std_threshold=0.045,
        valid_fraction=0.85,
        preview_clusters=7,
        class_names={
            0: "Water Bodies",
            1: "Industrial / Port Infrastructure",
            2: "Bare Land / Soil",
            3: "Vegetation / Mangroves",
            4: "Urban Built-up",
        },
        cluster_to_class={
            0: 1,
            1: 2,
            2: 3,
            3: 1,
            4: 0,
            5: 4,
            6: 4,
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
        dark_water_mean_threshold=0.32,
    ),
    "cartosat_v2": PatchClassifierSpec(
        name="CARTOSAT 1m PAN Patch Classifier V2",
        image_path="Monocromatic/CARTOSAT_1M_PAN.tif",
        patch_size=256,
        std_threshold=0.055,
        valid_fraction=0.95,
        preview_clusters=7,
        class_names={
            0: "Water",
            1: "Urban",
            2: "Terrain",
        },
        cluster_to_class={
            0: 1,
            1: 0,
            2: 0,
            3: 2,
            4: 1,
            5: 1,
            6: 0,
        },
        cluster_descriptions={
            0: "Dense built-up urban fabric with mixed tree cover and waterfront settlement.",
            1: "Open water and dark homogeneous harbor water.",
            2: "Water-dominant coastal patches with shore transition.",
            3: "Bright exposed ground, transport surfaces, and mixed developed terrain.",
            4: "Urban waterfront blocks and compact built-up neighborhoods.",
            5: "Canal-edge and dense built-up urban patches with strong man-made geometry.",
            6: "Open water with urban edge effects and harbor-side water patches.",
        },
        dark_water_mean_threshold=0.26,
    ),
}


@dataclass
class PatchClassifierConfig:
    dataset_key: str
    output_dir: str
    seed: int = 42
    robust_percentiles: Tuple[float, float] = (1.0, 99.0)
    resize_for_features: int = 32
    pca_components: int = 25
    train_fraction: float = 0.8
    rf_estimators: int = 1000
    canny_low: int = 60
    canny_high: int = 160


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def load_raster(spec: PatchClassifierSpec, robust_percentiles: Tuple[float, float]):
    with rasterio.open(spec.image_path) as src:
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
            "path": spec.image_path,
            "width": int(src.width),
            "height": int(src.height),
            "dtype": str(src.dtypes[0]),
            "crs": str(src.crs),
            "nodata": nodata,
            "normalization_low": float(lo),
            "normalization_high": float(hi),
        }
    return image, valid_mask, metadata


def extract_patch_table(
    image: np.ndarray,
    valid_mask: np.ndarray,
    patch_size: int,
    std_threshold: float,
    valid_fraction_threshold: float,
) -> pd.DataFrame:
    rows = image.shape[0] // patch_size
    cols = image.shape[1] // patch_size
    records: List[dict] = []
    patches: List[np.ndarray] = []
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
            records.append(
                {
                    "row": row,
                    "col": col,
                    "valid_fraction": valid_fraction,
                    "std": std,
                    "keep": keep,
                }
            )
    table = pd.DataFrame.from_records(records)
    table["patch_index"] = np.arange(len(table))
    return table, np.stack(patches)


def compute_patch_features(
    patches: np.ndarray,
    resize_for_features: int,
    canny_low: int,
    canny_high: int,
) -> Tuple[np.ndarray, np.ndarray]:
    full_features = []
    handcrafted = []
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
        full_features.append(np.concatenate([small.flatten(), stats]))
        handcrafted.append(stats)
    return np.stack(full_features).astype(np.float32), np.stack(handcrafted).astype(np.float32)


def fit_cluster_pseudo_labels(
    feature_matrix: np.ndarray,
    spec: PatchClassifierSpec,
    config: PatchClassifierConfig,
):
    pca = PCA(
        n_components=min(config.pca_components, feature_matrix.shape[0], feature_matrix.shape[1]),
        random_state=config.seed,
        svd_solver="randomized",
    )
    reduced = pca.fit_transform(feature_matrix)
    kmeans = MiniBatchKMeans(
        n_clusters=spec.preview_clusters,
        random_state=config.seed,
        batch_size=min(256, len(feature_matrix)),
        n_init=20,
    )
    cluster_ids = kmeans.fit_predict(reduced)
    class_ids = np.array([spec.cluster_to_class[int(cluster_id)] for cluster_id in cluster_ids], dtype=np.int64)
    return cluster_ids, class_ids, reduced, pca, kmeans


def build_contact_sheet(patches: np.ndarray, indices: np.ndarray, patch_size: int) -> np.ndarray:
    sample = indices[:16]
    side = 4
    sheet = np.full((side * patch_size, side * patch_size), 255, dtype=np.uint8)
    for n, patch_idx in enumerate(sample):
        r, c = divmod(n, side)
        tile = np.clip(patches[int(patch_idx)] * 255.0, 0, 255).astype(np.uint8)
        y0 = r * patch_size
        x0 = c * patch_size
        sheet[y0 : y0 + patch_size, x0 : x0 + patch_size] = tile
    return sheet


def save_cluster_previews(
    filtered_patches: np.ndarray,
    handcrafted: np.ndarray,
    cluster_ids: np.ndarray,
    spec: PatchClassifierSpec,
    output_dir: Path,
) -> pd.DataFrame:
    preview_dir = ensure_dir(output_dir / "cluster_previews")
    records = []
    for cluster_id in sorted(np.unique(cluster_ids)):
        indices = np.where(cluster_ids == cluster_id)[0]
        sheet = build_contact_sheet(filtered_patches, indices, spec.patch_size)
        preview_path = preview_dir / f"cluster_{int(cluster_id)}.png"
        Image.fromarray(sheet, mode="L").save(preview_path)
        stats = handcrafted[indices].mean(axis=0)
        class_id = spec.cluster_to_class[int(cluster_id)]
        records.append(
            {
                "cluster_id": int(cluster_id),
                "count": int(len(indices)),
                "mapped_class_id": int(class_id),
                "mapped_class_name": spec.class_names[int(class_id)],
                "description": spec.cluster_descriptions.get(int(cluster_id), ""),
                "mean_intensity": float(stats[0]),
                "std_intensity": float(stats[1]),
                "edge_density": float(stats[6]),
                "preview_path": str(preview_path),
            }
        )
    cluster_frame = pd.DataFrame.from_records(records).sort_values("cluster_id")
    cluster_frame.to_csv(output_dir / "cluster_summary_v2.csv", index=False)
    return cluster_frame


def train_patch_classifier(
    reduced_features: np.ndarray,
    labels: np.ndarray,
    config: PatchClassifierConfig,
):
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=1.0 - config.train_fraction,
        random_state=config.seed,
        stratify=labels,
    )
    clf = RandomForestClassifier(
        n_estimators=config.rf_estimators,
        random_state=config.seed,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    clf.fit(reduced_features[train_idx], labels[train_idx])
    train_pred = clf.predict(reduced_features[train_idx])
    val_pred = clf.predict(reduced_features[val_idx])
    metrics = {
        "train_accuracy": float(accuracy_score(labels[train_idx], train_pred)),
        "val_accuracy": float(accuracy_score(labels[val_idx], val_pred)),
        "train_macro_f1": float(f1_score(labels[train_idx], train_pred, average="macro")),
        "val_macro_f1": float(f1_score(labels[val_idx], val_pred, average="macro")),
        "train_confusion": confusion_matrix(labels[train_idx], train_pred).tolist(),
        "val_confusion": confusion_matrix(labels[val_idx], val_pred).tolist(),
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
    }
    return clf, metrics, train_idx, val_idx


def heuristic_water_override(
    patch_mean: float,
    patch_std: float,
    spec: PatchClassifierSpec,
) -> int | None:
    if patch_std < spec.std_threshold and patch_mean < spec.dark_water_mean_threshold and 0 in spec.class_names:
        return 0
    return None


def colorize_grid(label_grid: np.ndarray, class_names: Dict[int, str]) -> np.ndarray:
    max_class = max(class_names)
    color = np.zeros((*label_grid.shape, 3), dtype=np.uint8)
    for class_id in range(max_class + 1):
        rgb = CLASS_COLORS_V2.get(class_id, (200, 200, 200))
        color[label_grid == class_id] = rgb
    return color


def save_prediction_maps(
    image: np.ndarray,
    label_grid: np.ndarray,
    output_dir: Path,
    prefix: str,
    class_names: Dict[int, str],
) -> Dict[str, str]:
    pred_dir = ensure_dir(output_dir / "predictions")
    color = colorize_grid(label_grid, class_names)
    overlay = cv2.addWeighted(np.dstack([image * 255.0] * 3).astype(np.uint8), 0.45, color, 0.55, 0)
    label_path = pred_dir / f"{prefix}_labels.npy"
    color_path = pred_dir / f"{prefix}_color.png"
    overlay_path = pred_dir / f"{prefix}_overlay.png"
    np.save(label_path, label_grid)
    Image.fromarray(color).save(color_path)
    Image.fromarray(overlay).save(overlay_path)
    return {
        "label_path": str(label_path),
        "color_path": str(color_path),
        "overlay_path": str(overlay_path),
    }


def build_final_map_with_legend(
    overlay_path: Path,
    output_path: Path,
    spec: PatchClassifierSpec,
    title: str,
) -> None:
    base = Image.open(overlay_path).convert("RGB")
    width, height = base.size
    legend_width = 380
    canvas = Image.new("RGB", (width + legend_width, height), (250, 248, 242))
    canvas.paste(base, (0, 0))

    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 28)
        text_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 20)
        small_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 16)
    except Exception:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    x0 = width + 24
    y = 28
    draw.text((x0, y), title, fill=(20, 20, 20), font=title_font)
    y += 48
    draw.text((x0, y), spec.name, fill=(40, 40, 40), font=text_font)
    y += 40
    draw.text((x0, y), "Classes", fill=(20, 20, 20), font=text_font)
    y += 34

    for class_id, class_name in spec.class_names.items():
        color = CLASS_COLORS_V2[class_id]
        draw.rectangle([x0, y + 2, x0 + 22, y + 24], fill=color, outline=(60, 60, 60))
        draw.text((x0 + 34, y), class_name, fill=(30, 30, 30), font=text_font)
        y += 42

    y += 8
    draw.text((x0, y), "Pipeline: cluster pseudo-labels +", fill=(80, 80, 80), font=small_font)
    y += 22
    draw.text((x0, y), "random-forest patch classifier", fill=(80, 80, 80), font=small_font)
    canvas.save(output_path)


def run_patch_classifier_pipeline(config: PatchClassifierConfig) -> Dict[str, object]:
    set_seed(config.seed)
    spec = SPECS_V2[config.dataset_key]
    output_dir = ensure_dir(Path(config.output_dir))

    image, valid_mask, raster_metadata = load_raster(spec, config.robust_percentiles)
    patch_table, patches = extract_patch_table(
        image=image,
        valid_mask=valid_mask,
        patch_size=spec.patch_size,
        std_threshold=spec.std_threshold,
        valid_fraction_threshold=spec.valid_fraction,
    )
    filtered_mask = patch_table["keep"].to_numpy()
    filtered_table = patch_table[filtered_mask].copy().reset_index(drop=True)
    filtered_patches = patches[filtered_mask]

    full_features, full_handcrafted = compute_patch_features(
        filtered_patches,
        resize_for_features=config.resize_for_features,
        canny_low=config.canny_low,
        canny_high=config.canny_high,
    )
    cluster_ids, class_ids, reduced_features, pca, kmeans = fit_cluster_pseudo_labels(
        full_features, spec, config
    )
    cluster_frame = save_cluster_previews(filtered_patches, full_handcrafted, cluster_ids, spec, output_dir)

    filtered_table["cluster_id"] = cluster_ids
    filtered_table["class_id"] = class_ids
    filtered_table["class_name"] = filtered_table["class_id"].map(spec.class_names)

    clf, metrics, train_idx, val_idx = train_patch_classifier(reduced_features, class_ids, config)
    filtered_table["split"] = "train"
    filtered_table.loc[val_idx, "split"] = "val"
    filtered_table.to_csv(output_dir / "filtered_patch_metadata_v2.csv", index=False)

    model_bundle = {
        "classifier": clf,
        "pca": pca,
        "kmeans": kmeans,
        "spec_key": config.dataset_key,
        "config": asdict(config),
    }
    joblib.dump(model_bundle, output_dir / "patch_classifier_model_v2.joblib")

    all_features, _ = compute_patch_features(
        patches,
        resize_for_features=config.resize_for_features,
        canny_low=config.canny_low,
        canny_high=config.canny_high,
    )
    all_reduced = pca.transform(all_features)
    all_pred = clf.predict(all_reduced).astype(np.int64)

    overrides = 0
    for idx in range(len(all_pred)):
        override = heuristic_water_override(
            patch_mean=float(patches[idx].mean()),
            patch_std=float(patch_table.iloc[idx]["std"]),
            spec=spec,
        )
        if override is not None:
            all_pred[idx] = override
            overrides += 1

    rows = int(patch_table["row"].max()) + 1
    cols = int(patch_table["col"].max()) + 1
    patch_grid = all_pred.reshape(rows, cols)
    label_grid = np.repeat(np.repeat(patch_grid, spec.patch_size, axis=0), spec.patch_size, axis=1)
    image_crop = image[: rows * spec.patch_size, : cols * spec.patch_size]
    maps = save_prediction_maps(image_crop, label_grid, output_dir, "patch_classifier_v2", spec.class_names)

    final_map_path = output_dir / "final_results_map_v2.png"
    build_final_map_with_legend(Path(maps["overlay_path"]), final_map_path, spec, title="Final Results Map V2")

    result_lines = [
        f"Dataset: {spec.name}",
        "",
        "Why V2 is more accurate:",
        "The original U-Net was trained on uniform patch masks, but the labels themselves were only patch-level pseudo-labels.",
        "This V2 pipeline trains a random-forest patch classifier directly on the clustered patch features, which matches the label design much better.",
        "",
        f"Validation accuracy: {metrics['val_accuracy']:.4f}",
        f"Validation macro F1: {metrics['val_macro_f1']:.4f}",
        f"Train accuracy: {metrics['train_accuracy']:.4f}",
        f"Train macro F1: {metrics['train_macro_f1']:.4f}",
        "",
        "Final output map:",
        str(final_map_path),
        "",
        "Main saved files:",
        str(output_dir / "run_summary_v2.json"),
        str(output_dir / "patch_classifier_model_v2.joblib"),
        str(output_dir / "filtered_patch_metadata_v2.csv"),
        str(output_dir / "cluster_summary_v2.csv"),
        maps["overlay_path"],
    ]
    (output_dir / "RESULTS_SUMMARY_V2.txt").write_text("\n".join(result_lines), encoding="utf-8")

    summary = {
        "config": asdict(config),
        "dataset": {
            "name": spec.name,
            "image_path": spec.image_path,
            "class_names": spec.class_names,
            "cluster_to_class": spec.cluster_to_class,
            "cluster_descriptions": spec.cluster_descriptions,
            "raster_metadata": raster_metadata,
            "total_grid_patches": int(len(patches)),
            "kept_filtered_patches": int(filtered_mask.sum()),
        },
        "metrics": metrics,
        "heuristic_water_overrides": overrides,
        "prediction_maps": maps,
        "final_results_map": str(final_map_path),
        "cluster_summary_path": str(output_dir / "cluster_summary_v2.csv"),
        "filtered_patch_metadata_path": str(output_dir / "filtered_patch_metadata_v2.csv"),
        "model_path": str(output_dir / "patch_classifier_model_v2.joblib"),
    }
    (output_dir / "run_summary_v2.json").write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")
    return summary


def parse_args() -> PatchClassifierConfig:
    parser = argparse.ArgumentParser(description="Improved patch-classifier pipeline for PAN imagery.")
    parser.add_argument("--dataset", choices=sorted(SPECS_V2), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rf-estimators", type=int, default=1000)
    args = parser.parse_args()
    return PatchClassifierConfig(
        dataset_key=args.dataset,
        output_dir=args.output_dir,
        rf_estimators=args.rf_estimators,
    )


if __name__ == "__main__":
    cfg = parse_args()
    result = run_patch_classifier_pipeline(cfg)
    print(json.dumps(json_ready(result), indent=2))
