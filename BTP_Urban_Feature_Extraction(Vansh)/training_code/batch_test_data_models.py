from __future__ import annotations

import csv
import json
import shutil
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from urban_feature_platform.platform_app.config import Settings
from urban_feature_platform.platform_app.image_utils import ensure_dir
from urban_feature_platform.platform_app.inference import InferenceService
from urban_feature_platform.platform_app.model_registry import ModelRegistry


ROOT = Path("/Users/vansharora665/BTP_GP")
TEST_DATA_DIR = ROOT / "Test_Data"
OUTPUT_DIR = ROOT / "documentation" / "test_data_model_results_2026_04_19_final"
CONVERTED_INPUT_DIR = OUTPUT_DIR / "converted_inputs"
RESULT_COPY_DIR = OUTPUT_DIR / "model_outputs"

PRIMARY_MODEL_IDS = [
    "jnpa_dense_v128_prod",
    "cartosat_dense_v128_prod",
    "jnpa_v2_prod",
    "cartosat_v3_4class_prod",
    "jnpa_unet_v1_compare",
    "jnpa_unet_smoke_compare",
]

SKIP_SUFFIXES = {
    ".aux",
    ".xml",
    ".ovr",
}


@dataclass
class RasterCandidate:
    name: str
    source_path: Path
    converted_path: Path
    driver: str
    width: int
    height: int
    count: int
    dtype: str
    crs: str | None


def safe_name(value: str) -> str:
    allowed = []
    for char in value.lower():
        if char.isalnum():
            allowed.append(char)
        elif char in {"-", "_"}:
            allowed.append(char)
        else:
            allowed.append("_")
    compact = "".join(allowed).strip("_")
    while "__" in compact:
        compact = compact.replace("__", "_")
    return compact or "unnamed"


def should_skip_path(path: Path) -> bool:
    if path.name == "info":
        return True
    if path.is_file():
        lower_name = path.name.lower()
        if lower_name.endswith(".aux.xml"):
            return True
        if path.suffix.lower() in SKIP_SUFFIXES:
            return True
    return False


def openable_rasters(test_dir: Path) -> list[tuple[Path, rasterio.DatasetReader]]:
    candidates: list[tuple[Path, rasterio.DatasetReader]] = []
    for path in sorted(test_dir.iterdir()):
        if should_skip_path(path):
            continue
        try:
            src = rasterio.open(path)
        except Exception:
            continue
        candidates.append((path, src))
    return candidates


def convert_to_geotiff(source_path: Path, src: rasterio.DatasetReader) -> RasterCandidate:
    name = safe_name(source_path.name)
    converted_path = CONVERTED_INPUT_DIR / f"{name}.tif"
    ensure_dir(CONVERTED_INPUT_DIR)

    profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )

    if not converted_path.exists():
        with rasterio.open(converted_path, "w", **profile) as dst:
            for band_index in range(1, src.count + 1):
                dst.write(src.read(band_index), band_index)

    dtype = src.dtypes[0] if src.dtypes else "unknown"
    return RasterCandidate(
        name=name,
        source_path=source_path,
        converted_path=converted_path,
        driver=str(src.driver),
        width=int(src.width),
        height=int(src.height),
        count=int(src.count),
        dtype=str(dtype),
        crs=str(src.crs) if src.crs else None,
    )


def copy_result_artifacts(result_payload: dict[str, Any], destination: Path) -> list[str]:
    ensure_dir(destination)
    copied: list[str] = []
    result_id = result_payload["result_id"]
    result_dir = Settings().result_dir / result_id
    for filename in [
        "RESULTS_SIMPLE.txt",
        "prediction_summary.json",
        "class_percentages.json",
        "patch_predictions.csv",
        "input_preview.png",
        "marked_preview.png",
        "segmentation_preview.png",
        "segmentation_labels.npy",
        "segmentation_labels_georef.tif",
        "class_shapefiles_georef.zip",
        "marked_full.png",
        "segmentation_full.png",
    ]:
        source = result_dir / filename
        if source.exists():
            shutil.copy2(source, destination / filename)
            copied.append(filename)
    return copied


def compact_percentages(rows: list[dict[str, Any]]) -> str:
    parts = []
    for item in rows:
        parts.append(f"{item['class_name']}={item['percentage']:.2f}%")
    return "; ".join(parts)


def write_summary_files(rows: list[dict[str, Any]], skipped: list[dict[str, Any]], output_dir: Path) -> None:
    ensure_dir(output_dir)
    csv_path = output_dir / "test_data_model_results_summary.csv"
    fieldnames = [
        "dataset",
        "source_path",
        "converted_path",
        "driver",
        "width",
        "height",
        "bands",
        "crs",
        "run_mode",
        "requested_model",
        "status",
        "selected_model_id",
        "selected_model_name",
        "dominant_class",
        "model_fit_score",
        "result_id",
        "copied_result_folder",
        "class_percentages",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    json_path = output_dir / "test_data_model_results_summary.json"
    json_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "test_data_dir": str(TEST_DATA_DIR),
                "output_dir": str(output_dir),
                "runs": rows,
                "skipped": skipped,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    success_rows = [row for row in rows if row["status"] == "success"]
    failure_rows = [row for row in rows if row["status"] != "success"]
    markdown_lines = [
        "# Test Data Model Evaluation Results",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Scope",
        "",
        f"- Test data folder: `{TEST_DATA_DIR}`",
        f"- Converted input folder: `{CONVERTED_INPUT_DIR}`",
        f"- Copied output folder: `{RESULT_COPY_DIR}`",
        f"- Successful runs: `{len(success_rows)}`",
        f"- Failed runs: `{len(failure_rows)}`",
        "",
        "The ArcInfo Grid datasets were converted to GeoTIFF before inference so they could pass through the same platform loader used for normal uploads.",
        "",
        "## Valid Raster Inputs",
        "",
    ]

    seen = {}
    for row in rows:
        seen[row["dataset"]] = row
    for dataset, row in seen.items():
        markdown_lines.append(
            f"- `{dataset}`: {row['width']} x {row['height']} pixels, bands={row['bands']}, driver={row['driver']}, CRS={row['crs'] or 'None'}"
        )

    markdown_lines.extend(["", "## Auto Mode Results", ""])
    auto_rows = [row for row in success_rows if row["run_mode"] == "auto"]
    if auto_rows:
        markdown_lines.append("| Dataset | Selected model | Dominant class | Top percentages | Result folder |")
        markdown_lines.append("|---|---|---|---|---|")
        for row in auto_rows:
            markdown_lines.append(
                f"| {row['dataset']} | {row['selected_model_name']} | {row['dominant_class']} | {row['class_percentages']} | `{row['copied_result_folder']}` |"
            )
    else:
        markdown_lines.append("No successful auto-mode runs were completed.")

    markdown_lines.extend(["", "## Manual Model Runs", ""])
    manual_rows = [row for row in success_rows if row["run_mode"] == "manual"]
    if manual_rows:
        markdown_lines.append("| Dataset | Requested model | Dominant class | Score | Result folder |")
        markdown_lines.append("|---|---|---|---:|---|")
        for row in manual_rows:
            markdown_lines.append(
                f"| {row['dataset']} | {row['requested_model']} | {row['dominant_class']} | {row['model_fit_score']} | `{row['copied_result_folder']}` |"
            )
    else:
        markdown_lines.append("No successful manual model runs were completed.")

    if failure_rows:
        markdown_lines.extend(["", "## Failed Runs", ""])
        markdown_lines.append("| Dataset | Mode | Model | Error |")
        markdown_lines.append("|---|---|---|---|")
        for row in failure_rows:
            markdown_lines.append(
                f"| {row['dataset']} | {row['run_mode']} | {row['requested_model']} | {row['error']} |"
            )

    if skipped:
        markdown_lines.extend(["", "## Skipped Items", ""])
        for item in skipped:
            markdown_lines.append(f"- `{item['path']}`: {item['reason']}")

    markdown_lines.extend(
        [
            "",
            "## Important Interpretation Note",
            "",
            "The models are trained using pseudo-labels derived from clustering and dense pseudo-mask generation. These test outputs are valid predictions from the trained project models, but they are not ground-truth accuracy measurements unless manually annotated masks are later added for these test rasters.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(markdown_lines), encoding="utf-8")


def run_batch() -> None:
    ensure_dir(OUTPUT_DIR)
    ensure_dir(RESULT_COPY_DIR)

    settings = Settings()
    registry = ModelRegistry(settings)
    service = InferenceService(settings, registry)

    available_model_ids = {model["id"] for model in registry.list_models()}
    model_ids = [model_id for model_id in PRIMARY_MODEL_IDS if model_id in available_model_ids]

    skipped: list[dict[str, Any]] = []
    for path in sorted(TEST_DATA_DIR.iterdir()):
        if should_skip_path(path):
            skipped.append({"path": str(path), "reason": "support file or auxiliary folder"})

    raster_sources = openable_rasters(TEST_DATA_DIR)
    candidates: list[RasterCandidate] = []
    for path, src in raster_sources:
        try:
            candidates.append(convert_to_geotiff(path, src))
        finally:
            src.close()

    rows: list[dict[str, Any]] = []
    runs = [("auto", None)] + [("manual", model_id) for model_id in model_ids]

    for candidate in candidates:
        print(f"\n=== Dataset: {candidate.name} ===", flush=True)
        for mode, model_id in runs:
            requested = model_id or "auto"
            print(f"Running {candidate.name} with {requested}...", flush=True)
            row: dict[str, Any] = {
                "dataset": candidate.name,
                "source_path": str(candidate.source_path),
                "converted_path": str(candidate.converted_path),
                "driver": candidate.driver,
                "width": candidate.width,
                "height": candidate.height,
                "bands": candidate.count,
                "crs": candidate.crs,
                "run_mode": mode,
                "requested_model": requested,
                "status": "failed",
                "selected_model_id": "",
                "selected_model_name": "",
                "dominant_class": "",
                "model_fit_score": "",
                "result_id": "",
                "copied_result_folder": "",
                "class_percentages": "",
                "error": "",
            }
            try:
                payload = service.predict_file(candidate.converted_path, mode=mode, model_id=model_id)
                selected_model = payload["selected_model"]
                result_folder = RESULT_COPY_DIR / candidate.name / requested / payload["result_id"]
                copied = copy_result_artifacts(payload, result_folder)
                row.update(
                    {
                        "status": "success",
                        "selected_model_id": selected_model["id"],
                        "selected_model_name": selected_model["display_name"],
                        "dominant_class": payload["dominant_class"],
                        "model_fit_score": payload.get("score_breakdown", {}).get("final_score", ""),
                        "result_id": payload["result_id"],
                        "copied_result_folder": str(result_folder.relative_to(OUTPUT_DIR)),
                        "class_percentages": compact_percentages(payload["class_percentages"]),
                        "copied_files": copied,
                    }
                )
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                error_dir = ensure_dir(RESULT_COPY_DIR / candidate.name / requested)
                (error_dir / "ERROR.txt").write_text(
                    "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                    encoding="utf-8",
                )
                print(f"FAILED {candidate.name} with {requested}: {exc}", flush=True)
            rows.append(row)
            write_summary_files(rows, skipped, OUTPUT_DIR)

    write_summary_files(rows, skipped, OUTPUT_DIR)
    print("\nBatch test complete.")
    print(f"Summary: {OUTPUT_DIR / 'README.md'}")
    print(f"CSV: {OUTPUT_DIR / 'test_data_model_results_summary.csv'}")


if __name__ == "__main__":
    run_batch()
