from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import urban_feature_patch_classifier_v2 as base


SPECS_V3 = {
    "cartosat_v3_4class": base.PatchClassifierSpec(
        name="CARTOSAT 1m PAN Patch Classifier V3 (4-Class)",
        image_path="Monocromatic/CARTOSAT_1M_PAN.tif",
        patch_size=256,
        std_threshold=0.055,
        valid_fraction=0.95,
        preview_clusters=7,
        class_names={
            0: "Water",
            1: "Dense Urban Built-up",
            2: "Port / Waterfront Infrastructure",
            3: "Terrain / Open Ground",
        },
        cluster_to_class={
            0: 1,
            1: 0,
            2: 0,
            3: 3,
            4: 2,
            5: 2,
            6: 0,
        },
        cluster_descriptions={
            0: "Dense built-up urban neighborhoods with compact housing blocks and mixed tree cover.",
            1: "Open harbor water and dark homogeneous water surfaces.",
            2: "Water-dominant coastal and shoreline transition patches.",
            3: "Bright exposed ground, transport surfaces, bare terrain, and mixed open land.",
            4: "Waterfront industrial blocks, docks, quay-side structures, and organized port-side layouts.",
            5: "Canal-edge institutional or port-adjacent built-up patches with strong engineered geometry.",
            6: "Harbor water patches with urban edge effects and coastal transition.",
        },
        dark_water_mean_threshold=0.26,
    )
}


def json_ready(obj):
    return base.json_ready(obj)


def finalize_output_names(output_dir: Path, dataset_key: str) -> None:
    rename_map = {
        output_dir / "patch_classifier_model_v2.joblib": output_dir / "patch_classifier_model_v3.joblib",
        output_dir / "cluster_summary_v2.csv": output_dir / "cluster_summary_v3.csv",
        output_dir / "filtered_patch_metadata_v2.csv": output_dir / "filtered_patch_metadata_v3.csv",
        output_dir / "RESULTS_SUMMARY_V2.txt": output_dir / "RESULTS_SUMMARY_V3.txt",
        output_dir / "run_summary_v2.json": output_dir / "run_summary_v3.json",
        output_dir / "final_results_map_v2.png": output_dir / "final_results_map_v3.png",
        output_dir / "predictions" / "patch_classifier_v2_labels.npy": output_dir / "predictions" / "patch_classifier_v3_labels.npy",
        output_dir / "predictions" / "patch_classifier_v2_color.png": output_dir / "predictions" / "patch_classifier_v3_color.png",
        output_dir / "predictions" / "patch_classifier_v2_overlay.png": output_dir / "predictions" / "patch_classifier_v3_overlay.png",
    }
    for src, dst in rename_map.items():
        if src.exists():
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))

    summary_path = output_dir / "run_summary_v3.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary["config"]["dataset_key"] = dataset_key
        summary["config"]["output_dir"] = str(output_dir)
        summary["prediction_maps"] = {
            "label_path": str(output_dir / "predictions" / "patch_classifier_v3_labels.npy"),
            "color_path": str(output_dir / "predictions" / "patch_classifier_v3_color.png"),
            "overlay_path": str(output_dir / "predictions" / "patch_classifier_v3_overlay.png"),
        }
        summary["final_results_map"] = str(output_dir / "final_results_map_v3.png")
        summary["cluster_summary_path"] = str(output_dir / "cluster_summary_v3.csv")
        summary["filtered_patch_metadata_path"] = str(output_dir / "filtered_patch_metadata_v3.csv")
        summary["model_path"] = str(output_dir / "patch_classifier_model_v3.joblib")
        summary_path.write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")

    results_path = output_dir / "RESULTS_SUMMARY_V3.txt"
    if results_path.exists():
        text = results_path.read_text(encoding="utf-8")
        text = text.replace("V2", "V3")
        text = text.replace("patch_classifier_model_v2.joblib", "patch_classifier_model_v3.joblib")
        text = text.replace("filtered_patch_metadata_v2.csv", "filtered_patch_metadata_v3.csv")
        text = text.replace("cluster_summary_v2.csv", "cluster_summary_v3.csv")
        text = text.replace("patch_classifier_v2_overlay.png", "patch_classifier_v3_overlay.png")
        text = text.replace("run_summary_v2.json", "run_summary_v3.json")
        text = text.replace("final_results_map_v2.png", "final_results_map_v3.png")
        results_path.write_text(text, encoding="utf-8")


def run_pipeline(config: base.PatchClassifierConfig):
    base.SPECS_V2.update(SPECS_V3)
    summary = base.run_patch_classifier_pipeline(config)
    finalize_output_names(Path(config.output_dir), config.dataset_key)
    return summary


def parse_args() -> base.PatchClassifierConfig:
    parser = argparse.ArgumentParser(description="CARTOSAT V3 4-class patch-classifier pipeline.")
    parser.add_argument("--dataset", choices=sorted(SPECS_V3), default="cartosat_v3_4class")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rf-estimators", type=int, default=1000)
    args = parser.parse_args()
    return base.PatchClassifierConfig(
        dataset_key=args.dataset,
        output_dir=args.output_dir,
        rf_estimators=args.rf_estimators,
    )


if __name__ == "__main__":
    cfg = parse_args()
    result = run_pipeline(cfg)
    print(json.dumps(json_ready(result), indent=2))
