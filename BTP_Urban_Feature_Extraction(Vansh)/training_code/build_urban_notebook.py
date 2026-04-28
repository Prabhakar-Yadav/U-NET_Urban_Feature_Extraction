from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [dedent(text).strip() + "\n"],
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [dedent(text).strip() + "\n"],
    }


NOTEBOOK_PATH = Path("urban_feature_extraction_end_to_end.ipynb")


cells = [
    md(
        """
        # Urban Feature Extraction from Panchromatic Satellite Imagery

        This notebook implements an end-to-end pipeline for **urban feature extraction** from
        high-resolution **single-band panchromatic imagery**. It is designed for:

        - `JNPA/JNPA_2_5.tif` with 5 classes
        - `Monocromatic/CARTOSAT_1M_PAN.tif` with a simpler 4-class setup

        The workflow covers:

        1. Raster loading and robust normalization
        2. Patch extraction and low-information filtering
        3. Texture-driven unsupervised clustering for pseudo-label creation
        4. Manual cluster-to-semantic mapping based on visual inspection
        5. U-Net training with CE + Dice loss
        6. Evaluation, reconstruction, and visualization of the final segmentation map

        `Important:` the reported accuracy/IoU values are measured against **pseudo-labels**
        generated from clustering, not independent ground-truth annotations.
        """
    ),
    code(
        """
        from pathlib import Path
        import json

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import torch
        from IPython.display import Image as IPyImage, display

        from urban_feature_pipeline import (
            CLASS_COLORS,
            DATASET_SPECS,
            PipelineConfig,
            build_dataloaders,
            cluster_patch_features,
            collect_metrics,
            compute_class_weights,
            compute_patch_features,
            extract_patch_table,
            load_raster,
            predict_clustering_baseline,
            predict_full_map,
            resolve_device,
            save_cluster_previews,
            save_dataset,
            save_preview_images,
            save_training_preview,
            set_seed,
            train_model,
        )

        plt.style.use("seaborn-v0_8")
        """
    ),
    md(
        """
        ## Configuration

        Change `dataset_key` to `"cartosat"` if you want to run the simpler CARTOSAT configuration.
        The default notebook configuration below targets the **JNPA** dataset that was used for the
        main training run.
        """
    ),
    code(
        """
        config = PipelineConfig(
            dataset_key="jnpa",
            output_dir="outputs/jnpa_training_run",
            epochs=20,
            batch_size=8,
            learning_rate=1e-3,
            seed=42,
        )

        spec = DATASET_SPECS[config.dataset_key]
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        set_seed(config.seed)

        print("Dataset:", spec.name)
        print("Image path:", spec.image_path)
        print("Output dir:", output_dir)
        print("Classes:", spec.class_names)
        print("Fine cluster to class mapping:", spec.cluster_to_class)
        """
    ),
    md("## Semantic Class Definitions and Manual Fine-Cluster Mapping"),
    code(
        """
        cluster_table = pd.DataFrame(
            {
                "fine_cluster": list(spec.cluster_to_class.keys()),
                "mapped_class_id": list(spec.cluster_to_class.values()),
                "mapped_class_name": [spec.class_names[c] for c in spec.cluster_to_class.values()],
                "cluster_description": [
                    spec.cluster_descriptions.get(k, "") for k in spec.cluster_to_class.keys()
                ],
            }
        ).sort_values("fine_cluster")

        display(cluster_table)
        print(spec.notes)
        """
    ),
    md("## Step 1-2: Load and Normalize the Panchromatic Raster"),
    code(
        """
        bundle = load_raster(spec, config.robust_percentiles)
        preview_paths = save_preview_images(bundle, output_dir)

        raster_info = pd.DataFrame([bundle.metadata])
        display(raster_info)

        display(IPyImage(filename=preview_paths["full_scene_preview"]))
        """
    ),
    md("## Step 3-4: Patch Extraction and Low-Information Filtering"),
    code(
        """
        patch_table = extract_patch_table(
            image=bundle.image,
            valid_mask=bundle.valid_mask,
            patch_size=spec.patch_size,
            std_threshold=spec.std_threshold,
            valid_fraction_threshold=spec.valid_fraction,
        )
        filtered_table = patch_table.filtered()

        debug_stats = {
            "total_grid_patches": int(len(patch_table.patches)),
            "training_patches_kept": int(len(filtered_table.patches)),
            "patch_size": spec.patch_size,
            "std_threshold": spec.std_threshold,
            "valid_fraction_threshold": spec.valid_fraction,
            "mean_std_all_patches": float(patch_table.std.mean()),
            "mean_std_kept_patches": float(filtered_table.std.mean()),
        }
        debug_stats
        """
    ),
    code(
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].hist(patch_table.std, bins=40, color="slategray")
        axes[0].axvline(spec.std_threshold, color="crimson", linestyle="--", label="threshold")
        axes[0].set_title("Patch Standard Deviation")
        axes[0].legend()

        axes[1].hist(patch_table.valid_fraction, bins=30, color="olive")
        axes[1].axvline(spec.valid_fraction, color="crimson", linestyle="--", label="threshold")
        axes[1].set_title("Patch Valid Fraction")
        axes[1].legend()

        plt.tight_layout()
        plt.show()
        """
    ),
    md("## Step 5-7: Feature Extraction, PCA, Clustering, and Cluster-to-Class Mapping"),
    code(
        """
        feature_matrix, handcrafted = compute_patch_features(
            filtered_table.patches,
            resize_for_features=config.resize_for_features,
            canny_low=config.canny_low,
            canny_high=config.canny_high,
        )

        clusters, reduced_features, pca_model, kmeans_model = cluster_patch_features(
            feature_matrix,
            num_clusters=spec.preview_clusters,
            pca_components=config.pca_components,
            seed=config.seed,
        )

        pseudo_class_ids = np.array(
            [spec.cluster_to_class[int(cluster_id)] for cluster_id in clusters],
            dtype=np.uint8,
        )

        cluster_summary = save_cluster_previews(
            output_dir=output_dir,
            patches=filtered_table.patches,
            cluster_ids=clusters,
            spec=spec,
            contact_sheet_limit=config.contact_sheet_limit,
        )
        display(cluster_summary)

        print("PCA explained variance sum:", float(pca_model.explained_variance_ratio_.sum()))
        print("KMeans inertia:", float(kmeans_model.inertia_))
        """
    ),
    code(
        """
        for preview_path in cluster_summary["preview_path"]:
            display(IPyImage(filename=preview_path))
        """
    ),
    md("## Step 8-10: Pseudo-Labeled Dataset Creation, Split, and Data Loader Setup"),
    code(
        """
        metadata = save_dataset(
            output_dir=output_dir,
            patch_table=patch_table,
            clusters=clusters,
            class_ids=pseudo_class_ids,
            spec=spec,
        )
        display(metadata.head())

        class_distribution = metadata["class_name"].value_counts().rename_axis("class_name").reset_index(name="count")
        display(class_distribution)
        """
    ),
    code(
        """
        train_loader, val_loader, train_frame, val_frame = build_dataloaders(
            metadata=metadata,
            num_classes=len(spec.class_names),
            config=config,
        )

        print("Train patches:", len(train_frame))
        print("Validation patches:", len(val_frame))

        batch_images, batch_masks = next(iter(train_loader))
        print("Image tensor shape:", tuple(batch_images.shape))
        print("Mask tensor shape:", tuple(batch_masks.shape))
        print("Unique labels in first batch:", torch.unique(batch_masks))
        """
    ),
    md("## Step 11-14: U-Net Training, CE + Dice Loss, and Evaluation"),
    code(
        """
        device = resolve_device()
        class_weights = compute_class_weights(train_frame["class_id"], len(spec.class_names), device)
        print("Training device:", device)
        print("Class weights:", class_weights.detach().cpu().numpy())

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

        save_training_preview(
            model=model,
            val_frame=val_frame,
            spec=spec,
            device=device,
            output_dir=output_dir,
            limit=config.val_preview_count,
        )

        print("Best epoch:", train_summary["best_epoch"])
        print("Final train metrics:", json.dumps(final_train_metrics, indent=2))
        print("Final val metrics:", json.dumps(final_val_metrics, indent=2))
        """
    ),
    code(
        """
        history_frame.tail()
        """
    ),
    code(
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history_frame["epoch"], history_frame["train_loss"], label="train")
        axes[0].plot(history_frame["epoch"], history_frame["val_loss"], label="val")
        axes[0].set_title("Loss")
        axes[0].legend()

        axes[1].plot(history_frame["epoch"], history_frame["train_mean_iou"], label="train")
        axes[1].plot(history_frame["epoch"], history_frame["val_mean_iou"], label="val")
        axes[1].set_title("Mean IoU")
        axes[1].legend()

        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        preview_dir = output_dir / "validation_previews"
        for preview_path in sorted(preview_dir.glob("*_overlay.png"))[: config.val_preview_count]:
            display(IPyImage(filename=str(preview_path)))
        """
    ),
    md("## Step 15-16: Full Scene Reconstruction and Final Visualization"),
    code(
        """
        reconstruction = predict_full_map(
            model=model,
            patch_table=patch_table,
            spec=spec,
            device=device,
            output_dir=output_dir,
        )

        clustering_baseline = predict_clustering_baseline(
            patch_table=patch_table,
            spec=spec,
            config=config,
            pca=pca_model,
            kmeans=kmeans_model,
            output_dir=output_dir,
        )

        {
            "unet_reconstruction": reconstruction,
            "clustering_baseline": clustering_baseline,
        }
        """
    ),
    code(
        """
        display(IPyImage(filename=clustering_baseline["color_path"]))
        display(IPyImage(filename=clustering_baseline["overlay_path"]))
        display(IPyImage(filename=reconstruction["color_path"]))
        display(IPyImage(filename=reconstruction["overlay_path"]))
        """
    ),
    md("## Final Run Summary"),
    code(
        """
        run_summary = {
            "dataset": spec.name,
            "output_dir": str(output_dir),
            "best_epoch": int(train_summary["best_epoch"]),
            "train_pixel_accuracy": float(final_train_metrics["pixel_accuracy"]),
            "train_mean_iou": float(final_train_metrics["mean_iou"]),
            "val_pixel_accuracy": float(final_val_metrics["pixel_accuracy"]),
            "val_mean_iou": float(final_val_metrics["mean_iou"]),
            "clustering_baseline_overlay": clustering_baseline["overlay_path"],
            "full_scene_overlay": reconstruction["overlay_path"],
        }

        with open(output_dir / "notebook_run_summary.json", "w", encoding="utf-8") as fh:
            json.dump(run_summary, fh, indent=2)

        run_summary
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.9",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH}")
