# Version Comparison

## Purpose

This document compares the two main versions used for:

- `JNPA` dataset
- `CARTOSAT / Panchromatic` dataset

The goal is to show, on similar parameters and metrics, what changed between versions and why the results changed.

## Important Evaluation Note

All versions here were trained and evaluated using **pseudo-labels** generated from clustering and manual cluster-to-class interpretation.

- These are **not** manually drawn ground-truth masks.
- So the reported scores show how well each model learned the cluster-derived semantic labeling scheme.
- This still gives a fair version-to-version comparison because the data source and labeling logic were kept consistent within each dataset family.

---

## 1. JNPA Comparison

### 1.1 What Stayed Common

| Parameter | JNPA Version 1 | JNPA Version 2 |
|---|---:|---:|
| Dataset | `JNPA/JNPA_2_5.tif` | `JNPA/JNPA_2_5.tif` |
| Image type | Single-band panchromatic | Single-band panchromatic |
| Patch size | `256 x 256` | `256 x 256` |
| Std threshold | `0.045` | `0.045` |
| Valid fraction threshold | `0.85` | `0.85` |
| Total grid patches | `1015` | `1015` |
| Filtered/kept patches | `735` | `735` |
| PCA components | `25` | `25` |
| Canny thresholds | `60 / 160` | `60 / 160` |
| Train split | `80%` | `80%` |
| Final classes | `5` | `5` |

### 1.2 What Actually Changed

| Parameter | JNPA Version 1 | JNPA Version 2 |
|---|---|---|
| Modeling approach | U-Net semantic segmentation | Random Forest patch classification |
| Prediction unit | Pixel-wise mask | One class per patch |
| Label structure used for training | Uniform patch masks from pseudo-labels | Direct patch labels from pseudo-labels |
| Augmentation | Yes | No |
| Main augmentations | Flip, rotate, brightness/contrast, noise | Not used |
| Optimizer / training style | AdamW, epoch-based deep learning | Classical ML fit, no backprop epochs |
| Loss | Cross-Entropy + Dice + patch CE | Random Forest impurity optimization |
| Water stabilization | Not explicit | Added dark-water heuristic override |

### 1.3 Classes Used

Both JNPA versions used the same semantic classes:

- Class 0: Water Bodies
- Class 1: Industrial / Port Infrastructure
- Class 2: Bare Land / Soil
- Class 3: Vegetation / Mangroves
- Class 4: Urban Built-up

### 1.4 Overall Metrics

| Metric | JNPA Version 1 | JNPA Version 2 |
|---|---:|---:|
| Best epoch | `17` | Not applicable |
| Final validation accuracy metric | Pixel Accuracy = `48.96%` | Validation Accuracy = `87.76%` |
| Mean IoU | `33.50%` | Not primary metric for this model |
| Validation Macro F1 | Not saved as primary score | `85.56%` |

### 1.5 Class-wise Accuracy

| Class | JNPA Version 1 | JNPA Version 2 |
|---|---:|---:|
| Water Bodies | `69.25%` | `63.16%` |
| Industrial / Port Infrastructure | `57.46%` | `80.00%` |
| Bare Land / Soil | `53.33%` | `100.00%` |
| Vegetation / Mangroves | `37.70%` | `100.00%` |
| Urban Built-up | `40.86%` | `80.56%` |

### 1.6 Why Version 2 Performed Better

The main reason is **task-model alignment**.

- In Version 1, the U-Net was trained for dense pixel segmentation.
- But the pseudo-labels were effectively patch-level labels, because each training patch had one dominant semantic class.
- That means the model was asked to predict detailed pixel boundaries from labels that did not contain true pixel-level boundary information.

In Version 2:

- the training label unit was the **patch**
- the model output unit was also the **patch**
- therefore the learning problem matched the label design much better

This is the biggest reason the JNPA score improved from a medium-quality result to a much stronger and more stable result.

### 1.7 JNPA Conclusion

`JNPA Version 2` is the better production choice because:

- it matches the pseudo-label design
- it is much faster to train
- it gives much better held-out validation performance
- it produces more stable class separation for industrial, bare land, vegetation, and urban classes

Relevant outputs:

- Old map: `outputs/jnpa_training_run/predictions/full_prediction_overlay.png`
- Improved map: `outputs/jnpa_patch_classifier_v2/final_results_map_v2.png`

---

## 2. CARTOSAT / Panchromatic Comparison

### 2.1 What Stayed Common

| Parameter | CARTOSAT Version 2 | CARTOSAT Version 3 |
|---|---:|---:|
| Dataset | `Monocromatic/CARTOSAT_1M_PAN.tif` | `Monocromatic/CARTOSAT_1M_PAN.tif` |
| Image type | Single-band panchromatic | Single-band panchromatic |
| Model family | Random Forest patch classifier | Random Forest patch classifier |
| Patch size | `256 x 256` | `256 x 256` |
| Std threshold | `0.055` | `0.055` |
| Valid fraction threshold | `0.95` | `0.95` |
| Total grid patches | `1716` | `1716` |
| Filtered/kept patches | `681` | `681` |
| PCA components | `25` | `25` |
| Random Forest trees | `1000` | `1000` |
| Train split | `80%` | `80%` |
| Augmentation | No | No |

### 2.2 What Actually Changed

| Parameter | CARTOSAT Version 2 | CARTOSAT Version 3 |
|---|---|---|
| Number of final classes | `3` | `4` |
| Class granularity | Coarser | More detailed |
| Urban-related labeling | Single `Urban` class | Split into `Dense Urban Built-up` and `Port / Waterfront Infrastructure` |
| Terrain definition | `Terrain` | `Terrain / Open Ground` |
| Cluster interpretation | Simpler mapping | More refined mapping after reviewing cluster previews |

### 2.3 Classes Used

#### CARTOSAT Version 2

- Class 0: Water
- Class 1: Urban
- Class 2: Terrain

#### CARTOSAT Version 3

- Class 0: Water
- Class 1: Dense Urban Built-up
- Class 2: Port / Waterfront Infrastructure
- Class 3: Terrain / Open Ground

### 2.4 Overall Metrics

| Metric | CARTOSAT Version 2 | CARTOSAT Version 3 |
|---|---:|---:|
| Validation Accuracy | `94.16%` | `89.78%` |
| Validation Macro F1 | `63.58%` | `74.64%` |

### 2.5 Class-wise Accuracy

#### CARTOSAT Version 2

| Class | Accuracy |
|---|---:|
| Water | `97.30%` |
| Urban | `96.61%` |
| Terrain | `0.00%` |

#### CARTOSAT Version 3

| Class | Accuracy |
|---|---:|
| Water | `97.30%` |
| Dense Urban Built-up | `94.74%` |
| Port / Waterfront Infrastructure | `66.67%` |
| Terrain / Open Ground | `25.00%` |

### 2.6 Why Version 3 Is Still Better for Presentation and Use

At first glance, `Version 2` seems better because overall accuracy is higher.

But that is misleading for this dataset because:

- Version 2 had only `3` coarse classes
- the dataset is strongly dominated by water and urban patches
- the terrain class almost failed completely with `0.00%` class accuracy

Version 3 is more meaningful because:

- it separates waterfront/port infrastructure from dense urban built-up
- it improves overall class balance understanding
- macro F1 improves from `63.58%` to `74.64%`
- minority classes become more interpretable, even though the task becomes harder

So the trade-off is:

- `Version 2`: better coarse overall accuracy
- `Version 3`: better semantic detail and better balanced performance

### 2.7 CARTOSAT Conclusion

`CARTOSAT Version 3` is the better final model for demonstration and platform use because it identifies more meaningful land-use structure in panchromatic imagery.

Relevant outputs:

- Previous map: `outputs/cartosat_patch_classifier_v2/final_results_map_v2.png`
- Final map: `outputs/cartosat_patch_classifier_v3_4class/final_results_map_v3.png`

---

## 3. Final Summary of What Differed

### JNPA

- The biggest change was the **modeling strategy**
- Version 1 used U-Net for pseudo-dense segmentation
- Version 2 used Random Forest for patch classification
- Since the labels were patch-driven, Version 2 matched the problem better and improved performance strongly

### CARTOSAT / Panchromatic

- The model family stayed the same
- The biggest change was **semantic class refinement**
- Version 3 introduced a more meaningful 4-class structure instead of the simpler 3-class structure
- This reduced raw overall accuracy slightly but improved macro F1 and interpretability

---

## 4. Best Version to Use

| Dataset | Best Version to Use | Reason |
|---|---|---|
| JNPA | `Version 2` | Best alignment between pseudo-label design and model type |
| CARTOSAT / Panchromatic | `Version 3` | Best semantic usefulness and clearer feature identification |

---

## 5. File References

- JNPA old summary: `outputs/jnpa_training_run/run_summary.json`
- JNPA improved summary: `outputs/jnpa_patch_classifier_v2/run_summary_v2.json`
- CARTOSAT previous summary: `outputs/cartosat_patch_classifier_v2/run_summary_v2.json`
- CARTOSAT improved summary: `outputs/cartosat_patch_classifier_v3_4class/run_summary_v3.json`
- JNPA old pipeline code: `urban_feature_pipeline.py`
- Patch-classifier code: `urban_feature_patch_classifier_v2.py`
- CARTOSAT 4-class extension: `urban_feature_patch_classifier_v3.py`
