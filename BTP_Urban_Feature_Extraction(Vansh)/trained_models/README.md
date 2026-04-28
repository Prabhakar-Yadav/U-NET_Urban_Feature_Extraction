# Trained Models

This folder stores the trained models that are referenced by the web platform.

## Final production models

- `jnpa_dense_unet_v128/` - final JNPA dense U-Net model used in production
- `cartosat_dense_unet_v128/` - final CARTOSAT dense U-Net model used in production

## Supporting production and comparison models

- `jnpa_patch_classifier_v2/` - JNPA patch-classifier teacher/baseline
- `cartosat_patch_classifier_v3_4class/` - CARTOSAT 4-class patch-classifier teacher/baseline
- `cartosat_patch_classifier_v2/` - earlier CARTOSAT patch-classifier version
- `jnpa_unet_v1_compare/` - JNPA earlier U-Net used for comparison in the platform
- `jnpa_unet_smoke_compare/` - smoke-run JNPA U-Net compare model

Each subfolder contains the checkpoint and the corresponding run-summary JSON used by the registry-driven platform.
