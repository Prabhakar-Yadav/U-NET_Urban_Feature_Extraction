# Test Data Model Evaluation Results

Generated: 2026-04-19 02:33:06

## Scope

- Test data folder: `/Users/vansharora665/BTP_GP/Test_Data`
- Converted input folder: `/Users/vansharora665/BTP_GP/documentation/test_data_model_results_2026_04_19_final/converted_inputs`
- Copied output folder: `/Users/vansharora665/BTP_GP/documentation/test_data_model_results_2026_04_19_final/model_outputs`
- Auto-mode visual sheet: `auto_results_contact_sheet.png`
- Successful runs: `42`
- Failed runs: `0`

The ArcInfo Grid datasets were converted to GeoTIFF before inference so they could pass through the same platform loader used for normal uploads.

## Valid Raster Inputs

- `test_1_6mc1`: 2220 x 1583 pixels, bands=1, driver=AIG, CRS=EPSG:4326
- `test_1_6mc2`: 2220 x 1583 pixels, bands=1, driver=AIG, CRS=EPSG:4326
- `test_1_6mc3`: 2220 x 1583 pixels, bands=1, driver=AIG, CRS=EPSG:4326
- `test_1_6mc4`: 2220 x 1583 pixels, bands=1, driver=AIG, CRS=EPSG:4326
- `test_1m`: 4931 x 4020 pixels, bands=1, driver=AIG, CRS=EPSG:32643
- `test_2_5m`: 1973 x 1608 pixels, bands=1, driver=AIG, CRS=EPSG:32643

## Auto Mode Results

| Dataset | Selected model | Dominant class | Top percentages | Result folder |
|---|---|---|---|---|
| test_1_6mc1 | CARTOSAT Dense U-Net V128 (Hard Masks) | Water | Water=35.78%; Port / Waterfront Infrastructure=29.40%; Dense Urban Built-up=24.73%; Terrain / Open Ground=10.10% | `model_outputs/test_1_6mc1/auto/20260419_023023_6bce9b0f` |
| test_1_6mc2 | CARTOSAT Dense U-Net V128 (Hard Masks) | Port / Waterfront Infrastructure | Port / Waterfront Infrastructure=34.36%; Water=31.10%; Dense Urban Built-up=25.08%; Terrain / Open Ground=9.47% | `model_outputs/test_1_6mc2/auto/20260419_023046_96679509` |
| test_1_6mc3 | CARTOSAT Dense U-Net V128 (Hard Masks) | Port / Waterfront Infrastructure | Port / Waterfront Infrastructure=43.32%; Dense Urban Built-up=25.91%; Water=23.50%; Terrain / Open Ground=7.27% | `model_outputs/test_1_6mc3/auto/20260419_023107_5d1d70d2` |
| test_1_6mc4 | CARTOSAT Dense U-Net V128 (Hard Masks) | Port / Waterfront Infrastructure | Port / Waterfront Infrastructure=53.63%; Dense Urban Built-up=24.76%; Terrain / Open Ground=15.68%; Water=5.93% | `model_outputs/test_1_6mc4/auto/20260419_023129_d0aa05cd` |
| test_1m | CARTOSAT Dense U-Net V128 (Hard Masks) | Water | Water=38.08%; Dense Urban Built-up=31.10%; Port / Waterfront Infrastructure=21.11%; Terrain / Open Ground=9.70% | `model_outputs/test_1m/auto/20260419_023153_68819267` |
| test_2_5m | CARTOSAT Dense U-Net V128 (Hard Masks) | Dense Urban Built-up | Dense Urban Built-up=34.13%; Port / Waterfront Infrastructure=34.01%; Water=25.01%; Terrain / Open Ground=6.85% | `model_outputs/test_2_5m/auto/20260419_023250_dec8b7ed` |

## Manual Model Runs

| Dataset | Requested model | Dominant class | Score | Result folder |
|---|---|---|---:|---|
| test_1_6mc1 | jnpa_dense_v128_prod | Vegetation / Mangroves | 0.533 | `model_outputs/test_1_6mc1/jnpa_dense_v128_prod/20260419_023027_d17820c7` |
| test_1_6mc1 | cartosat_dense_v128_prod | Water | 0.7688 | `model_outputs/test_1_6mc1/cartosat_dense_v128_prod/20260419_023030_8252a4de` |
| test_1_6mc1 | jnpa_v2_prod | Vegetation / Mangroves | 0.5945 | `model_outputs/test_1_6mc1/jnpa_v2_prod/20260419_023033_2859c366` |
| test_1_6mc1 | cartosat_v3_4class_prod | Water | 0.6325 | `model_outputs/test_1_6mc1/cartosat_v3_4class_prod/20260419_023035_223af932` |
| test_1_6mc1 | jnpa_unet_v1_compare | Vegetation / Mangroves | 0.4334 | `model_outputs/test_1_6mc1/jnpa_unet_v1_compare/20260419_023039_ebeb30d1` |
| test_1_6mc1 | jnpa_unet_smoke_compare | Industrial / Port Infrastructure | 0.4444 | `model_outputs/test_1_6mc1/jnpa_unet_smoke_compare/20260419_023042_0e899647` |
| test_1_6mc2 | jnpa_dense_v128_prod | Vegetation / Mangroves | 0.5209 | `model_outputs/test_1_6mc2/jnpa_dense_v128_prod/20260419_023049_d0f76f18` |
| test_1_6mc2 | cartosat_dense_v128_prod | Port / Waterfront Infrastructure | 0.7521 | `model_outputs/test_1_6mc2/cartosat_dense_v128_prod/20260419_023052_3acacff9` |
| test_1_6mc2 | jnpa_v2_prod | Vegetation / Mangroves | 0.5793 | `model_outputs/test_1_6mc2/jnpa_v2_prod/20260419_023055_7ed25a6f` |
| test_1_6mc2 | cartosat_v3_4class_prod | Water | 0.655 | `model_outputs/test_1_6mc2/cartosat_v3_4class_prod/20260419_023057_facdbb91` |
| test_1_6mc2 | jnpa_unet_v1_compare | Vegetation / Mangroves | 0.4109 | `model_outputs/test_1_6mc2/jnpa_unet_v1_compare/20260419_023101_71c43972` |
| test_1_6mc2 | jnpa_unet_smoke_compare | Industrial / Port Infrastructure | 0.421 | `model_outputs/test_1_6mc2/jnpa_unet_smoke_compare/20260419_023104_9721ad40` |
| test_1_6mc3 | jnpa_dense_v128_prod | Vegetation / Mangroves | 0.5086 | `model_outputs/test_1_6mc3/jnpa_dense_v128_prod/20260419_023110_1044dfed` |
| test_1_6mc3 | cartosat_dense_v128_prod | Port / Waterfront Infrastructure | 0.7138 | `model_outputs/test_1_6mc3/cartosat_dense_v128_prod/20260419_023113_e605e555` |
| test_1_6mc3 | jnpa_v2_prod | Vegetation / Mangroves | 0.5528 | `model_outputs/test_1_6mc3/jnpa_v2_prod/20260419_023116_d85fb316` |
| test_1_6mc3 | cartosat_v3_4class_prod | Water | 0.6396 | `model_outputs/test_1_6mc3/cartosat_v3_4class_prod/20260419_023119_b8713a73` |
| test_1_6mc3 | jnpa_unet_v1_compare | Vegetation / Mangroves | 0.3947 | `model_outputs/test_1_6mc3/jnpa_unet_v1_compare/20260419_023122_42bec37d` |
| test_1_6mc3 | jnpa_unet_smoke_compare | Industrial / Port Infrastructure | 0.3907 | `model_outputs/test_1_6mc3/jnpa_unet_smoke_compare/20260419_023125_426e695c` |
| test_1_6mc4 | jnpa_dense_v128_prod | Bare Land / Soil | 0.4937 | `model_outputs/test_1_6mc4/jnpa_dense_v128_prod/20260419_023132_86e32c8e` |
| test_1_6mc4 | cartosat_dense_v128_prod | Port / Waterfront Infrastructure | 0.7006 | `model_outputs/test_1_6mc4/cartosat_dense_v128_prod/20260419_023135_f386a092` |
| test_1_6mc4 | jnpa_v2_prod | Bare Land / Soil | 0.552 | `model_outputs/test_1_6mc4/jnpa_v2_prod/20260419_023137_e6db470e` |
| test_1_6mc4 | cartosat_v3_4class_prod | Dense Urban Built-up | 0.5941 | `model_outputs/test_1_6mc4/cartosat_v3_4class_prod/20260419_023140_4a619e51` |
| test_1_6mc4 | jnpa_unet_v1_compare | Bare Land / Soil | 0.4347 | `model_outputs/test_1_6mc4/jnpa_unet_v1_compare/20260419_023143_5a16127f` |
| test_1_6mc4 | jnpa_unet_smoke_compare | Bare Land / Soil | 0.4211 | `model_outputs/test_1_6mc4/jnpa_unet_smoke_compare/20260419_023146_1a2af595` |
| test_1m | jnpa_dense_v128_prod | Vegetation / Mangroves | 0.4796 | `model_outputs/test_1m/jnpa_dense_v128_prod/20260419_023201_c7e40d5a` |
| test_1m | cartosat_dense_v128_prod | Water | 0.6206 | `model_outputs/test_1m/cartosat_dense_v128_prod/20260419_023210_c03b7a67` |
| test_1m | jnpa_v2_prod | Vegetation / Mangroves | 0.6351 | `model_outputs/test_1m/jnpa_v2_prod/20260419_023217_53bce0b5` |
| test_1m | cartosat_v3_4class_prod | Water | 0.6505 | `model_outputs/test_1m/cartosat_v3_4class_prod/20260419_023223_68389659` |
| test_1m | jnpa_unet_v1_compare | Vegetation / Mangroves | 0.5031 | `model_outputs/test_1m/jnpa_unet_v1_compare/20260419_023233_ccbc872b` |
| test_1m | jnpa_unet_smoke_compare | Industrial / Port Infrastructure | 0.4818 | `model_outputs/test_1m/jnpa_unet_smoke_compare/20260419_023244_4638f9d4` |
| test_2_5m | jnpa_dense_v128_prod | Vegetation / Mangroves | 0.4935 | `model_outputs/test_2_5m/jnpa_dense_v128_prod/20260419_023252_7290b0c3` |
| test_2_5m | cartosat_dense_v128_prod | Dense Urban Built-up | 0.6284 | `model_outputs/test_2_5m/cartosat_dense_v128_prod/20260419_023255_2e27ad5a` |
| test_2_5m | jnpa_v2_prod | Vegetation / Mangroves | 0.6455 | `model_outputs/test_2_5m/jnpa_v2_prod/20260419_023257_98973ce4` |
| test_2_5m | cartosat_v3_4class_prod | Water | 0.6429 | `model_outputs/test_2_5m/cartosat_v3_4class_prod/20260419_023259_74d9913e` |
| test_2_5m | jnpa_unet_v1_compare | Vegetation / Mangroves | 0.5042 | `model_outputs/test_2_5m/jnpa_unet_v1_compare/20260419_023302_bb6e49fa` |
| test_2_5m | jnpa_unet_smoke_compare | Industrial / Port Infrastructure | 0.4885 | `model_outputs/test_2_5m/jnpa_unet_smoke_compare/20260419_023304_40e69c21` |

## Skipped Items

- `/Users/vansharora665/BTP_GP/Test_Data/TEST_2_5M.aux.xml`: support file or auxiliary folder
- `/Users/vansharora665/BTP_GP/Test_Data/TEST_2_5M.ovr`: support file or auxiliary folder
- `/Users/vansharora665/BTP_GP/Test_Data/info`: support file or auxiliary folder
- `/Users/vansharora665/BTP_GP/Test_Data/test_1_6m.aux.xml`: support file or auxiliary folder
- `/Users/vansharora665/BTP_GP/Test_Data/test_1_6m.ovr`: support file or auxiliary folder
- `/Users/vansharora665/BTP_GP/Test_Data/test_1_6mc1.aux.xml`: support file or auxiliary folder
- `/Users/vansharora665/BTP_GP/Test_Data/test_1_6mc2.aux.xml`: support file or auxiliary folder
- `/Users/vansharora665/BTP_GP/Test_Data/test_1_6mc3.aux.xml`: support file or auxiliary folder
- `/Users/vansharora665/BTP_GP/Test_Data/test_1_6mc4.aux.xml`: support file or auxiliary folder
- `/Users/vansharora665/BTP_GP/Test_Data/test_1m.aux.xml`: support file or auxiliary folder
- `/Users/vansharora665/BTP_GP/Test_Data/test_1m.ovr`: support file or auxiliary folder

## Important Interpretation Note

The models are trained using pseudo-labels derived from clustering and dense pseudo-mask generation. These test outputs are valid predictions from the trained project models, but they are not ground-truth accuracy measurements unless manually annotated masks are later added for these test rasters.
