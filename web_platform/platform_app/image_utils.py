from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image, ImageOps


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_allowed_file(filename: str, allowed_extensions: tuple[str, ...]) -> bool:
    return Path(filename).suffix.lower() in allowed_extensions


def robust_normalize(
    image: np.ndarray,
    valid_mask: np.ndarray,
    percentiles: tuple[float, float] = (1.0, 99.0),
) -> tuple[np.ndarray, dict[str, float]]:
    valid_pixels = image[valid_mask]
    if valid_pixels.size == 0:
        normalized = np.zeros_like(image, dtype=np.float32)
        return normalized, {"low": 0.0, "high": 1.0}

    low, high = np.percentile(valid_pixels, percentiles)
    if not np.isfinite(low):
        low = float(np.min(valid_pixels))
    if not np.isfinite(high):
        high = float(np.max(valid_pixels))
    if high <= low:
        high = low + 1.0

    normalized = np.clip(image, low, high)
    normalized = (normalized - low) / max(high - low, 1e-6)
    normalized = normalized.astype(np.float32)
    normalized[~valid_mask] = 0.0
    return normalized, {"low": float(low), "high": float(high)}


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def compute_preview_size(width: int, height: int, max_dimension: int) -> tuple[int, int]:
    longest = max(width, height)
    if longest <= max_dimension:
        return width, height
    scale = max_dimension / float(longest)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def resize_rgb(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(rgb, mode="RGB")
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def resize_grayscale(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(to_uint8(image), mode="L")
    if pil_image.size != size:
        pil_image = pil_image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(pil_image, dtype=np.uint8)


@dataclass
class LoadedImage:
    source_path: Path
    grayscale: np.ndarray | None
    valid_mask: np.ndarray | None
    preview_rgb: np.ndarray
    width: int
    height: int
    raster_profile: dict[str, Any] | None
    source_metadata: dict[str, Any]
    normalization_low: float | None
    normalization_high: float | None
    nodata: float | None
    processing_mode: str


@dataclass
class TiledImage:
    patches: np.ndarray
    patch_means: np.ndarray
    patch_stds: np.ndarray
    patch_coverages: np.ndarray
    rows: int
    cols: int
    original_height: int
    original_width: int


def _read_raster_grayscale(
    src: rasterio.io.DatasetReader,
    *,
    window: rasterio.windows.Window | None = None,
    out_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    read_kwargs: dict[str, Any] = {}
    if window is not None:
        read_kwargs["window"] = window
    if out_shape is not None:
        if src.count >= 3:
            read_kwargs["out_shape"] = (3, out_shape[0], out_shape[1])
        else:
            read_kwargs["out_shape"] = out_shape
        read_kwargs["resampling"] = rasterio.enums.Resampling.bilinear

    nodata = src.nodata
    if src.count >= 3:
        bands = src.read([1, 2, 3], **read_kwargs).astype(np.float32)
        valid_mask = np.ones(bands.shape[1:], dtype=bool)
        if nodata is not None:
            valid_mask &= np.all(np.not_equal(bands, nodata), axis=0)
        gray = (
            0.299 * bands[0]
            + 0.587 * bands[1]
            + 0.114 * bands[2]
        ).astype(np.float32)
    else:
        band = src.read(1, **read_kwargs).astype(np.float32)
        valid_mask = np.ones_like(band, dtype=bool)
        if nodata is not None:
            valid_mask &= np.not_equal(band, nodata)
        gray = band
    return gray, valid_mask


def _build_preview_rgb_from_raster(
    src: rasterio.io.DatasetReader,
    preview_height: int,
    preview_width: int,
    single_band_preview_gray: np.ndarray | None = None,
) -> np.ndarray:
    nodata = src.nodata
    if src.count >= 3:
        preview_bands = src.read(
            [1, 2, 3],
            out_shape=(3, preview_height, preview_width),
            resampling=rasterio.enums.Resampling.bilinear,
        ).astype(np.float32)
        preview_channels = []
        for idx in range(preview_bands.shape[0]):
            channel_valid = np.ones_like(preview_bands[idx], dtype=bool)
            if nodata is not None:
                channel_valid &= np.not_equal(preview_bands[idx], nodata)
            preview_channels.append(to_uint8(robust_normalize(preview_bands[idx], channel_valid)[0]))
        return np.stack(preview_channels, axis=-1)

    assert single_band_preview_gray is not None
    return np.dstack([single_band_preview_gray] * 3)


def read_normalized_raster_patch(
    src: rasterio.io.DatasetReader,
    *,
    x0: int,
    y0: int,
    patch_size: int,
    normalization_low: float,
    normalization_high: float,
) -> tuple[np.ndarray, np.ndarray]:
    width = int(src.width)
    height = int(src.height)
    window = rasterio.windows.Window(
        col_off=x0,
        row_off=y0,
        width=min(patch_size, width - x0),
        height=min(patch_size, height - y0),
    )
    gray, valid_mask = _read_raster_grayscale(src, window=window)
    normalized = np.clip(gray, normalization_low, normalization_high)
    normalized = (normalized - normalization_low) / max(normalization_high - normalization_low, 1e-6)
    normalized = normalized.astype(np.float32)
    normalized[~valid_mask] = 0.0

    pad_bottom = patch_size - normalized.shape[0]
    pad_right = patch_size - normalized.shape[1]
    if pad_bottom or pad_right:
        if normalized.size:
            normalized = np.pad(normalized, ((0, pad_bottom), (0, pad_right)), mode="edge")
        else:
            normalized = np.zeros((patch_size, patch_size), dtype=np.float32)
        valid_mask = np.pad(valid_mask, ((0, pad_bottom), (0, pad_right)), mode="constant", constant_values=False)

    return normalized.astype(np.float32), valid_mask.astype(bool)


def read_raster_valid_mask(
    src: rasterio.io.DatasetReader,
    *,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> np.ndarray:
    window = rasterio.windows.Window(
        col_off=x0,
        row_off=y0,
        width=width,
        height=height,
    )
    _, valid_mask = _read_raster_grayscale(src, window=window)
    return valid_mask.astype(bool)


def _load_with_rasterio(
    path: Path,
    preview_max_dimension: int,
    raster_stream_pixel_limit: int,
    raster_stream_file_size_mb: int,
    raster_stats_sample_max_dimension: int,
) -> LoadedImage:
    with rasterio.open(path) as src:
        band_count = int(src.count)
        width = int(src.width)
        height = int(src.height)
        nodata = src.nodata
        file_size_mb = path.stat().st_size / (1024 * 1024)
        preview_width, preview_height = compute_preview_size(width, height, preview_max_dimension)

        sample_max_dimension = min(max(preview_max_dimension, raster_stats_sample_max_dimension), max(width, height))
        sample_width, sample_height = compute_preview_size(width, height, sample_max_dimension)
        sample_gray, sample_valid = _read_raster_grayscale(src, out_shape=(sample_height, sample_width))
        sample_gray_norm, sampled_norm_stats = robust_normalize(sample_gray, sample_valid)
        sampled_preview_gray = resize_grayscale(sample_gray_norm, (preview_width, preview_height))
        preview_rgb = _build_preview_rgb_from_raster(
            src,
            preview_height=preview_height,
            preview_width=preview_width,
            single_band_preview_gray=sampled_preview_gray,
        )

        pixel_count = width * height
        use_streaming = pixel_count >= raster_stream_pixel_limit or file_size_mb >= raster_stream_file_size_mb
        profile = src.profile.copy()

        source_metadata = {
            "loader": "rasterio",
            "width": width,
            "height": height,
            "band_count": band_count,
            "nodata": nodata,
            "crs": str(src.crs) if src.crs else None,
            "format": path.suffix.lower(),
            "normalization_low": sampled_norm_stats["low"],
            "normalization_high": sampled_norm_stats["high"],
            "file_size_mb": round(file_size_mb, 2),
            "processing_mode": "streaming_raster" if use_streaming else "memory",
        }
        if use_streaming:
            return LoadedImage(
                source_path=path,
                grayscale=None,
                valid_mask=None,
                preview_rgb=preview_rgb,
                width=width,
                height=height,
                raster_profile=profile,
                source_metadata=source_metadata,
                normalization_low=sampled_norm_stats["low"],
                normalization_high=sampled_norm_stats["high"],
                nodata=nodata,
                processing_mode="streaming_raster",
            )

        full_gray, valid_mask = _read_raster_grayscale(src)
        normalized_gray, norm_stats = robust_normalize(full_gray, valid_mask)
        if band_count < 3:
            preview_gray = resize_grayscale(normalized_gray, (preview_width, preview_height))
            preview_rgb = np.dstack([preview_gray] * 3)
        source_metadata["normalization_low"] = norm_stats["low"]
        source_metadata["normalization_high"] = norm_stats["high"]
        return LoadedImage(
            source_path=path,
            grayscale=normalized_gray,
            valid_mask=valid_mask,
            preview_rgb=preview_rgb,
            width=width,
            height=height,
            raster_profile=profile,
            source_metadata=source_metadata,
            normalization_low=norm_stats["low"],
            normalization_high=norm_stats["high"],
            nodata=nodata,
            processing_mode="memory",
        )


def _load_with_pillow(path: Path, preview_max_dimension: int) -> LoadedImage:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    try:
        image.seek(0)
    except Exception:
        pass

    rgba = image.convert("RGBA")
    rgba_np = np.asarray(rgba, dtype=np.uint8)
    alpha = rgba_np[..., 3] > 0
    rgb = rgba_np[..., :3]
    if not np.any(alpha):
        alpha = np.ones(rgb.shape[:2], dtype=bool)

    gray = (
        0.299 * rgb[..., 0].astype(np.float32)
        + 0.587 * rgb[..., 1].astype(np.float32)
        + 0.114 * rgb[..., 2].astype(np.float32)
    )
    gray_norm, norm_stats = robust_normalize(gray, alpha)

    width, height = image.size
    preview_width, preview_height = compute_preview_size(width, height, preview_max_dimension)
    preview_rgb = resize_rgb(rgb, (preview_width, preview_height))

    return LoadedImage(
        source_path=path,
        grayscale=gray_norm,
        valid_mask=alpha,
        preview_rgb=preview_rgb,
        width=width,
        height=height,
        raster_profile=None,
        source_metadata={
            "loader": "pillow",
            "width": width,
            "height": height,
            "band_count": 3,
            "nodata": None,
            "crs": None,
            "format": image.format,
            "normalization_low": norm_stats["low"],
            "normalization_high": norm_stats["high"],
            "file_size_mb": round(path.stat().st_size / (1024 * 1024), 2),
            "processing_mode": "memory",
        },
        normalization_low=norm_stats["low"],
        normalization_high=norm_stats["high"],
        nodata=None,
        processing_mode="memory",
    )


def load_uploaded_image(
    path: Path,
    preview_max_dimension: int,
    raster_stream_pixel_limit: int,
    raster_stream_file_size_mb: int,
    raster_stats_sample_max_dimension: int,
) -> LoadedImage:
    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            return _load_with_rasterio(
                path,
                preview_max_dimension=preview_max_dimension,
                raster_stream_pixel_limit=raster_stream_pixel_limit,
                raster_stream_file_size_mb=raster_stream_file_size_mb,
                raster_stats_sample_max_dimension=raster_stats_sample_max_dimension,
            )
        except Exception:
            return _load_with_pillow(path, preview_max_dimension=preview_max_dimension)
    return _load_with_pillow(path, preview_max_dimension=preview_max_dimension)


def tile_image(
    image: np.ndarray,
    valid_mask: np.ndarray,
    patch_size: int,
) -> TiledImage:
    height, width = image.shape
    rows = int(np.ceil(height / patch_size))
    cols = int(np.ceil(width / patch_size))
    padded_height = rows * patch_size
    padded_width = cols * patch_size

    pad_bottom = padded_height - height
    pad_right = padded_width - width

    if pad_bottom or pad_right:
        padded_image = np.pad(image, ((0, pad_bottom), (0, pad_right)), mode="edge")
        padded_mask = np.pad(valid_mask, ((0, pad_bottom), (0, pad_right)), mode="constant", constant_values=False)
    else:
        padded_image = image
        padded_mask = valid_mask

    patches = []
    means = []
    stds = []
    coverages = []

    for row in range(rows):
        for col in range(cols):
            y0 = row * patch_size
            x0 = col * patch_size
            patch = padded_image[y0 : y0 + patch_size, x0 : x0 + patch_size]
            patch_valid = padded_mask[y0 : y0 + patch_size, x0 : x0 + patch_size]
            patches.append(patch)
            means.append(float(patch.mean()))
            stds.append(float(patch.std()))
            coverages.append(int(patch_valid.sum()))

    return TiledImage(
        patches=np.stack(patches).astype(np.float32),
        patch_means=np.asarray(means, dtype=np.float32),
        patch_stds=np.asarray(stds, dtype=np.float32),
        patch_coverages=np.asarray(coverages, dtype=np.int64),
        rows=rows,
        cols=cols,
        original_height=height,
        original_width=width,
    )


def build_full_label_map(patch_grid: np.ndarray, patch_size: int, height: int, width: int) -> np.ndarray:
    repeated = np.repeat(np.repeat(patch_grid, patch_size, axis=0), patch_size, axis=1)
    return repeated[:height, :width].astype(np.uint8)
