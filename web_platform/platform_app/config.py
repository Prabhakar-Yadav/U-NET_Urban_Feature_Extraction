from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    root_dir: Path = Path(__file__).resolve().parents[1]
    registry_path: Path = root_dir / "config" / "model_registry.json"
    runtime_dir: Path = root_dir / "runtime"
    upload_dir: Path = runtime_dir / "uploads"
    result_dir: Path = runtime_dir / "results"
    max_content_length: int = 550 * 1024 * 1024
    preview_max_dimension: int = 1800
    full_visual_pixel_limit: int = 25_000_000
    raster_stream_pixel_limit: int = 40_000_000
    raster_stream_file_size_mb: int = 150
    prediction_batch_patches: int = 64
    raster_stats_sample_max_dimension: int = 2048
    secret_key: str = os.environ.get("URBAN_PLATFORM_SECRET", "urban-feature-platform-dev")
    allowed_extensions: tuple[str, ...] = (
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
        ".webp",
        ".gif",
    )
