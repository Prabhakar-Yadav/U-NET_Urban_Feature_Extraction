from __future__ import annotations

import uuid
from pathlib import Path

from flask import Blueprint, current_app, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from .config import Settings
from .image_utils import ensure_dir, is_allowed_file
from .inference import InferenceService
from .model_registry import ModelRegistry


platform_bp = Blueprint("platform", __name__)


def get_settings() -> Settings:
    return current_app.config["PLATFORM_SETTINGS"]


def get_registry() -> ModelRegistry:
    registry = current_app.extensions.get("urban_model_registry")
    if registry is None:
        registry = ModelRegistry(get_settings())
        current_app.extensions["urban_model_registry"] = registry
    return registry


def get_inference_service() -> InferenceService:
    service = current_app.extensions.get("urban_inference_service")
    if service is None:
        service = InferenceService(get_settings(), get_registry())
        current_app.extensions["urban_inference_service"] = service
    return service


@platform_bp.get("/")
def index():
    settings = get_settings()
    return render_template(
        "index.html",
        models=get_registry().list_models(),
        allowed_extensions=", ".join(settings.allowed_extensions),
        max_upload_mb=round(settings.max_content_length / (1024 * 1024)),
    )


@platform_bp.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "enabled_models": len(get_registry().list_models()),
        }
    )


@platform_bp.get("/api/models")
def list_models():
    return jsonify({"models": get_registry().list_models()})


@platform_bp.post("/api/predict")
def predict():
    settings = get_settings()
    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "Please upload an image file."}), 400

    if not is_allowed_file(uploaded.filename, settings.allowed_extensions):
        return jsonify(
            {
                "error": f"Unsupported file type. Allowed extensions: {', '.join(settings.allowed_extensions)}",
            }
        ), 400

    mode = request.form.get("model_mode", "auto").strip().lower()
    if mode not in {"auto", "manual"}:
        return jsonify({"error": "Invalid model mode."}), 400

    model_id = request.form.get("model_id", "").strip() or None
    if mode == "manual" and not model_id:
        return jsonify({"error": "Choose a model or switch back to Auto mode."}), 400

    original_name = secure_filename(uploaded.filename)
    suffix = Path(original_name).suffix.lower()
    upload_id = uuid.uuid4().hex[:8]
    upload_path = ensure_dir(settings.upload_dir) / f"{Path(original_name).stem}_{upload_id}{suffix}"
    uploaded.save(upload_path)

    try:
        result = get_inference_service().predict_file(upload_path, mode=mode, model_id=model_id)
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 400
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        current_app.logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return jsonify(result)


@platform_bp.get("/results/<result_id>/<path:filename>")
def serve_result_file(result_id: str, filename: str):
    result_dir = get_settings().result_dir / result_id
    return send_from_directory(result_dir, filename, as_attachment=False)
