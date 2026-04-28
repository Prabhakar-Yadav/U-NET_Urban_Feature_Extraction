from __future__ import annotations

from flask import Flask, jsonify, request
from werkzeug.exceptions import RequestEntityTooLarge

from .config import Settings
from .image_utils import ensure_dir
from .routes import platform_bp


def create_app() -> Flask:
    settings = Settings()
    ensure_dir(settings.runtime_dir)
    ensure_dir(settings.upload_dir)
    ensure_dir(settings.result_dir)
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config.from_mapping(
        SECRET_KEY=settings.secret_key,
        MAX_CONTENT_LENGTH=settings.max_content_length,
        PLATFORM_SETTINGS=settings,
    )

    @app.errorhandler(RequestEntityTooLarge)
    def handle_request_entity_too_large(_: RequestEntityTooLarge):
        message = (
            f"Upload is too large. This platform currently accepts files up to "
            f"{round(settings.max_content_length / (1024 * 1024))} MB."
        )
        if request.path.startswith("/api/"):
            return jsonify({"error": message}), 413
        return message, 413

    app.register_blueprint(platform_bp)
    return app
