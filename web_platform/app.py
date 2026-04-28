import os

from platform_app import create_app

app = create_app()


if __name__ == "__main__":
    host = os.environ.get("URBAN_PLATFORM_HOST", "0.0.0.0")
    port = int(os.environ.get("URBAN_PLATFORM_PORT", "5050"))
    debug = os.environ.get("URBAN_PLATFORM_DEBUG", "1") == "1"
    app.run(host=host, port=port, debug=debug, use_reloader=False)
