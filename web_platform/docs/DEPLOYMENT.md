# Deployment Notes

## Local development

```bash
cd web_platform
python3 app.py
```

Open:

```text
http://127.0.0.1:5050
```

## Production server

```bash
cd web_platform
gunicorn --config gunicorn.conf.py wsgi:app
```

## Current production architecture

- Flask web application
- registry-driven model loading
- in-memory model cache
- cached domain-profile signatures for auto model selection
- synchronous single-image prediction endpoint
- streaming raster inference path for large GeoTIFF uploads
- browser UI plus JSON API

## Runtime folders

- `runtime/uploads/`
- `runtime/results/`
- `runtime/cache/domain_profiles/`

These folders should be kept on persistent storage if you want outputs to remain available after restart.

## First-request behavior

The first time a model is used after a clean start, the app may take a little longer because it builds and caches a domain profile from that model's original source imagery.

This cache is then reused for later requests.

## Scaling guidance

For larger deployments:

- run Gunicorn behind a reverse proxy such as Nginx
- keep model bundles on local SSD or mounted persistent storage
- move `runtime/results` to object storage if multiple replicas need shared access
- place authentication in front of the app if it is exposed publicly
- add a worker queue if you later support large batches or many concurrent full-scene jobs

## Large-image behavior

The app is built to stay stable on large scenes:

- uploads up to roughly `500 MB` are accepted
- prediction is patch-based
- large TIFF rasters are processed from disk in streaming batches instead of loading the whole scene into RAM
- preview outputs are always generated
- full-size PNG visuals are skipped automatically for very large images to keep memory usage safe
- georeferenced label GeoTIFF export is still produced when raster metadata is available

## Gunicorn notes

Current `gunicorn.conf.py`:

- `workers = 2`
- `threads = 4`
- `timeout = 180`

You can tune these based on machine RAM and expected image size.

If users mostly upload small crops, you can increase workers.
If users upload very large scenes, prioritize available memory over worker count.

## Suggested next production upgrades

- Redis-backed request/result cache
- background task queue
- object storage integration
- authentication and user workspaces
- request auditing
- cleanup policy for old uploads and results
