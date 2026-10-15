# OpenCV motion detection and Flask dashboard

A CPU-based USB-camera motion experiment using OpenCV MOG2 background subtraction, morphological filtering and contour area thresholds. This project uses neither ONNX nor the NPU.

## Implemented behavior

- Five-second background warmup and a 1,000-pixel contour area threshold.
- Motion events begin after two consecutive motion frames and finish after one second without motion.
- MJPEG AVI recordings and a peak-motion JPEG snapshot for each event.
- The continuous monitor and web app reconnect after camera failures and manage a 5 GiB recording budget with a 100 MiB reserve, deleting oldest event directories as needed. Use a dedicated recordings directory.
- The web app serves an annotated MJPEG live stream (up to 15 published frames/second), status polling, a paginated recording gallery, snapshot viewing and video downloads. Actual performance depends on the camera and CPU.

## Programs and layout

| File | Purpose |
| --- | --- |
| `motion_events.py` | Initial 30-second event capture experiment; writes to `events/` |
| `motion_monitor.py` | Continuous recorder with reconnect and retention; writes to `monitor/` |
| `motion_web.py` | Camera worker, Flask routes and recording logic; writes to `monitor/` |
| `templates/dashboard.html` | Dashboard structure and initial event rendering |
| `static/dashboard.css` | Responsive dashboard styles |
| `static/dashboard.js` | Status polling, gallery updates and pagination |

The web UI was extracted from the Python inline template into Flask's standard template/static layout. It works without an external CDN or frontend build step. Default output paths now resolve beside the scripts rather than requiring `/root/motion_test`.

## Install and run on the board

Use Python 3 with OpenCV's Linux V4L2 backend and Flask. If the board image provides OpenCV through the system package manager, use that build and install Flask into the same Python environment. Otherwise:

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 motion_web.py
```

Open `http://<board-ip>:5000` from a browser on the same trusted network. The app binds to all interfaces and has no authentication; it is intended for local network experiments. Run one process using the provided entry point: importing the Flask app alone does not start the camera worker. Do not enable the Flask reloader, which could open the camera twice.

Optional web configuration:

```sh
MOTION_DATA_DIR=/path/to/dedicated/recordings MOTION_CAMERA=0 MOTION_PORT=5000 python3 motion_web.py
```

`MOTION_CAMERA` is an integer OpenCV camera index (default 0). All three scripts support `MOTION_DATA_DIR`; the camera index and HTTP port overrides apply to the web app. Stop the web app with Ctrl+C to finalize the active event and release the camera. Run only one camera program at a time.

For the earlier experiments:

```sh
python3 motion_events.py
python3 motion_monitor.py
```

AVI videos are offered as downloads; inline browser video playback is not assumed. Recording metadata uses a nominal 20 FPS, not measured capture timestamps. Retention is periodic and is not a strict filesystem quota. Recordings, caches and virtual environments are excluded from publication by the root `.gitignore`.

## Camera-free checks

With Flask and OpenCV installed, run from this directory:

```sh
python3 -m unittest -v test_motion_web.py
```

These checks cover template/static delivery, event pagination and downloads, invalid requests, status and MJPEG framing without opening a camera. They do not verify physical camera capture or recording codecs.
