#!/usr/bin/env python3
"""PaDiM monitor: python3 padim_monitor/app.py --help."""
import argparse
import logging
import re
from pathlib import Path
import signal

from flask import Flask, Response, jsonify, render_template, request, send_file, url_for

if __package__:
    from .dashboard import EventStore, ThermalReader
    from .pipeline import MODES, Monitor, load_config
else:
    from dashboard import EventStore, ThermalReader
    from pipeline import MODES, Monitor, load_config


def create_app(monitor):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 1024
    events = EventStore(monitor.config["events_dir"])
    thermal = ThermalReader()

    @app.get("/")
    def dashboard():
        return render_template("index.html", width=monitor.config["width"], height=monitor.config["height"],
                               heatmap_min=monitor.config["heatmap_min"]+(monitor.config["heatmap_max"]-monitor.config["heatmap_min"])*monitor.config.get("heatmap_visible_fraction",.6), heatmap_max=monitor.config["heatmap_max"])

    @app.get("/video_feed")
    def video_feed():
        mode = request.args.get("mode", "heatmap")
        if mode not in MODES:
            return jsonify(error="Choose original, heatmap or side-by-side"), 400
        return Response(monitor.stream(mode), mimetype="multipart/x-mixed-replace; boundary=frame",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/frame.png")
    def diagnostic_frame():
        # Download the exact unannotated frame used for the current score.
        import time
        import cv2
        with monitor.condition:
            if not monitor._healthy(time.monotonic()):
                return jsonify(error="No fresh inference frame available"), 503
            frame = monitor.result[0].copy()
            score = monitor.result[2]
        ok, encoded = cv2.imencode(".png", frame)
        if not ok:
            return jsonify(error="Could not encode diagnostic frame"), 500
        return Response(encoded.tobytes(), mimetype="image/png", headers={
            "Cache-Control": "no-store", "Content-Disposition": 'attachment; filename="frame.png"',
            "X-Anomaly-Score": repr(score)})

    @app.get("/api/status")
    def status():
        payload = monitor.status()
        payload["temperatures"] = thermal.read()
        response = jsonify(payload)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/events")
    def event_list():
        try:
            offset = int(request.args.get("offset", 0))
            limit = int(request.args.get("limit", 12))
            before = request.args.get("before")
            if offset < 0 or not 1 <= limit <= 48:
                raise ValueError()
            if before is not None:
                if not re.fullmatch(r"[0-9]{8}_[0-9]{6}_[0-9]{6}", before):
                    raise ValueError()
        except ValueError:
            return jsonify(error="Invalid pagination parameters"), 400
        entries = events.entries()
        total = len(entries)
        if before:
            entries = [entry for entry in entries if entry["id"] < before]
        batch = entries[offset:offset+limit]
        for entry in batch:
            entry["files"] = {name: url_for("event_file", event_id=entry["id"], filename=name)
                              for name in ("original.jpg", "heatmap.jpg", "anomaly_map.npy", "event.json")}
        response = jsonify(events=batch, total=total, has_more=offset+limit < len(entries))
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/events/<event_id>/<filename>")
    def event_file(event_id, filename):
        try:
            path = events.file(event_id, filename)
        except ValueError:
            return jsonify(error="Invalid event or filename"), 400
        except FileNotFoundError:
            return jsonify(error="Event file not found"), 404
        return send_file(path, as_attachment=filename in ("anomaly_map.npy", "event.json"))

    @app.post("/api/reset_alarm")
    def reset_alarm():
        if not request.is_json:
            return jsonify(error="Send application/json"), 415
        return jsonify(monitor.reset_alarm())

    return app


def parse_config(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config")/"config.json")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--compute", choices=("npu", "cpu"), help="npu: NB/VIP; cpu: ONNX Runtime")
    selection.add_argument("--backend", choices=("nb", "onnx"), help="Legacy alias for compute selection")
    parser.add_argument("--stale-seconds", type=float, help="Maximum camera/result age; increase for slow CPU diagnostics")
    parser.add_argument("--model", type=Path, help="Override selected NB/ONNX model path")
    parser.add_argument("--library", type=Path, help="Override NB library path")
    parser.add_argument("--metadata", type=Path, help="Matching NB quantization metadata")
    parser.add_argument("--scoring", choices=("cpu","npu"), help="PaDiM distance calculation backend")
    parser.add_argument("--score-model", type=Path, help="FP16 scoring NB")
    parser.add_argument("--score-metadata", type=Path, help="Scoring NB metadata")
    parser.add_argument("--statistics", type=Path, help="FP32-fitted PaDiM statistics NPZ")
    parser.add_argument("--preprocessing", choices=("pil","neon"), help="Reference PIL or fast native bilinear resize")
    parser.add_argument("--threshold", type=float, help="Explicit image score threshold")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.preprocessing and args.preprocessing!=cfg["preprocessing"]:
        cfg["preprocessing"]=args.preprocessing
        cfg["threshold_calibrated"]=False
    if args.stale_seconds is not None:
        import math
        if not math.isfinite(args.stale_seconds) or args.stale_seconds <= 0:
            parser.error("--stale-seconds must be finite and positive")
        cfg["stale_seconds"] = args.stale_seconds
    if args.compute:
        cfg["backend"] = {"npu": "nb", "cpu": "onnx"}[args.compute]
    elif args.backend:
        cfg["backend"] = args.backend
    if cfg["backend"] == "onnx":
        cfg["threshold_calibrated"] = False
    if args.scoring and args.scoring!=cfg["scoring"]:
        cfg["scoring"]=args.scoring;cfg["threshold_calibrated"]=False
    if cfg["backend"]=="onnx":cfg["scoring"]="cpu"
    for name in ("metadata", "statistics", "score_model", "score_metadata"):
        value = getattr(args,name)
        if value:
            cfg[name] = str(value.resolve())
            cfg["threshold_calibrated"] = False
    if args.threshold is not None:
        import math
        if not math.isfinite(args.threshold):
            parser.error("--threshold must be finite")
        cfg["threshold"] = args.threshold
        cfg["threshold_calibrated"] = False
    if args.model:
        cfg["model" if cfg["backend"] == "nb" else "onnx_model"] = str(args.model.resolve())
        cfg["threshold_calibrated"] = False
    if args.library:
        cfg["library"] = str(args.library.resolve())
    return cfg


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = parse_config()
    selected = cfg["model" if cfg["backend"] == "nb" else "onnx_model"]
    logging.info("Compute=%s model=%s threshold=%.17g", "npu" if cfg["backend"] == "nb" else "cpu", selected, cfg["threshold"])
    logging.info("Preprocessing: shortest-side resize + center crop + external ImageNet normalization; source view=%s", cfg["roi"])
    if not cfg["threshold_calibrated"]:
        logging.warning("Threshold is not calibrated for this selected model; compare scores before changing it")
    # Bound OpenCV's CPU use so JPEG rendering does not monopolize all cores.
    import cv2
    cv2.setNumThreads(2)
    monitor = Monitor(cfg)
    app = create_app(monitor)

    def shutdown(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, shutdown)
    try:
        monitor.start()
        app.run(host=cfg["host"], port=cfg["port"], threaded=True, use_reloader=False)
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()
