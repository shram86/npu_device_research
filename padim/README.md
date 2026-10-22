# PaDiM anomaly monitor for Orange Pi 4 Pro

A USB-camera anomaly detector with INT8 ResNet-18 feature extraction and a separate FP16 PaDiM scoring network on the NPU. A Flask dashboard displays the live camera, localized anomaly heatmap, alarms, event gallery, CPU/NPU temperatures and detailed latency metrics.

The project was developed from experiment B: original FP32-fitted PaDiM statistics are used with quantized feature extraction, without refitting statistics on quantized features. Its historical AUROC was 0.990625, but the NB version used for that experiment was not recorded. The current feature model is v4; that historical accuracy must not be assumed for the current feature/scoring pair.

## Build and run

Run the following from the repository root (the directory containing `padim/`). Requirements: the Orange Pi's working NPU driver and vendor headers/libraries, GCC/OpenMP, a V4L2 USB camera, and the model assets listed below.

```sh
python3 -m pip install -r padim/requirements.txt

gcc -O2 -Wall -Wextra -fopenmp -fPIC -shared \
  -I/usr/include padim/padim_camera.c \
  -L/lib -L/usr/lib -lNBGlinker -lVIPhal -lm \
  -o padim/libpadim_camera.so

OMP_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE OMP_PROC_BIND=false \
OPENBLAS_NUM_THREADS=2 \
python3 padim/app.py --compute=npu --scoring=npu --preprocessing=neon
```

Open `http://<board-ip>:5000`. Run one process, without Flask's reloader or multiple worker processes. Stop other applications using the camera first. Ctrl+C/SIGTERM shuts down. Config paths resolve relative to `config/config.json`; CLI file paths resolve relative to the working directory.

Passive OpenMP waiting avoids idle workers competing with capture/readback/rendering. Binding threads or restricting the entire app with `taskset` previously produced regressions; benchmark complete pipeline latency before adding affinity. Native resizing is single-threaded with NEON; CPU distance scoring can use OpenMP.

## Required assets

| Local path | Purpose |
| --- | --- |
| `models/network_binary_v4.nb` | INT8 ResNet-18 feature graph |
| `models/nbg_meta.json` | Matching feature tensor/quantization metadata |
| `models/network_binary_score.nb` | FP16 full-covariance squared-distance graph |
| `models/score_meta.json` | Matching scoring tensor metadata |
| `models/padim_statistics.npz` | Saved channel indices; CPU comparison mean/precision |
| `models/resnet18_features.onnx` | Optional FP32 CPU feature reference |
| `models/assets.json` | File hashes and original development-workspace provenance |

NB/ONNX/NPZ assets are present locally but ignored by Git. A fresh source checkout needs them supplied separately. The statistics archive is about 118 MiB; use LFS or external/release storage rather than ordinary GitHub file storage. Original `build/...`, `resnet18/...` and experiment paths in the manifest describe the development workspace and are not directories shipped in this repository.

The feature NB and its metadata matched the development v4 build; the score NB matched the supplied scoring build by hash. Metadata/hash matching does not prove which statistics were embedded in the score model. Its mean/precision must correspond to the NPZ's selected channel indices and intended crop/preprocessing.

`install_nb.py` can install a different feature NB and metadata into a fresh destination, records hashes and refuses overwrites. It does not install/rebuild the scoring graph. Existing config/model filenames must be updated explicitly when selecting another pair.

## Implemented pipeline

1. **Camera and preprocessing:** request 1280×720 MJPG at 60 FPS with four capture buffers. Resize the shortest side to 256, take a position-configurable 256×256 crop, convert BGR→RGB and apply ImageNet mean/std. Normalize externally before signed INT8 input encoding.
2. **Feature graph:** input INT8 NCHW `[1,3,256,256]`, 196608 bytes. Three feature layers have CHW shapes `[64,64,64]`, `[128,32,32]` and `[256,16,16]`. C matches semantic layers by shape, independent of physical output order, and queries actual quantization.
3. **Selected readback in C:** Python loads `stats["indices"]` and calls `padim_set_channels` once. `padim_run_selected` invalidates/maps outputs, dequantizes only the saved channels and resizes them with half-pixel bilinear interpolation (`align_corners=False`). Saved channel order is preserved. For the supplied indices: 18/27/55 channels from layer1/2/3, decoding **115456 source values instead of 458752**. INT8 decoding uses a small FP32 lookup table. Python receives `[100,64,64]` and does not repeat selection/resizing.
4. **Scoring graph:** pack selected **features**, not mean-subtracted deltas, as FP16 NCHW `[1,100,64,64]` (819200 bytes). The graph embeds mean and full precision matrices, subtracts mean and computes squared Mahalanobis distances. Output is FP16 `[1,1,64,64]` (8192 bytes). Both networks share one initialized VIP runtime and execute sequentially.
5. **CPU finishing:** convert distances to FP32, reject nonfinite values, clamp negative roundoff, square-root, Gaussian-smooth with sigma 2, then take the smoothed map maximum as image score.

The C bridge validates shapes, storage, quantization and buffer sizes. NPU failures report inference unavailable; there is no silent CPU fallback. Full-feature `padim_run` remains for diagnostics, but the live NB path uses selected readback.

Native NEON preprocessing reuses its tensor buffer and avoids PIL conversion/intermediate resized images. Its point-bilinear interpolation differs from Pillow's antialiasing/uint8 rounding. Config defaults to `preprocessing: "pil"`; the launch command above explicitly selects NEON. Validate quality when changing preprocessing.

## Latest device timings

Live Orange Pi 4 Pro dashboard snapshot, **22 October 2026**, with native preprocessing, NPU features, selected C readback and NPU scoring:

| Stage | Latency |
| --- | ---: |
| RGB / resize / tensor | 1.87 ms |
| **Backend total** | **50.63 ms** |
| ↳ Input encode + flush | 0.84 ms |
| ↳ ResNet feature extraction (NPU) | 2.03 ms |
| ↳ Output read / selected decode / resize | 24.01 ms |
| ↳ Selected features / FP16 packing | 4.70 ms |
| ↳ Score input copy + flush | 0.38 ms |
| ↳ PaDiM scoring (NPU) | 14.79 ms |
| ↳ Score output read | 0.12 ms |
| ↳ Sqrt / smoothing / max | 2.76 ms |
| ↳ Python / C bridge | 1.01 ms |
| Output checks | 0.35 ms |

Backend subrows are included in the 50.63 ms total. Adding preprocessing and output checks gives approximately **52.85 ms**, or **18.9 processing cycles/s** arithmetically. These are snapshot values, not a benchmark mean or measured sustained FPS; capture, JPEG rendering and web transport are outside the sum. Rounding/measurement overhead explains small differences between sums. Readback/selected decoding/resizing remains the largest stage; its timing includes cache invalidation, mapping and unmapping, not just arithmetic.

## Crop, threshold and heatmap configuration

Edit `padim/config/config.json` and restart:

```json
"crop_position": {"x": 0.5, "y": 0.5},
"threshold": 25.0,
"threshold_calibrated": false,
"heatmap_min": 0.0,
"heatmap_max": 35.0,
"heatmap_visible_fraction": 0.6,
"heatmap_alpha": 0.5
```

Crop coordinates are fractions of available movement: x=0 left, 0.5 center, 1 right; y=0 top, 0.5 center, 1 bottom. For example x=0.25 moves the crop left. Crop size stays unchanged. The short dimension fills the crop and therefore has no movement: y has no effect on a landscape frame, x has no effect on a portrait frame. Python, NEON, diagnostics and the displayed crop rectangle use the same setting. Pixels outside that rectangle are not inspected.

PaDiM statistics are spatially dependent. Fit/validate using the same crop and rebuild the scoring NB if its embedded statistics change. Scores are raw smoothed distances, not probabilities. The configured threshold 25.0 is an uncalibrated starting value; choose a threshold from representative held-out device captures. Do not reuse an EfficientAD threshold.

The heatmap keeps a fixed score scale. Only stronger responses are overlaid yellow→orange→red; weaker areas retain original camera pixels. With range 0–35 and visible fraction 0.6, painting begins at score 21. This visibility cutoff only affects display, not anomaly scores or alarms. Alpha controls overlay opacity.

## Dashboard and events

Capture, inference, rendering and event recording use separate workers. Inference consumes the latest available frame; stale/disconnected results become UNKNOWN and reconnects discard obsolete in-flight results. The default web stream rate is 10 FPS, independently of camera and inference rates.

Three consecutive above-threshold inferences latch an alarm until Reset. Each new latched event saves the original image, heatmap JPEG, anomaly map NPY and JSON metadata. The gallery supports pagination and image/map/metadata downloads. No automatic event retention or disk cleanup is configured. CPU/NPU temperatures come from available sysfs sensors.

Endpoints: `/api/status`, `/api/frame.png`, `/api/events`, `/video_feed?mode=original|heatmap|side-by-side`, `POST /api/reset_alarm` with JSON `{}`, and `/events/<id>/<filename>`. The dashboard has no authentication; deploy on a trusted local network.

## CPU comparison and image diagnostics

Keep NPU feature extraction but compare CPU versus NPU distance scoring:

```sh
python3 padim/check_image.py normal.png missing.png --compute=npu --scoring=cpu --preprocessing=neon --output-dir /tmp/padim_cpu_maps
python3 padim/check_image.py normal.png missing.png --compute=npu --scoring=npu --preprocessing=neon --output-dir /tmp/padim_npu_maps
```

These commands print scores and stage timings and save maps. Use identical captures, feature NB and preprocessing to isolate the scoring difference. FP16 scoring can change numerical results or overflow; nonfinite output is rejected. Validate threshold/quality on the board.

For FP32 ONNX feature extraction and CPU scoring:

```sh
python3 -m pip install onnxruntime
OPENBLAS_NUM_THREADS=2 python3 padim/app.py --compute=cpu --stale-seconds=30
```

CPU ONNX mode uses Python feature selection/resizing. CPU scoring with NB features uses the selected C readback and native NEON/OpenMP distance calculation when available. CPU scoring loads about 164 MB of precision matrices into RAM; NPU scoring avoids that host-side precision load.

## Verification scope

Development tests covered reference preprocessing, configurable crop placement, interpolation, selected readback versus full decode, port permutation, FP16 packing, scoring postprocessing, graph lifetimes/errors, stale camera handling, alarm persistence and gallery behavior. C code was checked with strict compiler warnings and mock VIP fixtures. Real ONNX/statistics comparisons used the original experiment workspace.

The copied `tests/` retain development package names (`padim_monitor`) and external experiment fixtures; they are not yet a self-contained test suite runnable directly from this renamed publication folder. Device timing/accuracy validation is separate from mock tests. Build with the vendor header in `/usr/include`, never the test-only `tests/vip_lite.h`.
