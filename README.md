# Orange Pi 4 Pro: NPU vision and camera research

Six USB-camera projects developed and tested on the Orange Pi 4 Pro: object detection, relative depth, optical flow, pose estimation, OpenCV motion monitoring and **PaDiM anomaly detection**. The neural pipelines use ONNX models converted to the board's native NB format and execute through the vendor VIP/NBG runtime.

## Projects and findings

| Project | What was implemented | Precision used / preferred in device experiments |
| --- | --- | --- |
| [YOLO26n](yolo26n/README.md) | Live object detection and annotated video | Mixed INT8/INT16 |
| [MiDaS](midas/README.md) | Monocular relative-depth visualization | INT16 |
| [PWCNet](pwcnet/README.md) | Optical flow between consecutive camera frames | INT16 preferred; deployed tensor contract documented in its README |
| [BlazePose](blazepose/README.md) | Live pose landmarks | INT16 |
| [OpenCV motion detection](motion_detect_opencv/README.md) | CPU motion detection, event recordings and Flask gallery | CPU; no NB model |
| [PaDiM](padim/README.md) | Live anomaly maps, alarms, event gallery and device metrics | INT8 ResNet-18 features + a separate FP16 scoring graph |

The first four networks established the ONNX → NB conversion and USB-camera workflows. Quantization was a quality/latency tradeoff: mixed precision worked best for YOLO, while INT16 was preferred for MiDaS, PWCNet and BlazePose. Those preferences are experimental findings, not a common numerical accuracy benchmark across models. External tensor dtype alone does not establish internal layer precision.

## PaDiM: completed pipeline

The new `padim/` project extends the Flask monitoring approach with a two-network anomaly detector:

1. Capture BGR camera frames; resize and crop to 256×256, convert to RGB and apply ImageNet normalization. The optional native NEON path uses reusable buffers.
2. Run an INT8 ResNet-18 feature extractor on the NPU.
3. Read only the **100 saved PaDiM channels**. C code performs dequantization and bilinear resizing to 64×64 in saved channel order. With the supplied statistics, this decodes **115,456 source values instead of 458,752**; Python does not repeat selection/resizing.
4. Pack selected features as FP16 `[1,100,64,64]` and run a second NB graph for full-covariance Mahalanobis squared distances. Mean and precision matrices are embedded in that graph.
5. On CPU, clamp negative roundoff, take square roots, Gaussian-smooth with sigma 2, and use the map maximum as the anomaly score.

The monitor includes original/heatmap/side-by-side views, configurable crop position, yellow-to-red overlays only on stronger responses, persistent latched alarms, event snapshots/gallery, CPU/NPU temperatures and stage timings. CPU ONNX inference and CPU scoring remain available for comparison. See the [PaDiM README](padim/README.md) for exact contracts, configuration, build and launch commands.

### Latest PaDiM device timings

Live dashboard snapshot supplied from the Orange Pi 4 Pro on **22 October 2026**, using native preprocessing, NPU feature extraction, selected-channel C readback and NPU scoring:

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

Backend rows marked ↳ are parts of **Backend total**; do not add that total a second time. Preprocessing + backend + output checks is approximately **52.85 ms**, equivalent to about **18.9 processing cycles/s** by arithmetic. This is a live timing snapshot, not a warmed-up benchmark average, measured sustained FPS or camera-to-browser latency. Capture, JPEG rendering, HTTP transport and browser display are outside this sum. Small discrepancies between stage sums and totals result from rounding and timing overhead.

The remaining largest stage is output readback/selected decoding/resizing, not either NPU execution. Scoring previously ran on CPU; moving its quadratic-distance operation to the second NPU graph removed that CPU bottleneck. Historical experiment-B AUROC was **0.990625**, but its feature NB build was not recorded; it must not be attributed to this exact deployed pair without a new evaluation.

## Runtime and build requirements

Use the Orange Pi Linux image with a working NPU driver, vendor `vip_lite.h`, `libNBGlinker`, `libVIPhal`, GCC/OpenMP and a V4L2 USB camera. Vendor runtime and conversion tools are not supplied here. Camera permissions must allow access to `/dev/video0`; inference does not inherently require a root shell.

The four standalone C demos use **640×480 YUYV**. From the relevant project directory:

```sh
gcc -O2 -Wall -fopenmp \
  -I/usr/include <network>_camera.c \
  -L/lib -L/usr/lib -lNBGlinker -lVIPhal -lm \
  -o <output>
```

Each project README provides its concrete command and model arguments. PaDiM instead builds a shared C library and uses a Flask/OpenCV frontend; its default camera configuration requests **1280×720 MJPG at 60 FPS with four buffers**. Requested camera FPS, inference throughput and web stream FPS are separate metrics.

## Stream annotated results

The four standalone C demos (YOLO26n, MiDaS, PWCNet and BlazePose) write raw **640×480 YUYV422** frames to file descriptor 3 when it is open; diagnostic output stays separate. With FFmpeg installed on the board, replace `<name>` with the executable (and any model arguments) and `<ip>` with the receiving host:

```sh
OMP_NUM_THREADS=2 \
OMP_PROC_BIND=true \
OMP_PLACES=cores \
taskset -c 6,7 \
./<name> 3>&1 1>&2 | ffmpeg \
  -loglevel warning \
  -f rawvideo \
  -pixel_format yuyv422 \
  -video_size 640x480 \
  -framerate 30 \
  -i - \
  -an \
  -c:v libx264 \
  -preset ultrafast \
  -tune zerolatency \
  -pix_fmt yuv420p \
  -f mpegts \
  "udp://<ip>:5000?pkt_size=1316"
```

On the receiving host:

```sh
ffplay -fflags nobuffer -flags low_delay 'udp://0.0.0.0:5000?fifo_size=1000000&overrun_nonfatal=1'
```

Allow UDP port 5000 on the receiver. CPU affinity assumes cores 6 and 7 exist and are available. FFmpeg's `-framerate 30` assigns timestamps to raw frames; it is **not** a claim that inference reaches 30 FPS. Adjust it to the measured output rate. The motion and PaDiM Flask dashboards use HTTP/TCP port 5000 separately; run them on different ports if needed.

## Conversion and reproducibility

The completed work includes ONNX export/conversion, quantization comparisons, device-native integration and camera visualization. This publication folder contains runtime sources and selected local model artifacts, but not the complete training datasets, original converter commands/versions, calibration datasets or mixed-precision layer recipes. The PaDiM score exporter and original experiment workspace referenced in provenance records are outside this publication folder.

Preserve preprocessing, channel order, statistics, model metadata and tensor contracts when replacing a model. For PaDiM, the score graph's embedded statistics must match the selected indices. INT8 feature extraction and FP16 scoring are separate graph contracts. Validate scores and thresholds after a model, preprocessing or crop change.

## Repository contents and model assets

Publish this directory as the repository root. Per-project READMEs describe sources, required assets and launch commands. Generated shared libraries, executable builds, Python caches, logs and recordings are excluded by `.gitignore` rules.

PaDiM's local `models/` directory contains the feature NB, score NB, FP32 ONNX reference, statistics, metadata and hash/provenance records. **Its NB/ONNX/NPZ files are ignored by `padim/.gitignore` and will not be included by an ordinary `git add`.** Supply them separately or choose an explicit model distribution policy. The statistics archive is approximately **118 MiB**, exceeding GitHub's ordinary 100 MiB per-file limit; use Git LFS or release/download storage if publishing it. JSON metadata and asset manifests remain publishable as source files.

No project license has been selected. Source/model provenance and redistribution terms remain for the owner to establish before public release.
