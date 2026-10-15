# Orange Pi 4 Pro: NPU vision and camera research

Five working camera experiments developed on the Orange Pi 4 Pro: four vision networks running through the board's VIP/NBG runtime, and an OpenCV motion detector with a Flask dashboard.

## Final results

| Project | Task | Preferred quantization from device experiments |
| --- | --- | --- |
| [YOLO26n](yolo26n/README.md) | Object detection | Mixed INT8/INT16 |
| [MiDaS](midas/README.md) | Monocular relative depth | INT16 |
| [PWCNet](pwcnet/README.md) | Optical flow between consecutive frames | INT16 |
| [BlazePose](blazepose/README.md) | Pose landmarks | INT16 |
| [OpenCV motion detection](motion_detect_opencv/README.md) | Motion events, recordings and web monitoring | CPU; no NPU model |

All four networks were converted from ONNX to device-native `.nb` network binaries and integrated with a USB camera. Quantization was the main quality/latency tradeoff: the preferred settings above reflect the completed device experiments. INT8 was not the best choice for every network. No numerical benchmark report is included here, so these are qualitative findings rather than measured FPS or accuracy claims.

## Build on the device

Use Linux on the Orange Pi with its working NPU driver, vendor headers (including `vip_lite.h`), `libNBGlinker`, `libVIPhal`, GCC/OpenMP, and a V4L2 USB camera supporting **640×480 YUYV**. These are vendor runtime dependencies, not supplied by this repository. Compilation and NPU execution require the target environment.

Run from each model's directory, replacing the source and output names:

```sh
gcc -O2 -Wall -fopenmp \
    -I/usr/include \
    <network>_camera.c \
    -L/lib -L/usr/lib \
    -lNBGlinker -lVIPhal -lm \
    -o <output>
```

Each project README gives a concrete command. The default camera is `/dev/video0`. Run from the project directory so relative model paths resolve. Camera permissions must permit access to the device; inference does not inherently require a root shell.

## Stream annotated results

The C demos write raw **640×480 YUYV422** frames to file descriptor 3 when it is open; diagnostic output stays separate. With FFmpeg installed on the board, replace `<name>` with the executable (and any model arguments) and `<ip>` with the receiving host:

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

Allow UDP port 5000 on the receiver. CPU affinity assumes cores 6 and 7 exist and are available. FFmpeg's `-framerate 30` assigns timestamps to raw frames; it is **not** a claim that inference reaches 30 FPS. Adjust it to the measured output rate. The Flask motion dashboard uses HTTP/TCP port 5000 separately.

## ONNX → NB conversion status

The work completed for each network was: prepare/export ONNX, convert with the board's vendor conversion toolchain, compare quantization variants, and integrate the resulting tensor contracts into the camera application. This snapshot contains runtime sources and selected ONNX/NB artifacts, but **does not contain the original conversion commands, tool versions, calibration datasets or mixed-precision layer configuration**. It therefore supports running the supplied binaries, not fully reproducing their conversion.

Do not infer internal precision from a filename or input/output dtype alone. PWCNet's default file is named `network_binary_mixed.nb` despite the experimental preference for INT16; its camera code expects INT8 inputs and an asymmetric quantized flow output. Preserve that contract when replacing a model. Record the converter version, preprocessing, calibration set, layer precision and measured quality/latency for future conversions.

## Publication contents

Model artifacts are kept as ordinary files (largest artifact is below 40 MiB); Git LFS is optional. `.gitignore` excludes executable builds, recordings, Python caches, validation scratch files and the YOLO input scratch buffer. No camera recordings need to be published.

An inherited MMFlow Git checkout and its upstream CI/configuration were moved out of `pwcnet/` to the sibling `../npu_device_research_local_backup/` folder. They are local backup material, outside this publication root. Publish **this directory**, not its parent. The backup preserves the previous repository history.

This folder has no chosen project license. Model provenance, weight redistribution terms and the intended source license still need to be established by the owner before public release. No new ownership or licensing claims are made here.
