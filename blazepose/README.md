# BlazePose pose landmarks

The camera application runs BlazePose lite on a centered 480×480 crop resized to 256×256, and overlays pose landmarks on the original frame. This is a centered pose demo; keep the subject inside the indicated crop.

The experimental preference was **INT16**. The supplied `network_binary.nb` must match the checked tensor contract: NHWC 1×256×256×3 INT16 DFP input with fixed-point position 15, and the expected INT16 outputs. Preprocessing normalizes RGB to [0, 1]. The ONNX source and converter recipe are not included.

```sh
./blazepose_camera network_binary.nb /dev/video0
POSE_THRESHOLD=0.5 LANDMARK_THRESHOLD=0.5 ./blazepose_camera
```

The positional arguments override the model and camera. Environment variables control pose and landmark confidence thresholds (default 0.5).

## Build and run

Build on the Orange Pi 4 Pro with its installed vendor NPU runtime:

```sh
gcc -O2 -Wall -fopenmp \
    -I/usr/include \
    blazepose_camera.c \
    -L/lib -L/usr/lib \
    -lNBGlinker -lVIPhal -lm \
    -o blazepose_camera
./blazepose_camera
```

Run from this directory; the default model is `network_binary.nb` and the camera is `/dev/video0` (640×480 YUYV). Stop with Ctrl+C. For annotated video, use `./blazepose_camera` in the [shared FFmpeg streaming command](../README.md#stream-annotated-results). The program publishes video on fd 3; launching it directly is useful for diagnostics.

See the [research overview](../README.md) for quantization findings, ONNX-to-NB conversion limitations, dependencies and publication notes.
