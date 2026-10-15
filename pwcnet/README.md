# PWCNet optical flow

PWCNet estimates optical flow between consecutive camera frames using two 128×128 network inputs, then displays motion as a heatmap over the 640×480 stream. The experiments favored **INT16** for flow quality.

Artifacts: `pwcnet_128x128.onnx`, `network_binary.nb`, `network_binary_int8.nb`, and `network_binary_mixed.nb`. The application currently loads `network_binary_mixed.nb`; `network_binary.nb` has identical contents in this snapshot. The exact internal layer precision is not recorded, so the filename should not be treated as proof of an all-INT16 conversion. The application uses INT8 image inputs and requires asymmetric quantization for its flow output.

Runtime visualization options:

```sh
PWC_FLOW_OVERLAY=1 PWC_HEATMAP_MAX=8 PWC_HEATMAP_ALPHA=0.60 ./pwcnet_camera
```

`PWC_FLOW_OVERLAY=0` disables the overlay. `PWC_HEATMAP_MAX` controls the full-scale motion magnitude in camera pixels per frame pair, and `PWC_HEATMAP_ALPHA` controls maximum overlay opacity. Open fd 3 using the shared streaming command to see the visualization. Model and camera paths are compile-time constants (`NBG_FILE`, `CAMERA_DEVICE`); rebuild after changing them.

## Build and run

Build on the Orange Pi 4 Pro with its installed vendor NPU runtime:

```sh
gcc -O2 -Wall -fopenmp \
    -I/usr/include \
    pwcnet_camera.c \
    -L/lib -L/usr/lib \
    -lNBGlinker -lVIPhal -lm \
    -o pwcnet_camera
./pwcnet_camera
```

Run from this directory; the default model is `network_binary_mixed.nb` and the camera is `/dev/video0` (640×480 YUYV). Stop with Ctrl+C. For annotated video, use `./pwcnet_camera` in the [shared FFmpeg streaming command](../README.md#stream-annotated-results). The program publishes video on fd 3; launching it directly is useful for diagnostics.

See the [research overview](../README.md) for quantization findings, ONNX-to-NB conversion limitations, dependencies and publication notes.
