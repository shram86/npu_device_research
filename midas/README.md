# MiDaS relative depth

MiDaS estimates relative depth from a single camera image and visualizes it over the live image. It does not provide calibrated metric distances. The experimental preference was **INT16** for depth quality, accepting the latency tradeoff.

Supplied variants: `network_binary_int16.nb` (the default) and `network_binary_mixed.nb`. The ONNX source and converter recipe are not present in this snapshot.

```sh
./midas_camera network_binary_int16.nb /dev/video0 0.6 --input-type int16
./midas_camera network_binary_mixed.nb /dev/video0 0.6 --input-type auto
```

The optional alpha is in the range 0–1. `--input-type auto|int8|int16` validates the model input type; it does not convert or requantize the network. Use `./midas_camera --help` for the argument format.

## Build and run

Build on the Orange Pi 4 Pro with its installed vendor NPU runtime:

```sh
gcc -O2 -Wall -fopenmp \
    -I/usr/include \
    midas_camera.c \
    -L/lib -L/usr/lib \
    -lNBGlinker -lVIPhal -lm \
    -o midas_camera
./midas_camera
```

Run from this directory; the default model is `network_binary_int16.nb` and the camera is `/dev/video0` (640×480 YUYV). Stop with Ctrl+C. For annotated video, use `./midas_camera` in the [shared FFmpeg streaming command](../README.md#stream-annotated-results). The program publishes video on fd 3; launching it directly is useful for diagnostics.

See the [research overview](../README.md) for quantization findings, ONNX-to-NB conversion limitations, dependencies and publication notes.
