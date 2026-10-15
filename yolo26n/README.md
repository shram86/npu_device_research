# YOLO26n object detection

The camera pipeline resizes/letterboxes 640×480 camera frames to a 640×640 network input, runs detection on the NPU and draws detections on the camera image. The final experimental preference was **mixed INT8/INT16 precision**, balancing latency and detection quality.

`yolo26n_direct_outputs.onnx` is the ONNX reference; `yolo_direct_calibrate.nb` is the deployed network. `input_0.dat` is local scratch data and is ignored by Git; the camera application does not read it.

### Validation utilities

`run_validation_yolo26n.sh` converts UINT8 NCHW calibration/validation inputs to signed INT8 using `u - 128`, runs the vendor sample runner, and saves boxes/scores and logs. It requires external `vpm_run`, `sample_pc_i8.txt` configured for the intended model, and `validation_inputs/*.dat` (50 files by default, each 3×640×640 bytes). Those supporting files are not included. The script overwrites local `input_0.dat` and removes previous `output_0.txt`/`output_1.txt` before each run.

```sh
bash run_validation_yolo26n.sh
```

`evalute_validation_yolo26n.py` (original filename retained) compares saved NPU results with ONNX predictions using the same converted inputs. This is a comparison with the ONNX reference, not ground-truth dataset accuracy.

On a host with ONNX Runtime:

```sh
python3 -m pip install -r requirements-validation.txt
python3 evalute_validation_yolo26n.py --help
python3 evalute_validation_yolo26n.py
```

Input and output paths and expected count can be overridden through the script options/environment variables. The C application uses compile-time model and camera paths; edit `NBG_FILE` or `CAMERA_DEVICE` and rebuild to change them.

## Build and run

Build on the Orange Pi 4 Pro with its installed vendor NPU runtime:

```sh
gcc -O2 -Wall -fopenmp \
    -I/usr/include \
    yolo_camera.c \
    -L/lib -L/usr/lib \
    -lNBGlinker -lVIPhal -lm \
    -o yolo_camera
./yolo_camera
```

Run from this directory; the default model is `yolo_direct_calibrate.nb` and the camera is `/dev/video0` (640×480 YUYV). Stop with Ctrl+C. For annotated video, use `./yolo_camera` in the [shared FFmpeg streaming command](../README.md#stream-annotated-results). The program publishes video on fd 3; launching it directly is useful for diagnostics.

See the [research overview](../README.md) for quantization findings, ONNX-to-NB conversion limitations, dependencies and publication notes.
