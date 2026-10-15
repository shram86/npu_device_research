import argparse
from pathlib import Path
import numpy as np
import onnxruntime as ort

BASE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Compare ONNX with NPU results using the saved INT8 input bytes")
parser.add_argument("--model", type=Path, default=BASE / "yolo26n_direct_outputs.onnx")
parser.add_argument("--npu-dir", type=Path, default=BASE / "validation_npu_50cal_yolo26n_mixedprecision_i8")
parser.add_argument("--input-dir", type=Path, help="INT8 inputs; defaults to NPU_DIR/inputs_int8 (not original UINT8 files)")
parser.add_argument("--expected-count", type=int, default=50)
args = parser.parse_args()
MODEL = str(args.model)
NPU_DIR = args.npu_dir
INPUT_DIR = args.input_dir if args.input_dir is not None else NPU_DIR / "inputs_int8"


NMS_IOU = 0.45
MATCH_IOU = 0.50

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def xywh_to_xyxy(b):
    r = np.empty_like(b)
    r[:, 0] = b[:, 0] - b[:, 2] / 2
    r[:, 1] = b[:, 1] - b[:, 3] / 2
    r[:, 2] = b[:, 0] + b[:, 2] / 2
    r[:, 3] = b[:, 1] + b[:, 3] / 2
    return r


def iou_one(box, boxes):
    if len(boxes) == 0:
        return np.empty(0, dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = (
        np.maximum(0, x2 - x1) *
        np.maximum(0, y2 - y1)
    )

    area1 = (
        np.maximum(0, box[2] - box[0]) *
        np.maximum(0, box[3] - box[1])
    )

    area2 = (
        np.maximum(0, boxes[:, 2] - boxes[:, 0]) *
        np.maximum(0, boxes[:, 3] - boxes[:, 1])
    )

    return inter / (area1 + area2 - inter + 1e-9)


def detect(boxes, scores, conf_threshold):
    boxes = boxes[0].T
    scores = scores[0].T

    cls = np.argmax(scores, axis=1)
    conf = np.max(scores, axis=1)

    keep = conf > conf_threshold

    boxes = xywh_to_xyxy(boxes[keep])
    conf = conf[keep]
    cls = cls[keep]

    selected = []

    # Class-aware NMS
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        idx = idx[np.argsort(conf[idx])[::-1]]

        while len(idx):
            best = idx[0]
            selected.append(best)

            if len(idx) == 1:
                break

            rest = idx[1:]
            ious = iou_one(boxes[best], boxes[rest])
            idx = rest[ious <= NMS_IOU]

    if not selected:
        return []

    selected = np.asarray(selected, dtype=np.int64)
    selected = selected[np.argsort(conf[selected])[::-1]]

    return [
        {
            "class": int(cls[i]),
            "conf": float(conf[i]),
            "box": boxes[i]
        }
        for i in selected
    ]


def match(cpu, npu):
    """
    Greedy matching:
      same class
      highest IoU
      IoU >= MATCH_IOU
    """

    candidates = []

    for ci, c in enumerate(cpu):
        for ni, n in enumerate(npu):
            if c["class"] != n["class"]:
                continue

            iou = float(
                iou_one(
                    c["box"],
                    np.asarray([n["box"]])
                )[0]
            )

            if iou >= MATCH_IOU:
                candidates.append((iou, ci, ni))

    # Highest IoU first
    candidates.sort(reverse=True)

    used_cpu = set()
    used_npu = set()
    matches = []

    for iou, ci, ni in candidates:
        if ci in used_cpu or ni in used_npu:
            continue

        used_cpu.add(ci)
        used_npu.add(ni)

        matches.append(
            (
                cpu[ci],
                npu[ni],
                iou
            )
        )

    return matches, len(cpu) - len(used_cpu), len(npu) - len(used_npu)


# ------------------------------------------------------------
# ORT
# ------------------------------------------------------------

# ------------------------------------------------------------
# ORT + load outputs ONCE
# ------------------------------------------------------------

sess = ort.InferenceSession(
    MODEL,
    providers=["CPUExecutionProvider"]
)

inputs = sorted(INPUT_DIR.glob("*.dat"))
if len(inputs) != args.expected_count or args.expected_count < 1:
    raise RuntimeError(f"Expected {args.expected_count} saved INT8 inputs in {INPUT_DIR}, found {len(inputs)}")

cached = []

print(f"Loading/running {len(inputs)} images once...")

for number, path in enumerate(inputs, 1):
    name = path.stem

    q = np.fromfile(path, dtype=np.int8)
    if q.size != 3 * 640 * 640:
        raise RuntimeError(f"Wrong input size: {path}: {q.size}")

    # Undo the NBG input zero point (-128) after widening, avoiding INT8 overflow.
    # This exactly reconstructs the original UINT8 pixels normalized by 255.
    x = (q.reshape(1, 3, 640, 640).astype(np.float32) + 128.0) / 255.0

    cpu_boxes, cpu_scores = sess.run(
        None,
        {"images": x}
    )

    # vpm_run --save_txt already dequantizes outputs: do not scale them again.
    npu_boxes = np.loadtxt(
        NPU_DIR / f"{name}_boxes.txt",
        dtype=np.float32
    ).reshape(1, 4, 8400)

    npu_scores = np.loadtxt(
        NPU_DIR / f"{name}_scores.txt",
        dtype=np.float32
    ).reshape(1, 80, 8400)

    cached.append(
        (cpu_boxes, cpu_scores, npu_boxes, npu_scores)
    )

    print(f"\r{number}/50", end="", flush=True)

print("\n")

# ------------------------------------------------------------
# Threshold sweep
# ------------------------------------------------------------

thresholds = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
]

print(
    "CONF   CPU   NPU   MATCH  CPU_ONLY  NPU_ONLY  "
    "CPU_MATCH  NPU_MATCH  MEAN_IOU  CONF_ERR"
)

for threshold in thresholds:

    total_cpu = 0
    total_npu = 0
    total_matches = 0
    total_missing = 0
    total_extra = 0

    all_ious = []
    all_conf_errors = []

    for (
        cpu_boxes,
        cpu_scores,
        npu_boxes,
        npu_scores,
    ) in cached:

        cpu_det = detect(
            cpu_boxes,
            cpu_scores,
            threshold
        )

        npu_det = detect(
            npu_boxes,
            npu_scores,
            threshold
        )

        matches, missing, extra = match(
            cpu_det,
            npu_det
        )

        total_cpu += len(cpu_det)
        total_npu += len(npu_det)
        total_matches += len(matches)
        total_missing += missing
        total_extra += extra

        for c, n, iou in matches:
            all_ious.append(iou)
            all_conf_errors.append(
                abs(c["conf"] - n["conf"])
            )

    cpu_rate = (
        100 * total_matches / total_cpu
        if total_cpu else 0
    )

    npu_rate = (
        100 * total_matches / total_npu
        if total_npu else 0
    )

    mean_iou = (
        np.mean(all_ious)
        if all_ious else float("nan")
    )

    mean_conf_error = (
        np.mean(all_conf_errors)
        if all_conf_errors else float("nan")
    )

    print(
        f"{threshold:4.2f} "
        f"{total_cpu:5d} "
        f"{total_npu:5d} "
        f"{total_matches:7d} "
        f"{total_missing:9d} "
        f"{total_extra:9d} "
        f"{cpu_rate:8.2f}% "
        f"{npu_rate:8.2f}% "
        f"{mean_iou:9.4f} "
        f"{mean_conf_error:9.4f}"
    )