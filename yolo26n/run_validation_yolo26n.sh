#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Source files are UINT8 pixels in NCHW order. Never modify them in place.
INPUT_DIR="${INPUT_DIR:-validation_inputs}"
RESULT_DIR="${RESULT_DIR:-validation_npu_50cal_yolo26n_mixedprecision_i8}"
INT8_DIR="$RESULT_DIR/inputs_int8"
EXPECTED_COUNT="${EXPECTED_COUNT:-50}"

mkdir -p "$RESULT_DIR"

# Convert once before inference. A byte translation implements signed (u - 128)
# without requiring NumPy on the board; a reinterpret cast would be incorrect.
python3 - "$INPUT_DIR" "$INT8_DIR" "$EXPECTED_COUNT" <<'PYCONVERT'
from pathlib import Path
import sys

source, destination = map(Path, sys.argv[1:3])
expected = int(sys.argv[3])
inputs = sorted(source.glob("*.dat"))
if len(inputs) != expected or expected < 1:
    raise SystemExit(f"Expected {expected} UINT8 inputs in {source}, found {len(inputs)}")
if source.resolve() == destination.resolve():
    raise SystemExit("Source and converted-input directories must differ")
for path in inputs:
    if path.stat().st_size != 3 * 640 * 640:
        raise SystemExit(f"Wrong input size: {path}")
destination.mkdir(parents=True, exist_ok=True)
# Reject stale files from a different dataset instead of including them in evaluation.
extra = {p.name for p in destination.glob("*.dat")} - {p.name for p in inputs}
if extra:
    raise SystemExit(f"Stale converted inputs found: {sorted(extra)}; choose a new RESULT_DIR")
translation = bytes((u - 128) % 256 for u in range(256))
for path in inputs:
    (destination / path.name).write_bytes(path.read_bytes().translate(translation))
print(f"Converted {len(inputs)} UINT8 inputs to INT8 in {destination}")
PYCONVERT

count=0

for input in "$INT8_DIR"/*.dat; do
    name=$(basename "$input" .dat)
    count=$((count + 1))

    echo "[$count/$EXPECTED_COUNT] $name"

    # vpm_run expects this exact filename from sample.txt
    cp "$input" input_0.dat

    # Remove previous output so stale files can never be copied.
    rm -f output_0.txt output_1.txt

    ./vpm_run \
        -s sample_pc_i8.txt \
        -b 0 \
        --show_top5 0 \
        --save_txt 1 \
        > "$RESULT_DIR/${name}.log" 2>&1

    test -f output_0.txt
    test -f output_1.txt

    # Verify inference actually succeeded.
    grep -q "vpm run ret=0" "$RESULT_DIR/${name}.log" || {
        echo "ERROR: NPU run failed for $name"
        exit 1
    }

    mv output_0.txt "$RESULT_DIR/${name}_boxes.txt"
    mv output_1.txt "$RESULT_DIR/${name}_scores.txt"
done

echo
echo "Completed: $count images"

echo -n "boxes:  "
find "$RESULT_DIR" -name '*_boxes.txt' | wc -l

echo -n "scores: "
find "$RESULT_DIR" -name '*_scores.txt' | wc -l

echo -n "logs:   "
find "$RESULT_DIR" -name '*.log' | wc -l
