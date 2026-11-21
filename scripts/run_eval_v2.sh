#!/usr/bin/env bash
# Sequential driver for the v2 full evaluation (metrics then full-split loss).
set -u
cd "$(dirname "$0")/.."
P=./venv/Scripts/python.exe
L=docs/eval_v2
mkdir -p "$L"

run () {  # run <logfile> <args...>
  log="$1"; shift
  echo "=== $(date +%H:%M:%S) start: $* ==="
  "$P" -u src/evaluate_v2_full.py "$@" > "$L/$log" 2>&1
  echo "=== $(date +%H:%M:%S) exit $? : $log ==="
}

run log_metrics_v2_val.txt  --phase metrics --model v2 --split val --max-new 50
run log_metrics_v1_val.txt  --phase metrics --model v1 --split val --max-new 50
run log_metrics_v2_test.txt --phase metrics --model v2 --split test --max-new 50
run log_metrics_v1_test.txt --phase metrics --model v1 --split test --max-new 50
run log_loss_v1_128.txt     --phase loss --model v1 --block 128
run log_loss_v2_128.txt     --phase loss --model v2 --block 128
run log_loss_v1_64.txt      --phase loss --model v1 --block 64 --max-windows 400
echo "=== ALL DONE $(date +%H:%M:%S) ==="
