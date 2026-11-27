#!/bin/bash
# Follow a training run with the eval watcher (archives every N-th checkpoint, evaluates on held-out seeds).
#   tools/b100/eval.sh <name> <engine_dir> <gpu> [every=100] [n_seeds=200] [--sample] [ascension]
set -euo pipefail
NAME=$1; ENGINE=$2; GPU=$3; EVERY=${4:-100}; N=${5:-200}; EXTRA=${6:-}
ASC=()
if [[ -n ${7:-} ]]; then ASC=(--ascension "$7"); fi
cd "$(dirname "$0")/../.."
setsid nohup .venv/bin/python -m rl.eval_watch --run_dir "runs/$NAME" --every "$EVERY" --n_seeds "$N" \
  --engine_dir "$ENGINE" --device "cuda:$GPU" $EXTRA "${ASC[@]}" >> "runs/$NAME.eval.log" 2>&1 < /dev/null &
echo "eval watcher for $NAME on GPU $GPU"
