#!/bin/bash
# Start a training run in the background (expects a virtualenv in .venv at the repository root).
#   tools/b100/train.sh <name> <engine_dir> <gpu> [extra rl.train args...]
# Logs to runs/<name>.log; resumes automatically if runs/<name>/ckpt exists.
set -euo pipefail
NAME=$1; ENGINE=$2; GPU=$3; shift 3
cd "$(dirname "$0")/../.."
mkdir -p runs
CUDA_VISIBLE_DEVICES=$GPU setsid nohup .venv/bin/python -m rl.train --name "$NAME" --resume auto \
  --engine real --engine_dir "$ENGINE" "$@" >> "runs/$NAME.log" 2>&1 < /dev/null &
echo "started $NAME on GPU $GPU (engine $ENGINE)"
