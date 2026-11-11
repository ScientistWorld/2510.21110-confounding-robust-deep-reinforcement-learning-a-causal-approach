#!/bin/bash
# Run the paper's proposed method (Causal-DQN) on every experiment.
#
# Trains independent runs k = 0 1 2 (seed = k) on the logged demonstrations of
# both the train and the test split, and writes
#   checkpoints/causal_dqn/<experiment>/run<k>.json
# Extra arguments are passed to method/causal_dqn.py.
#
# Usage:
#   bash scripts/method.sh [extra args]

set -euo pipefail

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

for EXP in ${EXPERIMENTS:-confounded_expert_demos pervasive_confounding}; do
  for RUN in ${RUNS:-0 1 2}; do
    if [ "${SKIP_EXISTING:-0}" = "1" ] && [ -f "checkpoints/causal_dqn/$EXP/run$RUN.json" ]; then continue; fi
    python method/causal_dqn.py --experiment "$EXP" --run "$RUN" "$@"
  done
done
