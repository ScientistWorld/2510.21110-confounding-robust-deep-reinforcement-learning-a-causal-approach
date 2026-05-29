#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

python method/causal_dqn.py \
  --output checkpoints/causal_dqn_policy.json \
  --seed "${SEED:-0}" \
  --demo-episodes "${DEMO_EPISODES:-500}" \
  --updates "${UPDATES:-1000}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --log-interval "${LOG_INTERVAL:-100}"

if [ "${EVALUATE_AFTER:-1}" = "1" ]; then
  bash scripts/evaluate.sh checkpoints/causal_dqn_policy.json
fi
