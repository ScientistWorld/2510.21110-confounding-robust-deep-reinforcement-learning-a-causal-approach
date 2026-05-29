#!/bin/bash
# Compute-node entry point for the default scoring audit.

set -euo pipefail

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
export EVAL_EPISODES="${EVAL_EPISODES:-2000}"

rm -f scoring/scores.json

bash scripts/evaluate.sh \
  checkpoints/conf_dqn_base_seed0.json \
  checkpoints/conf_dqn_base_seed1.json \
  checkpoints/conf_dqn_base_seed2.json \
  checkpoints/causal_dqn_base_seed0.json \
  checkpoints/causal_dqn_base_seed1.json \
  checkpoints/causal_dqn_base_seed2.json \
  checkpoints/conf_dqn_weaker_bias_seed0.json \
  checkpoints/conf_dqn_weaker_bias_seed1.json \
  checkpoints/conf_dqn_weaker_bias_seed2.json \
  checkpoints/causal_dqn_weaker_bias_seed0.json \
  checkpoints/causal_dqn_weaker_bias_seed1.json \
  checkpoints/causal_dqn_weaker_bias_seed2.json

python validate.py --compare
