#!/bin/bash
# Reproduce the paper's results end-to-end.
#
# This script runs inside the compute container and should call
# scripts/baseline.sh and scripts/method.sh to run baselines and
# the paper's method, each of which calls scripts/evaluate.sh.
#
# Assumes:
#   - Container is built (environment/container.sif)
#   - Data and models are downloaded (via scripts/download.sh)
#   - Working directory is /home/user
#
# This is the "gold standard" reproduction — 
# You do not need to run this, but keep this in good shape
# so other agents can easily run a full loop of reproduction and get the paper's numbers.

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

mkdir -p checkpoints scoring

ARTIFACTS=()
for RUN_VARIANT in ${VARIANTS:-base}; do
  export VARIANT="$RUN_VARIANT"
  for RUN_SEED in ${SEEDS:-0}; do
    export SEED="$RUN_SEED"
    CONF_OUT="checkpoints/conf_dqn_${RUN_VARIANT}_seed${RUN_SEED}.json"
    CAUSAL_OUT="checkpoints/causal_dqn_${RUN_VARIANT}_seed${RUN_SEED}.json"
    OUTPUT="$CONF_OUT" EVALUATE_AFTER=0 bash scripts/baseline.sh conf_dqn
    OUTPUT="$CAUSAL_OUT" EVALUATE_AFTER=0 bash scripts/method.sh
    ARTIFACTS+=("$CONF_OUT" "$CAUSAL_OUT")
  done
done

if [ -f checkpoints/conf_dqn_base_seed0.json ]; then
  cp checkpoints/conf_dqn_base_seed0.json checkpoints/conf_dqn_policy.json
fi
if [ -f checkpoints/causal_dqn_base_seed0.json ]; then
  cp checkpoints/causal_dqn_base_seed0.json checkpoints/causal_dqn_policy.json
fi

bash scripts/evaluate.sh "${ARTIFACTS[@]}"
