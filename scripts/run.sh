#!/bin/bash
# Example job script submitted via the end-of-turn YAML block.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Make sure all data, code, and dependencies are already in the
# workspace or baked into the container before submitting.

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
export DEMO_EPISODES="${DEMO_EPISODES:-500}"
export UPDATES="${UPDATES:-1000}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export EVAL_EPISODES="${EVAL_EPISODES:-1000}"
export LOG_INTERVAL="${LOG_INTERVAL:-100}"
export SEEDS="${SEEDS:-0 1 2}"
export VARIANTS="${VARIANTS:-base weaker_bias}"

bash scripts/reproduce.sh
python validate.py --compare
