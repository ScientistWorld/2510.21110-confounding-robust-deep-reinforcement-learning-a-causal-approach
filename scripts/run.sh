#!/bin/bash
# Compute-node entry point for a fresh end-to-end reproduction.

set -euo pipefail

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
export EVAL_EPISODES="${EVAL_EPISODES:-2000}"

rm -f scoring/scores.json

bash scripts/reproduce.sh

python validate.py --compare
