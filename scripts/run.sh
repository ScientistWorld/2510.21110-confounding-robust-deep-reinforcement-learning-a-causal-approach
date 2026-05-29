#!/bin/bash
# Compute-node entry point for the train/test evaluation split.

set -euo pipefail

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
export EVAL_EPISODES="${EVAL_EPISODES:-2000}"

rm -f scoring/scores_train.json scoring/scores_test.json

bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
