#!/bin/bash
# Reproduce the reference results end to end (CPU works; a GPU is much faster).
#
#  1. build the train/test splits if they are missing (they are committed);
#  2. train Causal-DQN (runs 0-2) and the baselines on both splits;
#  3. score: train slice, hidden test slice, development scores.json.

set -euo pipefail

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

python eval/train/build_train_split.py
if [ -f eval/test/build_test_split.py ]; then
  python eval/test/build_test_split.py
fi

bash scripts/method.sh
bash scripts/baseline.sh all

bash scripts/evaluate_train.sh
if [ -d eval/test ]; then
  bash scripts/evaluate_test.sh
fi
bash scripts/evaluate.sh
