#!/bin/bash
# Visible train-slice evaluation: scores every method directory under
# checkpoints/ on the public training instances -> scoring/scores_train.json.

set -euo pipefail

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
mkdir -p scoring
python eval/train/evaluate_policies.py --output scoring/scores_train.json
