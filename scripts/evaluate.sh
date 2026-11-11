#!/bin/bash
# Development evaluation: score every method under checkpoints/ on the public
# training instances and write scoring/scores.json (same numbers as
# scripts/evaluate_train.sh). Uses only eval/; never imports method/.
#
# Method output contract (any method):
#   checkpoints/<method>/<experiment>/run<k>.json, k = 0 1 2, each containing
#   {"policies": {"<instance_id>": [one action id per observed state], ...}}
# for every instance id of both splits.

set -euo pipefail

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
mkdir -p scoring
python eval/train/evaluate_policies.py --output scoring/scores.json
