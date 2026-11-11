#!/bin/bash
# Held-out test-slice evaluation. Takes no arguments (any are ignored).
#
# Scores every method directory under checkpoints/ on the hidden test
# instances (true dynamics in eval/test/envs/) -> scoring/scores_test.json.
# Test size, instances, runs and normalisation are fixed by eval/test/.
# A method with missing or invalid policies receives the worst attainable
# value of every metric.

set -euo pipefail

cd /home/user
rm -f scoring/scores_test.json
mkdir -p scoring
python eval/test/evaluate_policies.py --checkpoints checkpoints --output scoring/scores_test.json
