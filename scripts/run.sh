#!/bin/bash
# TEMPORARY (hardening iteration 13): rescore-only follow-up. Adds a grid of
# honest tabular variants (scored objective gamma = 1, Hurwicz weights) to the
# hidden-test cases and rebuilds the summary. No retraining.
# Restored to the end-to-end reproduction entry point afterwards.
set -euo pipefail
cd /home/user
export PYTHONPATH="/home/user"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
python -m redteam.hardening.suite grid
python -m redteam.hardening.suite summary
