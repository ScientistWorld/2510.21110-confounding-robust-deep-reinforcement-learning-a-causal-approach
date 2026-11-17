#!/bin/bash
# Compute-node entry point: reproduce the reference end to end.
#   builds the splits if missing (they are committed), trains Causal-DQN and
#   the baselines (3 runs each, both experiments), scores train/test/dev and
#   validates. A GPU makes training take minutes instead of hours (~25 min on
#   one A100 for everything).
# The hardening checks are separate (see HARDENING.md):
#   python -m redteam.hardening.suite train|cases|grid|summary

set -euo pipefail

cd /home/user
export PYTHONPATH="/home/user"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

python -c "import torch, numpy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'numpy', numpy.__version__, flush=True)"

bash scripts/reproduce.sh
python validate.py --compare
