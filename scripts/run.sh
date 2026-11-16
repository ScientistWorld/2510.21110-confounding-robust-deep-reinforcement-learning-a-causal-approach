#!/bin/bash
# Compute-node entry point: reproduce the reference end to end.
#   builds the splits if missing (they are committed), trains Causal-DQN and
#   the baselines (3 runs each, both experiments), scores train/test/dev and
#   validates. A GPU makes training take minutes instead of hours.
# This hardening job (iteration 13) also runs the hardening checks:
#   python -m redteam.hardening.suite train|cases|summary
# SKIP_EXISTING=1 lets a restarted job resume from finished runs.

set -euo pipefail

cd /home/user
export PYTHONPATH="/home/user"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

python -c "import torch, numpy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'numpy', numpy.__version__, flush=True)"

SKIP_EXISTING=1 bash scripts/reproduce.sh
python -m redteam.hardening.suite train
python -m redteam.hardening.suite cases
python -m redteam.hardening.suite summary
python validate.py --compare || true
