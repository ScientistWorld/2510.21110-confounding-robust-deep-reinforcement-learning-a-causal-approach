#!/bin/bash
# Compute-node entry point: reproduce the reference end to end, then rerun the
# hardening checks (SKIP_EXISTING=1 reuses finished training runs).

set -euo pipefail

cd /home/user
export PYTHONPATH="/home/user"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"

python -c "import torch, numpy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'numpy', numpy.__version__, flush=True)"

bash scripts/reproduce.sh

python -m redteam.hardening.suite train
python -m redteam.hardening.suite cases
python -m redteam.hardening.suite summary
python validate.py --compare
