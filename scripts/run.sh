#!/bin/bash
# TEMPORARY (hardening iteration 13): family-design exploration job.
# Restored to the end-to-end reproduction entry point afterwards.
set -euo pipefail
cd /home/user
export PYTHONPATH="/home/user"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
python -c "import torch, numpy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'numpy', numpy.__version__, flush=True)"
python -m redteam.hardening.explore 256
