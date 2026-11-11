#!/bin/bash
# Compute-node job (iteration 12 hardening check):
#   build the committed splits if missing, reproduce the reference and the
#   baselines, then run the hardening suite (noise replicates, better/weaker
#   variants, shortcuts) through scripts/evaluate_test.sh.

set -euo pipefail

cd /home/user
export PYTHONPATH="/home/user"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export SKIP_EXISTING=1

nvidia-smi || true
python -c "import torch, numpy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'numpy', numpy.__version__, flush=True)"

python eval/train/build_train_split.py
python eval/test/build_test_split.py

# Fail-fast smoke test of every entry point.
rm -rf tmp/smoke && mkdir -p tmp/smoke/empty
python method/causal_dqn.py --experiment confounded_expert_demos --run 0 --updates 20 --log-interval 10 --output-root tmp/smoke/ckpt/causal
python baseline/confounded_dqn.py --experiment pervasive_confounding --run 0 --updates 20 --log-interval 10 --output-root tmp/smoke/ckpt/conf
python baseline/behavior_cloning.py --experiment confounded_expert_demos --output-root tmp/smoke/ckpt/bc
python eval/test/evaluate_policies.py --checkpoints tmp/smoke/ckpt --output tmp/smoke/test.json > tmp/smoke/test.log
python eval/train/evaluate_policies.py --checkpoints tmp/smoke/ckpt --output tmp/smoke/train.json > tmp/smoke/train.log
tail -n 5 tmp/smoke/test.log
rm -rf tmp/smoke

bash scripts/reproduce.sh

python -m redteam.hardening.suite train
python -m redteam.hardening.suite cases
python -m redteam.hardening.suite summary
python validate.py --compare || true
