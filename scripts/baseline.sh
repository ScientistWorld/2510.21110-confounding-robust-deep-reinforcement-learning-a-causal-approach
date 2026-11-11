#!/bin/bash
# Run a baseline on every experiment (runs k = 0 1 2).
#
#   conf_dqn          the paper's confounded DQN baseline (standard DQN loss on
#                     the logged action, same data/scale as the reference)
#   behavior_cloning  most frequent logged action per state (not in the paper;
#                     simplest reward-free learner, kept as a sanity floor)
#
# Usage:
#   bash scripts/baseline.sh <conf_dqn|behavior_cloning|all>

set -euo pipefail

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

NAME="${1:-all}"
EXPS="${EXPERIMENTS:-confounded_expert_demos pervasive_confounding}"

run_conf() {
  for EXP in $EXPS; do
    for RUN in ${RUNS:-0 1 2}; do
      if [ "${SKIP_EXISTING:-0}" = "1" ] && [ -f "checkpoints/conf_dqn/$EXP/run$RUN.json" ]; then continue; fi
      python baseline/confounded_dqn.py --experiment "$EXP" --run "$RUN"
    done
  done
}
run_bc() {
  for EXP in $EXPS; do
    python baseline/behavior_cloning.py --experiment "$EXP" --runs 3
  done
}

case "$NAME" in
  conf_dqn) run_conf ;;
  behavior_cloning) run_bc ;;
  all) run_conf; run_bc ;;
  *) echo "Unknown baseline: $NAME" >&2; exit 2 ;;
esac
