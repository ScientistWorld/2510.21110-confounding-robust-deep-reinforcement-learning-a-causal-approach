#!/bin/bash
# Visible train-slice evaluation entry point.
#
# Mirrors scripts/evaluate.sh: pass policy JSON artifacts explicitly, or omit
# arguments to score every JSON policy artifact found under checkpoints/.

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

mkdir -p scoring

if [ "$#" -eq 0 ]; then
  mapfile -t ARTIFACTS < <(find checkpoints -maxdepth 1 -type f -name '*.json' | sort)
  if [ "${#ARTIFACTS[@]}" -eq 0 ]; then
    echo "No policy JSON artifacts found under checkpoints/." >&2
    exit 1
  fi
  set -- "${ARTIFACTS[@]}"
fi

python eval/train/evaluate_cmdp.py "$@" \
  --episodes "${EVAL_EPISODES:-2000}" \
  --seeds ${EVAL_TRAIN_SEEDS:-137 409 811} \
  --output scoring/scores_train.json
