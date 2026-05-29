#!/bin/bash
# Held-out test-slice evaluation entry point.
#
# Mirrors scripts/evaluate.sh: pass policy JSON artifacts explicitly, or omit
# arguments to score every JSON policy artifact found under checkpoints/.
# When scoring/scores_train.json exists, eval/test filters to the same
# experiment/method entries and fails if any train-scored entry lacks an
# artifact on disk.

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

python eval/test/evaluate_cmdp.py "$@" \
  --episodes "${EVAL_EPISODES:-2000}" \
  --seeds ${EVAL_TEST_SEEDS:-1499 2137 3779} \
  --output scoring/scores_test.json \
  --require-methods-from scoring/scores_train.json
