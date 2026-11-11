#!/bin/bash
set -euo pipefail
cd /home/user
if [ ! -f redteam/reference/artifacts/conf_base_seed0.json ]; then
  bash redteam/reference/run.sh
fi
python -m redteam.harness shortcut_data_reward
