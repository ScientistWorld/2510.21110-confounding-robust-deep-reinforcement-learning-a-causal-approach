#!/bin/bash
# Download all data and models needed for this environment.
#
# Estimated total download size: 0 GB
# Estimated disk usage after extraction: 0 GB
#
# Nothing is fetched: the problem is a procedurally generated family of
# confounded MDPs. The generator is data/cmdp_family.py and the logged
# demonstration datasets (data/train/*.json, data/test/*.json) are small,
# agent-authored files committed to git. The public training split can be
# rebuilt with:  PYTHONPATH=. python eval/train/build_train_split.py --force
# Python dependencies (numpy, torch) are installed by environment/setup.sh.

set -e

cd /home/user
mkdir -p data/downloads
