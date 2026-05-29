#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: 0 GB
# Estimated disk usage after extraction: 0 GB
#
# This environment is self-contained. If future work adds external resources,
# write idempotent fetches here that populate data/downloads/.

set -e

cd /home/user

# This scaled reproduction uses a tracked procedural CMDP in data/.
# No external datasets, model weights, or package downloads are required here.
