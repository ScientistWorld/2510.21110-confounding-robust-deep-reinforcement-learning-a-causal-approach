# Reproduction Milestones

**Current: method_runs**

<!-- Milestone levels (update "Current" above as you progress):
  none             — just started, no meaningful progress yet
  method_runs      — the paper's method executes end-to-end without errors
  core_claim       — minimum experiment supports the paper's central claim
  core_claim_plus  — core claim reproduced on additional settings
  secondary_claims — secondary results or contributions reproduced
  majority         — more than half of reported results reproduced
  near_complete    — most results reproduced, only minor gaps remain
  full             — all reported results reproduced
-->

## Progress Log

### [2026-05-29 00:00] - none
- Read the paper and identified Causal-DQN's causal Bellman lower-bound update as the core algorithm.
- Implemented a scaled hidden-confounder CMDP, Causal-DQN, a confounded DQN baseline, and method-agnostic evaluation scripts.
- Prepared a compute job to verify the full reproduction pipeline before claiming `method_runs`.

### [2026-05-29 00:35] - none
- First compute job reached the container and installed dependencies.
- Training did not start due to a Python import-path issue; scripts now export `PYTHONPATH=/home/user`.

### [2026-05-29 05:50] - none
- Second compute job completed the baseline once, but was preempted and restarted before the method artifact was written.
- Reduced the smoke-run network and update count, pinned PyTorch to one thread, and added progress logs before resubmitting.

### [2026-05-29 06:05] - method_runs
- CPU smoke job completed end-to-end and wrote both baseline and Causal-DQN policy artifacts plus `scoring/scores.json`.
- Causal-DQN improved interventional return over Confounded DQN, but still selected the risky action in most states, so this does not yet justify `core_claim`.
- Tightened the scaled CMDP confounder probabilities and increased the smoke update budget for the next run.

## Stop Justification
- Not stopped.
