# Reproduction Milestones

**Current: none**

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

## Stop Justification
- Not stopped.
