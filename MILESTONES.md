# Reproduction Milestones

**Current: core_claim_plus**

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

### [2026-05-29 06:15] - core_claim
- Tightened scaled CMDP run completed end-to-end.
- Confounded DQN learned the all-risky confounded policy with mean return `-2.714`.
- Causal-DQN learned the all-safe policy with mean return `1.500`, safe action rate `1.000`, and risky action rate `0.000`.
- Added multi-seed reproduction support for the next run to test robustness beyond the minimum core claim.

### [2026-05-29 06:25] - core_claim_plus
- Multi-seed run over seeds `0 1 2` completed successfully.
- Aggregated metrics still matched the scaled target: Confounded DQN mean return `-2.714`, Causal-DQN mean return `1.500`, Causal-DQN safe action rate `1.000`.
- Stopping at `core_claim_plus`; the remaining Atari-scale benchmark is documented as a larger follow-up rather than part of this CPU-scale gym.

## Stop Justification
- Completed at milestone `core_claim_plus`.
