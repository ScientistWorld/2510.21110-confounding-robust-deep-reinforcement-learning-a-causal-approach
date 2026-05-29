# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-05-29 04:31] - none
- Started from the paper markdown and identified the central claim: Causal-DQN should learn robustly from hidden-confounded demonstrator data where standard off-policy DQN fails.

### [2026-05-29 05:22] - method_runs
- Implemented a scaled hidden-confounder CMDP, Causal-DQN's all-action causal Bellman backup, a confounded DQN baseline, and an evaluator that reads saved policy artifacts.
- First end-to-end CPU job completed but did not yet separate the method clearly enough from the baseline.

### [2026-05-29 05:39] - core_claim
- Tightened the CMDP so the logged risky action is observationally inflated but harmful under intervention.
- Job `76913160-30f` showed confounded DQN learned the all-risky policy while Causal-DQN learned the all-safe policy.

### [2026-05-29 05:49] - core_claim_plus
- Job `3ffe5bba-0a3` ran both `base` and `weaker_bias` variants over seeds `0 1 2`.
- Causal-DQN outperformed confounded DQN in both settings while reporting constraint metrics for risky and safe action rates.

### [2026-05-29 08:12] - core_claim
- Continuing-agent audit found the implementation and evaluator structurally sound, but the latest Double-DQN bootstrap cleanup has not yet been validated by a fresh job.
- Temporarily tracking the active milestone as `core_claim` until the current-code validation run restores `core_claim_plus`.
