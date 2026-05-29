# Reproduction Milestones

**Current: core_claim_plus**

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

### [2026-05-29 05:54] - core_claim
- Continuing-agent audit found the implementation and evaluator structurally sound, but the latest Double-DQN bootstrap cleanup has not yet been validated by a fresh job.
- Temporarily tracking the active milestone as `core_claim` until the current-code validation run restores `core_claim_plus`.

### [2026-05-29 06:04] - core_claim_plus
- Job `e6f5da67-7e9` validated the current Double-DQN Causal-DQN and confounded DQN code across `base` and `weaker_bias` variants over seeds `0 1 2`.
- The run passed `python validate.py --compare` and restored the prior `core_claim_plus` evidence on the current code revision.

### [2026-05-29 06:19] - core_claim_plus
- Job `99d1d814-ec3` ran the independent train/test CMDP split on the saved final policy artifacts.
- Both slices preserved the central claim signal: Causal-DQN beat confounded DQN on `mean_return` in both CMDP variants while maintaining substantially higher safe-action rates and lower risky-action rates.
- `python validate.py --compare` passed after narrowing `scoring/reference.json` to the two executable CMDP experiments.
