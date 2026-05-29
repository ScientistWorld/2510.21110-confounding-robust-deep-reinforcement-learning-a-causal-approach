# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-05-29 00:00] - core_claim
- Implemented and validated Causal-DQN's causal Bellman all-action lower-bound update on a small hidden-confounder CMDP.
- Evaluated against a standard confounded DQN baseline across seeds `0 1 2`; Causal-DQN learns the safe interventional policy while the baseline learns the confounded risky policy.
- Full masked-Atari reproduction remains out of scope for this packaged core-claim gym.

### [2026-05-29 00:00] - core_claim
- Audited the existing implementation and evaluation; the method implements the paper's causal all-action pessimistic backup and eval reads saved artifacts independently of `method/`.
- Added a second tracked CMDP variant, `weaker_bias_cmdp`, to test the same Causal-DQN computation under a different hidden-confounding strength.
- Submitted a CPU test job to run both variants over seeds `0 1 2`; milestone remains `core_claim` until that job completes successfully.

### [2026-05-29 01:45] - core_claim_plus
- CPU test job `3ffe5bba-0a3` completed both CMDP variants over seeds `0 1 2` and passed `python validate.py --compare`.
- In `scaled_confounded_cmdp`, Causal-DQN reached mean return `1.500` versus Confounded DQN `-2.714`.
- In `weaker_bias_cmdp`, Causal-DQN reached mean return `0.923` versus Confounded DQN `-1.504`, with safe action rate `0.889`.

## Stop Justification
- Completed at milestone `core_claim_plus`.
