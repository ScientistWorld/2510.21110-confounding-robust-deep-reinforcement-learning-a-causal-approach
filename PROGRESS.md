# Progress

## What Works

- Paper analysis is complete for the core claim: Causal-DQN learns from confounded off-policy demonstrator data using a pessimistic causal Bellman lower-bound target.
- Implemented a small tracked CMDP where the demonstrator sees a hidden confounder that the learner cannot observe.
- Implemented `method/causal_dqn.py` with the paper's all-action target: standard backup for the logged action and reward-floor plus worst-next-state backup for non-logged actions.
- Implemented `baseline/confounded_dqn.py` as the causally unaware off-policy DQN baseline.
- Implemented `eval/evaluate_cmdp.py`; it reads saved policy artifacts and does not import from `method/`.

## Results

- No completed compute-job results yet. The next job runs baseline, Causal-DQN, evaluation, and validation.

## Remaining

- Run the full pipeline on a compute node.
- If Causal-DQN beats the confounded DQN baseline and chooses the safe policy, update `MILESTONES.md` to `core_claim`.
- Optionally expand to more seeds or a second CMDP setting for `core_claim_plus`.

## Issues

- The paper does not provide an author Causal-DQN code release. It cites open demonstrator resources for Atari, but reproducing the full 12-game benchmark is too large for a first-pass core-claim job.

## Deviations from Paper

- Paper setup: 12 masked Atari games, diamond/sebulba demonstrators, Nature DQN CNNs, 1M steps, 5 seeds.
- Reproduction setup: a small finite-horizon hidden-confounder CMDP with an MLP Q-network, generated demonstrator replay, and interventional evaluation.
- Reason: the author code is not released and the full Atari setup would require substantial demonstrator integration and many GPU-hours. The algorithmic computation is preserved: replay data from a hidden-confounder demonstrator and the causal all-action Bellman lower-bound update from Algorithm 2 / Eq. 26.
