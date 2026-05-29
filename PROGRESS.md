# Progress

## What Works

- Paper analysis is complete for the core claim: Causal-DQN learns from confounded off-policy demonstrator data using a pessimistic causal Bellman lower-bound target.
- Implemented a small tracked CMDP where the demonstrator sees a hidden confounder that the learner cannot observe.
- Implemented `method/causal_dqn.py` with the paper's all-action target: standard backup for the logged action and reward-floor plus worst-next-state backup for non-logged actions.
- Implemented `baseline/confounded_dqn.py` as the causally unaware off-policy DQN baseline.
- Implemented `eval/evaluate_cmdp.py`; it reads saved policy artifacts and does not import from `method/`.

## Results

- No completed compute-job results yet. The next job runs baseline, Causal-DQN, evaluation, and validation.
- CPU test job `f4b7194a-87e` completed the confounded DQN baseline once, then was preempted and restarted before producing the Causal-DQN artifact. I cancelled it after 40 minutes because the default smoke workload was too slow for a one-CPU test node.

## Remaining

- Run the full pipeline on a compute node with the reduced smoke workload: 200 demonstration episodes, 300 updates per learner, batch size 64, and 1000 evaluation episodes.
- If Causal-DQN beats the confounded DQN baseline and chooses the safe policy, update `MILESTONES.md` to `core_claim`.
- Optionally expand to more seeds or a second CMDP setting for `core_claim_plus`.

## Issues

- The paper does not provide an author Causal-DQN code release. It cites open demonstrator resources for Atari, but reproducing the full 12-game benchmark is too large for a first-pass core-claim job.
- Job `ddbc01d9-67a` installed dependencies successfully, then failed before training because direct script execution from `baseline/` did not include the workspace root on Python's import path. Fixed by exporting `PYTHONPATH=/home/user` in run/evaluate/method/baseline scripts.
- Job `f4b7194a-87e` showed the pipeline could start training but the CPU smoke defaults were too expensive. The next submission uses a smaller MLP, one PyTorch thread, shorter defaults, and progress logging every 50 updates.

## Deviations from Paper

- Paper setup: 12 masked Atari games, diamond/sebulba demonstrators, Nature DQN CNNs, 1M steps, 5 seeds.
- Reproduction setup: a small finite-horizon hidden-confounder CMDP with an MLP Q-network, generated demonstrator replay, and interventional evaluation.
- Reason: the author code is not released and the full Atari setup would require substantial demonstrator integration and many GPU-hours. The algorithmic computation is preserved: replay data from a hidden-confounder demonstrator and the causal all-action Bellman lower-bound update from Algorithm 2 / Eq. 26.
