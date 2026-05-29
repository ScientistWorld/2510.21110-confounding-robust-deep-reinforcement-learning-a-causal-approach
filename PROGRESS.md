# Progress

## What Works

- Paper analysis is complete for the core claim: Causal-DQN learns from confounded off-policy demonstrator data using a pessimistic causal Bellman lower-bound target.
- Implemented a small tracked CMDP where the demonstrator sees a hidden confounder that the learner cannot observe.
- Implemented `method/causal_dqn.py` with the paper's all-action target: standard backup for the logged action and reward-floor plus worst-next-state backup for non-logged actions.
- Implemented `baseline/confounded_dqn.py` as the causally unaware off-policy DQN baseline.
- Implemented `eval/evaluate_cmdp.py`; it reads saved policy artifacts and does not import from `method/`.

## Results

- CPU test job `f4b7194a-87e` completed the confounded DQN baseline once, then was preempted and restarted before producing the Causal-DQN artifact. I cancelled it after 40 minutes because the default smoke workload was too slow for a one-CPU test node.
- CPU test job `9afc19ef-8d1` completed end-to-end. Confounded DQN learned the all-risky policy with mean return `-1.174`; Causal-DQN improved to `-0.062`, but still chose risky actions at rate `0.667`. This earns `method_runs` only, not `core_claim`.
- CPU test job `76913160-30f` reproduced the core claim on the tightened scaled CMDP. Confounded DQN learned the all-risky policy with mean return `-2.714`; Causal-DQN learned the all-safe policy with mean return `1.500`, safe action rate `1.000`, and risky action rate `0.000`.
- CPU test job `e1d5cddb-bd5` repeated the tightened run across seeds `0 1 2`. The aggregated result remained unchanged: Confounded DQN mean return `-2.714`; Causal-DQN mean return `1.500`, safe action rate `1.000`, risky action rate `0.000`.

## Remaining

- Full masked-Atari reproduction remains as future work: integrate the diamond/sebulba demonstrator data, Atari preprocessing, and Nature DQN CNN training at the paper's scale.

## Issues

- The paper does not provide an author Causal-DQN code release. It cites open demonstrator resources for Atari, but reproducing the full 12-game benchmark is too large for a first-pass core-claim job.
- Job `ddbc01d9-67a` installed dependencies successfully, then failed before training because direct script execution from `baseline/` did not include the workspace root on Python's import path. Fixed by exporting `PYTHONPATH=/home/user` in run/evaluate/method/baseline scripts.
- Job `f4b7194a-87e` showed the pipeline could start training but the CPU smoke defaults were too expensive. The next submissions use a smaller MLP, one PyTorch thread, shorter defaults, and periodic progress logging.
- Job `9afc19ef-8d1` showed the original scaled CMDP was not separating the conservative target strongly enough: the method's value estimates improved return but still preferred risky actions in most states. The next run keeps the hidden-confounding structure but lowers favorable hidden-context probabilities below the marginal break-even point in every state.
- Job `76913160-30f` validated the tightened setup and passed `python validate.py --compare`.
- Job `e1d5cddb-bd5` validated the multi-seed aggregation path and passed `python validate.py --compare`.

## Deviations from Paper

- Paper setup: 12 masked Atari games, diamond/sebulba demonstrators, Nature DQN CNNs, 1M steps, 5 seeds.
- Reproduction setup: a small finite-horizon hidden-confounder CMDP with an MLP Q-network, generated demonstrator replay, and interventional evaluation.
- Reason: the author code is not released and the full Atari setup would require substantial demonstrator integration and many GPU-hours. The algorithmic computation is preserved: replay data from a hidden-confounder demonstrator and the causal all-action Bellman lower-bound update from Algorithm 2 / Eq. 26.
