# Progress

## What works

- **Problem.** `data/cmdp_family.py` generates confounded MDPs following the paper's Definition 2.1:
  - a hidden context drawn afresh at every step from a continuous space;
  - a demonstrator that sees it;
  - every action's rewards and transitions reacting to the context to a degree that differs per state-action pair;
  - per-instance randomised hyperparameters.
- **Splits.**
  - Public train split: `data/train/` (32 instances per experiment), with true dynamics in `eval/train/envs/`.
  - Hidden test split: logs in `data/test/` (64 instances per experiment); true dynamics only in `eval/test/envs/`, built from a secret seed by `eval/test/build_test_split.py`. That builder has its own independent copy of the family.
- **Method.** `method/causal_dqn.py` implements Causal-DQN as in Algorithm 2:
  - streamed replay;
  - the all-action causal target;
  - the worst next state sampled from replay;
  - Double DQN with a target network;
  - batch 512, learning rate 5e-4, γ 0.99.
- **Baselines.**
  - `baseline/confounded_dqn.py`: the paper's Conf. DQN.
  - `baseline/behavior_cloning.py`: a reward-free sanity floor.
- **Evaluators.** `eval/train/evaluate_policies.py` and `eval/test/evaluate_policies.py` are independent copies. Both:
  - evaluate every `checkpoints/<method>/` exactly;
  - normalise per instance (random = 0, demonstrator = 1);
  - require runs 0–2 for both experiments;
  - give failed methods the worst value of every metric.

## Results (hidden test, 64 instances × 3 runs; normalised mean / IQM / worst decile)

| Method | confounded_expert_demos | pervasive_confounding |
|---|---|---|
| causal_dqn (reference) | 1.185 / 1.181 / 0.760 | 0.959 / 0.948 / 0.670 |
| behavior_cloning | 1.098 / 1.097 / 0.646 | 0.897 / 0.892 / 0.584 |
| conf_dqn | 1.028 / 1.004 / 0.523 | 0.709 / 0.689 / 0.268 |
| best possible policy | 1.424 / 1.412 / 1.013 | 1.177 / 1.159 / 0.859 |

Seed noise of the reference is about ±0.015 in the mean. An honest sharper bound reaches 1.346 / 1.083. See `HARDENING.md` for the variants, the noise analysis and the shortcuts.

## Deviations from the paper

- Small procedurally generated MDPs with an MLP Q-network, instead of 12 masked Atari games with CNNs and the DIAMOND/sebulba demonstrators.
- 30k logged transitions per instance instead of 1M steps.
- 30k updates, one per logged transition. The paper's ratio of one update per 20 steps scores the same within noise.
- 3 runs instead of 5 seeds.
- Interv. DQN and LSTM-DQN are not reproduced.

## Remaining

- A prior fitted on simulated instances keeps a small edge over the best honest method in `confounded_expert_demos` (+0.022 ± 0.007). It is out of scope by the constraints; see `HARDENING.md`.
