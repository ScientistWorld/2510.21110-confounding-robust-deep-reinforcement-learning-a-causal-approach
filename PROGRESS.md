# Progress

## What works

- **Problem.** `data/cmdp_family.py` generates confounded MDPs following the paper's Definition 2.1: a hidden context drawn fresh at every step, and a demonstrator that sees it.
  - Public train split: `data/train/`, with true dynamics in `eval/train/envs/`.
  - Hidden test split: logs in `data/test/`; true dynamics only in `eval/test/envs/`, built from a secret seed by `eval/test/build_test_split.py`.
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

## Results (hidden test, 3 runs; normalised mean / IQM / worst decile)

| Method | confounded_expert_demos | pervasive_confounding |
|---|---|---|
| causal_dqn (reference) | 0.804 / 0.820 / 0.491 | 0.695 / 0.677 / 0.485 |
| behavior_cloning | 0.717 / 0.726 / 0.434 | 0.607 / 0.582 / 0.395 |
| conf_dqn | 0.353 / 0.377 / −0.216 | 0.133 / 0.143 / −0.266 |
| best possible policy (mean) | 1.181 | 1.049 |

Seed noise of the reference is about ±0.01 in the mean. See `HARDENING.md` for the variants, the noise analysis and the shortcuts.

## Deviations from the paper

- Small procedurally generated MDPs with an MLP Q-network, instead of 12 masked Atari games with CNNs and the DIAMOND/sebulba demonstrators.
- 10k logged transitions per instance instead of 1M steps.
- 3 runs instead of 5 seeds.
- 3 updates per logged transition instead of 1 per 20. With the paper's ratio the method scores slightly lower.
- Interv. DQN and LSTM-DQN are not reproduced.

## Remaining

- `pervasive_confounding` has open headroom: no tried method is clearly better than the reference on all three metrics there.
