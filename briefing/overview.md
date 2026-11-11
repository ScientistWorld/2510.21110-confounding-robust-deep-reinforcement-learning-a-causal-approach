# Overview

The paper (Li, Zhang & Bareinboim, "Confounding Robust Deep Reinforcement Learning: A Causal Approach") studies deep RL from off-policy demonstrations when unobserved confounders affect both the demonstrator's actions and their outcomes. It proposes Causal-DQN, which replaces the DQN backup with a causal Bellman lower bound. On 12 masked Atari games it reports a demonstrator-normalised mean of 1.04, against about 0.10 for DQN baselines.

## This workspace

The workspace scales that experiment down to a procedurally generated family of confounded MDPs:

- The demonstrator sees a hidden context; the learner does not.
- 32 public train instances and 32 held-out test instances per experiment.
- Policies are scored by exact interventional return, normalised per instance as in the paper's Table 1.

## Reproduced claim (hidden test, 3 runs)

Causal-DQN clearly beats the paper's Conf. DQN baseline in both experiments:

- normalised mean 0.80 vs 0.35 in `confounded_expert_demos`;
- 0.69 vs 0.13 in `pervasive_confounding`.

It also beats a behaviour-cloning floor (0.72 / 0.61) and has a much better worst decile. It does not reach the best achievable policy (1.18 / 1.05), so there is headroom. In the second experiment it falls further short, matching the paper's own weaker results with a stronger, more information-reliant demonstrator (Appendix E).
