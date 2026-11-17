# Overview

The paper (Li, Zhang & Bareinboim, "Confounding Robust Deep Reinforcement Learning: A Causal Approach") studies deep RL from off-policy demonstrations when unobserved confounders affect both the demonstrator's actions and their outcomes. It proposes Causal-DQN, which replaces the DQN backup with a causal Bellman lower bound. On 12 masked Atari games it reports a demonstrator-normalised mean of 1.04, against about 0.10 for DQN baselines.

## This workspace

The workspace scales that experiment down to a procedurally generated family of confounded MDPs:

- The demonstrator sees a hidden context drawn afresh at every step from a continuous space; the learner does not. Every action's reward and transitions react to the context, to a degree that differs by state-action pair.
- 32 public train instances and 64 held-out test instances per experiment, 30,000 logged transitions each.
- Policies are scored by exact interventional return, normalised per instance as in the paper's Table 1 (random 0, demonstrator 1).

## Reproduced claim (hidden test, 3 runs)

Causal-DQN clearly beats the paper's Conf. DQN baseline in both experiments:

- normalised mean 1.18 vs 1.03 in `confounded_expert_demos`;
- 0.96 vs 0.71 in `pervasive_confounding`.

It also beats a behaviour-cloning floor (1.10 / 0.90) and has a much better worst decile (0.76 / 0.67 against 0.52 / 0.27 for Conf. DQN). It does not reach the best achievable policy (1.42 / 1.18), so there is headroom. The paper's own numbers are on a different scale: here the demonstrator is noisy, so a learner can exceed it, as Causal-DQN does in 7 of the paper's 12 games.
