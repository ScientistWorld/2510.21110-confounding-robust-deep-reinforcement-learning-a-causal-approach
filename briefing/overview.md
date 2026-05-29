# Overview

This paper studies deep RL from confounded off-policy data. It proposes Causal-DQN, which replaces the standard DQN backup with a causal Bellman lower-bound backup so the learned policy is robust to hidden action-outcome confounders.

The original experiments use masked Atari games and expert demonstrators. This workspace implements the paper's actual Causal-DQN update on a small confounded CMDP, plus a standard confounded DQN baseline, and evaluates both by interventional rollouts in the same hidden-confounder environment.

The packaged reproduction targets `core_claim_plus`: across seeds `0 1 2` and two confounded CMDP variants, Confounded DQN should learn the confounded risky policy while Causal-DQN should learn the safe interventional policy.
