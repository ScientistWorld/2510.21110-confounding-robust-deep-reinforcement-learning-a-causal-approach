# Experiments

## scaled_confounded_cmdp

This experiment checks whether a learner trained from biased demonstrator trajectories chooses a policy that performs well when actually intervened in the environment. It compares standard confounded DQN against Causal-DQN on average return and on whether the learned policy avoids the confounded risky action.

## diamond_atari_aggregate

This records the paper's main reported Atari aggregate results for context. The current runnable reproduction does not attempt the full Atari benchmark because the author Causal-DQN code was not released and the original 12-game, 5-seed setup would exceed the first-pass budget.

