# Targets

Scored on the hidden test slice (`scripts/evaluate_test.sh`). Both experiments have weight 0.5.

| Experiment | Metric | Role | Coefficient | Reference (causal_dqn, test) |
|---|---|---|---|---|
| confounded_expert_demos | normalized_mean_return | primary benefit | 1.0 | 0.804 |
| confounded_expert_demos | normalized_iqm_return | benefit | 0.5 | 0.820 |
| confounded_expert_demos | worst_decile_normalized_return | robustness constraint | 0.5 | 0.491 |
| pervasive_confounding | normalized_mean_return | primary benefit | 1.0 | 0.695 |
| pervasive_confounding | normalized_iqm_return | benefit | 0.5 | 0.677 |
| pervasive_confounding | worst_decile_normalized_return | robustness constraint | 0.5 | 0.485 |

Baselines on the same test slice:

- `conf_dqn`: 0.353 / 0.133 normalised mean return (expert / pervasive).
- `behavior_cloning`: 0.717 / 0.608.

A method must beat the reference on return without lowering the worst-decile constraint. A method that fails to produce valid policies for every instance, experiment and run gets the worst attainable value of every metric.

The best possible policy for each test instance has a mean normalised return of 1.18 (expert) and 1.05 (pervasive). The reference leaves substantial headroom below that bound.
