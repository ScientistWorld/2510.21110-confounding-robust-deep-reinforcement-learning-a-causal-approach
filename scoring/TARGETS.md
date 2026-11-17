# Targets

Scored on the hidden test slice (`scripts/evaluate_test.sh`, 64 instances × 3 runs per experiment). Both experiments have weight 0.5.

| Experiment | Metric | Role | Coefficient | Reference (causal_dqn, test) |
|---|---|---|---|---|
| confounded_expert_demos | normalized_mean_return | primary benefit | 1.0 | 1.185 |
| confounded_expert_demos | normalized_iqm_return | benefit | 0.5 | 1.181 |
| confounded_expert_demos | worst_decile_normalized_return | robustness constraint | 0.5 | 0.760 |
| pervasive_confounding | normalized_mean_return | primary benefit | 1.0 | 0.959 |
| pervasive_confounding | normalized_iqm_return | benefit | 0.5 | 0.948 |
| pervasive_confounding | worst_decile_normalized_return | robustness constraint | 0.5 | 0.670 |

Baselines on the same test slice (normalised mean, expert / pervasive):

- `conf_dqn`: 1.028 / 0.709.
- `behavior_cloning`: 1.098 / 0.897.

A method must beat the reference on return without lowering the worst-decile constraint. A method that fails to produce valid policies for every instance, experiment and run gets the worst attainable value of every metric.

The best possible policy for each held-out instance scores, as mean / IQM / worst decile:

- `confounded_expert_demos`: 1.424 / 1.412 / 1.013;
- `pervasive_confounding`: 1.177 / 1.159 / 0.859.

The reference leaves substantial headroom below these bounds in every metric. Seed-to-seed variation of the reference is about ±0.015 in the mean.
