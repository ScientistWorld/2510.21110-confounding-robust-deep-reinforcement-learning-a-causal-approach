# Evaluation

Evaluation is method-agnostic. A method writes a policy artifact containing an action for each observed state. `scripts/evaluate.sh` calls `eval/evaluate_cmdp.py`, which loads saved policy artifacts and evaluates them by intervening in the CMDP with the saved policy.

Metrics:

- `mean_return`: average interventional episode return.
- `safe_action_rate`: fraction of evaluation decisions choosing the robust safe action.
- `risky_action_rate`: fraction of evaluation decisions choosing the confounded risky action.
- `covered_states`: fraction of observed states with an explicit policy entry.
- `std_return`: standard deviation of evaluation returns.

The benefit metric is return. The constraint metrics are action-composition metrics: a method should not improve apparent logged value by exploiting the hidden-confounded risky action.

