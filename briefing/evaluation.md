# Evaluation

Evaluation is method-agnostic. A method writes a policy artifact containing an action for each observed state and, optionally, the CMDP variant name. `scripts/evaluate.sh` calls `eval/evaluate_cmdp.py`, which loads saved policy artifacts and evaluates them by intervening in the corresponding CMDP with the saved policy.

Metrics:

- `mean_return`: average interventional episode return.
- `tail_return_10pct`: average return over the lowest-return decile of evaluation episodes.
- `safe_action_rate`: fraction of evaluation decisions choosing the robust safe action.
- `risky_action_rate`: fraction of evaluation decisions choosing the confounded risky action.
- `explicit_safe_state_rate`: fraction of observed states explicitly mapped to the safe action in the policy artifact.
- `valid_action_rate`: fraction of explicit policy entries using a legal action ID.
- `covered_states`: fraction of observed states with an explicit policy entry.
- `std_return`: standard deviation of evaluation returns.

The benefit metrics are return and lower-tail return. The constraint metrics check that a method covers the observed state space with legal actions and does not improve apparent logged value by exploiting the hidden-confounded risky action.
