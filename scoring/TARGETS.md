# Targets

Optimize `scaled_confounded_cmdp.mean_return` while maintaining high `safe_action_rate` and low `risky_action_rate`.

A successful method should beat `conf_dqn` on interventional return and should not rely on the action whose value is inflated by hidden-confounded demonstrator data.

