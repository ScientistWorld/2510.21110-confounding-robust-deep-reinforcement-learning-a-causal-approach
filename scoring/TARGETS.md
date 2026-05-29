# Targets

Optimize `scaled_confounded_cmdp.mean_return` and `weaker_bias_cmdp.mean_return` while maintaining high `safe_action_rate` and low `risky_action_rate`.

A successful method should beat `conf_dqn` on interventional return and should not rely on the action whose value is inflated by hidden-confounded demonstrator data.

The default reproduction runs variants `base weaker_bias` over seeds `0 1 2` and reports the mean metric per method. Future improvements should preserve this multi-setting, multi-seed evaluation unless they document an expanded setting or seed set.
