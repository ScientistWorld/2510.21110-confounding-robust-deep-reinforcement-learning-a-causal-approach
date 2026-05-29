# Experiments

## scaled_confounded_cmdp

This experiment checks whether a learner trained from biased demonstrator trajectories chooses a policy that performs well when actually intervened in the environment. It compares submitted policy artifacts on average return, lower-tail return, state coverage, action validity, and whether the learned policy avoids the confounded risky action. When several policy artifacts for the same method are supplied, the evaluator reports the mean metric across those runs.

## weaker_bias_cmdp

This repeats the same evaluation under a second hidden-confounder setting where the logged risky action is less selectively biased but still harmful under intervention. It checks that the robust policy advantage is not specific to one confounding strength.
