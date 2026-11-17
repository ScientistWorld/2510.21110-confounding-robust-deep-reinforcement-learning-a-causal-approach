# Direction

Work within off-policy learning from confounded demonstrations: learning a policy for an agent that acts itself, from logs produced by a demonstrator who acted on unobserved information.

In scope:
- better robust or partially identified value estimates from confounded logs;
- better use of what is known about how the log was collected;
- finite-sample corrections;
- decision rules under partial identification;
- better function approximation or optimisation of such estimates;
- ensembles.

Every gain should hold on both the average and the worst-decile metric, and on the more confounded experiment as well as the main one.

Out of scope:
- policies that observe the hidden context at execution time;
- interacting with the true environment or using its dynamics;
- learning the synthetic generator's prior: supervised models from log statistics to the true values of simulated or train instances;
- hand-coding the generator's structure or parameters.
