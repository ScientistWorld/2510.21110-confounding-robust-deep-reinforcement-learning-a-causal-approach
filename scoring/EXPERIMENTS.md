# Experiments

Both experiments ask the same question. Can a learner that only has a demonstrator's logged transitions find a policy that does well when it acts itself, even though the demonstrator's choices were driven by information the learner never sees? Scores are exact interventional returns on 32 held-out instances per experiment. Each instance is normalised so the uniform-random policy is 0 and the demonstrator is 1, averaged over 3 independent runs.

## confounded_expert_demos

This is the main setting. The demonstrator is capable and uses the hidden context for part of its decisions; 10% of logged actions are uniformly random. The experiment shows whether a method avoids being misled by actions that only look good in the log. It also shows whether a method can recover, or exceed, the demonstrator's level from the log alone.

- Benefit: normalised mean and interquartile mean.
- Constraint: the worst-decile normalised return. A method must not buy a higher mean by failing badly on some instances.

## pervasive_confounding

This is the same measurement in a harder regime. The demonstrator relies on the hidden context more often, and only 5% of logged actions are random, so the log is more biased and covers the action space more thinly. Current methods recover a clearly smaller fraction of the achievable return here, so this experiment separates methods that use the log more efficiently and safely.
