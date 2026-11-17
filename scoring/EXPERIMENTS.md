# Experiments

Both experiments ask the same question. Can a learner that only has a demonstrator's logged transitions find a policy that does well when it acts itself, even though the demonstrator's choices were driven by information the learner never sees? Scores are exact interventional returns on 64 held-out instances per experiment. Each instance is normalised so the uniform-random policy is 0 and the demonstrator is 1, averaged over 3 independent runs.

## confounded_expert_demos

This is the main setting. Actions react moderately to the hidden context, and 10% of logged actions are uniformly random. The experiment shows whether a method avoids being misled by actions that only look good in the log. It also shows how much of the return achievable without the hidden context a method recovers from the log alone. The demonstrator is noisy, so good methods exceed 1.

- Benefit: normalised mean and interquartile mean.
- Constraint: the worst-decile normalised return. A method must not buy a higher mean by failing badly on some instances.

## pervasive_confounding

This is the same measurement in a harder regime. Actions react more strongly to the hidden context, so the demonstrator relies on it more, and only 5% of logged actions are random. The log is therefore more biased and covers the action space more thinly. Methods recover a smaller share of the achievable return here, so this experiment separates methods that use the log more efficiently and safely.
