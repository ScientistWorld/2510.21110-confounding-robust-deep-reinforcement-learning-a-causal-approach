# Evaluation

## Output contract

A method is a directory `checkpoints/<method>/`. It must contain, for each experiment in (`confounded_expert_demos`, `pervasive_confounding`) and each run `k` in (0, 1, 2), the file:

```
checkpoints/<method>/<experiment>/run<k>.json
{"policies": {"<instance_id>": [a_0, a_1, ..., a_29], ...}}
```

- There is one integer action in `[0, 4)` per observed state.
- Every instance id of both splits must be present (`train_00`..`train_31`, `test_00`..`test_63`).
- The three runs are independent repetitions of the method (for example, different seeds). A deterministic method may write the same policy three times.
- Any other files or keys are ignored.

## Scoring

Scores are computed only from these policies:

- **Exact return.** Each policy's expected interventional return (total reward until termination) is computed exactly from the instance's true dynamics, with the hidden context drawn from its true distribution at every step. There is no rollout noise.
- **Per-instance normalisation.** Each return is normalised as `(return - random) / (demonstrator - random)`. Here `random` is the uniform-random policy and `demonstrator` is the logging demonstrator, who sees the hidden context. A value of 1 matches the demonstrator. The demonstrator is noisy, so values above 1 are common: the best possible observed-state policy averages about 1.42 (`confounded_expert_demos`) and 1.18 (`pervasive_confounding`) on the held-out instances.

Metrics (per experiment, over the 64 held-out instances × 3 runs; the train slice uses its 32 instances):

| Metric | Direction | Coefficient | Meaning |
|---|---|---|---|
| `normalized_mean_return` | higher | 1.0 | mean normalised return (primary) |
| `normalized_iqm_return` | higher | 0.5 | interquartile mean of normalised returns |
| `worst_decile_normalized_return` | higher | 0.5 | mean of the lowest 10% of normalised returns: robustness constraint |

Both experiments have weight 0.5.

**Failure rule.** A method with any missing file, missing instance, wrong-length list, non-integer action or out-of-range action is scored as failed. A failed method gets the worst attainable value of every metric in every experiment: the value of the worst possible policy on every instance.

## Scripts

- **`scripts/evaluate_train.sh`** (or `scripts/evaluate.sh`): scores every method directory on the public training instances and writes `scoring/scores_train.json` (or `scoring/scores.json`).
- **`scripts/evaluate_test.sh`**:
  - takes no arguments;
  - scores every method directory on the held-out instances and writes `scoring/scores_test.json`;
  - deletes any previous `scores_test.json` first.

The test instances, run count and normalisation are fixed by the evaluator, not by any method output.
