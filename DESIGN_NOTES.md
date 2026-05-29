# Train/Test Evaluation Design Notes

## Checks Performed

- Read `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` side by side.
- Confirmed the test script invokes `eval/test/evaluate_cmdp.py` and writes `scoring/scores_test.json`.
- Confirmed the train script invokes `eval/train/evaluate_cmdp.py` and writes `scoring/scores_train.json`.
- Grepped for train/test cross-imports and hardcoded method-name defaults; none were found.
- Confirmed `scoring/scores_train.json` and `scoring/scores_test.json` do not exist.
- Ran `bash -n` on both split scripts.

## Changes Made

- Fixed an evaluator shortcut in both train and test slices: action IDs other than `0` are now counted and executed as risky actions instead of being executed as risky while only action ID `1` was counted as risky.
- Added `valid_action_rate` so malformed policy artifacts are visible in scoring instead of silently blending into the safe/risky rates.
- Added `explicit_safe_state_rate` so a method cannot omit states and receive the default safe action without exposing the omission.
- Added `tail_return_10pct` to measure downside risk, not only average return. This better matches the paper's robust-policy claim under hidden confounding.
- Added `_slice: "train"` and `_slice: "test"` markers to the emitted split score files.
- Updated `scoring/reference.json` so all metrics have required descriptions.
- Reweighted `scoring/reference.json` toward the two CMDP experiments actually computed by the train/test split while leaving the paper Atari aggregate as low-weight context.

## Runtime Concerns for Runner Iteration

- Runner job `99d1d814-ec3` executed `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` successfully with `EVAL_EPISODES=2000`.
- The new `tail_return_10pct` values were stable at the method level: the deterministic safe base policy scored 1.5 on both slices, while the weaker-bias policy and confounded baseline showed only small train/test differences.
- The visible train and held-out test slices currently differ by rollout seeds, not by a new hidden CMDP variant. This is adequate for catching finite-seed and malformed-artifact shortcuts, but a future design pass could strengthen OOD pressure by adding a third CMDP variant in `data/` and making it held out.
- `covered_states` and `valid_action_rate` saturated at 1.0 for all current artifacts. They remain emitted as diagnostics, but their reference coefficients are zero so they do not create constant ranking credit.
- The unscored `diamond_atari_aggregate` context was removed from active `reference.json`; the split scripts only score saved CMDP policy artifacts, and the two CMDP experiments now carry the full active score weight.
