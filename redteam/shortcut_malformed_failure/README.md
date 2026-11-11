This shortcut attempt **failed**: the hidden evaluator exited 1 on a null
action. `scores_test.json` here is the unchanged, deliberately planted stale
sentinel, **not a score produced by the hidden test**. Its values must not be
ranked. `execution.json` records both the nonzero test exit and unchanged score
file; `test.log` records the error. The outer case runner exits successfully
because observing this expected failure is part of the audit.

The finding is missing failure-value handling: no worst-value results are
written, and the stale file is not removed. Whether an external ranker rejects
a failed job is outside this workspace. No paying shortcut is claimed here.
