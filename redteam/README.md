This audit runs only on a compute node. Run `python -m redteam.run_suite` from
the repository root, or invoke a case's `run.sh`. Each case installs ordinary
policy artifacts, invokes the visible train evaluator, invokes the restored
hidden-test script with **no arguments**, saves its scores and exit status,
and restores checkpoint, data, and score files byte for byte.

`reference/recorded_scores_test.json` and `audit/metric_spec.json` are audit
evidence, not inputs to any candidate or shortcut. Candidate construction uses
the visible method (for improvement-mode scientific variants), baseline,
environment definitions, and policy interface. Shortcuts never inspect the
paper, hidden evaluator, held-out seeds, or reference score files. The analyst
has inspected the hidden evaluator to audit its behavior.

The reference uses the original 500 demonstration episodes, 1,000 updates,
batch size 64, and seeds 0, 1, 2 for each of the two settings. The larger
training check uses 2,000 episodes and 4,000 updates. The ensemble combines
the three reference fits by per-state majority vote, without knowing the safe
action's ID. The causal-target ablation reuses the matched standard-DQN fits.

`shortcut_*` cases are deliberately shortcuts, including any ties. A shortcut
that exits nonzero is recorded as a failure, not a successful scored attempt.
No case edits `eval/`, the hidden shell script, Python packages, interpreter
hooks, or shell startup files. Mutation of the fixed reward data is isolated
and explicitly labeled a shortcut.

The ranking service is not present in this repository. Any standardized rank
reported by the audit must name its comparison cohort; future-attempt
normalization cannot be known in advance. Raw test metrics and metricwise
dominance are the primary evidence.
