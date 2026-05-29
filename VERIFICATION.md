# Verification Report

## Initial Assessment

The workspace claimed `core_claim_plus` and had working method, baseline, evaluation, train/test split, and prior CPU job evidence. The milestone is justified: the implemented method preserves the paper's confounded off-policy value-learning computation at small scale, and it outperforms the causally unaware baseline on two tracked CMDP variants.

## Issues Found

- Minor: `scoring/EXPERIMENTS.md` named the reproduced method directly, which leaked solution information into a method-agnostic scoring document.
- Minor: `briefing/evaluation.md` did not list the full scored metric set now present in `reference.json`.
- Minor: `scoring/scores.json` lacked the full metrics already computed by the train/test evaluators.
- Minor: `scripts/reproduce.sh` defaulted to only the base variant and one seed instead of the two-variant, three-seed setup supporting `core_claim_plus`.
- Packaging: `scripts/run.sh` evaluated ignored checkpoint artifacts directly, which was not fresh-clone robust.

## Fixes Applied

- Reworded `scoring/EXPERIMENTS.md` to describe the experiment without naming the reproduced method or baseline architecture.
- Expanded `briefing/evaluation.md` to document `tail_return_10pct`, `explicit_safe_state_rate`, and `valid_action_rate`.
- Filled in the missing metrics in `scoring/scores.json` using the existing evaluated policy results.
- Updated `scripts/reproduce.sh` defaults to run `base weaker_bias` over seeds `0 1 2`.
- Updated `scripts/run.sh` to call `scripts/reproduce.sh`, regenerating artifacts before `python validate.py --compare`.
- Re-ran `python validate.py --compare`; all checks passed.

## Final Judgment

The workspace is ready to serve as a research gym for the scaled version of the paper's claim. Problem/solution separation is clean, evaluation reads policy artifacts rather than method internals, train/test scoring is wired through independent seeds, and the scoring metrics reward interventional return while penalizing reliance on the hidden-confounded risky action.

- Milestone: core_claim_plus
- Ready for gym use: yes
- Confidence: medium
- Key limitation: The gym uses a small synthetic hidden-confounder CMDP rather than the paper's full masked-Atari benchmark.
