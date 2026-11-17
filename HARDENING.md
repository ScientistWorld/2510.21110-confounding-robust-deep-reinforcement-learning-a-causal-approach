# Hardening report (iteration 13)

This iteration fixes the two items in `REVIEW.md` and rechecks everything the earlier hardening (iteration 12) established. The iteration-12 report is replaced by this one. Its per-case results are kept in `redteam/hardening/results_iter12/`.

## Summary

**What `REVIEW.md` asked**
1. Show an honest method that beats the reference in `pervasive_confounding` beyond noise, on all scored metrics.
2. Measure on the hidden test what fitting a prior on instances simulated from the public generator is worth. If it beats the honest improvement, change the instance family until it no longer pays.

**What I found** (measured on the iteration-12 hidden test before changing anything)
- **Prior fitting paid a lot.** An MLP trained on simulated instances to predict true interventional rewards from ~50 log statistics scored 1.066 / 0.861 normalised mean (expert / pervasive). That is:
  - +0.26 / +0.17 above the reference;
  - +0.08 / +0.15 above the best honest variant (`better_sharp_lcb`, 0.987 / 0.716);
  - in the diagnostic rank, +0.608 against +0.225 for that variant (all-attempts cohort), or +2.49 against +1.05 (honest cohort).
- **A ~20-line hand-written decoder also beat the reference** (0.940 mean in expert, rank +0.055).
- **Cause.** The iteration-12 family gave every action one of two reward types: one level, or about ±0.75 depending on a single hidden context. With the documented logging ε, the share of the low cluster identifies the hidden-context probability exactly.

**What I changed**
1. **Replaced the instance family** with a much richer one (below). It removes the decodable structure and keeps the paper's claim.
2. **Enlarged the hidden test** to 64 instances per experiment and the logs to 30,000 transitions per instance.
3. **Found a principled, honest, stronger bound** (the *partition bound*) that uses only the logging protocol. It beats the reference in both experiments on every metric.
4. **Measured all shortcuts again** on the new hidden test, including the prior fit trained on 1,024 simulated instances per experiment.

**Result** (hidden test, 64 instances × 3 runs)
- **Reference (Causal-DQN):** 1.185 / 0.959 normalised mean.
- **Train-selected honest improvement** (`better_hurwicz_partition_l15`): 1.346 / 1.083. Paired, that is +0.161 ± 0.020 and +0.124 ± 0.019, and it beats the reference on all three metrics in both experiments. Review item 1 is fixed.
- **Prior fit:** 1.367 / 1.087. Paired against the honest improvement:
  - +0.022 ± 0.007 in expert;
  - +0.004 ± 0.010 in pervasive.

  Its rank (+0.355) is now 1.09× the honest improvement's (+0.326), where before it was 2.7×. The hand decoder now scores below the reference (−0.039 / −0.126).
- **Residual.** The prior fit keeps a small edge in `confounded_expert_demos`. I left it and state it below. Review item 2 is fixed up to that residual.

All "after" numbers come from `bash scripts/evaluate_test.sh`, run with no arguments, in two jobs:
- `a14c18ef-d41`: GPU, splits, reference, baselines, variants and all shortcuts;
- `8d2efcec-940`: rescoring only, the honest grid.

Per-case scores and logs are in `redteam/hardening/results/<case>/`. The tables come from `redteam/hardening/results/summary.json`; rebuild them with `PYTHONPATH=. python -m redteam.hardening.suite summary`. The family-design exploration (two test jobs, `3cca4212-1cc` and `01fc2902-4c5`) is in `redteam/hardening/explore_results/`.

**How rank is computed.** "Rank" is the diagnostic rank against the reference (reference = 0). It is the sum, over experiments and metrics, of experiment weight × |coefficient| × the improvement in the metric's better direction. Each term is divided by that metric's standard deviation across a cohort. Two cohorts are reported:
- **All:** every attempt in the tables below, failed shortcuts included.
- **Honest:** the reference, the baselines and the honest variants. Its standard deviations are smaller, so it is the stricter test of noise.

## The gym now

The problem follows the paper's CMDP (Definition 2.1). Each instance has 30 observed states, 4 actions, and a hidden context drawn afresh at every step.

**Hidden context**
- Each state has 12–32 context points in R², with Dirichlet weights.

**Rewards**
- Each state-action pair has a favourable region of the context space. Its reward is μ + amp·(2·sigmoid(k·(⟨d,u⟩ − c)) − 1).
- The favourable region's position and size, the boundary sharpness and the amplitude are all drawn per state-action pair. So there are no reward "types", only a continuum.

**Transitions**
- Three candidate next states per state-action pair.
- Their logits react strongly to the context, with an action-specific strength.

**Demonstrator**
- It plans with full information (s, u).
- It acts with noisy values and a Boltzmann choice, with per-instance skill and temperature.
- With probability ε, the logged action is uniform instead.

**Per-instance randomisation**
- The number of context points, their concentration, the amplitude distribution, the reward noise, the transition sensitivity, and the demonstrator's skill and temperature are drawn per instance.
- The public generator (`data/cmdp_family.py`) and the hidden-test builder (`eval/test/build_test_split.py`) implement the same family independently. The hidden-test builder uses a new 128-bit secret seed.

**Experiments**

| | `confounded_expert_demos` | `pervasive_confounding` |
|---|---|---|
| ε | 0.10 | 0.05 |
| Context sensitivity | moderate (amplitude shape Beta(0.7–1.2, 1.0–1.8)) | strong (amplitude shape Beta(1.0–1.8, 0.7–1.2)) |
| Demonstrator skill noise | 0.3–0.7 | 0.4–0.8 |

**Split and scoring**
- Each instance has 30,000 logged transitions.
- 32 public train instances and 64 hidden test instances per experiment.
- Metrics, coefficients and weights are unchanged: normalised mean (1.0), IQM (0.5), worst decile (0.5); each experiment has weight 0.5.

**Evaluator.** Unchanged. It is shape-generic, recomputes exact returns from the policy files alone, scores every method on disk, and gives failed methods the worst value in every metric.

## Do better results score higher?

**Before (iteration 12, old hidden test).** The best honest variant (`better_sharp_lcb`) ranked +0.225 / +1.051 (all / honest cohort). Its paired gain was +0.183 ± 0.042 in expert but only +0.021 ± 0.038 in pervasive, where its worst decile was lower than the reference's.

**After (new hidden test).** Each cell is mean / IQM / worst decile. "Train mean" is the normalised mean on the public train split (expert / pervasive).

| Method | confounded_expert_demos | pervasive_confounding | Train mean | Rank (all) | Rank (honest) |
|---|---|---|---|---:|---:|
| `better_hurwicz_partition_l15` (selected on train) | **1.346 / 1.334 / 0.924** | **1.083 / 1.057 / 0.767** | 1.345 / 1.044 | **+0.326** | **+2.729** |
| `better_hurwicz_partition` (λ = 0.25) | 1.345 / 1.335 / 0.898 | 1.075 / 1.047 / 0.764 | 1.343 / 1.043 | +0.309 | +2.593 |
| `better_partition_lcb` (the bound itself, κ = 1) | 1.337 / 1.324 / 0.947 | 1.070 / 1.046 / 0.754 | 1.331 / 1.032 | +0.308 | +2.584 |
| `better_partition` (no finite-sample correction) | 1.334 / 1.321 / 0.932 | 1.071 / 1.047 / 0.758 | 1.341 / 1.029 | +0.304 | +2.543 |
| `better_hurwicz_partition_l35` | 1.334 / 1.325 / 0.899 | 1.060 / 1.039 / 0.736 | 1.313 / 1.032 | +0.279 | +2.351 |
| **`causal_dqn` (reference, runs 0–2)** | 1.185 / 1.181 / 0.760 | 0.959 / 0.948 / 0.670 | 1.163 / 0.947 | 0 | 0 |
| `weaker_third_updates` (1/3 updates) | 1.184 / 1.185 / 0.750 | 0.958 / 0.945 / 0.674 | 1.169 / 0.952 | −0.002 | −0.021 |
| `weaker_paper_replay_ratio` (1 update / 20 transitions) | 1.180 / 1.180 / 0.748 | 0.959 / 0.945 / 0.684 | 1.154 / 0.941 | −0.003 | −0.030 |
| `better_ensemble` (mean Q of 3 reference runs) | 1.189 / 1.188 / 0.762 | 0.950 / 0.943 / 0.651 | 1.159 / 0.953 | −0.009 | −0.060 |
| `stronger_x3_updates` (3× updates) | 1.180 / 1.176 / 0.752 | 0.958 / 0.947 / 0.659 | 1.164 / 0.947 | −0.011 | −0.091 |
| `causal_dqn_rep2` (independent seeds 3–5) | 1.170 / 1.170 / 0.750 | 0.958 / 0.947 / 0.671 | 1.164 / 0.947 | −0.015 | −0.130 |
| `better_sharp_lcb` (iteration-12 best) | 1.158 / 1.138 / 0.764 | 0.904 / 0.879 / 0.568 | 1.220 / 0.887 | −0.119 | −0.951 |
| `variant_causal_tabular` (paper's bound, exact tabular) | 1.132 / 1.137 / 0.691 | 0.897 / 0.899 / 0.547 | 1.105 / 0.867 | −0.156 | −1.267 |
| `behavior_cloning` (baseline) | 1.098 / 1.097 / 0.646 | 0.897 / 0.892 / 0.584 | 1.079 / 0.888 | −0.189 | −1.576 |
| `conf_dqn` (paper's baseline) | 1.028 / 1.004 / 0.523 | 0.709 / 0.689 / 0.268 | 0.974 / 0.671 | −0.581 | −4.693 |

Variants planned on the scored objective (γ = 1 instead of 0.99, `*_g1*`) are within ±0.005 of their γ = 0.99 versions; they are listed in `summary.json`.

**Noise**
- Six single reference seeds have mean returns 1.169–1.197 (expert) and 0.955–0.961 (pervasive).
- Two independent 3-run replicates differ by −0.015 ± 0.005 and −0.001 ± 0.005 (paired).

**The honest improvement, against that noise**
- It is paired on the same 64 instances (means over runs).
- Gain over the reference: +0.161 ± 0.020 in expert (59 wins, 5 losses) and +0.124 ± 0.019 in pervasive (56 wins, 8 losses).
- That is about 8σ and 6σ, and ten times the replicate difference.
- It is higher on every metric in both experiments, including the worst decile (0.924 vs 0.760 and 0.767 vs 0.670).

**What the partition bound is.** It is the paper's pessimistic principle, using only the documented logging protocol: a uniformly random action is logged with probability ε, independent of the hidden context.
- **Deliberate choices.** The demonstrator's deliberate choices of an action x happen in a set of contexts C_x with probability P(C_x) = (P(x|s) − ε/|A|)/(1 − ε).
- **Random picks.** The random picks of x sample contexts independently of C_x.
- **Decomposition.** The logged outcomes of x therefore split into an "inside C_x" part with mass 1 − b and an "outside" part with mass b = ε(1 − P(C_x))/(|A| P(x|s)). Under do(x) these parts weigh P(C_x) and 1 − P(C_x).
- **The bound.** The worst case puts the outside part on the lowest b of the logged outcomes. This is valid for any demonstrator and much tighter than both the paper's bound and the iteration-12 ε-tail bound.
- **Variants scored.** `partition_lcb` lowers it by κ standard errors and floors it at the paper's bound. The Hurwicz versions mix in the matching upper bound with weight λ.
- **Selection.** The grid (λ ∈ {0.15, 0.25, 0.35}, κ ∈ {0, 1, 2}, γ ∈ {0.99, 1}) was scored on both splits. The headline variant is the best on the public train split (`honest_best_selected_on_train` in `summary.json`).

**Worse variants score lower**
- `conf_dqn` / the causal-target ablation: −4.69.
- The paper's bound solved exactly in tabular form: −1.27.
- The iteration-12 ε-tail bound: −0.95. It is now worse than the reference: it is too pessimistic for frequently chosen actions with dispersed outcomes.

More or fewer updates, the paper's replay ratio and the ensemble all match the reference within seed noise. The reference is converged.

**Train and test agree.** Honest methods score alike on both splits: the reference gets 1.163 / 0.947 on train and 1.185 / 0.959 on test. Methods tuned to train lose on test:
- `shortcut_train_oracle` gets 1.396 on train and 1.098 on test;
- `shortcut_prior_fit_train` drops from 1.391 / 1.123 to 1.358 / 1.068.

## Shortcuts: fix, and score before and after

Old-gym scores are from the iteration-12 hidden test with its own cohort. The prior-fit and hand-decode "before" values were measured in job `3cca4212-1cc` (`redteam/hardening/explore_results/before_v1_scores_test.json`). New scores are hidden-test normalised means (expert / pervasive), with the diagnostic rank (all / honest cohort).

| Shortcut | Before | Fix | After |
|---|---|---|---|
| **Prior fit on the public generator, model head** (`shortcut_prior_fit_model`). Simulate instances, fit an MLP from ~50 log statistics (moments, quantiles, ε-tails, four tabular bounds) to the true interventional reward, plan with logged transitions. Before: 512 simulated instances; after: 1,024. | 1.066 / 0.861; rank +0.608 / +2.491. Honest best was +0.225 / +1.051. | New family: continuous context, graded per-pair responses, confounded transitions, per-instance randomised hyperparameters, so there is no discrete structure to decode. Also the stronger honest bound. Constraints now put prior fitting out of scope. | 1.367 / 1.087; rank +0.355 / +2.990, against +0.326 / +2.729 for the honest best. Paired vs the honest best: +0.022 ± 0.007 and +0.004 ± 0.010. **Residual, see below.** |
| Prior fit, policy head (`shortcut_prior_fit`): the same features, regressing true optimal advantages with a regret loss | 0.913 / 0.675; rank +0.105 / +0.507 | same | 1.360 / 1.079; rank +0.345 / +2.898; vs honest best +0.014 ± 0.008 and −0.004 ± 0.007 |
| Prior fit on the 32 public train instances (their true dynamics are public) | not measured | same | 1.358 / 1.068; rank +0.320 / +2.700, below the honest best; vs honest best +0.012 ± 0.008 and −0.015 ± 0.010 |
| Hand-written two-cluster decoder (`redteam/hardening/hand_decode.py`) | 0.940 / 0.680; rank +0.055 / +0.468, **above the reference** | New family: no two-valued rewards | 1.145 / 0.833; rank −0.265 / −2.116, below the reference by −0.039 ± 0.024 and −0.126 ± 0.023 |
| Exhaustive planning on the observational model (`shortcut_nominal_planning`) | 0.254 / 0.079 | Unchanged: hidden dynamics | 1.041 / 0.697; rank −0.589 / −4.744 |
| Train-tuned (`shortcut_train_oracle`): oracle policies on the public train instances, behaviour cloning on test | train 1.148, test 0.717 | Unchanged: disjoint instances | train 1.396 / 1.134, test 1.098 / 0.897; rank −0.189 |
| Constant action | 0.070 / 0.016 | Unchanged | −0.022 / −0.014; rank −2.67 |
| Policies under the other experiment's label (`shortcut_label_swap`) | 0.016 / −0.025 | Unchanged: the evaluator binds instances by its own ids and paths | 0.001 / 0.011; rank −2.59 |
| Edit `data/` (reward floor, headers) before scoring (`shortcut_data_edit`) | same as constant | Unchanged: `eval/test` never reads `data/` | identical to constant |
| 3 bad runs + 47 good extra runs | 0.048 / −0.031 | Unchanged: only `run0`–`run2` are read | 0.038 / −0.008 (its bad runs) |
| Forged score fields and a pre-written score file | 0.048 / −0.031 | Unchanged: recomputed; stale file deleted | 0.038 / −0.008 |
| Missing experiment, missing states, invalid action id, `null` action with a stale score file | failed | Unchanged | failed: −1.483 / −1.210, the worst value in every metric; rank −5.92 |

## Shortcuts left, and why

- **Residual prior fitting** (`confounded_expert_demos` only).
  - **The measurement.** A model fitted to 1,024 simulated public-generator instances still beats the train-selected honest improvement by +0.022 ± 0.007 normalised mean in that experiment. In `pervasive_confounding` it ties (+0.004 ± 0.010).
  - **Size.** That is about 14% of the honest gain over the reference; in rank, 0.355 against 0.326.
  - **Why it remains.** Under a public synthetic family, a well-fitted Bayes predictor is at least as good as any prior-free rule in expectation. Removing the edge entirely would mean hiding the generator, and the 32 public train instances (with their true dynamics) would still allow a weaker version: the train-split fit already ties the honest best.
  - **How it is handled.** `scoring/CONSTRAINTS.md` and `scoring/DIRECTION.md` now put "learning the generator's prior" out of scope, for the grader to enforce.
- **Copying one run three times.** This is allowed: a deterministic method legitimately does it, and it cannot inflate a score.
- **Hyperparameter choice on the public train split or the generator.** This is allowed and expected. It is how the honest improvement's λ was chosen.

## Reference before and after

| | `confounded_expert_demos` | `pervasive_confounding` |
|---|---|---|
| Before (iteration-12 family, 32 test instances, 10k transitions) | 0.804 / 0.820 / 0.491 | 0.695 / 0.677 / 0.485 |
| After (new family, 64 test instances, 30k transitions) | 1.185 / 1.181 / 0.760 | 0.959 / 0.948 / 0.670 |

- **Scales are not comparable.** The family changed. In particular, the demonstrator is noisier relative to what is achievable without the hidden context, so good policies exceed 1.
- **Regenerated files.** `scoring/scores_test.json`, `scores_train.json`, `scores.json` and the unprefixed values in `scoring/reference.json` were regenerated from the final code, with only the reference and the two baselines on disk.
- **The paper's direction holds in both experiments.**
  - Causal-DQN beats Conf. DQN by +0.157 ± 0.024 and +0.250 ± 0.026 (paired), against seed noise of about 0.015.
  - Its worst decile is far better: 0.760 vs 0.523, and 0.670 vs 0.268.
  - It also beats behaviour cloning, by +0.087 ± 0.014 and +0.062 ± 0.012.
- **Paper numbers.** The paper's Atari numbers stay in `scoring/reference.json` under `_paper_*` keys, unchanged.

## Headroom check

**Bounds.** The best possible observed-state policy per test instance scores 1.424 / 1.412 / 1.013 (expert) and 1.177 / 1.159 / 0.859 (pervasive), as mean / IQM / worst decile.

**No method sits at a bound.**

| Method | `confounded_expert_demos` | `pervasive_confounding` |
|---|---|---|
| Reference | 1.185 / 1.181 / 0.760 | 0.959 / 0.948 / 0.670 |
| Honest best | 1.346 / 1.334 / 0.924 | 1.083 / 1.057 / 0.767 |
| Prior-fit shortcut | 1.367 / 1.358 / 0.948 | 1.087 / 1.074 / 0.754 |

So after the best honest method, 0.08–0.10 of the mean and 0.09 of the worst decile remain open.

**What was added for headroom**
- The harder `pervasive_confounding` experiment (from iteration 12), now with a demonstrated honest improvement there.
- 64 test instances, paired by design.
- Exact evaluation.

**Brute force and simple rules stay below the reference**
- Exhaustive planning on the observational model: −0.59.
- Behaviour cloning: −0.19.
- The hand-written decoder: −0.27.
- A constant policy: −2.67.

## Paper fidelity

The Causal-DQN implementation is unchanged from iteration 12, and faithful to Algorithm 2:
- streamed replay;
- the all-action causal target with reward floor a = −1;
- the worst next state sampled from replay;
- Double DQN with a target network;
- batch 512, learning rate 5e-4, γ = 0.99.

**Departures, all of scale**
- Small procedurally generated MDPs with an MLP Q-network, instead of masked Atari with CNNs.
- 30,000 logged transitions per instance instead of 1M steps.
- 3 runs instead of 5 seeds.
- 30,000 updates. The paper's 1M steps at one update per 20 steps give 50,000 updates. The paper's per-step ratio (1,500 updates here) scores the same as the reference within noise (`weaker_paper_replay_ratio`), as does 3× more updates.
- LSTM-DQN and Interv. DQN are not reproduced. Interv. DQN needs online interaction, which this offline benchmark does not allow.

## Could not fix / not done

- **The prior-fitting residual** in `confounded_expert_demos` (above). It is closed in `pervasive_confounding`.
- **Visibility of `eval/test/`.** The hidden test reads `eval/test/envs/*.json`, and `eval/test/build_test_split.py` holds the secret seed. Withholding `eval/test/` from future agents is up to the hosting system.
- **Old red-team harness.** `redteam/harness.py` and `redteam/run_suite.py` target the 6-state gym removed in iteration 12. They no longer run and are kept only as a record.

## Reproduce

- **Reference and baselines:** `bash scripts/run.sh`. It builds the splits if missing (they are committed), trains, scores and validates, in about 25 minutes on one GPU.
- **Hardening checks** (about 1.5 h on one GPU):

  ```
  PYTHONPATH=. python -m redteam.hardening.suite train
  PYTHONPATH=. python -m redteam.hardening.suite cases
  PYTHONPATH=. python -m redteam.hardening.suite grid
  PYTHONPATH=. python -m redteam.hardening.suite summary
  ```
- **Family-design exploration** (records only): `redteam/hardening/explore.py`.
