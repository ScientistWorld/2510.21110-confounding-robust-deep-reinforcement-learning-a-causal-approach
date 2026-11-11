# Hardening report (iteration 12)

## Summary

**Before this iteration**
- The gym did not reward science.
- Its 6-state problem was solved by "always take action 0". The base setting was saturated.
- Exhaustive search over 64 policies tied the reference.
- Several evaluator loopholes let a failed method tie or beat the reference.

**What I changed**
- I replaced the problem with a procedurally generated family of confounded MDPs. The test instances are hidden, and policies are scored by exact interventional return.
- I rewrote the test evaluator so it trusts nothing the method writes and scores failures at the worst value.
- I restored the paper's Causal-DQN procedure.

**What I checked**
- The reference keeps the paper's claim in the paper's direction.
- A variant selected on train beats it on the hidden test by far more than seed noise.
- Every shortcut the red team ran is either far below the reference or scored as failed.

**Evidence**

All numbers below come from running `bash scripts/evaluate_test.sh` with no arguments:
- Job `cf004967-974`: GPU, training plus all cases.
- Job `2112f426-6f6`: CPU, reran every case with the new variant.

Per-case scores and logs are in `redteam/hardening/results/<case>/`. The tables come from `redteam/hardening/results/summary.json`; rebuild them with `PYTHONPATH=. python -m redteam.hardening.suite summary`.

**How rank is computed**

"Rank" below is the diagnostic rank against the reference (reference = 0). It is the sum, over experiments and metrics, of experiment weight × |coefficient| × the improvement in the metric's better direction, divided by that metric's standard deviation across a cohort. Two cohorts are reported:
- **All:** every attempt in this report, including the failed shortcuts.
- **Honest:** only the reference, the baselines and the honest variants. Its standard deviations are smaller, so it is the stricter test of noise.

## What the gym is now

The problem follows the paper's CMDP (Definition 2.1):

- **Instance size and dynamics.** 30 observed states, 4 actions, and 3 hidden contexts drawn fresh at every step. The context changes rewards and next-state distributions.
- **Demonstrator.** It sees the context and plans with full information, but with noisy values and Boltzmann choices. With the logged probability ε, a logged action is uniform-random.
- **Learner's data.** For each instance, the learner gets 10,000 logged transitions.
- **Experiments.**
  - `confounded_expert_demos` (ε = 0.10, 40% context-dependent actions): the paper's Table 1 analogue.
  - `pervasive_confounding` (ε = 0.05, 55% context-dependent actions): the analogue of where the paper's method recovers less of the demonstrator (Appendix E, Table 2; Appendix G).
- **Size.** 32 public train instances and 32 hidden test instances per experiment.
- **Metrics.** Exact expected interventional return, normalised per instance (random = 0, demonstrator = 1, as in the paper's Table 1), over 3 runs:

  | Metric | Coefficient | Role |
  |---|---|---|
  | `normalized_mean_return` | 1.0 | primary benefit |
  | `normalized_iqm_return` | 0.5 | benefit (paper reports IQM) |
  | `worst_decile_normalized_return` | 0.5 | robustness constraint: the paper's "safe for the worst case" claim |

  Both experiments have weight 0.5. The two experiments matter equally. The second is where headroom remains, so down-weighting it would hide exactly what a better method should improve.

## Requirements and where they are met

| Requirement | Now |
|---|---|
| Test answers only in `eval/test/` | The true test dynamics are in `eval/test/envs/*.json`, built by `eval/test/build_test_split.py` from a 128-bit secret seed. Only the logs (`data/test/`) are public. The test evaluator imports nothing from `data/`, `method/` or `eval/train/`. |
| Recompute everything from raw outputs | Scores are computed from the policy lists alone, by exact linear-system policy evaluation. Any other keys in the artifacts are ignored. |
| Comparison side fixed by `eval/` | Instances, run count (exactly `run0`–`run2`), random and demonstrator normalisers, and worst values are all fixed by the evaluator. `evaluate_test.sh` accepts no options. |
| Failed run gets the worst value everywhere | Any missing, malformed or illegal policy makes the method failed. It then gets, in every metric of every experiment, the value of the worst possible policy on every instance: −1.117 / −1.101 / −1.494 (expert), −0.952 / −0.939 / −1.261 (pervasive). Stale `scores_test.json` is deleted before scoring. |
| Scores every method on disk | Every directory under `checkpoints/` is scored. Nothing is read from `scores_train.json`. |
| No dependence on wall-clock; no saturation | Exact evaluation; no metric is at its bound for any method (see the headroom section). |

## Do better results score higher?

**Before (REDTEAM.md, old gym).** The better ensemble scored +0.237. That was below the reference seed spread (0.357) and equal to the best single seed, so it was indistinguishable from noise. Training with 4× the data and updates made things worse (−0.228). The causal-target ablation scored −4.18.

**After (hidden test).** Each cell is mean / IQM / worst decile.

| Variant | confounded_expert_demos | pervasive_confounding | Rank (all) | Rank (honest) |
|---|---|---|---:|---:|
| `better_sharp_lcb`: finite-sample sharp causal bound (below) | **0.987 / 0.968 / 0.635** | 0.716 / 0.709 / 0.392 | **+0.225** | **+1.051** |
| `causal_dqn` (reference, runs 0–2) | 0.804 / 0.820 / 0.491 | 0.695 / 0.677 / 0.485 | 0 | 0 |
| `causal_dqn_rep2` (independent seeds 3–5) | 0.811 / 0.825 / 0.484 | 0.694 / 0.674 / 0.507 | +0.010 | +0.037 |
| `stronger_x3_updates` (3× updates) | 0.810 / 0.825 / 0.490 | 0.694 / 0.676 / 0.487 | +0.005 | +0.025 |
| `weaker_third_updates` (1/3 updates) | 0.817 / 0.829 / 0.508 | 0.699 / 0.680 / 0.514 | +0.034 | +0.124 |
| `better_ensemble` (mean Q of the 3 reference runs) | 0.801 / 0.816 / 0.493 | 0.696 / 0.684 / 0.466 | −0.008 | −0.026 |
| `weaker_paper_replay_ratio` (paper's 1 update / 20 transitions) | 0.789 / 0.808 / 0.456 | 0.675 / 0.655 / 0.471 | −0.059 | −0.222 |
| `variant_causal_tabular` (paper's bound, solved exactly in tabular form) | 0.744 / 0.742 / 0.478 | 0.635 / 0.617 / 0.417 | −0.178 | −0.717 |
| `behavior_cloning` (baseline) | 0.717 / 0.726 / 0.434 | 0.607 / 0.582 / 0.395 | −0.264 | −1.047 |
| `better_sharp_bound` (sharp bound without finite-sample care) | 0.755 / 0.818 / −0.048 | 0.490 / 0.488 / 0.091 | −0.629 | −2.075 |
| `conf_dqn` (paper's baseline / causal-target ablation) | 0.353 / 0.377 / −0.216 | 0.133 / 0.143 / −0.266 | −1.715 | −6.455 |

**Noise.**
- Single reference seeds 0–5 have mean returns between 0.796 and 0.815 (expert) and between 0.690 and 0.701 (pervasive).
- Their ranks range from −0.050 to +0.145 (honest cohort).
- Two independent 3-run replicates differ by +0.037.

**The better variant, against that noise.**
- Paired per instance (same instances, means over runs), `better_sharp_lcb` gains:
  - +0.183 ± 0.042 standard error in the expert setting (22 wins, 10 losses);
  - +0.021 ± 0.038 in the pervasive setting (17 wins, 15 losses).
- Its rank gain of +1.05 (honest cohort) is about 7× the largest single-seed rank and 28× the replicate difference.
- It was selected on the public train split only: two settings, z = 2 and κ = 2, both chosen there. On train it scores 0.987 / 0.715 mean against 0.770 / 0.662 for the reference, so honest methods score alike on train and test.

**What `better_sharp_lcb` does.** It is the paper's pessimistic principle with a sharper bound. Under the documented logging protocol, a uniformly random action is logged with probability ε, independently of the hidden context. So for any action, the interventional outcome density is at most |A|·P(x|s)/ε times the logged density. The interventional lower bound is therefore a lower-tail mean of logged outcomes, rather than the reward floor. The variant:
1. uses an upper confidence bound on P(x|s);
2. subtracts 2 standard errors from the tail mean;
3. never goes below the paper's own bound.

The uncorrected version (`better_sharp_bound`) fails catastrophically on rarely logged actions, through the winner's curse.

The constraint metric does its job: in `pervasive_confounding` the variant's worst decile is lower (0.392 vs 0.485) and costs it rank there.

**Worse variants score lower:**
- `conf_dqn` / ablation at −6.46;
- `variant_causal_tabular` at −0.72;
- the paper replay ratio at −0.22.

More or fewer updates score the same as the reference within noise. The reference is converged, so extra compute is not a free improvement.

## Shortcuts: fix, and score before and after

Old-gym scores are from REDTEAM.md (its own audit cohort). New scores are hidden-test values, with each cell giving the normalised mean in the expert / pervasive setting.

| Shortcut | Old score (rank) | Fix | New result | New rank (all / honest) |
|---|---|---|---|---|
| Constant visible action | +0.237, beat the reference | New problem: the optimal action depends on the state, and no constant policy is good | `shortcut_constant_action` 0.070 / 0.016 | −2.22 / −8.65 |
| Brute force over all 64 policies with the public reward model | +0.237 | Test dynamics hidden: 4^30 policies and no evaluator access. The best exhaustive planner on the visible logs (`shortcut_nominal_planning`, exact VI on the observational model) | 0.254 / 0.079 | −1.88 / −7.24 |
| Variant/label swap in artifact metadata | +0.334 | The evaluator binds instance ↔ dynamics by its own ids and paths; labels are ignored. `shortcut_label_swap` puts each experiment's policies under the other experiment | 0.016 / −0.025 | −2.38 / −9.29 |
| Editing the reward table in `data/` | +3.02 | `eval/test` never reads `data/`. `shortcut_data_edit` rewrote `REWARD_FLOOR` and the dataset headers before scoring | identical to the constant policy, 0.070 / 0.016 | −2.22 / −8.65 |
| Partial manifest (drop a failed setting from `scores_train.json`) | 0, exact tie | No manifest; every experiment is required. `shortcut_missing_experiment` | failed, worst values | −5.48 / −21.6 |
| Empty manifest | 0, tie | Same | (covered by the line above) | — |
| Duplicate dilution (50 extra good copies) | +0.151 | Exactly `run0`–`run2` are read; extra files ignored. `shortcut_extra_runs` (3 bad + 47 good) | scored as its 3 bad runs, 0.048 / −0.031 | −2.33 / −9.07 |
| Empty policy (missing states default to safe) | −0.20 / +1.65 | Missing states make the method failed. `shortcut_missing_states` | failed, worst values | −5.48 / −21.6 |
| Forged score fields and pre-written score file | −4.18, already failed | Unchanged: recomputed, and the file is deleted first. `shortcut_forged_fields` | 0.048 / −0.031 | −2.33 / −9.07 |
| Invalid action ids (mapped to "risky") | −4.18 | Now a hard failure. `shortcut_invalid_action` | failed, worst values | −5.48 / −21.6 |
| Null action leaving a stale score file (evaluator crashed, nothing written) | not ranked | Stale file deleted; failures write worst values with exit 0. `shortcut_null_action` | failed, worst values | −5.48 / −21.6 |
| (new) Train-tuned: optimal policies from the visible train dynamics, behaviour cloning on test | — | Train and test are different instances | train 1.148 / 0.999, test 0.717 / 0.607 | −0.26 / −1.05 |

## Shortcuts left, and why

- **Learning the generator's prior.** The family generator (`data/cmdp_family.py`) is public and is the same family as the test instances. A scientist can simulate unlimited instances and fit a model from log statistics to good actions. This "writes down the generating process" in a learned form. It was not run here and its effect is unmeasured; it is the main residual risk of a synthetic gym. The test instances themselves cannot be regenerated (secret seed), and the hidden context is not decodable from single outcomes: rewards are noisy and next-state supports are shared across contexts.
- **Copying one run three times.** This is allowed. A deterministic method legitimately does it, and it cannot inflate scores beyond what that policy earns.
- **The old red-team harness** (`redteam/harness.py`, `redteam/run_suite.py`) targets the removed 6-state gym (`data/confounded_cmdp.py`) and no longer runs. It is kept only as the record of the previous audit.

## Reference before and after

**Before (old gym, raw 6-step returns, 3 seeds):**

| Setting | Mean return | Note |
|---|---:|---|
| Base | 1.500 | the optimum: saturated |
| Weaker-bias | 0.940 | |

**After (new metrics):** Causal-DQN scores 0.804 / 0.820 / 0.491 (expert) and 0.695 / 0.677 / 0.485 (pervasive), as mean / IQM / worst decile.

The scales are not comparable. The problem and metrics were replaced, so the reference scores were regenerated from the final code (`scoring/scores_test.json`, `scores_train.json`, `scores.json`, with the reference and baselines only).

The paper's direction holds in both experiments. Causal-DQN beats Conf. DQN by 0.45 / 0.56 normalised mean, against seed noise of about 0.02, and has a far better worst decile.

The paper's Atari numbers stay in `scoring/reference.json` under `_paper_*` keys: Causal-DQN 1.04 / 1.02, Conf. DQN 0.10 / 0.13 (Table 1); Causal-DQN 0.55 / 0.59 with sebulba (Table 2). They are context, not targets.

## Headroom check

**Bounds.** The best possible policy per test instance averages 1.181 (expert) and 1.049 (pervasive) normalised return. The worst policy averages −1.117 / −0.952.

**Where the reference sits.** At 0.804 / 0.695 it is at the bound of no scored metric; no method in this report is.

**What was added.**
- The `pervasive_confounding` experiment, built from where the paper's own method falls short (Appendix E: less recovery with a demonstrator that relies more on hidden information; Appendix G: thin support).
- Exact evaluation, so no rollout noise.
- Paired instances, so instance-to-instance variation cancels out of comparisons.

**Demonstrated headroom.** `better_sharp_lcb` beats the reference clearly in the main experiment and still leaves room above it (0.987 vs 1.181). In `pervasive_confounding`, no method tried here is clearly better than the reference (best +0.02 in mean, with a lower worst decile), so most of the 0.35 gap to the bound is open.

**Simple baselines and brute force stay below the reference:**
- behaviour cloning: −0.26 / −1.05 rank (all / honest cohort);
- nominal exhaustive planning: −1.88 / −7.24;
- constant policy: −2.22 / −8.65.

## Paper fidelity: restored and still departing

Restored to the paper (Algorithm 2 and Sec. 4):
- demonstrator transitions streamed into replay, with updates interleaved;
- batch size 512 and learning rate 5e-4 (was 64 and 1e-3);
- γ = 0.99 (was 0.95);
- Double DQN with a target network;
- reward floor a = −1 for rewards in [−1, 1];
- the worst next state estimated by sampling next states from replay (was enumeration of all states);
- the same termination mask on the logged-action target and the pessimistic target (previously only the logged-action target was masked).

Still scaled:
- small procedurally generated MDPs with 16-dimensional state features and an MLP Q-network, instead of masked Atari with Nature-DQN CNNs, frame stacks and DIAMOND/sebulba demonstrators;
- 10,000 logged transitions per instance instead of 1M environment steps;
- 3 runs instead of 5 seeds;
- 30,000 updates (3 per logged transition) instead of the paper's 1 per 20.

With the paper's replay ratio (475 updates), the method scores slightly lower: 0.789 / 0.675.

LSTM-DQN and Interv. DQN are not reproduced. Interv. DQN needs online interaction, which this offline benchmark does not allow.

## Could not fix / not done

- The meta-learning / prior-fitting risk above is not measured.
- In `pervasive_confounding`, I did not find a method that is better on all three metrics. Headroom is only shown in principle there, by the oracle bound, plus a mean/IQM gain with a worse worst decile.
- The hidden test reads `eval/test/envs/*.json` from the workspace. Withholding `eval/test/` from future agents is up to the hosting system.
