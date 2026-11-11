"""Lightweight analysis of saved scores; never runs training or rollouts.

The external ranking cohort is unknowable. We therefore save its full formula,
the exact diagnostic audit cohort and per-metric normalizers, and raw metrics.
Coefficient signs are ignored; higher_is_better supplies direction. Missing
metrics contribute zero, as specified by the audit's ranking contract.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text())


def dump(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def get(payload, experiment, method, metric):
    return payload.get("experiments", {}).get(experiment, {}).get("results", {}).get(method, {}).get(metric)


spec = load(ROOT / "audit/metric_spec.json")["experiments"]
reference = load(ROOT / "reference/recorded_scores_test.json")
suite = load(ROOT / "audit/suite.json")
attempts = {}
for name in suite:
    execution = load(ROOT / name / "execution.json")
    if execution["exit_code"] == 0:
        attempts[name] = load(ROOT / name / "scores_test.json")

# Per-experiment, per-method, per-metric population standard deviations across
# all successfully completed audit attempts; absent values are not observations.
normalizers = {}
for experiment, definition in spec.items():
    normalizers[experiment] = {}
    for method in reference["experiments"][experiment]["results"]:
        normalizers[experiment][method] = {}
        for metric in definition["metrics"]:
            values = [get(payload, experiment, method, metric) for payload in attempts.values()]
            values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
            normalizers[experiment][method][metric] = statistics.pstdev(values) if len(values) > 1 else 0.0


def rank(payload, norms=normalizers):
    total = 0.0
    contributions = {}
    missing = []
    for experiment, definition in spec.items():
        contributions[experiment] = {}
        for method, ref_metrics in reference["experiments"][experiment]["results"].items():
            for metric, rule in definition["metrics"].items():
                coefficient = abs(rule["coefficient"])
                if not coefficient:
                    continue
                value = get(payload, experiment, method, metric)
                if value is None:
                    missing.append(f"{experiment}/{method}/{metric}")
                    continue
                delta = (value - ref_metrics[metric]) * (1 if rule["higher_is_better"] else -1)
                spread = norms[experiment][method][metric]
                # Constant metrics and floating point roundoff are ties.
                contribution = 0.0 if abs(delta) < 1e-12 or spread < 1e-12 else (
                    definition["weight"] * coefficient * delta / spread)
                contributions[experiment][f"{method}/{metric}"] = contribution
                total += contribution
    return {"rank_delta": total, "contributions": contributions,
            "missing_scored_values": missing}


summary = {}
for name in suite:
    execution = load(ROOT / name / "execution.json")
    payload = load(ROOT / name / "scores_test.json")
    item = {"hidden_test_exit_code": execution["exit_code"],
            "test_means": {e: get(payload, e, "causal_dqn", "mean_return")
                           if execution["exit_code"] == 0 else None for e in spec},
            "scores_source": f"redteam/{name}/scores_test.json"}
    if execution["exit_code"] == 0:
        item.update(rank(payload))
    else:
        item.update({"rank_delta": None, "note": "stale file; not written by successful hidden test",
                     "stale_file_values": {e: get(payload, e, "causal_dqn", "mean_return") for e in spec}})
    summary[name] = item

seed_scores = {}
seed_metrics = {}
for seed in (0, 1, 2):
    payload = load(ROOT / f"reference/scores_test_seed{seed}.json")
    seed_scores[str(seed)] = rank(payload)["rank_delta"]
    seed_metrics[str(seed)] = {
        e: payload["experiments"][e]["results"]["causal_dqn"] for e in spec}

metric_spreads = {}
for experiment, definition in spec.items():
    metric_spreads[experiment] = {}
    for metric in definition["metrics"]:
        values = [seed_metrics[str(seed)][experiment][metric] for seed in (0, 1, 2)]
        metric_spreads[experiment][metric] = {
            "min": min(values), "max": max(values), "range": max(values) - min(values)}

# The empty-policy shortcut illustrates why the ranking's normalization matters:
# use a possible future cohort consisting of just reference and that attempt.
pair = [attempts["reference"], attempts["shortcut_empty_policy"]]
pair_norms = {e: {method: {metric: statistics.pstdev([
    get(payload, e, method, metric) for payload in pair]) for metric in definition["metrics"]}
    for method in reference["experiments"][e]["results"]} for e, definition in spec.items()}
pair_result = rank(attempts["shortcut_empty_policy"], pair_norms)

output = {
    "formula": "sum(weight * abs(coefficient) * directional(candidate-reference) / cohort_population_std); missing values skip",
    "reference_source": "redteam/reference/recorded_scores_test.json",
    "cohort": list(attempts),
    "caveat": "Diagnostic audit-cohort ranks, not predictions of unknown future-attempt normalizers.",
    "normalizers": normalizers, "attempts": summary,
    "reference_seed_ranks": seed_scores,
    "reference_seed_rank_range": max(seed_scores.values()) - min(seed_scores.values()),
    "reference_seed_metrics": seed_metrics, "reference_seed_metric_spreads": metric_spreads,
    "empty_policy_two_attempt_cohort": pair_result,
}
dump(ROOT / "audit/summary.json", output)
for name, item in summary.items():
    score = item["rank_delta"]
    print(f"{name:31s} rank={score!s:22s} means={item['test_means']} exit={item['hidden_test_exit_code']}")
print("SEED RANKS", seed_scores, "RANGE", output["reference_seed_rank_range"])
print("EMPTY POLICY TWO-ATTEMPT COHORT", pair_result["rank_delta"])
