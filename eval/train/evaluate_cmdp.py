"""Visible train-slice evaluator for saved CMDP policy artifacts.

This evaluator is method-agnostic: it reads policy JSON artifacts produced by
any method that follows the checkpoint schema used by the base evaluator, then
scores those policies in the confounded CMDP.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from data.confounded_cmdp import N_STATES, evaluate_tabular_policy


DEFAULT_TRAIN_SEEDS = (137, 409, 811)


def load_policy(path: Path) -> tuple[str, str, str, dict[int, int]]:
    artifact = json.loads(path.read_text())
    method = artifact.get("method", path.stem)
    variant = artifact.get("variant", "base")
    experiment = artifact.get("experiment", "scaled_confounded_cmdp")
    raw_policy = artifact["policy"]
    return experiment, method, variant, {int(k): int(v) for k, v in raw_policy.items()}


def average_metrics(runs: list[dict[str, float]]) -> dict[str, float]:
    keys = runs[0].keys()
    return {key: float(sum(run[key] for run in runs) / len(runs)) for key in keys}


def evaluate_one(path: Path, episodes: int, seeds: list[int]) -> tuple[str, str, dict[str, float]]:
    experiment, method, variant, policy = load_policy(path)
    seed_runs = []
    for seed in seeds:
        metrics = evaluate_tabular_policy(policy, episodes=episodes, seed=seed, variant=variant)
        metrics["covered_states"] = float(len(policy) / N_STATES)
        seed_runs.append(metrics)
    return experiment, method, average_metrics(seed_runs)


def resolve_seeds(seed: int | None, seeds: list[int] | None) -> list[int]:
    if seeds:
        return seeds
    if seed is not None:
        return [seed]
    return list(DEFAULT_TRAIN_SEEDS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+", help="Policy JSON artifacts to score")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=None, help="Compatibility option for a single evaluation seed")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Visible train-slice evaluation seeds")
    parser.add_argument("--output", default="scoring/scores_train.json")
    args = parser.parse_args()

    seeds = resolve_seeds(args.seed, args.seeds)
    grouped: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for item in args.artifacts:
        experiment, method, metrics = evaluate_one(Path(item), args.episodes, seeds)
        grouped[(experiment, method)].append(metrics)

    experiments: dict[str, dict[str, dict[str, dict[str, float]]]] = defaultdict(lambda: {"results": {}})
    for (experiment, method), runs in sorted(grouped.items()):
        experiments[experiment]["results"][method] = average_metrics(runs)

    scores = {"experiments": dict(experiments)}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
