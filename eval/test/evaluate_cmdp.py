"""Held-out test-slice evaluator for saved CMDP policy artifacts.

The test slice scores the same artifact interface as the visible train slice,
but with held-out rollout seeds. When a train score file is present, this
evaluator filters and validates the output so every train-scored method entry
is scored on the test slice and no extra method entries are introduced.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from data.confounded_cmdp import N_STATES, evaluate_tabular_policy


DEFAULT_TEST_SEEDS = (1499, 2137, 3779)


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
    return list(DEFAULT_TEST_SEEDS)


def load_required_entries(path: Path) -> set[tuple[str, str]] | None:
    if not path.exists():
        return None

    payload = json.loads(path.read_text())
    required: set[tuple[str, str]] = set()
    for experiment, experiment_payload in payload.get("experiments", {}).items():
        for method in experiment_payload.get("results", {}):
            required.add((experiment, method))
    return required


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+", help="Policy JSON artifacts to score")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=None, help="Compatibility option for a single evaluation seed")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Held-out test-slice evaluation seeds")
    parser.add_argument("--output", default="scoring/scores_test.json")
    parser.add_argument(
        "--require-methods-from",
        default="scoring/scores_train.json",
        help="Optional train score file whose experiment/method entries define the required test entries",
    )
    args = parser.parse_args()

    seeds = resolve_seeds(args.seed, args.seeds)
    required_entries = load_required_entries(Path(args.require_methods_from))

    grouped: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for item in args.artifacts:
        experiment, method, metrics = evaluate_one(Path(item), args.episodes, seeds)
        entry = (experiment, method)
        if required_entries is None or entry in required_entries:
            grouped[entry].append(metrics)

    if required_entries is not None:
        missing = sorted(required_entries - set(grouped))
        if missing:
            formatted = ", ".join(f"{experiment}/{method}" for experiment, method in missing)
            raise SystemExit(f"missing test artifacts for train-scored entries: {formatted}")

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
