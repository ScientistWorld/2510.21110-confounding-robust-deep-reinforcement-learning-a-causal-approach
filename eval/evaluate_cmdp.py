"""Evaluate saved policies for the scaled confounded CMDP gym."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from data.confounded_cmdp import N_STATES, evaluate_tabular_policy


def load_policy(path: Path) -> tuple[str, dict[int, int]]:
    artifact = json.loads(path.read_text())
    method = artifact.get("method", path.stem)
    raw_policy = artifact["policy"]
    return method, {int(k): int(v) for k, v in raw_policy.items()}


def evaluate_one(path: Path, episodes: int, seed: int) -> tuple[str, dict[str, float]]:
    method, policy = load_policy(path)
    metrics = evaluate_tabular_policy(policy, episodes=episodes, seed=seed)
    metrics["covered_states"] = float(len(policy) / N_STATES)
    return method, metrics


def average_metrics(runs: list[dict[str, float]]) -> dict[str, float]:
    keys = runs[0].keys()
    return {key: float(sum(run[key] for run in runs) / len(runs)) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+", help="Policy JSON artifacts to score")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default="scoring/scores.json")
    args = parser.parse_args()

    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for item in args.artifacts:
        method, metrics = evaluate_one(Path(item), args.episodes, args.seed)
        grouped[method].append(metrics)

    results = {method: average_metrics(runs) for method, runs in grouped.items()}

    scores = {"experiments": {"scaled_confounded_cmdp": {"results": results}}}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
