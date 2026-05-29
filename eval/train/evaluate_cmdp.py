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

from data.confounded_cmdp import (
    N_STATES,
    RISKY_ACTION,
    SAFE_ACTION,
    HiddenConfounderCMDP,
)


DEFAULT_TRAIN_SEEDS = (137, 409, 811)


def load_policy(path: Path) -> tuple[str, str, str, dict[int, int]]:
    artifact = json.loads(path.read_text())
    method = artifact.get("method", path.stem)
    variant = artifact.get("variant", "base")
    experiment = artifact.get("experiment", "scaled_confounded_cmdp")
    raw_policy = artifact["policy"]
    return experiment, method, variant, {int(k): int(v) for k, v in raw_policy.items()}


def bottom_decile_mean(values: list[float]) -> float:
    ordered = sorted(values)
    n_tail = max(1, len(ordered) // 10)
    return float(sum(ordered[:n_tail]) / n_tail)


def policy_structure_metrics(policy: dict[int, int]) -> dict[str, float]:
    explicit = [policy.get(state) for state in range(N_STATES)]
    valid = [action in (SAFE_ACTION, RISKY_ACTION) for action in explicit if action is not None]
    valid_count = sum(valid)
    explicit_count = len(valid)
    return {
        "covered_states": float(explicit_count / N_STATES),
        "explicit_safe_state_rate": float(sum(action == SAFE_ACTION for action in explicit) / N_STATES),
        "valid_action_rate": float(valid_count / max(explicit_count, 1)),
    }


def average_metrics(runs: list[dict[str, float]]) -> dict[str, float]:
    keys = runs[0].keys()
    return {key: float(sum(run[key] for run in runs) / len(runs)) for key in keys}


def evaluate_policy_metrics(policy: dict[int, int], episodes: int, seed: int, variant: str) -> dict[str, float]:
    env = HiddenConfounderCMDP(seed=seed, variant=variant)
    returns: list[float] = []
    risky = 0
    total = 0

    for _ in range(episodes):
        state = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            raw_action = int(policy.get(int(state), SAFE_ACTION))
            action = SAFE_ACTION if raw_action == SAFE_ACTION else RISKY_ACTION
            risky += int(action == RISKY_ACTION)
            total += 1
            state, reward, done, _ = env.step(action)
            ep_return += reward
        returns.append(ep_return)

    mean_return = sum(returns) / len(returns)
    variance = sum((value - mean_return) ** 2 for value in returns) / len(returns)
    metrics = {
        "mean_return": float(mean_return),
        "std_return": float(variance**0.5),
        "tail_return_10pct": bottom_decile_mean(returns),
        "risky_action_rate": float(risky / max(total, 1)),
        "safe_action_rate": float(1.0 - risky / max(total, 1)),
    }
    metrics.update(policy_structure_metrics(policy))
    return metrics


def evaluate_one(path: Path, episodes: int, seeds: list[int]) -> tuple[str, str, dict[str, float]]:
    experiment, method, variant, policy = load_policy(path)
    seed_runs = []
    for seed in seeds:
        seed_runs.append(evaluate_policy_metrics(policy, episodes=episodes, seed=seed, variant=variant))
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

    scores = {"_slice": "train", "experiments": dict(experiments)}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
