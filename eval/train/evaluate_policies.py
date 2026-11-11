"""Train-slice evaluator for confounded off-policy control.

It reads the true dynamics of the public training instances in
``eval/train/envs/``
and the policy files written by methods. It imports nothing from ``data/``,
``method/`` or ``eval/test``. Scoring rules are the same as the
hidden test's, applied to the public training instances.

Every directory ``checkpoints/<method>/`` is a method. A method must provide,
for every experiment and every run ``k`` in ``0..N_RUNS-1``, the file
``checkpoints/<method>/<experiment>/run<k>.json`` containing
``{"policies": {"<instance_id>": [action for each observed state], ...}}``.
Other files and keys are ignored.

All scores are recomputed here from the policies by exact policy evaluation
on the instance dynamics. A method with any missing, malformed or illegal
policy is scored as failed: it receives the worst attainable value of every
metric in every experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ENV_DIR = Path(__file__).resolve().parent / "envs"
EXPERIMENTS = ("confounded_expert_demos", "pervasive_confounding")
N_RUNS = 3
TAIL_FRACTION = 0.10
METRICS = ("normalized_mean_return", "normalized_iqm_return", "worst_decile_normalized_return")


class PolicyError(Exception):
    pass


def _dense(cand: np.ndarray, weights: np.ndarray, n_states: int) -> np.ndarray:
    S, A, U, C = weights.shape
    T = np.zeros((S, A, U, n_states))
    for c in range(C):
        for u in range(U):
            np.add.at(T[:, :, u, :], (np.arange(S)[:, None], np.arange(A)[None, :], cand[:, :, c]),
                      weights[:, :, u, c])
    return T


def _joint_value(env: dict, probs: np.ndarray) -> float:
    w = env["Pu"][:, :, None] * probs  # [S,U,A]
    R = np.einsum("sua,sau->s", w, env["reward"])
    P = np.einsum("sua,saut->st", w, env["T"])
    V = np.linalg.solve(np.eye(env["S"]) - env["cont"] * P, R)
    return float(env["rho0"] @ V)


def _extreme_value(env: dict, best: bool) -> float:
    """Value of the best (or worst) deterministic policy on observed states."""
    V = np.zeros(env["S"])
    Rsa = np.einsum("su,sau->sa", env["Pu"], env["reward"])
    Psa = np.einsum("su,saut->sat", env["Pu"], env["T"])
    for _ in range(20000):
        Q = Rsa + env["cont"] * Psa @ V
        V_new = Q.max(1) if best else Q.min(1)
        if np.max(np.abs(V_new - V)) < 1e-12:
            V = V_new
            break
        V = V_new
    return float(env["rho0"] @ V)


def policy_value(env: dict, policy: np.ndarray) -> float:
    idx = np.arange(env["S"])
    R = (env["Pu"] * env["reward"][idx, policy, :]).sum(-1)
    P = np.einsum("su,sut->st", env["Pu"], env["T"][idx, policy])
    V = np.linalg.solve(np.eye(env["S"]) - env["cont"] * P, R)
    return float(env["rho0"] @ V)


def load_envs(experiment: str) -> dict:
    payload = json.loads((ENV_DIR / f"{experiment}.json").read_text())
    envs = []
    for rec in payload["instances"]:
        Pu = np.asarray(rec["context_probs"], dtype=float)
        S, U = Pu.shape
        reward = np.asarray(rec["reward"], dtype=float)
        env = {
            "id": rec["id"],
            "S": S,
            "A": reward.shape[1],
            "Pu": Pu,
            "reward": reward,
            "T": _dense(np.asarray(rec["next_candidates"]), np.asarray(rec["next_weights"], dtype=float), S),
            "rho0": np.asarray(rec["initial_dist"], dtype=float),
            "cont": 1.0 - float(rec["termination_prob"]),
        }
        demo = np.asarray(rec["demo_policy"], dtype=float)
        env["v_demo"] = _joint_value(env, demo)
        env["v_random"] = _joint_value(env, np.full_like(demo, 1.0 / env["A"]))
        env["v_worst"] = _extreme_value(env, best=False)
        env["v_best"] = _extreme_value(env, best=True)
        scale = env["v_demo"] - env["v_random"]
        if not scale > 0:
            raise RuntimeError(f"degenerate normalisation for {experiment}/{rec['id']}")
        envs.append(env)
    return {"n_states": payload["n_states"], "n_actions": payload["n_actions"], "instances": envs}


def normalized(env: dict, value: float) -> float:
    return (value - env["v_random"]) / (env["v_demo"] - env["v_random"])


def aggregate(scores: np.ndarray) -> dict[str, float]:
    flat = np.sort(scores.ravel())
    n = flat.size
    q = n // 4
    k = max(1, int(round(TAIL_FRACTION * n)))
    return {
        "normalized_mean_return": float(flat.mean()),
        "normalized_iqm_return": float(flat[q:n - q].mean()),
        "worst_decile_normalized_return": float(flat[:k].mean()),
    }


def read_policies(path: Path, ids: list[str], n_states: int, n_actions: int) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise PolicyError(f"missing {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise PolicyError(f"unreadable {path.name}: {exc}") from None
    policies = payload.get("policies") if isinstance(payload, dict) else None
    if not isinstance(policies, dict):
        raise PolicyError(f"{path.name}: no 'policies' mapping")
    out = {}
    for iid in ids:
        actions = policies.get(iid)
        if not isinstance(actions, list) or len(actions) != n_states:
            raise PolicyError(f"{path.name}: instance {iid} needs a list of {n_states} actions")
        for a in actions:
            if type(a) is not int or not 0 <= a < n_actions:
                raise PolicyError(f"{path.name}: instance {iid} has illegal action {a!r}")
        out[iid] = np.asarray(actions, dtype=np.int64)
    return out


def worst_results(all_envs: dict) -> dict[str, dict[str, float]]:
    out = {}
    for experiment, bundle in all_envs.items():
        worst = np.array([[normalized(e, e["v_worst"])] * N_RUNS for e in bundle["instances"]])
        out[experiment] = aggregate(worst)
    return out


def score_method(method_dir: Path, all_envs: dict) -> tuple[str, dict, dict]:
    detail: dict[str, list] = {}
    results: dict[str, dict[str, float]] = {}
    try:
        for experiment, bundle in all_envs.items():
            envs = bundle["instances"]
            ids = [e["id"] for e in envs]
            scores = np.zeros((len(envs), N_RUNS))
            for k in range(N_RUNS):
                path = method_dir / experiment / f"run{k}.json"
                policies = read_policies(path, ids, bundle["n_states"], bundle["n_actions"])
                for i, env in enumerate(envs):
                    scores[i, k] = normalized(env, policy_value(env, policies[env["id"]]))
            if not np.all(np.isfinite(scores)):
                raise PolicyError(f"{experiment}: non-finite score")
            lo = np.array([normalized(e, e["v_worst"]) for e in envs])[:, None] - 1e-9
            hi = np.array([normalized(e, e["v_best"]) for e in envs])[:, None] + 1e-9
            if np.any(scores < lo) or np.any(scores > hi):
                raise PolicyError(f"{experiment}: score outside the attainable range")
            results[experiment] = aggregate(scores)
            detail[experiment] = {"ids": ids, "scores": scores.tolist()}
    except PolicyError as exc:
        return f"failed: {exc}", worst_results(all_envs), {}
    return "ok", results, detail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", default=str(ROOT / "checkpoints"))
    parser.add_argument("--output", default=str(ROOT / "scoring" / "scores_train.json"))
    parser.add_argument("--detail-output", default=None,
                        help="Optional per-instance normalized scores for analysis; never used for ranking.")
    args = parser.parse_args()

    out = Path(args.output)
    if out.exists():
        out.unlink()
    all_envs = {experiment: load_envs(experiment) for experiment in EXPERIMENTS}

    ckpt = Path(args.checkpoints)
    methods = sorted(p for p in ckpt.iterdir() if p.is_dir()) if ckpt.is_dir() else []
    experiments = {experiment: {"results": {}} for experiment in EXPERIMENTS}
    status, details = {}, {}
    for method_dir in methods:
        state, results, detail = score_method(method_dir, all_envs)
        status[method_dir.name] = state
        details[method_dir.name] = detail
        for experiment, metrics in results.items():
            assert all(math.isfinite(v) for v in metrics.values())
            experiments[experiment]["results"][method_dir.name] = metrics

    payload = {"_slice": "train", "_status": status, "experiments": experiments}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, out)
    if args.detail_output:
        bounds = {
            experiment: {
                "ids": [e["id"] for e in bundle["instances"]],
                "best_policy_normalized": [normalized(e, e["v_best"]) for e in bundle["instances"]],
                "worst_policy_normalized": [normalized(e, e["v_worst"]) for e in bundle["instances"]],
            }
            for experiment, bundle in all_envs.items()
        }
        Path(args.detail_output).write_text(json.dumps({"bounds": bounds, "methods": details}))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
