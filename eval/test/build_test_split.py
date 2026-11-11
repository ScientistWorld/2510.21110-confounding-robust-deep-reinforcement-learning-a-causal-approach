"""Build the held-out test split (hidden-test side only).

Self-contained on purpose: it does not import ``data/`` or ``eval/train``.
The instance family matches the public generator in ``data/cmdp_family.py``;
the instances themselves come from an undisclosed seed, so the true dynamics
(``eval/test/envs/*.json``) cannot be regenerated from public code.

Writes ``data/test/<experiment>.json`` (logged demonstrations, visible to
methods) and ``eval/test/envs/<experiment>.json`` (true dynamics, used only by
``eval/test/evaluate_policies.py``). Idempotent unless ``--force``.
"""

from __future__ import annotations

import argparse
import base64
import json
from bisect import bisect_right
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SECRET_ENTROPY = 60516820836623623144672442943115842689
N_INSTANCES = 32
REWARD_SCALE = 32000.0

BASE = {
    "n_states": 30, "n_contexts": 3, "n_actions": 4, "obs_dim": 16,
    "termination_prob": 0.05, "demo_temperature": 0.10,
    "opportunistic_gain": (0.6, 0.9), "opportunistic_loss": (0.6, 0.9),
    "robust_mean": (-0.6, 0.6), "robust_context_sd": 0.15, "reward_noise": 0.10,
    "n_next_candidates": 3, "transition_concentration": 0.5,
    "n_transitions": 10_000, "min_demo_advantage": 1.0,
}
EXPERIMENTS = {
    "confounded_expert_demos": {**BASE, "logging_epsilon": 0.10, "demo_skill_noise": 0.60, "p_opportunistic": 0.40},
    "pervasive_confounding": {**BASE, "logging_epsilon": 0.05, "demo_skill_noise": 0.70, "p_opportunistic": 0.55},
}


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def dense(cand, weights, S):
    _, A, U, C = weights.shape
    T = np.zeros((S, A, U, S))
    for c in range(C):
        for u in range(U):
            np.add.at(T[:, :, u, :], (np.arange(S)[:, None], np.arange(A)[None, :], cand[:, :, c]), weights[:, :, u, c])
    return T


def value(Pu, reward, T, cont, probs):
    w = Pu[:, :, None] * probs
    R = np.einsum("sua,sau->s", w, reward)
    P = np.einsum("sua,saut->st", w, T)
    return np.linalg.solve(np.eye(len(R)) - cont * P, R)


def make_instance(rng, c):
    S, U, A = c["n_states"], c["n_contexts"], c["n_actions"]
    cont = 1.0 - c["termination_prob"]
    while True:
        Pu = rng.dirichlet(np.ones(U), size=S)
        opp = rng.random((S, A)) < c["p_opportunistic"]
        target = rng.integers(U, size=(S, A))
        gain = rng.uniform(*c["opportunistic_gain"], size=(S, A))
        loss = rng.uniform(*c["opportunistic_loss"], size=(S, A))
        r_opp = np.where(np.arange(U)[None, None, :] == target[..., None], gain[..., None], -loss[..., None])
        mu = rng.uniform(*c["robust_mean"], size=(S, A))
        r_rob = mu[..., None] + c["robust_context_sd"] * rng.standard_normal((S, A, U))
        lim = 1.0 - c["reward_noise"]
        reward = np.clip(np.where(opp[..., None], r_opp, r_rob), -lim, lim)
        cand = rng.integers(S, size=(S, A, c["n_next_candidates"]))
        weights = rng.gamma(c["transition_concentration"], size=(S, A, U, c["n_next_candidates"]))
        weights = weights / weights.sum(-1, keepdims=True)
        T = dense(cand, weights, S)
        V = np.zeros(S)
        for _ in range(5000):
            Qf = reward.transpose(0, 2, 1) + cont * np.einsum("saut,t->sua", T, V)
            V_new = (Pu * Qf.max(-1)).sum(-1)
            if np.max(np.abs(V_new - V)) < 1e-12:
                V = V_new
                break
            V = V_new
        Qf = reward.transpose(0, 2, 1) + cont * np.einsum("saut,t->sua", T, V)
        Qd = Qf + c["demo_skill_noise"] * rng.standard_normal((S, U, A))
        demo = (1 - c["logging_epsilon"]) * softmax(Qd / c["demo_temperature"]) + c["logging_epsilon"] / A
        obs = rng.standard_normal((S, c["obs_dim"]))
        rho0 = np.full(S, 1.0 / S)
        v_demo = rho0 @ value(Pu, reward, T, cont, demo)
        v_rand = rho0 @ value(Pu, reward, T, cont, np.full((S, U, A), 1.0 / A))
        if v_demo - v_rand >= c["min_demo_advantage"]:
            break
    return dict(Pu=Pu, reward=reward, cand=cand, weights=weights, T=T, demo=demo, obs=obs, rho0=rho0)


def collect(inst, c, rng):
    n = c["n_transitions"]
    cum_u = np.cumsum(inst["Pu"], -1).tolist()
    cum_x = np.cumsum(inst["demo"], -1).tolist()
    cum_t = np.cumsum(inst["T"], -1).tolist()
    cum_0 = np.cumsum(inst["rho0"]).tolist()
    reward = inst["reward"].tolist()
    draws = rng.random((n, 5)).tolist()
    out = {k: np.zeros(n, np.uint8) for k in ("state", "action", "next_state", "done")}
    y_arr = np.zeros(n)

    def pick(cum, r):
        return min(bisect_right(cum, r), len(cum) - 1)

    s = pick(cum_0, rng.random())
    for t in range(n):
        ru, rx, ry, rs, rd = draws[t]
        u = pick(cum_u[s], ru)
        x = pick(cum_x[s][u], rx)
        y_arr[t] = reward[s][x][u] + c["reward_noise"] * (2.0 * ry - 1.0)
        s2 = pick(cum_t[s][x][u], rs)
        done = rd < c["termination_prob"]
        out["state"][t], out["action"][t], out["next_state"][t], out["done"][t] = s, x, s2, done
        s = pick(cum_0, rng.random()) if done else s2
    out["reward"] = y_arr
    return out


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for exp_index, (experiment, c) in enumerate(EXPERIMENTS.items()):
        data_path = ROOT / "data" / "test" / f"{experiment}.json"
        env_path = ROOT / "eval" / "test" / "envs" / f"{experiment}.json"
        if data_path.exists() and env_path.exists() and not args.force:
            print(f"keep {data_path} and {env_path}")
            continue
        records, envs = [], []
        for i in range(N_INSTANCES):
            rng = np.random.default_rng(np.random.SeedSequence([SECRET_ENTROPY, exp_index, i]))
            inst = make_instance(rng, c)
            log = collect(inst, c, rng)
            iid = f"test_{i:02d}"
            records.append({
                "id": iid,
                "obs": np.round(inst["obs"], 6).tolist(),
                "n_transitions": int(len(log["state"])),
                "state": b64(log["state"]),
                "action": b64(log["action"]),
                "reward_q": b64(np.round(log["reward"] * REWARD_SCALE).astype("<i2")),
                "next_state": b64(log["next_state"]),
                "done": b64(log["done"]),
            })
            r = lambda a: np.round(np.asarray(a, dtype=float), 12).tolist()  # noqa: E731
            envs.append({
                "id": iid,
                "context_probs": r(inst["Pu"]),
                "reward": r(inst["reward"]),
                "next_candidates": inst["cand"].tolist(),
                "next_weights": r(inst["weights"]),
                "demo_policy": r(inst["demo"]),
                "initial_dist": r(inst["rho0"]),
                "termination_prob": c["termination_prob"],
            })
        header = {
            "experiment": experiment, "split": "test",
            "n_states": c["n_states"], "n_actions": c["n_actions"], "obs_dim": c["obs_dim"],
            "reward_range": [-1.0, 1.0], "termination_prob": c["termination_prob"],
            "logging_epsilon": c["logging_epsilon"],
            "format": "base64 little-endian arrays; reward = reward_q / 32000",
        }
        data_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text(json.dumps({**header, "instances": records}, separators=(",", ":")))
        env_path.write_text(json.dumps({**header, "instances": envs}, separators=(",", ":")))
        print(f"wrote {data_path} and {env_path}")


if __name__ == "__main__":
    main()
