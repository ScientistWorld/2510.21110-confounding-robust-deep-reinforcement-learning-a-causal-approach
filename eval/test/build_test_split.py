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
SECRET_ENTROPY = 24217207746181121126165528235288710488
N_INSTANCES = 64
REWARD_SCALE = 32000.0

BASE = {
    "n_states": 30, "n_actions": 4, "obs_dim": 16, "termination_prob": 0.05, "n_transitions": 30_000,
    "n_contexts": (12, 32), "context_dim": 2, "context_concentration": (0.3, 3.0),
    "reward_mean": (-0.3, 0.5), "sensitivity_max": (0.7, 0.9),
    "favorable_threshold": (-0.3, 1.0), "response_sharpness": (1.5, 6.0), "reward_noise": (0.05, 0.2),
    "n_next_candidates": 3, "transition_base_sd": 1.0, "transition_sensitivity_max": (2.0, 4.0),
    "demo_temperature": (0.03, 0.3), "min_demo_advantage": 1.0,
}
EXPERIMENTS = {
    "confounded_expert_demos": {**BASE, "logging_epsilon": 0.10, "sensitivity_shape_a": (0.7, 1.2),
                                "sensitivity_shape_b": (1.0, 1.8), "demo_skill_noise": (0.3, 0.7)},
    "pervasive_confounding": {**BASE, "logging_epsilon": 0.05, "sensitivity_shape_a": (1.0, 1.8),
                              "sensitivity_shape_b": (0.7, 1.2), "demo_skill_noise": (0.4, 0.8)},
}


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def draw(rng, r, log=False):
    lo, hi = float(r[0]), float(r[1])
    if lo == hi:
        return lo
    if log:
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return float(rng.uniform(lo, hi))


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
    S, A, D, C = c["n_states"], c["n_actions"], c["context_dim"], c["n_next_candidates"]
    eps = c["logging_epsilon"]
    cont = 1.0 - c["termination_prob"]
    hp = {
        "n_contexts": int(rng.integers(int(c["n_contexts"][0]), int(c["n_contexts"][1]) + 1)),
        "context_concentration": draw(rng, c["context_concentration"], log=True),
        "sensitivity_max": draw(rng, c["sensitivity_max"]),
        "sensitivity_shape_a": draw(rng, c["sensitivity_shape_a"]),
        "sensitivity_shape_b": draw(rng, c["sensitivity_shape_b"]),
        "reward_noise": draw(rng, c["reward_noise"]),
        "transition_sensitivity_max": draw(rng, c["transition_sensitivity_max"]),
        "demo_temperature": draw(rng, c["demo_temperature"], log=True),
        "demo_skill_noise": draw(rng, c["demo_skill_noise"]),
    }
    U = hp["n_contexts"]
    while True:
        Pu = rng.dirichlet(np.full(U, hp["context_concentration"]), size=S)
        atoms = rng.standard_normal((S, U, D))
        mu = rng.uniform(*c["reward_mean"], size=(S, A))
        amp = hp["sensitivity_max"] * rng.beta(hp["sensitivity_shape_a"], hp["sensitivity_shape_b"], size=(S, A))
        direction = unit(rng.standard_normal((S, A, D)))
        threshold = rng.uniform(*c["favorable_threshold"], size=(S, A))
        sharpness = rng.uniform(*c["response_sharpness"], size=(S, A))
        proj = np.einsum("sad,sud->sau", direction, atoms)
        favour = 1.0 / (1.0 + np.exp(-sharpness[..., None] * (proj - threshold[..., None])))
        lim = 1.0 - hp["reward_noise"]
        reward = np.clip(mu[..., None] + amp[..., None] * (2.0 * favour - 1.0), -lim, lim)
        cand = rng.integers(S, size=(S, A, C))
        base = c["transition_base_sd"] * rng.standard_normal((S, A, C))
        amp_t = hp["transition_sensitivity_max"] * rng.beta(hp["sensitivity_shape_a"], hp["sensitivity_shape_b"],
                                                            size=(S, A))
        dir_t = unit(rng.standard_normal((S, A, C, D)))
        proj_t = np.einsum("sacd,sud->sauc", dir_t, atoms)
        weights = softmax(base[:, :, None, :] + amp_t[:, :, None, None] * proj_t, axis=-1)
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
        Qd = Qf + hp["demo_skill_noise"] * rng.standard_normal((S, U, A))
        demo = (1 - eps) * softmax(Qd / hp["demo_temperature"]) + eps / A
        obs = rng.standard_normal((S, c["obs_dim"]))
        rho0 = np.full(S, 1.0 / S)
        v_demo = rho0 @ value(Pu, reward, T, cont, demo)
        v_rand = rho0 @ value(Pu, reward, T, cont, np.full((S, U, A), 1.0 / A))
        if v_demo - v_rand >= c["min_demo_advantage"]:
            break
    return dict(Pu=Pu, reward=reward, cand=cand, weights=weights, T=T, demo=demo, obs=obs, rho0=rho0,
                noise=hp["reward_noise"], hyper=hp)


def collect(inst, c, rng):
    n = c["n_transitions"]
    cum_u = np.cumsum(inst["Pu"], -1).tolist()
    cum_x = np.cumsum(inst["demo"], -1).tolist()
    cum_t = np.cumsum(inst["T"], -1).tolist()
    cum_0 = np.cumsum(inst["rho0"]).tolist()
    reward = inst["reward"].tolist()
    noise = inst["noise"]
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
        y_arr[t] = reward[s][x][u] + noise * (2.0 * ry - 1.0)
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
                "reward_q": b64(np.round(np.clip(log["reward"], -1.0, 1.0) * REWARD_SCALE).astype("<i2")),
                "next_state": b64(log["next_state"]),
                "done": b64(log["done"]),
            })
            r = lambda a: np.round(np.asarray(a, dtype=float), 10).tolist()  # noqa: E731
            envs.append({
                "id": iid,
                "context_probs": r(inst["Pu"]),
                "reward": r(inst["reward"]),
                "next_candidates": inst["cand"].tolist(),
                "next_weights": r(inst["weights"]),
                "demo_policy": r(inst["demo"]),
                "initial_dist": r(inst["rho0"]),
                "termination_prob": c["termination_prob"],
                "hyper": inst["hyper"],
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
