"""Public problem definition: a family of confounded Markov decision processes.

Each instance is a finite CMDP in the sense of the paper's Definition 2.1.
At every step an exogenous hidden context ``u`` is drawn from ``P(u | s)``.
The demonstrator observes ``(s, u)`` and acts; the learner only ever observes
``s`` (through a feature vector), the logged action, a noisy reward and the
next observed state. Rewards and next-state distributions depend on ``u``, so
the logged action-outcome association is confounded.

This module contains

* the generator for the family (used to build the public training split and
  free to be used for extra development instances), and
* the loader for the logged demonstration datasets in ``data/<split>/``.

Held-out test instances are drawn from the same family with an undisclosed
seed; their datasets are in ``data/test/`` but their true dynamics are not
part of the public workspace.
"""

from __future__ import annotations

import base64
import json
from bisect import bisect_right
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent
REWARD_FLOOR = -1.0
REWARD_CEILING = 1.0


@dataclass(frozen=True)
class FamilyConfig:
    n_states: int = 30
    n_contexts: int = 3
    n_actions: int = 4
    obs_dim: int = 16
    termination_prob: float = 0.05
    logging_epsilon: float = 0.10
    demo_temperature: float = 0.10
    demo_skill_noise: float = 0.60
    p_opportunistic: float = 0.40
    opportunistic_gain: tuple[float, float] = (0.6, 0.9)
    opportunistic_loss: tuple[float, float] = (0.6, 0.9)
    robust_mean: tuple[float, float] = (-0.6, 0.6)
    robust_context_sd: float = 0.15
    reward_noise: float = 0.10
    n_next_candidates: int = 3
    transition_concentration: float = 0.5
    n_transitions: int = 10_000
    min_demo_advantage: float = 1.0


EXPERIMENTS: dict[str, FamilyConfig] = {
    # Main setting: a capable demonstrator with moderate reliance on the hidden
    # context and 10% uniformly random logged actions.
    "confounded_expert_demos": FamilyConfig(
        logging_epsilon=0.10, demo_skill_noise=0.60, p_opportunistic=0.40
    ),
    # Harder setting: the demonstrator relies on the hidden context for more
    # of its choices and the log contains fewer random actions.
    "pervasive_confounding": FamilyConfig(
        logging_epsilon=0.05, demo_skill_noise=0.70, p_opportunistic=0.55
    ),
}


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------


def _softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def dense_transitions(cand: np.ndarray, weights: np.ndarray, n_states: int) -> np.ndarray:
    """cand [S,A,C] int, weights [S,A,U,C] -> dense T [S,A,U,S]."""
    S, A, U, C = weights.shape
    T = np.zeros((S, A, U, n_states))
    for c in range(C):
        idx = cand[:, :, c]
        for u in range(U):
            np.add.at(T[:, :, u, :], (np.arange(S)[:, None], np.arange(A)[None, :], idx), weights[:, :, u, c])
    return T


def joint_value(Pu, reward, T, cont, probs) -> np.ndarray:
    """Exact state values of a context-dependent policy probs[S,U,A]."""
    S = Pu.shape[0]
    w = Pu[:, :, None] * probs  # [S,U,A]
    R = np.einsum("sua,sau->s", w, reward)
    P = np.einsum("sua,saut->st", w, T)
    return np.linalg.solve(np.eye(S) - cont * P, R)


def generate_instance(rng: np.random.Generator, cfg: FamilyConfig) -> dict:
    S, U, A = cfg.n_states, cfg.n_contexts, cfg.n_actions
    cont = 1.0 - cfg.termination_prob
    while True:
        Pu = rng.dirichlet(np.ones(U), size=S)
        opp = rng.random((S, A)) < cfg.p_opportunistic
        target = rng.integers(U, size=(S, A))
        gain = rng.uniform(*cfg.opportunistic_gain, size=(S, A))
        loss = rng.uniform(*cfg.opportunistic_loss, size=(S, A))
        r_opp = np.where(np.arange(U)[None, None, :] == target[..., None], gain[..., None], -loss[..., None])
        mu = rng.uniform(*cfg.robust_mean, size=(S, A))
        r_rob = mu[..., None] + cfg.robust_context_sd * rng.standard_normal((S, A, U))
        lim = REWARD_CEILING - cfg.reward_noise
        reward = np.clip(np.where(opp[..., None], r_opp, r_rob), -lim, lim)  # [S,A,U]
        cand = rng.integers(S, size=(S, A, cfg.n_next_candidates))
        weights = rng.gamma(cfg.transition_concentration, size=(S, A, U, cfg.n_next_candidates))
        weights = weights / weights.sum(-1, keepdims=True)
        T = dense_transitions(cand, weights, S)

        # Demonstrator: plans with full information (s, u), with an imperfect
        # (noisy) value estimate, Boltzmann action choice and epsilon-uniform
        # logging noise.
        V = np.zeros(S)
        for _ in range(5000):
            Qf = reward.transpose(0, 2, 1) + cont * np.einsum("saut,t->sua", T, V)
            V_new = (Pu * Qf.max(-1)).sum(-1)
            if np.max(np.abs(V_new - V)) < 1e-12:
                V = V_new
                break
            V = V_new
        Qf = reward.transpose(0, 2, 1) + cont * np.einsum("saut,t->sua", T, V)
        Qd = Qf + cfg.demo_skill_noise * rng.standard_normal((S, U, A))
        demo = (1 - cfg.logging_epsilon) * _softmax(Qd / cfg.demo_temperature) + cfg.logging_epsilon / A
        obs = rng.standard_normal((S, cfg.obs_dim))
        rho0 = np.full(S, 1.0 / S)

        v_demo = rho0 @ joint_value(Pu, reward, T, cont, demo)
        v_rand = rho0 @ joint_value(Pu, reward, T, cont, np.full((S, U, A), 1.0 / A))
        if v_demo - v_rand >= cfg.min_demo_advantage:
            break
    return {
        "context_probs": Pu,
        "reward": reward,
        "next_candidates": cand,
        "next_weights": weights,
        "demo_policy": demo,
        "obs": obs,
        "initial_dist": rho0,
        "termination_prob": cfg.termination_prob,
    }


def collect_demonstrations(inst: dict, cfg: FamilyConfig, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Log ``cfg.n_transitions`` consecutive demonstrator transitions."""
    S, U, A = cfg.n_states, cfg.n_contexts, cfg.n_actions
    n = cfg.n_transitions
    cum_u = np.cumsum(inst["context_probs"], -1).tolist()
    cum_x = np.cumsum(inst["demo_policy"], -1).tolist()
    T = dense_transitions(inst["next_candidates"], inst["next_weights"], S)
    cum_t = np.cumsum(T, -1).tolist()
    cum_0 = np.cumsum(inst["initial_dist"]).tolist()
    reward = inst["reward"].tolist()
    draws = rng.random((n, 5)).tolist()
    s_arr = np.zeros(n, np.uint8)
    x_arr = np.zeros(n, np.uint8)
    y_arr = np.zeros(n, np.float64)
    s2_arr = np.zeros(n, np.uint8)
    d_arr = np.zeros(n, np.uint8)

    def pick(cum, r):
        return min(bisect_right(cum, r), len(cum) - 1)

    s = pick(cum_0, rng.random())
    for t in range(n):
        ru, rx, ry, rs, rd = draws[t]
        u = pick(cum_u[s], ru)
        x = pick(cum_x[s][u], rx)
        y = reward[s][x][u] + cfg.reward_noise * (2.0 * ry - 1.0)
        s2 = pick(cum_t[s][x][u], rs)
        done = rd < inst["termination_prob"]
        s_arr[t], x_arr[t], y_arr[t], s2_arr[t], d_arr[t] = s, x, y, s2, done
        s = pick(cum_0, rng.random()) if done else s2
    return {"state": s_arr, "action": x_arr, "reward": y_arr, "next_state": s2_arr, "done": d_arr}


# ---------------------------------------------------------------------------
# Dataset files
# ---------------------------------------------------------------------------

_REWARD_SCALE = 32000.0


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def _unb64(text: str, dtype) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text), dtype=dtype).copy()


def dataset_record(instance_id: str, inst: dict, log: dict[str, np.ndarray]) -> dict:
    return {
        "id": instance_id,
        "obs": np.round(inst["obs"], 6).tolist(),
        "n_transitions": int(len(log["state"])),
        "state": _b64(log["state"].astype(np.uint8)),
        "action": _b64(log["action"].astype(np.uint8)),
        "reward_q": _b64(np.round(log["reward"] * _REWARD_SCALE).astype("<i2")),
        "next_state": _b64(log["next_state"].astype(np.uint8)),
        "done": _b64(log["done"].astype(np.uint8)),
    }


def dataset_header(experiment: str, split: str, cfg: FamilyConfig) -> dict:
    return {
        "experiment": experiment,
        "split": split,
        "n_states": cfg.n_states,
        "n_actions": cfg.n_actions,
        "obs_dim": cfg.obs_dim,
        "reward_range": [REWARD_FLOOR, REWARD_CEILING],
        "termination_prob": cfg.termination_prob,
        "logging_epsilon": cfg.logging_epsilon,
        "format": "base64 little-endian arrays; reward = reward_q / 32000",
    }


def load_dataset(split: str, experiment: str, root: Path | str = DATA_DIR) -> dict:
    """Load one split/experiment dataset file into numpy arrays."""
    payload = json.loads((Path(root) / split / f"{experiment}.json").read_text())
    out = {k: v for k, v in payload.items() if k != "instances"}
    instances = []
    for rec in payload["instances"]:
        instances.append({
            "id": rec["id"],
            "obs": np.asarray(rec["obs"], dtype=np.float32),
            "state": _unb64(rec["state"], np.uint8).astype(np.int64),
            "action": _unb64(rec["action"], np.uint8).astype(np.int64),
            "reward": _unb64(rec["reward_q"], "<i2").astype(np.float32) / _REWARD_SCALE,
            "next_state": _unb64(rec["next_state"], np.uint8).astype(np.int64),
            "done": _unb64(rec["done"], np.uint8).astype(np.float32),
        })
    out["instances"] = instances
    return out


def env_record(instance_id: str, inst: dict) -> dict:
    """Serialise the true dynamics of one instance (kept with its evaluator)."""
    r = lambda a: np.round(np.asarray(a, dtype=float), 12).tolist()  # noqa: E731
    return {
        "id": instance_id,
        "context_probs": r(inst["context_probs"]),
        "reward": r(inst["reward"]),
        "next_candidates": np.asarray(inst["next_candidates"]).tolist(),
        "next_weights": r(inst["next_weights"]),
        "demo_policy": r(inst["demo_policy"]),
        "initial_dist": r(inst["initial_dist"]),
        "termination_prob": float(inst["termination_prob"]),
    }


def config_dict(cfg: FamilyConfig) -> dict:
    return asdict(cfg)


__all__ = [
    "EXPERIMENTS",
    "FamilyConfig",
    "REWARD_FLOOR",
    "collect_demonstrations",
    "config_dict",
    "dataset_header",
    "dataset_record",
    "env_record",
    "generate_instance",
    "load_dataset",
    "replace",
]
