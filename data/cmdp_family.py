"""Public problem definition: a family of confounded Markov decision processes.

Each instance is a finite CMDP in the sense of the paper's Definition 2.1.
At every step an exogenous hidden context ``u`` is drawn afresh from a
state-dependent distribution ``P(u | s)`` over a set of context points in a
continuous space. The demonstrator observes ``(s, u)`` and acts; the learner
only ever observes ``s`` (through a feature vector), the logged action, a
noisy reward and the next observed state. Every action's reward and
next-state distribution respond smoothly, and to a different degree, to the
hidden context, so the logged action-outcome association is confounded.

Instances are deliberately heterogeneous: the number and spread of context
points, how strongly each action reacts to the context, the reward noise and
the demonstrator's skill and decisiveness are drawn per instance.

Every action's reward is higher in a favourable region of the hidden context
space and lower elsewhere; how far apart the two are, how large the
favourable region is and how sharp the boundary is differ by state-action
pair. Next-state probabilities also react to the context.

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

Range = tuple[float, float]


@dataclass(frozen=True)
class FamilyConfig:
    """Family parameters. ``(lo, hi)`` ranges are drawn once per instance
    (per-(state, action) ranges are drawn for every state-action pair)."""

    n_states: int = 30
    n_actions: int = 4
    obs_dim: int = 16
    termination_prob: float = 0.05
    logging_epsilon: float = 0.10
    n_transitions: int = 30_000
    # Hidden context: per state, ``n_contexts`` points in R^context_dim with
    # Dirichlet(concentration) probabilities; one is drawn at every step.
    n_contexts: tuple[int, int] = (12, 32)
    context_dim: int = 2
    context_concentration: Range = (0.3, 3.0)          # log-uniform, per instance
    # Reward of action a in state s under context u:
    #   mu + amp * (2 * sigmoid(sharpness * (<dir, u> - threshold)) - 1)
    # i.e. +amp above mu in a favourable region of context space, -amp
    # elsewhere, with a graded transition between the two. amp is
    # sensitivity_max * Beta(shape_a, shape_b); everything is per (s, a).
    reward_mean: Range = (-0.3, 0.5)
    sensitivity_max: Range = (0.7, 0.9)                # per instance
    sensitivity_shape_a: Range = (0.7, 1.2)            # per instance
    sensitivity_shape_b: Range = (1.0, 1.8)            # per instance
    favorable_threshold: Range = (-0.3, 1.0)           # per (s, a)
    response_sharpness: Range = (1.5, 6.0)             # per (s, a)
    reward_noise: Range = (0.05, 0.2)                  # uniform half-width, per instance
    # Transitions: a few candidate next states per (s, a); their logits react
    # to the context with an action-specific strength.
    n_next_candidates: int = 3
    transition_base_sd: float = 1.0
    transition_sensitivity_max: Range = (2.0, 4.0)     # per instance
    # Demonstrator: plans with full information (s, u), then acts by a
    # Boltzmann choice over noisy action values; with probability
    # ``logging_epsilon`` the logged action is uniform instead.
    demo_temperature: Range = (0.03, 0.3)              # log-uniform, per instance
    demo_skill_noise: Range = (0.3, 0.7)               # per instance
    min_demo_advantage: float = 1.0


EXPERIMENTS: dict[str, FamilyConfig] = {
    # Main setting: actions react moderately to the hidden context; 10%
    # uniformly random logged actions.
    "confounded_expert_demos": FamilyConfig(
        logging_epsilon=0.10,
        sensitivity_shape_a=(0.7, 1.2), sensitivity_shape_b=(1.0, 1.8),
        demo_skill_noise=(0.3, 0.7),
    ),
    # Harder setting: actions react more strongly to the hidden context, so
    # the demonstrator relies on it more, and only 5% of logged actions are
    # uniformly random.
    "pervasive_confounding": FamilyConfig(
        logging_epsilon=0.05,
        sensitivity_shape_a=(1.0, 1.8), sensitivity_shape_b=(0.7, 1.2),
        demo_skill_noise=(0.4, 0.8),
    ),
}


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------


def _softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _draw(rng: np.random.Generator, r, log: bool = False) -> float:
    lo, hi = float(r[0]), float(r[1])
    if lo == hi:
        return lo
    if log:
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return float(rng.uniform(lo, hi))


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


def draw_hyperparameters(rng: np.random.Generator, cfg: FamilyConfig) -> dict:
    return {
        "n_contexts": int(rng.integers(int(cfg.n_contexts[0]), int(cfg.n_contexts[1]) + 1)),
        "context_concentration": _draw(rng, cfg.context_concentration, log=True),
        "sensitivity_max": _draw(rng, cfg.sensitivity_max),
        "sensitivity_shape_a": _draw(rng, cfg.sensitivity_shape_a),
        "sensitivity_shape_b": _draw(rng, cfg.sensitivity_shape_b),
        "reward_noise": _draw(rng, cfg.reward_noise),
        "transition_sensitivity_max": _draw(rng, cfg.transition_sensitivity_max),
        "demo_temperature": _draw(rng, cfg.demo_temperature, log=True),
        "demo_skill_noise": _draw(rng, cfg.demo_skill_noise),
    }


def generate_instance(rng: np.random.Generator, cfg: FamilyConfig) -> dict:
    S, A, D, C = cfg.n_states, cfg.n_actions, cfg.context_dim, cfg.n_next_candidates
    cont = 1.0 - cfg.termination_prob
    hp = draw_hyperparameters(rng, cfg)
    U = hp["n_contexts"]
    while True:
        Pu = rng.dirichlet(np.full(U, hp["context_concentration"]), size=S)  # [S,U]
        atoms = rng.standard_normal((S, U, D))
        mu = rng.uniform(*cfg.reward_mean, size=(S, A))
        amp = hp["sensitivity_max"] * rng.beta(hp["sensitivity_shape_a"], hp["sensitivity_shape_b"], size=(S, A))
        direction = _unit(rng.standard_normal((S, A, D)))
        threshold = rng.uniform(*cfg.favorable_threshold, size=(S, A))
        sharpness = rng.uniform(*cfg.response_sharpness, size=(S, A))
        proj = np.einsum("sad,sud->sau", direction, atoms)  # [S,A,U]
        favour = 1.0 / (1.0 + np.exp(-sharpness[..., None] * (proj - threshold[..., None])))
        lim = REWARD_CEILING - hp["reward_noise"]
        reward = np.clip(mu[..., None] + amp[..., None] * (2.0 * favour - 1.0), -lim, lim)  # [S,A,U]
        cand = rng.integers(S, size=(S, A, C))
        base = cfg.transition_base_sd * rng.standard_normal((S, A, C))
        amp_t = hp["transition_sensitivity_max"] * rng.beta(hp["sensitivity_shape_a"], hp["sensitivity_shape_b"],
                                                            size=(S, A))
        dir_t = _unit(rng.standard_normal((S, A, C, D)))
        proj_t = np.einsum("sacd,sud->sauc", dir_t, atoms)  # [S,A,U,C]
        weights = _softmax(base[:, :, None, :] + amp_t[:, :, None, None] * proj_t, axis=-1)
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
        Qd = Qf + hp["demo_skill_noise"] * rng.standard_normal((S, U, A))
        demo = (1 - cfg.logging_epsilon) * _softmax(Qd / hp["demo_temperature"]) + cfg.logging_epsilon / A
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
        "reward_noise": hp["reward_noise"],
        "hyper": hp,
    }


def collect_demonstrations(inst: dict, cfg: FamilyConfig, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Log ``cfg.n_transitions`` consecutive demonstrator transitions."""
    S = cfg.n_states
    n = cfg.n_transitions
    noise = float(inst["reward_noise"])
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
        y = reward[s][x][u] + noise * (2.0 * ry - 1.0)
        s2 = pick(cum_t[s][x][u], rs)
        done = rd < inst["termination_prob"]
        s_arr[t], x_arr[t], y_arr[t], s2_arr[t], d_arr[t] = s, x, y, s2, done
        s = pick(cum_0, rng.random()) if done else s2
    return {"state": s_arr, "action": x_arr, "reward": y_arr, "next_state": s2_arr, "done": d_arr}


def collect_batch(insts: list[dict], n_transitions: int, rng: np.random.Generator) -> list[dict[str, np.ndarray]]:
    """Vectorised ``collect_demonstrations`` for many instances at once.

    Same distribution as the sequential collector (not the same random
    stream); handy for simulating many development instances quickly.
    """
    N = len(insts)
    S = insts[0]["context_probs"].shape[0]
    A = insts[0]["reward"].shape[1]
    U = max(inst["context_probs"].shape[1] for inst in insts)
    n_ctx = np.array([inst["context_probs"].shape[1] for inst in insts])
    cum_u = np.ones((N, S, U))
    cum_x = np.ones((N, S, U, A))
    cum_t = np.ones((N, S, A, U, S))
    reward = np.zeros((N, S, A, U))
    for i, inst in enumerate(insts):
        k = n_ctx[i]
        cum_u[i, :, :k] = np.cumsum(inst["context_probs"], -1)
        cum_x[i, :, :k] = np.cumsum(inst["demo_policy"], -1)
        T = dense_transitions(inst["next_candidates"], inst["next_weights"], S)
        cum_t[i, :, :, :k] = np.cumsum(T, -1)
        reward[i, :, :, :k] = inst["reward"]
    cum_0 = np.cumsum(np.stack([inst["initial_dist"] for inst in insts]), -1)
    noise = np.array([float(inst["reward_noise"]) for inst in insts])
    term = np.array([float(inst["termination_prob"]) for inst in insts])
    idx = np.arange(N)

    def pick(cum, r, top):
        return np.minimum((cum <= r[:, None]).sum(-1), top)

    out = {k: np.zeros((N, n_transitions), np.uint8) for k in ("state", "action", "next_state", "done")}
    out["reward"] = np.zeros((N, n_transitions))
    s = pick(cum_0, rng.random(N), S - 1)
    for t in range(n_transitions):
        r = rng.random((N, 6))
        u = pick(cum_u[idx, s], r[:, 0], n_ctx - 1)
        x = pick(cum_x[idx, s, u], r[:, 1], A - 1)
        y = reward[idx, s, x, u] + noise * (2.0 * r[:, 2] - 1.0)
        s2 = pick(cum_t[idx, s, x, u], r[:, 3], S - 1)
        done = r[:, 4] < term
        out["state"][:, t], out["action"][:, t], out["reward"][:, t] = s, x, y
        out["next_state"][:, t], out["done"][:, t] = s2, done
        s = np.where(done, pick(cum_0, r[:, 5], S - 1), s2)
    return [{k: v[i] for k, v in out.items()} for i in range(N)]


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
        "reward_q": _b64(np.round(np.clip(log["reward"], REWARD_FLOOR, REWARD_CEILING) * _REWARD_SCALE).astype("<i2")),
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
    r = lambda a: np.round(np.asarray(a, dtype=float), 10).tolist()  # noqa: E731
    return {
        "id": instance_id,
        "context_probs": r(inst["context_probs"]),
        "reward": r(inst["reward"]),
        "next_candidates": np.asarray(inst["next_candidates"]).tolist(),
        "next_weights": r(inst["next_weights"]),
        "demo_policy": r(inst["demo_policy"]),
        "initial_dist": r(inst["initial_dist"]),
        "termination_prob": float(inst["termination_prob"]),
        "hyper": inst.get("hyper", {}),
    }


def config_dict(cfg: FamilyConfig) -> dict:
    return asdict(cfg)


__all__ = [
    "EXPERIMENTS",
    "FamilyConfig",
    "REWARD_FLOOR",
    "collect_batch",
    "collect_demonstrations",
    "config_dict",
    "dataset_header",
    "dataset_record",
    "env_record",
    "generate_instance",
    "load_dataset",
    "replace",
]
