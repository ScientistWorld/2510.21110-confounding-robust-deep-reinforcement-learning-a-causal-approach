"""Batched tabular estimators on logged confounded data, and exact scoring.

Used by the hardening checks only (honest variants, shortcuts, prior-fit
features). All estimators read nothing but the logs and the documented facts
in the dataset header (reward floor, termination probability, logging
epsilon). ``exact_*`` helpers read true dynamics and are used for labels on
simulated development instances and for diagnostics, never by a submitted
method on the test side.

Kinds of value iteration on the empirical observational model:

* ``nominal``   - observational Bellman optimality (treats the log as causal);
* ``manski``    - the paper's causal Bellman lower bound (Prop. 3.1, Eq. 8);
* ``sharp``     - lower bound that also uses the documented logging epsilon:
                  under do(x) the outcome density is at most |A| P(x|s) / eps
                  times the logged density, so the bound is a lower-tail mean
                  of the logged outcomes at level eps / (|A| P(x|s));
* ``sharp_lcb`` - finite-sample version of ``sharp`` (upper confidence bound
                  on P(x|s), tail mean lowered by ``kappa`` standard errors,
                  never below the paper's bound, which is also valid);
* ``sharp_upper`` - the matching upper bound (upper-tail mean);
* ``hurwicz``   - (1 - lam) * lower + lam * upper of the sharp interval;
* ``partition`` - the tighter bound implied by the same logging protocol.
                  The deliberate (non-random) choices of x happen in a set
                  C_x of hidden contexts with P(C_x) = (P(x|s) - eps/|A|) /
                  (1 - eps), and the random picks of x sample the contexts
                  independently, so the logged outcomes of x are a mixture
                  of outcomes in C_x (mass 1 - b) and outside it (mass
                  b = alpha (1 - P(C_x)), alpha = eps / (|A| P(x|s))). Under
                  do(x) the two parts weigh P(C_x) and 1 - P(C_x). The worst
                  case puts the outside part at the lowest b of the logged
                  outcomes:  P(C)(mean - b tail_b) / (1 - b) + (1 - P(C)) tail_b;
* ``partition_upper`` / ``partition_lcb`` / ``hurwicz_partition`` - its
                  upper counterpart, finite-sample version (mean lowered by
                  kappa standard errors, never below the paper's bound) and
                  (1 - lam) * lower + lam * upper.
"""

from __future__ import annotations

import numpy as np
import torch

GAMMA = 0.99
FLOOR = -1.0


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stack_logs(logs: list[dict], dev: torch.device | None = None) -> dict[str, torch.Tensor]:
    dev = dev or device()
    out = {}
    for k in ("state", "action", "next_state"):
        out[k] = torch.as_tensor(np.stack([np.asarray(l[k], dtype=np.int64) for l in logs]), device=dev)
    for k in ("reward", "done"):
        out[k] = torch.as_tensor(np.stack([np.asarray(l[k], dtype=np.float64) for l in logs]), device=dev)
    return out


def _partition(key, vals, mean, p_sa, eps, A, alpha, n_sa, starts, ar, lower: bool):
    pc = ((p_sa - eps / A) / (1.0 - eps)).clamp(0.0, 1.0)
    b = (alpha * (1.0 - pc)).clamp(0.0, 1.0)
    tail, den = _tail(key, vals, b, n_sa, starts, ar, lower)
    tail = torch.where(den > 1e-12, tail, mean)
    inside = (mean - b * tail) / (1.0 - b).clamp(min=1e-9)
    return torch.where(b >= 1.0 - 1e-9, tail, pc * inside + (1.0 - pc) * tail)


def _tail(key, vals, alpha, n_sa, starts, ar, lower: bool):
    comp = key.double() * 1000.0 + (vals if lower else -vals)
    order = torch.argsort(comp, dim=1)
    sk = key.gather(1, order)
    sv = vals.gather(1, order)
    pos = ar - starts.gather(1, sk)
    mass = (alpha * n_sa).gather(1, sk)
    w = (mass - pos).clamp(0.0, 1.0)
    num = torch.zeros_like(n_sa).scatter_add_(1, sk, w * sv)
    den = torch.zeros_like(n_sa).scatter_add_(1, sk, w)
    return num / den.clamp(min=1e-12), den


@torch.no_grad()
def solve(logs: dict[str, torch.Tensor], S: int, A: int, eps: float, term: float, kind: str,
          iters: int = 600, tol: float = 1e-8, z: float = 2.0, kappa: float = 2.0, lam: float = 0.25,
          chunk: int = 96) -> np.ndarray:
    """Fixed point of the chosen Bellman operator for every instance -> Q [N,S,A]."""
    N = logs["state"].shape[0]
    if N > chunk:
        return np.concatenate([
            solve({k: v[i:i + chunk] for k, v in logs.items()}, S, A, eps, term, kind, iters, tol, z, kappa, lam, chunk)
            for i in range(0, N, chunk)])
    s, x, s2 = logs["state"], logs["action"], logs["next_state"]
    y, d = logs["reward"].double(), logs["done"].double()
    dev = s.device
    T = s.shape[1]
    SA = S * A
    key = s * A + x
    ones = torch.ones_like(y)
    zeros = torch.zeros(N, SA, dtype=torch.float64, device=dev)
    n_sa = zeros.clone().scatter_add_(1, key, ones)
    n_s = torch.zeros(N, S, dtype=torch.float64, device=dev).scatter_add_(1, s, ones)
    n_s_sa = n_s.repeat_interleave(A, dim=1)
    p_sa = n_sa / n_s_sa.clamp(min=1.0)
    seen = torch.zeros(N, S, dtype=torch.float64, device=dev).scatter_(1, s2, 1.0) > 0
    cont_sa = zeros.clone().scatter_add_(1, key, 1.0 - d) / n_sa.clamp(min=1.0)
    cont = 1.0 - term
    alpha = (eps / (A * p_sa.clamp(min=1e-12))).clamp(0.0, 1.0)
    if kind == "sharp_lcb":
        p_ucb = (p_sa + z * torch.sqrt((p_sa * (1 - p_sa)).clamp(min=1e-12) / n_s_sa.clamp(min=1.0))
                 + 1.0 / n_s_sa.clamp(min=1.0)).clamp(max=1.0)
        alpha = (eps / (A * p_ucb.clamp(min=1e-12))).clamp(0.0, 1.0)
    starts = torch.cumsum(n_sa, 1) - n_sa
    ar = torch.arange(T, device=dev, dtype=torch.float64)[None, :].expand(N, T)
    has = n_sa > 0
    V = torch.zeros(N, S, dtype=torch.float64, device=dev)
    inf = torch.full_like(V, float("inf"))
    Q = zeros
    sharp_family = kind in ("sharp", "sharp_lcb", "sharp_upper", "hurwicz", "partition", "partition_upper",
                            "partition_lcb", "hurwicz_partition")
    for _ in range(iters):
        Vn = V.gather(1, s2)
        vals = y + GAMMA * cont * Vn if sharp_family else y + GAMMA * (1.0 - d) * Vn
        vmin = torch.where(seen, V, inf).min(1).values[:, None]
        mean = zeros.clone().scatter_add_(1, key, vals) / n_sa.clamp(min=1.0)
        fallback = (FLOOR + GAMMA * vmin).expand(N, SA)
        if kind == "nominal":
            Q = torch.where(has, mean, fallback)
        elif kind == "manski":
            Q = torch.where(has, p_sa * mean + (1 - p_sa) * (FLOOR + GAMMA * cont_sa * vmin), fallback)
        elif kind.startswith("partition") or kind == "hurwicz_partition":
            Qt = _partition(key, vals, mean, p_sa, eps, A, alpha, n_sa, starts, ar, lower=(kind != "partition_upper"))
            if kind == "hurwicz_partition":
                hi = _partition(key, vals, mean, p_sa, eps, A, alpha, n_sa, starts, ar, lower=False)
                Qt = (1 - lam) * Qt + lam * hi
            if kind == "partition_lcb":
                sq = zeros.clone().scatter_add_(1, key, vals ** 2)
                var = (sq - n_sa * mean ** 2) / (n_sa - 1.0).clamp(min=1.0)
                sd = torch.where(n_sa > 1, var.clamp(min=0.0).sqrt(), torch.full_like(var, 2.0))
                manski = p_sa * mean + (1 - p_sa) * (FLOOR + GAMMA * cont * vmin)
                Qt = torch.maximum(Qt - kappa * sd / n_sa.clamp(min=1.0).sqrt(), manski)
            Q = torch.where(has, Qt, fallback)
        else:
            if kind == "sharp_upper":
                Qt, den = _tail(key, vals, alpha, n_sa, starts, ar, lower=False)
            else:
                Qt, den = _tail(key, vals, alpha, n_sa, starts, ar, lower=True)
            if kind == "hurwicz":
                hi, _ = _tail(key, vals, alpha, n_sa, starts, ar, lower=False)
                Qt = (1 - lam) * Qt + lam * hi
            if kind == "sharp_lcb":
                sq = zeros.clone().scatter_add_(1, key, vals ** 2)
                var = (sq - n_sa * mean ** 2) / (n_sa - 1.0).clamp(min=1.0)
                sd = torch.where(n_sa > 1, var.clamp(min=0.0).sqrt(), torch.full_like(var, 2.0))
                lcb = Qt - kappa * sd / n_sa.clamp(min=1.0).sqrt()
                manski = p_sa * mean + (1 - p_sa) * (FLOOR + GAMMA * cont * vmin)
                Qt = torch.maximum(lcb, manski)
            Q = torch.where(has & (den > 0), Qt, fallback)
        V_new = Q.view(N, S, A).max(-1).values
        delta = (V_new - V).abs().max().item()
        V = V_new
        if delta < tol:
            break
    return Q.view(N, S, A).cpu().numpy()


# ---------------------------------------------------------------------------
# Exact scoring on true dynamics (diagnostics and simulated labels)
# ---------------------------------------------------------------------------


def dense(cand: np.ndarray, weights: np.ndarray, S: int) -> np.ndarray:
    _, A, U, C = weights.shape
    T = np.zeros((S, A, U, S))
    for c in range(C):
        for u in range(U):
            np.add.at(T[:, :, u, :], (np.arange(S)[:, None], np.arange(A)[None, :], cand[:, :, c]), weights[:, :, u, c])
    return T


def env_from_instance(inst: dict) -> dict:
    """Arrays of one instance (generator dict or env JSON record)."""
    Pu = np.asarray(inst["context_probs"], dtype=float)
    reward = np.asarray(inst["reward"], dtype=float)
    S = Pu.shape[0]
    T = dense(np.asarray(inst["next_candidates"]), np.asarray(inst["next_weights"], dtype=float), S)
    env = {"S": S, "A": reward.shape[1], "Pu": Pu, "reward": reward, "T": T,
           "rho0": np.asarray(inst["initial_dist"], dtype=float),
           "cont": 1.0 - float(inst["termination_prob"]),
           "demo": np.asarray(inst["demo_policy"], dtype=float)}
    env["Rsa"] = np.einsum("su,sau->sa", Pu, reward)
    env["Psa"] = np.einsum("su,saut->sat", Pu, T)
    env["v_demo"] = joint_value(env, env["demo"])
    env["v_random"] = joint_value(env, np.full_like(env["demo"], 1.0 / env["A"]))
    env["Qstar"] = optimal_q(env)
    env["v_best"] = float(env["rho0"] @ env["Qstar"].max(1))
    env["v_worst"] = float(env["rho0"] @ optimal_q(env, best=False).min(1))
    return env


def joint_value(env: dict, probs: np.ndarray) -> float:
    w = env["Pu"][:, :, None] * probs
    R = np.einsum("sua,sau->s", w, env["reward"])
    P = np.einsum("sua,saut->st", w, env["T"])
    return float(env["rho0"] @ np.linalg.solve(np.eye(env["S"]) - env["cont"] * P, R))


def optimal_q(env: dict, best: bool = True) -> np.ndarray:
    V = np.zeros(env["S"])
    Q = env["Rsa"]
    for _ in range(20000):
        Q = env["Rsa"] + env["cont"] * env["Psa"] @ V
        V_new = Q.max(1) if best else Q.min(1)
        if np.max(np.abs(V_new - V)) < 1e-12:
            V = V_new
            break
        V = V_new
    return env["Rsa"] + env["cont"] * env["Psa"] @ V


def policy_value(env: dict, policy) -> float:
    policy = np.asarray(policy, dtype=np.int64)
    idx = np.arange(env["S"])
    R = env["Rsa"][idx, policy]
    P = env["Psa"][idx, policy]
    return float(env["rho0"] @ np.linalg.solve(np.eye(env["S"]) - env["cont"] * P, R))


def normalized(env: dict, value: float) -> float:
    return (value - env["v_random"]) / (env["v_demo"] - env["v_random"])


def aggregate(scores: np.ndarray, tail: float = 0.10) -> dict[str, float]:
    flat = np.sort(np.asarray(scores, dtype=float).ravel())
    n = flat.size
    q = n // 4
    k = max(1, int(round(tail * n)))
    return {"mean": float(flat.mean()), "iqm": float(flat[q:n - q].mean()), "worst_decile": float(flat[:k].mean())}


def score_policies(envs: list[dict], policies: list) -> np.ndarray:
    return np.array([normalized(e, policy_value(e, p)) for e, p in zip(envs, policies)])
