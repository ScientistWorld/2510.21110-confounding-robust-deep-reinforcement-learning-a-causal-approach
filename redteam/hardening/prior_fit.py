"""Prior-fitting shortcut: learn the public generator's prior, apply it to logs.

The family generator is public, so a method can simulate unlimited instances
together with their true dynamics and fit a model from log statistics to the
true interventional answer. That is "writing down the generating process" in
learned form. This module implements the strongest simple version we could
think of, so the hidden test can measure what it is worth:

* simulate ``M`` instances from the public generator (same experiment config);
* per (state, action), compute ~50 log statistics: logged-action frequency,
  sample counts, reward moments and quantiles, lower/upper tail means at the
  documented logging level, next-state statistics, and the Q-values of four
  tabular estimators (nominal, the paper's bound, the sharp epsilon bound and
  its upper counterpart) plus their within-state advantages;
* label every (state, action) with its true optimal interventional advantage
  (``policy``) or true interventional reward (``model``);
* fit an MLP; act greedily on the predicted advantage (``policy``) or plan on
  the predicted rewards with the observed transitions (``model``).
"""

from __future__ import annotations

import numpy as np
import torch

from redteam.hardening import tabular

FEATURE_KINDS = ("nominal", "manski", "sharp", "sharp_upper")
QUANTILES = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)


def simulate(family, cfg, n: int, seed) -> tuple[list[dict], list[dict]]:
    """n instances (generator dicts) and their logs, from the public family."""
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    insts = []
    for _ in range(n):
        inst = family.generate_instance(rng, cfg)
        if "reward_noise" not in inst:
            inst["reward_noise"] = float(cfg.reward_noise)
        insts.append(inst)
    from data.cmdp_family import collect_batch  # vectorised collector; works for any instance dicts
    logs = []
    for i in range(0, n, 128):
        logs.extend(collect_batch(insts[i:i + 128], cfg.n_transitions, rng))
    for log in logs:
        log["reward"] = np.round(np.clip(log["reward"], -1, 1) * 32000) / 32000  # same quantisation as the files
    return insts, logs


def tabular_qs(logs: list[dict], S: int, A: int, eps: float, term: float) -> dict[str, np.ndarray]:
    t = tabular.stack_logs(logs)
    return {k: tabular.solve(t, S, A, eps, term, k) for k in FEATURE_KINDS}


def _group_tails(key, vals, alpha, n_sa, SA):
    order = np.lexsort((vals, key))
    ks, vs = key[order], vals[order]
    starts = np.concatenate([[0.0], np.cumsum(n_sa)[:-1]])
    pos = np.arange(len(vs)) - starts[ks]
    mass = alpha[ks] * n_sa[ks]
    w = np.clip(mass - pos, 0, 1)
    lo = np.bincount(ks, weights=w * vs, minlength=SA) / np.maximum(np.bincount(ks, weights=w, minlength=SA), 1e-12)
    pos_hi = (starts[ks] + n_sa[ks] - 1) - np.arange(len(vs))
    w_hi = np.clip(mass - pos_hi, 0, 1)
    hi = np.bincount(ks, weights=w_hi * vs, minlength=SA) / np.maximum(np.bincount(ks, weights=w_hi, minlength=SA), 1e-12)
    return lo, hi, ks, vs, starts


def features(log: dict, S: int, A: int, eps: float, qs: dict[str, np.ndarray]) -> np.ndarray:
    """Per (state, action) features of one logged instance -> [S*A, F]."""
    SA = S * A
    s = np.asarray(log["state"], dtype=np.int64)
    x = np.asarray(log["action"], dtype=np.int64)
    y = np.asarray(log["reward"], dtype=float)
    s2 = np.asarray(log["next_state"], dtype=np.int64)
    key = s * A + x
    n_sa = np.bincount(key, minlength=SA).astype(float)
    n_s = np.bincount(s, minlength=S).astype(float)
    p_sa = n_sa / np.maximum(np.repeat(n_s, A), 1.0)
    alpha = np.clip(eps / (A * np.maximum(p_sa, 1e-12)), 0.0, 1.0)
    has = (n_sa > 0).astype(float)
    lo, hi, ks, ys, starts = _group_tails(key, y, alpha, n_sa, SA)
    quants = []
    for f in QUANTILES:
        idx = (starts + np.floor(f * np.maximum(n_sa - 1, 0))).astype(np.int64)
        idx = np.minimum(idx, len(ys) - 1)
        quants.append(np.where(has > 0, ys[idx], 0.0))
    mean = np.bincount(key, weights=y, minlength=SA) / np.maximum(n_sa, 1)
    sq = np.bincount(key, weights=y ** 2, minlength=SA) / np.maximum(n_sa, 1)
    sd = np.sqrt(np.maximum(sq - mean ** 2, 0.0))
    v_nom = qs["nominal"].max(1)
    vnext = v_nom[s2]
    v_lo, v_hi, _, _, _ = _group_tails(key, vnext, alpha, n_sa, SA)
    v_mean = np.bincount(key, weights=vnext, minlength=SA) / np.maximum(n_sa, 1)
    cnt = np.bincount(key * S + s2, minlength=SA * S).reshape(SA, S)
    pr = cnt / np.maximum(n_sa[:, None], 1)
    ent = -(pr * np.log(np.where(pr > 0, pr, 1.0))).sum(1)
    n_next = (cnt > 0).sum(1).astype(float)
    P = p_sa.reshape(S, A)
    sorted_p = np.repeat(-np.sort(-P, axis=1), A, axis=0)
    rank = np.argsort(np.argsort(-P, axis=1), axis=1).ravel() / max(A - 1, 1)
    cols = [p_sa, np.log1p(n_sa), np.repeat(np.log1p(n_s), A), alpha, has, mean, sd, lo, hi, *quants,
            v_mean, v_lo, v_hi, ent, n_next, rank, np.full(SA, eps)]
    cols.extend(sorted_p.T)
    for kind in FEATURE_KINDS:
        Qk = qs[kind]
        cols.append(Qk.ravel())
        cols.append((Qk - Qk.max(1, keepdims=True)).ravel())
    return np.column_stack(cols).astype(np.float32)


def dataset(logs, qs_all, S, A, eps) -> np.ndarray:
    return np.stack([features(log, S, A, eps, {k: v[i] for k, v in qs_all.items()}) for i, log in enumerate(logs)])


class Net(torch.nn.Module):
    def __init__(self, f: int, hidden: int = 256):
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(f, hidden), torch.nn.ReLU(),
                                        torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
                                        torch.nn.Linear(hidden, 1))

    def forward(self, x):
        return self.body(x).squeeze(-1)


def fit(X: np.ndarray, target: np.ndarray, S: int, A: int, regret: bool, steps: int = 4000, seed: int = 0) -> dict:
    """X [N, S*A, F], target [N, S, A]. Returns a fitted predictor."""
    dev = tabular.device()
    torch.manual_seed(seed)
    F = X.shape[-1]
    Xs = torch.as_tensor(X.reshape(-1, A, F), device=dev)          # [N*S, A, F]
    Ys = torch.as_tensor(target.reshape(-1, A), device=dev, dtype=torch.float32)
    mu = Xs.reshape(-1, F).mean(0)
    sd = Xs.reshape(-1, F).std(0).clamp(min=1e-6)
    net = Net(F).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    g = torch.Generator(device=dev).manual_seed(seed)
    n = Xs.shape[0]
    for step in range(steps):
        idx = torch.randint(0, n, (min(1024, n),), generator=g, device=dev)
        xb, yb = (Xs[idx] - mu) / sd, Ys[idx]
        pred = net(xb)
        loss = ((pred - yb) ** 2).mean()
        if regret:
            pi = torch.softmax(pred / 0.05, -1)
            loss = loss + (pi * (yb.max(-1, keepdim=True).values - yb)).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    net.eval()
    return {"net": net, "mu": mu, "sd": sd}


@torch.no_grad()
def predict(model: dict, X: np.ndarray, A: int) -> np.ndarray:
    dev = tabular.device()
    F = X.shape[-1]
    Xs = torch.as_tensor(X.reshape(-1, A, F), device=dev)
    out = model["net"]((Xs - model["mu"]) / model["sd"])
    return out.cpu().numpy().reshape(X.shape[0], -1, A)


def plan_on_predicted_rewards(R: np.ndarray, log: dict, S: int, A: int, term: float) -> list[int]:
    """Value iteration with predicted interventional rewards and logged transitions."""
    s = np.asarray(log["state"], dtype=np.int64)
    x = np.asarray(log["action"], dtype=np.int64)
    s2 = np.asarray(log["next_state"], dtype=np.int64)
    cnt = np.bincount((s * A + x) * S + s2, minlength=S * A * S).reshape(S, A, S).astype(float)
    tot = cnt.sum(-1, keepdims=True)
    marg = cnt.sum(1)
    marg = marg / np.maximum(marg.sum(-1, keepdims=True), 1)
    P = np.where(tot > 0, cnt / np.maximum(tot, 1), marg[:, None, :])
    cont = 1.0 - term
    V = np.zeros(S)
    for _ in range(5000):
        Q = R + cont * P @ V
        V_new = Q.max(1)
        if np.max(np.abs(V_new - V)) < 1e-10:
            break
        V = V_new
    return [int(a) for a in (R + cont * P @ V).argmax(1)]


def train_prior_fit(family, cfg, n_sim: int, seed, kinds=("policy", "model")) -> dict:
    """Simulate from the public family and fit both predictors."""
    S, A = cfg.n_states, cfg.n_actions
    eps, term = cfg.logging_epsilon, cfg.termination_prob
    insts, logs = simulate(family, cfg, n_sim, seed)
    qs = tabular_qs(logs, S, A, eps, term)
    X = dataset(logs, qs, S, A, eps)
    envs = [tabular.env_from_instance(i) for i in insts]
    models = {}
    if "policy" in kinds:
        adv = np.stack([e["Qstar"] - e["Qstar"].max(1, keepdims=True) for e in envs])
        models["policy"] = fit(X, adv, S, A, regret=True)
    if "model" in kinds:
        R = np.stack([e["Rsa"] for e in envs])
        models["model"] = fit(X, R, S, A, regret=False)
    return models


def apply_prior_fit(models: dict, logs: list[dict], S: int, A: int, eps: float, term: float) -> dict[str, list]:
    qs = tabular_qs(logs, S, A, eps, term)
    X = dataset(logs, qs, S, A, eps)
    out = {}
    if "policy" in models:
        adv = predict(models["policy"], X, A)
        out["policy"] = [[int(a) for a in row.argmax(-1)] for row in adv]
    if "model" in models:
        R = predict(models["model"], X, A)
        out["model"] = [plan_on_predicted_rewards(R[i], logs[i], S, A, term) for i in range(len(logs))]
    return out
