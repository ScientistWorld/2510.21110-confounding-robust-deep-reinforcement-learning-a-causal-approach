"""Hand-written decoder of the iteration-12 instance family (a shortcut).

The iteration-12 family (``family_v1.py``) gave every action one of two
reward types: "robust" (one level, small context spread) or "opportunistic"
(about +0.75 in one hidden context, about -0.75 in the others). With the
documented logging epsilon, the uniformly random logged actions are an
unbiased sample of the hidden context, so for an opportunistic action the
share of the low cluster identifies P(bad context | s) exactly:

    share_low = (eps / |A|) * (1 - p_good) / P(x | s)

This rule reads that off the log, rebuilds every action's interventional
reward and plans with the logged transitions. It exploits the generator's
two-valued structure, not anything a learner would know on the paper's
problem. On the redesigned family the rule still runs (it falls back to the
logged mean when it does not see two clusters), which shows the structure is
gone.
"""

from __future__ import annotations

import numpy as np

from redteam.hardening.prior_fit import plan_on_predicted_rewards


def hand_decode(log: dict, S: int, A: int, eps: float, term: float, cut: float = 0.45) -> list[int]:
    s = np.asarray(log["state"], dtype=np.int64)
    x = np.asarray(log["action"], dtype=np.int64)
    y = np.asarray(log["reward"], dtype=float)
    key = s * A + x
    n_sa = np.bincount(key, minlength=S * A).astype(float)
    n_s = np.bincount(s, minlength=S).astype(float)
    p_sa = n_sa / np.maximum(np.repeat(n_s, A), 1.0)
    R = np.full(S * A, -1.0)
    order = np.argsort(key, kind="stable")
    bounds = np.concatenate([[0], np.cumsum(n_sa).astype(np.int64)])
    for k in range(S * A):
        ys = y[order[bounds[k]:bounds[k + 1]]]
        if len(ys) == 0:
            continue
        hi, lo = ys[ys > cut], ys[ys < -cut]
        mid = len(ys) - len(hi) - len(lo)
        if len(hi) and len(lo) and mid <= 0.02 * len(ys):
            p_bad = min(1.0, (len(lo) / len(ys)) * p_sa[k] * A / eps)
            R[k] = (1 - p_bad) * hi.mean() + p_bad * lo.mean()
        else:
            R[k] = ys.mean()
    return plan_on_predicted_rewards(R.reshape(S, A), log, S, A, term)
