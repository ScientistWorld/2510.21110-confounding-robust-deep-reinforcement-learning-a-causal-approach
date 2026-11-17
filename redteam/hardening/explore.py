"""Design exploration for the instance family (hardening iteration 13; record).

Round 1 compared the iteration-12 family with continuous-context candidates
(results in explore_results/round1_*); round 2, the version below, compared the
final family structure (``v3*``, with ``data/cmdp_family.py`` at that time
using favourable_threshold=(0.3, 1.5); the chosen ``v3c`` setting
(-0.3, 1.0) is now the default) - results in explore_results/round2_*.

For each candidate family and experiment it draws 32 fresh instances, and
scores (exact interventional return, demonstrator-normalised):

* tabular estimators: nominal, the paper's bound, sharp, sharp_lcb (two
  kappas), Hurwicz mixtures of the sharp interval; behaviour cloning; oracle;
* the hand-written two-cluster decoder (shortcut);
* the prior-fitting shortcut, trained on instances simulated from the same
  public family;
* Causal-DQN and Conf. DQN (one run each, GPU).

It also measures the prior-fit and hand-decode shortcuts on the CURRENT
hidden test (iteration-12 family) with ``eval/test/evaluate_policies.py``.

    PYTHONPATH=. python -m redteam.hardening.explore
"""

from __future__ import annotations

import functools
import json
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

import data.cmdp_family as family_v2
from redteam.hardening import family_v1, prior_fit, tabular
from redteam.hardening.hand_decode import hand_decode

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "redteam" / "hardening" / "explore_results"
TMP = ROOT / "tmp" / "explore"
EXPS = ("confounded_expert_demos", "pervasive_confounding")
N_EVAL = 32
N_SIM = int(sys.argv[1]) if len(sys.argv) > 1 else 512


CANDIDATES = {
    "v1_current": (family_v1, dict(family_v1.EXPERIMENTS)),
    "v3a": (family_v2, dict(family_v2.EXPERIMENTS)),
    "v3b_weak_transitions": (family_v2, {e: replace(c, transition_sensitivity_max=(0.5, 1.5))
                                         for e, c in family_v2.EXPERIMENTS.items()}),
    "v3c_wide_favourable": (family_v2, {e: replace(c, favorable_threshold=(-0.3, 1.0))
                                        for e, c in family_v2.EXPERIMENTS.items()}),
}
WITH_DQN = {"v3a", "v3b_weak_transitions", "v3c_wide_favourable"}


def bc_policy(log, S, A):
    counts = np.zeros((S, A))
    np.add.at(counts, (np.asarray(log["state"]), np.asarray(log["action"])), 1.0)
    fallback = int(counts.sum(0).argmax())
    return [int(r.argmax()) if r.sum() > 0 else fallback for r in counts]


def run_dqn(module_name: str, tag: str, ids: list[str]) -> list:
    import importlib
    mod = importlib.import_module(module_name)
    mod.load_dataset = functools.partial(family_v2.load_dataset, root=TMP / "data")
    args = mod.build_parser().parse_args(["--experiment", tag, "--splits", "explore", "--run", "0",
                                          "--log-interval", "0"])
    args.seed = 0
    art = mod.train(args)
    return [art["policies"][i] for i in ids]


def run_candidate(ci: int, cname: str, family, ei: int, exp: str, cfg, models_store: dict) -> dict:
    S, A = cfg.n_states, cfg.n_actions
    eps, term = cfg.logging_epsilon, cfg.termination_prob
    t0 = time.time()
    insts, logs = prior_fit.simulate(family, cfg, N_EVAL, [7001, ci, ei])
    envs = [tabular.env_from_instance(i) for i in insts]
    ids = [f"x{i:02d}" for i in range(N_EVAL)]
    res: dict = {"_bounds": {"best_mean": float(np.mean([tabular.normalized(e, e["v_best"]) for e in envs])),
                             "worst_mean": float(np.mean([tabular.normalized(e, e["v_worst"]) for e in envs]))}}
    if "hyper" in insts[0]:
        res["_hyper_example"] = insts[0]["hyper"]
    print(f"[{cname}/{exp}] simulated eval set in {time.time() - t0:.0f}s", flush=True)

    def record(name, policies):
        sc = tabular.score_policies(envs, policies)
        res[name] = {**tabular.aggregate(sc), "scores": np.round(sc, 5).tolist()}
        print(f"  {name:28s} mean {res[name]['mean']:+.3f} iqm {res[name]['iqm']:+.3f} "
              f"worst10 {res[name]['worst_decile']:+.3f}", flush=True)

    record("oracle", [e["Qstar"].argmax(1) for e in envs])
    record("behavior_cloning", [bc_policy(l, S, A) for l in logs])
    t = tabular.stack_logs(logs)
    for name, kind, kw in (("tab_nominal", "nominal", {}), ("tab_manski", "manski", {}),
                           ("tab_sharp", "sharp", {}), ("tab_sharp_lcb", "sharp_lcb", {}),
                           ("tab_partition", "partition", {}), ("tab_partition_lcb", "partition_lcb", {}),
                           ("tab_partition_lcb_k1", "partition_lcb", {"kappa": 1.0}),
                           ("tab_hurwicz25", "hurwicz", {"lam": 0.25}),
                           ("tab_hurwicz_partition25", "hurwicz_partition", {"lam": 0.25}),
                           ("tab_hurwicz_partition50", "hurwicz_partition", {"lam": 0.5})):
        try:
            Q = tabular.solve(t, S, A, eps, term, kind, **kw)
            record(name, [q.argmax(1) for q in Q])
        except Exception:
            traceback.print_exc()
    try:
        record("hand_decode", [hand_decode(l, S, A, eps, term) for l in logs])
    except Exception:
        traceback.print_exc()
    try:
        t1 = time.time()
        models = prior_fit.train_prior_fit(family, cfg, N_SIM, [8001, ci, ei])
        models_store[(cname, exp)] = models
        pf = prior_fit.apply_prior_fit(models, logs, S, A, eps, term)
        record("prior_fit_policy", pf["policy"])
        record("prior_fit_model", pf["model"])
        print(f"  prior fit took {time.time() - t1:.0f}s", flush=True)
    except Exception:
        traceback.print_exc()
    if torch.cuda.is_available() and cname in WITH_DQN:
        tag = f"{cname}__{exp}"
        path = TMP / "data" / "explore" / f"{tag}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        header = family_v2.dataset_header(exp, "explore", cfg)
        recs = [family_v2.dataset_record(i, inst, log) for i, inst, log in zip(ids, insts, logs)]
        path.write_text(json.dumps({**header, "instances": recs}))
        for name, module in (("causal_dqn", "method.causal_dqn"), ("conf_dqn", "baseline.confounded_dqn")):
            try:
                t1 = time.time()
                record(name, run_dqn(module, tag, ids))
                print(f"  {name} took {time.time() - t1:.0f}s", flush=True)
            except Exception:
                traceback.print_exc()
    ref = res.get("causal_dqn") or res.get("tab_manski")
    if ref:
        b = np.asarray(ref["scores"])
        paired = {}
        for name, r in res.items():
            if name.startswith("_") or "scores" not in r:
                continue
            dlt = np.asarray(r["scores"]) - b
            paired[name] = {"mean_diff": float(dlt.mean()), "se": float(dlt.std(ddof=1) / np.sqrt(len(dlt))),
                            "wins": int((dlt > 1e-9).sum()), "losses": int((dlt < -1e-9).sum())}
        res["_paired_vs_reference"] = paired
    res["_seconds"] = time.time() - t0
    return res


def before_hidden_test(models_store: dict) -> None:
    """Prior-fit and hand-decode shortcuts on the current hidden test (v1 family)."""
    ck = TMP / "ckpt_before"
    if ck.exists():
        shutil.rmtree(ck)
    for exp in EXPS:
        cfg = family_v1.EXPERIMENTS[exp]
        S, A, eps, term = cfg.n_states, cfg.n_actions, cfg.logging_epsilon, cfg.termination_prob
        insts = [i for split in ("train", "test") for i in family_v2.load_dataset(split, exp)["instances"]]
        logs = [{k: i[k] for k in ("state", "action", "reward", "next_state", "done")} for i in insts]
        ids = [i["id"] for i in insts]
        pols = {}
        models = models_store.get(("v1_current", exp))
        if models:
            pf = prior_fit.apply_prior_fit(models, logs, S, A, eps, term)
            pols["shortcut_prior_fit"] = pf["policy"]
            pols["shortcut_prior_fit_model"] = pf["model"]
        pols["shortcut_hand_decode"] = [hand_decode(l, S, A, eps, term) for l in logs]
        for name, plist in pols.items():
            for k in range(3):
                p = ck / name / exp / f"run{k}.json"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps({"policies": {i: [int(a) for a in pl] for i, pl in zip(ids, plist)}}))
    for name, src in (("causal_dqn", ROOT / "checkpoints/causal_dqn"), ("conf_dqn", ROOT / "checkpoints/conf_dqn"),
                      ("behavior_cloning", ROOT / "checkpoints/behavior_cloning"),
                      ("better_sharp_lcb", ROOT / "redteam/hardening/runs/better_sharp_lcb")):
        if src.exists():
            shutil.copytree(src, ck / name)
    for split in ("test", "train"):
        cmd = [sys.executable, f"eval/{split}/evaluate_policies.py", "--checkpoints", str(ck),
               "--output", str(OUT / f"before_v1_scores_{split}.json"),
               "--detail-output", str(OUT / f"before_v1_detail_{split}.json")]
        subprocess.run(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, check=False)
    f = OUT / "before_v1_scores_test.json"
    if f.exists():
        payload = json.loads(f.read_text())
        for exp, block in payload["experiments"].items():
            for m, v in block["results"].items():
                print(f"BEFORE hidden test {exp:26s} {m:28s} " + " ".join(f"{k}={x:+.3f}" for k, x in v.items()),
                      flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "n_sim", N_SIM, flush=True)
    models_store: dict = {}
    summary: dict = {}
    for ci, (cname, (family, exps)) in enumerate(CANDIDATES.items()):
        for ei, exp in enumerate(EXPS):
            try:
                summary[f"{cname}/{exp}"] = run_candidate(ci, cname, family, ei, exp, exps[exp], models_store)
            except Exception:
                traceback.print_exc()
            (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
        if cname == "v1_current" and not (OUT / "before_v1_scores_test.json").exists():
            try:
                before_hidden_test(models_store)
            except Exception:
                traceback.print_exc()
    print("\n=== summary (mean / iqm / worst decile) ===")
    for key, res in summary.items():
        print(key, res.get("_bounds"))
        for name, r in res.items():
            if name.startswith("_"):
                continue
            pr = res.get("_paired_vs_reference", {}).get(name, {})
            print(f"  {name:28s} {r['mean']:+.3f} / {r['iqm']:+.3f} / {r['worst_decile']:+.3f}"
                  f"   paired {pr.get('mean_diff', 0):+.3f} ± {pr.get('se', 0):.3f}")


if __name__ == "__main__":
    main()
