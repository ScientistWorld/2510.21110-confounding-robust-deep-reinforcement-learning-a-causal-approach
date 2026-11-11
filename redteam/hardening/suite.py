"""Hardening checks for the rebuilt gym (iteration 12).

Run from the repository root after ``scripts/reproduce.sh`` has produced the
reference (``checkpoints/causal_dqn``) and baselines (``checkpoints/conf_dqn``,
``checkpoints/behavior_cloning``):

    PYTHONPATH=. python -m redteam.hardening.suite train     # extra training runs
    PYTHONPATH=. python -m redteam.hardening.suite cases     # hidden-test cases
    PYTHONPATH=. python -m redteam.hardening.suite summary   # tables

Every case is scored by ``bash scripts/evaluate_test.sh`` with no arguments,
after installing the case's method directories next to the reference and the
baselines in a fresh ``checkpoints/``. The reference checkpoints are restored
at the end. Shortcut constructions only use public files (``data/``,
``eval/train/``, method outputs); ``eval/test/`` is used only by the scoring
script itself.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from data.cmdp_family import EXPERIMENTS, REWARD_FLOOR, load_dataset

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"          # stashed method outputs (gitignored)
RESULTS = HERE / "results"    # per-case scores (tracked)
CKPT = ROOT / "checkpoints"
REFERENCE_DIRS = ("causal_dqn", "conf_dqn", "behavior_cloning")
EXPS = list(EXPERIMENTS)
GAMMA = 0.99


def sh(args, log=None, check=True, env_extra=None):
    import os
    env = dict(os.environ, PYTHONPATH=str(ROOT), CUBLAS_WORKSPACE_CONFIG=":4096:8")
    if env_extra:
        env.update(env_extra)
    print("RUN", " ".join(map(str, args)), flush=True)
    t0 = time.time()
    with (open(log, "w") if log else open("/dev/null", "w")) as fh:
        res = subprocess.run(args, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    print(f"  exit {res.returncode} in {time.time() - t0:.0f}s", flush=True)
    if check and res.returncode:
        if log:
            print(Path(log).read_text()[-4000:])
        raise RuntimeError(f"command failed: {args}")
    return res.returncode


def write_runs(root: Path, policies_by_exp: dict[str, list[dict]], extra: dict | None = None):
    for exp, runs in policies_by_exp.items():
        for k, pol in enumerate(runs):
            out = root / exp / f"run{k}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"policies": pol, **(extra or {})}))


def read_run(root: Path, exp: str, k: int) -> dict:
    return json.loads((root / exp / f"run{k}.json").read_text())


def all_datasets(exp: str) -> list[tuple[dict, dict]]:
    out = []
    for split in ("train", "test"):
        payload = load_dataset(split, exp)
        for inst in payload["instances"]:
            out.append((payload, inst))
    return out


# ---------------------------------------------------------------------------
# Tabular algorithms on the logged data (variants and shortcuts)
# ---------------------------------------------------------------------------


def tabular_q(payload: dict, inst: dict, kind: str, iters: int = 3000) -> np.ndarray:
    """Fixed point of a Bellman operator estimated from one logged dataset.

    kind = "nominal": observational Bellman optimality (treats the log as causal).
    kind = "causal":  the paper's causal Bellman lower bound (Prop. 3.1),
                      solved exactly on the empirical observational model.
    kind = "sharp":   lower bound that also uses the documented logging
                      epsilon: under do(x) the outcome density is at most
                      |A| P(x|s) / eps times the logged density, so the bound is
                      a lower-tail (CVaR) mean of the logged outcomes.
    """
    S, A = payload["n_states"], payload["n_actions"]
    eps = payload["logging_epsilon"]
    s, x, y, s2, d = (inst[k] for k in ("state", "action", "reward", "next_state", "done"))
    key = s * A + x
    n_sa = np.bincount(key, minlength=S * A).astype(float)
    n_s = np.bincount(s, minlength=S).astype(float)
    p_sa = n_sa / np.maximum(np.repeat(n_s, A), 1.0)
    seen_next = np.unique(s2)
    cont_sa = np.bincount(key, weights=1.0 - d, minlength=S * A) / np.maximum(n_sa, 1.0)
    alpha = np.clip(eps / (A * np.maximum(p_sa, 1e-12)), 0.0, 1.0)
    V = np.zeros(S)
    Q = np.zeros(S * A)
    cont = 1.0 - payload["termination_prob"]  # documented, independent of everything else
    for _ in range(iters):
        if kind == "sharp":
            # Termination is an exogenous coin, so only (reward, next state)
            # can be confounded; bound E_do[y + gamma * cont * V(s')].
            vals = y + GAMMA * cont * V[s2]
        else:
            vals = y + GAMMA * (1.0 - d) * V[s2]
        vmin = V[seen_next].min()
        mean = np.bincount(key, weights=vals, minlength=S * A) / np.maximum(n_sa, 1.0)
        fallback = REWARD_FLOOR + GAMMA * vmin
        if kind == "nominal":
            Q = np.where(n_sa > 0, mean, fallback)
        elif kind == "causal":
            Q = p_sa * mean + (1.0 - p_sa) * (REWARD_FLOOR + GAMMA * cont_sa * vmin)
            Q = np.where(n_sa > 0, Q, fallback)
        elif kind == "sharp":
            order = np.lexsort((vals, key))
            sk, sv = key[order], vals[order]
            starts = np.concatenate([[0], np.cumsum(n_sa)[:-1]])
            pos = np.arange(len(sk)) - starts[sk]
            mass = alpha[sk] * n_sa[sk]
            w = np.clip(mass - pos, 0.0, 1.0)
            num = np.bincount(sk, weights=w * sv, minlength=S * A)
            den = np.bincount(sk, weights=w, minlength=S * A)
            Q = np.where(den > 0, num / np.maximum(den, 1e-12), fallback)
        else:
            raise ValueError(kind)
        V_new = Q.reshape(S, A).max(1)
        if np.max(np.abs(V_new - V)) < 1e-9:
            V = V_new
            break
        V = V_new
    return Q.reshape(S, A)


def tabular_policies(kind: str) -> dict[str, dict]:
    out = {}
    for exp in EXPS:
        pol = {}
        for payload, inst in all_datasets(exp):
            pol[inst["id"]] = [int(a) for a in tabular_q(payload, inst, kind).argmax(1)]
        out[exp] = pol
    return out


def train_oracle_policies() -> dict[str, dict]:
    """Optimal policies for the PUBLIC train instances from eval/train/envs."""
    out = {}
    for exp in EXPS:
        env_payload = json.loads((ROOT / "eval/train/envs" / f"{exp}.json").read_text())
        pol = {}
        for rec in env_payload["instances"]:
            Pu = np.asarray(rec["context_probs"])
            R = np.asarray(rec["reward"])
            S, A, U = R.shape
            T = np.zeros((S, A, U, S))
            cand = np.asarray(rec["next_candidates"])
            wts = np.asarray(rec["next_weights"])
            for c in range(cand.shape[-1]):
                for u in range(U):
                    np.add.at(T[:, :, u, :], (np.arange(S)[:, None], np.arange(A)[None, :], cand[:, :, c]),
                              wts[:, :, u, c])
            Rsa = np.einsum("su,sau->sa", Pu, R)
            Psa = np.einsum("su,saut->sat", Pu, T)
            V = np.zeros(S)
            cont = 1 - rec["termination_prob"]
            for _ in range(5000):
                V = (Rsa + cont * Psa @ V).max(1)
            pol[rec["id"]] = [int(a) for a in (Rsa + cont * Psa @ V).argmax(1)]
        out[exp] = pol
    return out


# ---------------------------------------------------------------------------
# Stage 1: extra training runs and variant artifacts
# ---------------------------------------------------------------------------

DQN_VARIANTS = {
    # name: (script, extra args, seed offset)
    "causal_dqn_rep2": ("method/causal_dqn.py", [], 3),
    "stronger_x3_updates": ("method/causal_dqn.py", ["--updates", "90000", "--target-interval", "1500"], 0),
    "weaker_third_updates": ("method/causal_dqn.py", ["--updates", "10000", "--target-interval", "167"], 0),
    "weaker_paper_replay_ratio": ("method/causal_dqn.py", ["--updates", "475", "--target-interval", "8"], 0),
}


def stage_train():
    RUNS.mkdir(parents=True, exist_ok=True)
    logs = HERE / "logs"
    logs.mkdir(exist_ok=True)
    for name, (script, extra, offset) in DQN_VARIANTS.items():
        for exp in EXPS:
            for k in range(3):
                out = RUNS / name / exp / f"run{k}.json"
                if out.exists():
                    continue
                sh(["python", script, "--experiment", exp, "--run", str(k), "--seed", str(k + offset),
                    "--output-root", str(RUNS / name), *extra], log=logs / f"{name}_{exp}_run{k}.log")
    # Ensemble of the three reference runs: average Q-values, act greedily.
    ens = {}
    for exp in EXPS:
        runs = [read_run(CKPT / "causal_dqn", exp, k) for k in range(3)]
        ids = list(runs[0]["policies"])
        ens[exp] = {iid: [int(a) for a in np.mean([np.asarray(r["q_values"][iid]) for r in runs], 0).argmax(1)]
                    for iid in ids}
    write_runs(RUNS / "better_ensemble", {e: [p] * 3 for e, p in ens.items()})
    for name, kind in (("better_sharp_bound", "sharp"), ("variant_causal_tabular", "causal"),
                       ("shortcut_nominal_planning", "nominal")):
        if not (RUNS / name).exists():
            pols = tabular_policies(kind)
            write_runs(RUNS / name, {e: [p] * 3 for e, p in pols.items()})
    # Per-seed pseudo-methods (three identical copies of one seed's policy).
    for seed in range(6):
        src = CKPT / "causal_dqn" if seed < 3 else RUNS / "causal_dqn_rep2"
        write_runs(RUNS / f"seed_causal_{seed}",
                   {e: [read_run(src, e, seed % 3)["policies"]] * 3 for e in EXPS})
    print("stage train done", flush=True)


# ---------------------------------------------------------------------------
# Stage 2: hidden-test cases
# ---------------------------------------------------------------------------


def snapshot_reference():
    ref = RUNS / "_reference"
    if not ref.exists():
        ref.mkdir(parents=True)
        for name in REFERENCE_DIRS:
            shutil.copytree(CKPT / name, ref / name)
    return ref


def install(method_dirs: dict[str, Path]):
    if CKPT.exists():
        shutil.rmtree(CKPT)
    CKPT.mkdir()
    ref = RUNS / "_reference"
    for name in REFERENCE_DIRS:
        shutil.copytree(ref / name, CKPT / name)
    for name, src in method_dirs.items():
        shutil.copytree(src, CKPT / name)


def constant_policies(action: int, length: int = 30) -> dict[str, dict]:
    out = {}
    for exp in EXPS:
        out[exp] = {inst["id"]: [action] * length for _, inst in all_datasets(exp)}
    return out


def build_shortcut(name: str, folder: Path) -> tuple[dict[str, Path], callable, callable]:
    """Return (method dirs to install, before_test hook, cleanup hook)."""
    tmp = folder / "artifacts"
    if tmp.exists():
        shutil.rmtree(tmp)
    target = tmp / name
    before = after = None
    if name == "shortcut_constant_action":
        write_runs(target, {e: [p] * 3 for e, p in constant_policies(0).items()})
    elif name == "shortcut_invalid_action":
        write_runs(target, {e: [p] * 3 for e, p in constant_policies(7).items()})
    elif name == "shortcut_null_action":
        pols = constant_policies(0)
        for e in pols:
            for iid in pols[e]:
                pols[e][iid][0] = None
        write_runs(target, {e: [p] * 3 for e, p in pols.items()})

        def before():
            sentinel = {"experiments": {e: {"results": {name: {"normalized_mean_return": 999}}} for e in EXPS}}
            (ROOT / "scoring/scores_test.json").write_text(json.dumps(sentinel))
    elif name == "shortcut_missing_experiment":
        good = {e: [read_run(RUNS / "better_sharp_bound", e, 0)["policies"]] * 3 for e in EXPS[:1]}
        write_runs(target, good)
    elif name == "shortcut_missing_states":
        pols = {e: {iid: [] for iid in p} for e, p in constant_policies(0).items()}
        write_runs(target, {e: [p] * 3 for e, p in pols.items()})
    elif name == "shortcut_extra_runs":
        bad = constant_policies(1)
        good = {e: read_run(RUNS / "better_sharp_bound", e, 0)["policies"] for e in EXPS}
        write_runs(target, {e: [bad[e]] * 3 + [good[e]] * 47 for e in EXPS})
    elif name == "shortcut_forged_fields":
        forged = {m: 1e9 for m in ("normalized_mean_return", "normalized_iqm_return",
                                   "worst_decile_normalized_return")}
        write_runs(target, {e: [p] * 3 for e, p in constant_policies(1).items()},
                   extra={"scores": forged, "status": "ok", "valid": True})

        def before():
            payload = {"_slice": "test", "experiments": {e: {"results": {name: forged}} for e in EXPS}}
            (ROOT / "scoring/scores_test.json").write_text(json.dumps(payload))
    elif name == "shortcut_label_swap":
        good = {e: read_run(CKPT / "causal_dqn", e, k)["policies"] for e in EXPS for k in [0]}
        swapped = {EXPS[0]: good[EXPS[1]], EXPS[1]: good[EXPS[0]]}
        write_runs(target, {e: [p] * 3 for e, p in swapped.items()},
                   extra={"experiment": "swapped", "variant": "base"})
    elif name == "shortcut_data_edit":
        write_runs(target, {e: [p] * 3 for e, p in constant_policies(0).items()})
        backup = folder / "data_backup"

        def before():
            if backup.exists():
                shutil.rmtree(backup)
            shutil.copytree(ROOT / "data", backup)
            src = (ROOT / "data/cmdp_family.py").read_text()
            (ROOT / "data/cmdp_family.py").write_text(src.replace("REWARD_FLOOR = -1.0", "REWARD_FLOOR = 100.0"))
            for exp in EXPS:
                for split in ("train", "test"):
                    p = ROOT / "data" / split / f"{exp}.json"
                    payload = json.loads(p.read_text())
                    payload["n_actions"] = 1
                    payload["note"] = "edited by shortcut_data_edit"
                    p.write_text(json.dumps(payload))

        def after():
            shutil.rmtree(ROOT / "data")
            shutil.copytree(backup, ROOT / "data")
            shutil.rmtree(backup)
    elif name == "shortcut_nominal_planning":
        shutil.copytree(RUNS / "shortcut_nominal_planning", target)
    elif name == "shortcut_train_oracle":
        oracle = train_oracle_policies()
        bc = {e: read_run(CKPT / "behavior_cloning", e, 0)["policies"] for e in EXPS}
        mixed = {e: {**{i: a for i, a in bc[e].items() if i.startswith("test_")}, **oracle[e]} for e in EXPS}
        write_runs(target, {e: [p] * 3 for e, p in mixed.items()})
    else:
        raise ValueError(name)
    return {name: target}, before, after


HONEST = ["causal_dqn_rep2", "better_ensemble", "better_sharp_bound", "variant_causal_tabular",
          "stronger_x3_updates", "weaker_third_updates", "weaker_paper_replay_ratio",
          *[f"seed_causal_{k}" for k in range(6)]]
SHORTCUTS = ["shortcut_constant_action", "shortcut_invalid_action", "shortcut_null_action",
             "shortcut_missing_experiment", "shortcut_missing_states", "shortcut_extra_runs",
             "shortcut_forged_fields", "shortcut_label_swap", "shortcut_data_edit",
             "shortcut_nominal_planning", "shortcut_train_oracle"]


def run_case(case: str, method_dirs: dict[str, Path], before=None, after=None):
    folder = RESULTS / case
    folder.mkdir(parents=True, exist_ok=True)
    install(method_dirs)
    try:
        if before:
            before()
        code = sh(["bash", "scripts/evaluate_test.sh"], log=folder / "test.log", check=False)
        test_scores = ROOT / "scoring/scores_test.json"
        record = {"case": case, "evaluate_test_exit_code": code,
                  "scores_test_written": test_scores.exists()}
        if test_scores.exists():
            shutil.copy2(test_scores, folder / "scores_test.json")
        sh(["bash", "scripts/evaluate_train.sh"], log=folder / "train.log", check=False)
        if (ROOT / "scoring/scores_train.json").exists():
            shutil.copy2(ROOT / "scoring/scores_train.json", folder / "scores_train.json")
        sh(["python", "eval/test/evaluate_policies.py", "--output", str(folder / "scores_test_detail_run.json"),
            "--detail-output", str(RUNS / f"_detail_{case}.json")], log=folder / "detail.log", check=False)
        (folder / "scores_test_detail_run.json").unlink(missing_ok=True)
        (folder / "execution.json").write_text(json.dumps(record, indent=2))
    finally:
        if after:
            after()


def stage_cases():
    snapshot_reference()
    RESULTS.mkdir(parents=True, exist_ok=True)
    run_case("reference", {})
    run_case("honest_variants", {n: RUNS / n for n in HONEST})
    for name in SHORTCUTS:
        dirs, before, after = build_shortcut(name, RESULTS / name)
        run_case(name, dirs, before, after)
        shutil.rmtree(RESULTS / name / "artifacts", ignore_errors=True)
    # Restore the reference-only checkpoints and final score files.
    install({})
    sh(["bash", "scripts/evaluate_train.sh"], log=RESULTS / "final_train.log")
    sh(["bash", "scripts/evaluate_test.sh"], log=RESULTS / "final_test.log")
    sh(["bash", "scripts/evaluate.sh"], log=RESULTS / "final_dev.log")
    print("stage cases done", flush=True)


# ---------------------------------------------------------------------------
# Stage 3: summary
# ---------------------------------------------------------------------------


def stage_summary():
    ref_json = json.loads((ROOT / "scoring/reference.json").read_text())["experiments"]
    rows = {}
    statuses = {}
    for case_dir in sorted(p for p in RESULTS.iterdir() if p.is_dir()):
        f = case_dir / "scores_test.json"
        if not f.exists():
            continue
        payload = json.loads(f.read_text())
        statuses.update(payload.get("_status", {}))
        for exp, block in payload["experiments"].items():
            for method, metrics in block["results"].items():
                rows.setdefault(method, {})[exp] = metrics
    ref = rows["causal_dqn"]
    cohorts = {
        "all_attempts": [m for m in rows if not m.startswith("seed_causal_")],
        "honest_only": [m for m in rows if not m.startswith(("seed_causal_", "shortcut_"))],
    }
    stds = {}
    for cname, cohort in cohorts.items():
        for exp, spec in ref_json.items():
            for metric in spec["metrics"]:
                vals = [rows[m][exp][metric] for m in cohort if exp in rows[m]]
                stds[(cname, exp, metric)] = float(np.std(vals)) or 1.0

    def rank(method, cname):
        total = 0.0
        for exp, spec in ref_json.items():
            for metric, mspec in spec["metrics"].items():
                sign = 1.0 if mspec["higher_is_better"] else -1.0
                delta = sign * (rows[method][exp][metric] - ref[exp][metric])
                total += spec["weight"] * abs(mspec["coefficient"]) * delta / stds[(cname, exp, metric)]
        return total

    table = {m: {"rank_vs_reference": rank(m, "all_attempts"),
                 "rank_vs_reference_honest_cohort": rank(m, "honest_only"),
                 "status": statuses.get(m), "metrics": rows[m]} for m in rows}
    # Paired per-instance comparisons against the reference (same instances).
    paired = {}
    detail_ref = json.loads((RUNS / "_detail_reference.json").read_text())["methods"]["causal_dqn"]
    detail_var = json.loads((RUNS / "_detail_honest_variants.json").read_text())["methods"]
    for m, det in detail_var.items():
        if not det:
            continue
        paired[m] = {}
        for exp in EXPS:
            a = np.asarray(det[exp]["scores"]).mean(1)
            b = np.asarray(detail_ref[exp]["scores"]).mean(1)
            diff = a - b
            paired[m][exp] = {"mean_diff": float(diff.mean()),
                              "se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
                              "wins": int((diff > 1e-9).sum()), "losses": int((diff < -1e-9).sum())}
    bounds = json.loads((RUNS / "_detail_reference.json").read_text())["bounds"]
    oracle = {exp: {"best_policy_mean": float(np.mean(b["best_policy_normalized"])),
                    "worst_policy_mean": float(np.mean(b["worst_policy_normalized"]))}
              for exp, b in bounds.items()}
    seeds = {f"seed_causal_{k}": rows[f"seed_causal_{k}"] for k in range(6) if f"seed_causal_{k}" in rows}
    summary = {"cohorts": cohorts, "std": {f"{c}/{e}/{m}": v for (c, e, m), v in stds.items()},
               "table": table, "paired_vs_reference": paired, "test_bounds": oracle, "per_seed": seeds}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    for m in sorted(table, key=lambda k: -table[k]["rank_vs_reference"]):
        mm = table[m]["metrics"]
        line = " | ".join(f"{mm[e]['normalized_mean_return']:.3f}/{mm[e]['normalized_iqm_return']:.3f}/"
                          f"{mm[e]['worst_decile_normalized_return']:.3f}" for e in EXPS)
        print(f"{m:28s} rank {table[m]['rank_vs_reference']:+.3f} honest {table[m]['rank_vs_reference_honest_cohort']:+.3f}"
              f" | {line} | {table[m]['status']}")
    print(json.dumps(paired, indent=1))
    print(json.dumps(oracle, indent=1))


if __name__ == "__main__":
    {"train": stage_train, "cases": stage_cases, "summary": stage_summary}[sys.argv[1]]()
