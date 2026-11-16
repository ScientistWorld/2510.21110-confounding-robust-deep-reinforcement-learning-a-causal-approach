"""Hardening checks for the gym (iterations 12-13).

Run from the repository root after ``scripts/reproduce.sh`` has produced the
reference (``checkpoints/causal_dqn``) and baselines (``checkpoints/conf_dqn``,
``checkpoints/behavior_cloning``):

    PYTHONPATH=. python -m redteam.hardening.suite train     # extra runs, variants, shortcut policies
    PYTHONPATH=. python -m redteam.hardening.suite cases     # hidden-test cases
    PYTHONPATH=. python -m redteam.hardening.suite summary   # tables

Every case is scored by ``bash scripts/evaluate_test.sh`` with no arguments,
after installing the case's method directories next to the reference and the
baselines in a fresh ``checkpoints/``. The reference checkpoints are restored
at the end. Shortcut constructions only use public files (``data/``,
``eval/train/``, the public generator, method outputs); ``eval/test/`` is
used only by the scoring scripts themselves.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import data.cmdp_family as family
from data.cmdp_family import EXPERIMENTS, load_dataset
from redteam.hardening import prior_fit, tabular
from redteam.hardening.hand_decode import hand_decode

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"          # stashed method outputs (gitignored)
RESULTS = HERE / "results"    # per-case scores (tracked)
CKPT = ROOT / "checkpoints"
REFERENCE_DIRS = ("causal_dqn", "conf_dqn", "behavior_cloning")
EXPS = list(EXPERIMENTS)
N_PRIOR_SIM = int(os.environ.get("N_PRIOR_SIM", "1024"))


def sh(args, log=None, check=True):
    env = dict(os.environ, PYTHONPATH=str(ROOT), CUBLAS_WORKSPACE_CONFIG=":4096:8")
    print("RUN", " ".join(map(str, args)), flush=True)
    t0 = time.time()
    with (open(log, "w") if log else open(os.devnull, "w")) as fh:
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


def public_logs(exp: str) -> tuple[dict, list[str], list[dict]]:
    """Header, ids and logs of every public log (train and test splits)."""
    ids, logs, header = [], [], None
    for split in ("train", "test"):
        payload = load_dataset(split, exp)
        header = header or payload
        for inst in payload["instances"]:
            ids.append(inst["id"])
            logs.append({k: inst[k] for k in ("state", "action", "reward", "next_state", "done")})
    return header, ids, logs


def as_lists(pols) -> list[list[int]]:
    return [[int(a) for a in p] for p in pols]


# ---------------------------------------------------------------------------
# Policies built from the logs (variants and shortcuts)
# ---------------------------------------------------------------------------

TABULAR = {
    # name: (kind, kwargs)
    "better_partition_lcb": ("partition_lcb", {"kappa": 1.0}),
    "better_partition": ("partition", {}),
    "better_hurwicz_partition": ("hurwicz_partition", {"lam": 0.25}),
    "better_sharp_lcb": ("sharp_lcb", {"kappa": 2.0}),
    "variant_causal_tabular": ("manski", {}),
    "shortcut_nominal_planning": ("nominal", {}),
}


def tabular_policies(kind: str, kw: dict) -> dict[str, dict]:
    out = {}
    for exp in EXPS:
        h, ids, logs = public_logs(exp)
        Q = tabular.solve(tabular.stack_logs(logs), h["n_states"], h["n_actions"], h["logging_epsilon"],
                          h["termination_prob"], kind, **kw)
        out[exp] = dict(zip(ids, as_lists(Q.argmax(-1))))
    return out


def prior_fit_policies() -> dict[str, dict[str, dict]]:
    """Prior fitted on instances simulated from the public generator, and on
    the public train split (true train dynamics are public)."""
    out = {"shortcut_prior_fit": {}, "shortcut_prior_fit_model": {}, "shortcut_prior_fit_train": {}}
    for ei, exp in enumerate(EXPS):
        cfg = EXPERIMENTS[exp]
        h, ids, logs = public_logs(exp)
        S, A, eps, term = h["n_states"], h["n_actions"], h["logging_epsilon"], h["termination_prob"]
        t0 = time.time()
        models = prior_fit.train_prior_fit(family, cfg, N_PRIOR_SIM, [9001, ei])
        print(f"prior fit on {N_PRIOR_SIM} simulated {exp} instances: {time.time() - t0:.0f}s", flush=True)
        pf = prior_fit.apply_prior_fit(models, logs, S, A, eps, term)
        out["shortcut_prior_fit"][exp] = dict(zip(ids, pf["policy"]))
        out["shortcut_prior_fit_model"][exp] = dict(zip(ids, pf["model"]))
        env_payload = json.loads((ROOT / "eval/train/envs" / f"{exp}.json").read_text())
        envs = {r["id"]: tabular.env_from_instance(r) for r in env_payload["instances"]}
        tr = [i for i, iid in enumerate(ids) if iid in envs]
        models_tr = prior_fit.fit_from([envs[ids[i]] for i in tr], [logs[i] for i in tr], S, A, eps, term,
                                       kinds=("model",))
        out["shortcut_prior_fit_train"][exp] = dict(zip(ids, prior_fit.apply_prior_fit(models_tr, logs, S, A, eps,
                                                                                        term)["model"]))
    return out


def hand_decode_policies() -> dict[str, dict]:
    out = {}
    for exp in EXPS:
        h, ids, logs = public_logs(exp)
        out[exp] = {i: hand_decode(l, h["n_states"], h["n_actions"], h["logging_epsilon"], h["termination_prob"])
                    for i, l in zip(ids, logs)}
    return out


def train_oracle_policies() -> dict[str, dict]:
    """Optimal policies for the PUBLIC train instances from eval/train/envs."""
    out = {}
    for exp in EXPS:
        env_payload = json.loads((ROOT / "eval/train/envs" / f"{exp}.json").read_text())
        out[exp] = {r["id"]: [int(a) for a in tabular.env_from_instance(r)["Qstar"].argmax(1)]
                    for r in env_payload["instances"]}
    return out


# ---------------------------------------------------------------------------
# Stage 1: extra training runs and variant artifacts
# ---------------------------------------------------------------------------

DQN_VARIANTS = {
    # name: (script, extra args, seed offset)
    "causal_dqn_rep2": ("method/causal_dqn.py", [], 3),
    "stronger_x3_updates": ("method/causal_dqn.py", ["--updates", "90000", "--target-interval", "1500"], 0),
    "weaker_third_updates": ("method/causal_dqn.py", ["--updates", "10000", "--target-interval", "167"], 0),
    "weaker_paper_replay_ratio": ("method/causal_dqn.py", ["--updates", "1500", "--target-interval", "25"], 0),
}


def stage_train():
    RUNS.mkdir(parents=True, exist_ok=True)
    logs = HERE / "logs"
    logs.mkdir(exist_ok=True)
    for name, (kind, kw) in TABULAR.items():
        if not (RUNS / name).exists():
            t0 = time.time()
            pols = tabular_policies(kind, kw)
            write_runs(RUNS / name, {e: [p] * 3 for e, p in pols.items()})
            print(f"{name}: {time.time() - t0:.0f}s", flush=True)
    if not (RUNS / "shortcut_hand_decode").exists():
        write_runs(RUNS / "shortcut_hand_decode", {e: [p] * 3 for e, p in hand_decode_policies().items()})
    if not (RUNS / "shortcut_prior_fit").exists():
        for name, pols in prior_fit_policies().items():
            write_runs(RUNS / name, {e: [p] * 3 for e, p in pols.items()})
    # Ensemble of the three reference runs: average Q-values, act greedily.
    ens = {}
    for exp in EXPS:
        runs = [read_run(CKPT / "causal_dqn", exp, k) for k in range(3)]
        ids = list(runs[0]["policies"])
        ens[exp] = {iid: [int(a) for a in np.mean([np.asarray(r["q_values"][iid]) for r in runs], 0).argmax(1)]
                    for iid in ids}
    write_runs(RUNS / "better_ensemble", {e: [p] * 3 for e, p in ens.items()})
    for name, (script, extra, offset) in DQN_VARIANTS.items():
        for exp in EXPS:
            for k in range(3):
                out = RUNS / name / exp / f"run{k}.json"
                if out.exists():
                    continue
                sh(["python", script, "--experiment", exp, "--run", str(k), "--seed", str(k + offset),
                    "--output-root", str(RUNS / name), *extra], log=logs / f"{name}_{exp}_run{k}.log")
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
    if ref.exists():
        shutil.rmtree(ref)
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


def all_ids(exp: str) -> list[str]:
    return public_logs(exp)[1]


def constant_policies(action: int, length: int = 30) -> dict[str, dict]:
    return {exp: {iid: [action] * length for iid in all_ids(exp)} for exp in EXPS}


def build_shortcut(name: str, folder: Path):
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
        good = {e: [read_run(RUNS / "better_partition_lcb", e, 0)["policies"]] * 3 for e in EXPS[:1]}
        write_runs(target, good)
    elif name == "shortcut_missing_states":
        pols = {e: {iid: [] for iid in p} for e, p in constant_policies(0).items()}
        write_runs(target, {e: [p] * 3 for e, p in pols.items()})
    elif name == "shortcut_extra_runs":
        bad = constant_policies(1)
        good = {e: read_run(RUNS / "better_partition_lcb", e, 0)["policies"] for e in EXPS}
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
        good = {e: read_run(CKPT / "causal_dqn", e, 0)["policies"] for e in EXPS}
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
    elif name in PRECOMPUTED_SHORTCUTS:
        shutil.copytree(RUNS / name, target)
    elif name == "shortcut_train_oracle":
        oracle = train_oracle_policies()
        bc = {e: read_run(CKPT / "behavior_cloning", e, 0)["policies"] for e in EXPS}
        mixed = {e: {**{i: a for i, a in bc[e].items() if i.startswith("test_")}, **oracle[e]} for e in EXPS}
        write_runs(target, {e: [p] * 3 for e, p in mixed.items()})
    else:
        raise ValueError(name)
    return {name: target}, before, after


PRECOMPUTED_SHORTCUTS = ["shortcut_prior_fit", "shortcut_prior_fit_model", "shortcut_prior_fit_train",
                         "shortcut_hand_decode", "shortcut_nominal_planning"]
HONEST = ["causal_dqn_rep2", "better_ensemble", "better_partition_lcb", "better_partition",
          "better_hurwicz_partition", "better_sharp_lcb", "variant_causal_tabular",
          "stronger_x3_updates", "weaker_third_updates", "weaker_paper_replay_ratio",
          *[f"seed_causal_{k}" for k in range(6)]]
SHORTCUTS = [*PRECOMPUTED_SHORTCUTS, "shortcut_train_oracle", "shortcut_constant_action",
             "shortcut_invalid_action", "shortcut_null_action", "shortcut_missing_experiment",
             "shortcut_missing_states", "shortcut_extra_runs", "shortcut_forged_fields", "shortcut_label_swap",
             "shortcut_data_edit"]


def run_case(case: str, method_dirs: dict[str, Path], before=None, after=None):
    folder = RESULTS / case
    folder.mkdir(parents=True, exist_ok=True)
    install(method_dirs)
    try:
        if before:
            before()
        code = sh(["bash", "scripts/evaluate_test.sh"], log=folder / "test.log", check=False)
        test_scores = ROOT / "scoring/scores_test.json"
        record = {"case": case, "evaluate_test_exit_code": code, "scores_test_written": test_scores.exists()}
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
    if RESULTS.exists():
        shutil.rmtree(RESULTS)
    RESULTS.mkdir(parents=True)
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


def _paired(a: np.ndarray, b: np.ndarray) -> dict:
    diff = a - b
    return {"mean_diff": float(diff.mean()), "se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
            "wins": int((diff > 1e-9).sum()), "losses": int((diff < -1e-9).sum())}


def stage_summary():
    ref_json = json.loads((ROOT / "scoring/reference.json").read_text())["experiments"]
    rows, train_rows, statuses = {}, {}, {}
    for case_dir in sorted(p for p in RESULTS.iterdir() if p.is_dir()):
        for fname, store in (("scores_test.json", rows), ("scores_train.json", train_rows)):
            f = case_dir / fname
            if not f.exists():
                continue
            payload = json.loads(f.read_text())
            if fname == "scores_test.json":
                statuses.update(payload.get("_status", {}))
            for exp, block in payload["experiments"].items():
                for method, metrics in block["results"].items():
                    store.setdefault(method, {})[exp] = metrics
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

    def rank(method, cname, table=rows, base=ref):
        total = 0.0
        for exp, spec in ref_json.items():
            for metric, mspec in spec["metrics"].items():
                sign = 1.0 if mspec["higher_is_better"] else -1.0
                delta = sign * (table[method][exp][metric] - base[exp][metric])
                total += spec["weight"] * abs(mspec["coefficient"]) * delta / stds[(cname, exp, metric)]
        return total

    table = {m: {"rank_vs_reference": rank(m, "all_attempts"),
                 "rank_vs_reference_honest_cohort": rank(m, "honest_only"),
                 "status": statuses.get(m), "metrics": rows[m],
                 "train_metrics": train_rows.get(m)} for m in rows}
    # Paired per-instance comparisons (same instances, means over runs).
    detail = {}
    for f in sorted(RUNS.glob("_detail_*.json")):
        for m, det in json.loads(f.read_text())["methods"].items():
            if det:
                detail[m] = det
    paired = {}
    for m, det in detail.items():
        paired[m] = {}
        for exp in EXPS:
            a = np.asarray(det[exp]["scores"]).mean(1)
            paired[m][exp] = {"vs_reference": _paired(a, np.asarray(detail["causal_dqn"][exp]["scores"]).mean(1))}
            for other in ("better_partition_lcb", "better_hurwicz_partition"):
                if other in detail and m != other:
                    paired[m][exp][f"vs_{other}"] = _paired(a, np.asarray(detail[other][exp]["scores"]).mean(1))
    bounds = json.loads((RUNS / "_detail_reference.json").read_text())["bounds"]
    oracle = {exp: {"best_policy_mean": float(np.mean(b["best_policy_normalized"])),
                    "worst_policy_mean": float(np.mean(b["worst_policy_normalized"]))}
              for exp, b in bounds.items()}
    seeds = {f"seed_causal_{k}": rows[f"seed_causal_{k}"] for k in range(6) if f"seed_causal_{k}" in rows}
    summary = {"cohorts": cohorts, "std": {f"{c}/{e}/{m}": v for (c, e, m), v in stds.items()},
               "table": table, "paired": paired, "test_bounds": oracle, "per_seed": seeds}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    for m in sorted(table, key=lambda k: -table[k]["rank_vs_reference"]):
        mm = table[m]["metrics"]
        line = " | ".join(f"{mm[e]['normalized_mean_return']:.3f}/{mm[e]['normalized_iqm_return']:.3f}/"
                          f"{mm[e]['worst_decile_normalized_return']:.3f}" for e in EXPS)
        tr = table[m]["train_metrics"]
        trl = " | ".join(f"{tr[e]['normalized_mean_return']:.3f}" for e in EXPS) if tr else "-"
        print(f"{m:28s} rank {table[m]['rank_vs_reference']:+.3f} honest "
              f"{table[m]['rank_vs_reference_honest_cohort']:+.3f} | test {line} | train mean {trl} | "
              f"{table[m]['status']}")
    for m, d in paired.items():
        print(m, json.dumps({e: {k: f"{v['mean_diff']:+.3f}±{v['se']:.3f} ({v['wins']}/{v['losses']})"
                                 for k, v in x.items()} for e, x in d.items()}))
    print(json.dumps(oracle, indent=1))


if __name__ == "__main__":
    {"train": stage_train, "cases": stage_cases, "summary": stage_summary}[sys.argv[1]]()
