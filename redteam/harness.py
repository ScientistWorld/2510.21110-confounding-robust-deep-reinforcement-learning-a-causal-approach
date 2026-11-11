"""Audit runner. Candidate construction uses the visible gym interface only.

Every test invocation is `bash scripts/evaluate_test.sh` with no arguments,
in a new subprocess. No evaluator, shell startup, package, or Python hook is
modified. Each case restores its temporary workspace changes on exit.
"""

from __future__ import annotations

import contextlib
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "redteam"
EXPERIMENTS = {"base": "scaled_confounded_cmdp", "weaker_bias": "weaker_bias_cmdp"}
SEEDS = (0, 1, 2)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def command(args, log, *, allow_failure=False):
    print("RUN", " ".join(map(str, args)), flush=True)
    with Path(log).open("w") as stream:
        result = subprocess.run(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode and not allow_failure:
        print(Path(log).read_text()[-6000:], flush=True)
        raise RuntimeError(f"exit {result.returncode}: {args}")
    return result.returncode


@contextlib.contextmanager
def isolated(folder):
    """Restore files byte-for-byte, including ignored original checkpoints."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    state = folder / ".state"
    if state.exists():
        raise RuntimeError(f"unfinished previous case; inspect {state}")
    state.mkdir()
    saved = ["checkpoints", "scoring/scores_train.json", "scoring/scores_test.json",
             "scoring/scores.json", "data/confounded_cmdp.py"]
    existed = {}
    for relative in saved:
        path = ROOT / relative
        target = state / relative
        existed[relative] = path.exists()
        if path.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.is_dir():
                shutil.copytree(path, target)
            else:
                shutil.copy2(path, target)
    write(state / "manifest.json", existed)
    try:
        yield
    finally:
        for relative in saved:
            path = ROOT / relative
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
            if existed[relative]:
                source = state / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if source.is_dir():
                    shutil.copytree(source, path)
                else:
                    shutil.copy2(source, path)
        # Avoid timestamp-based reuse of a mutated environment bytecode cache.
        # This is ordinary cache cleanup after restoring an audited data file.
        shutil.rmtree(ROOT / "data/__pycache__", ignore_errors=True)
        shutil.rmtree(state)


def install(artifacts):
    checkpoint = ROOT / "checkpoints"
    checkpoint.mkdir(exist_ok=True)
    for path in checkpoint.glob("*.json"):
        path.unlink()
    for index, artifact in enumerate(artifacts):
        write(checkpoint / f"artifact_{index:03d}.json", artifact)


def reference_artifacts(*, baseline=False, seed=None):
    folder = AUDIT / "reference/artifacts"
    prefix = "conf" if baseline else "causal"
    paths = sorted(folder.glob(f"{prefix}_*.json"))
    if not paths:
        raise RuntimeError("run redteam/reference/run.sh first")
    artifacts = [read(path) for path in paths]
    return [a for a in artifacts if seed is None or a["seed"] == seed]


def policy(variant, entries, **extra):
    return {"method": "causal_dqn", "experiment": EXPERIMENTS[variant],
            "variant": variant, "policy": {str(k): v for k, v in entries.items()}, **extra}


def save_artifacts(folder, artifacts):
    destination = folder / "artifacts"
    destination.mkdir(exist_ok=True)
    for index, artifact in enumerate(artifacts):
        write(destination / f"artifact_{index:03d}.json", artifact)


def evaluate(folder, artifacts, *, suffix="", mutate_train=None, before_test=None,
             allow_failure=False):
    """Persist artifacts, visible train scores, and actual hidden-test outputs."""
    install(artifacts)
    command(["bash", "scripts/evaluate_train.sh"], folder / f"train{suffix}.log")
    if mutate_train:
        mutate_train()
    shutil.copy2(ROOT / "scoring/scores_train.json", folder / f"scores_train{suffix}.json")
    if before_test:
        before_test()
    prior = ROOT / "scoring/scores_test.json"
    previous_bytes = prior.read_bytes() if prior.exists() else None
    previous_mtime = prior.stat().st_mtime_ns if prior.exists() else None
    code = command(["bash", "scripts/evaluate_test.sh"], folder / f"test{suffix}.log",
                   allow_failure=allow_failure)
    if prior.exists():
        shutil.copy2(prior, folder / f"scores_test{suffix}.json")
    metadata = {"exit_code": code, "test_command": "bash scripts/evaluate_test.sh",
                "test_arguments": [], "episodes": 2000,
                "score_file_bytes_changed": prior.exists() and prior.read_bytes() != previous_bytes,
                "score_file_mtime_changed": prior.exists() and prior.stat().st_mtime_ns != previous_mtime}
    write(folder / f"execution{suffix}.json", metadata)
    return code


def train_reference(folder, *, stronger=False):
    output = folder / "artifacts"
    output.mkdir(exist_ok=True)
    models = [("causal", "method/causal_dqn.py")]
    if not stronger:
        models.append(("conf", "baseline/confounded_dqn.py"))
    for variant in EXPERIMENTS:
        for seed in SEEDS:
            for name, source in models:
                path = output / f"{name}_{variant}_seed{seed}.json"
                command(["python", "-m", source[:-3].replace("/", "."), "--variant", variant, "--seed", str(seed),
                         "--demo-episodes", "2000" if stronger else "500",
                         "--updates", "4000" if stronger else "1000",
                         "--batch-size", "64", "--log-interval", "1000",
                         "--output", str(path)],
                        folder / f"training_{name}_{variant}_seed{seed}.log")
    artifacts = [read(p) for p in sorted(output.glob("*.json"))]
    baselines = reference_artifacts(baseline=True) if stronger else [
        a for a in artifacts if a["method"] == "conf_dqn"]
    proposed = [a for a in artifacts if a["method"] == "causal_dqn"]
    for seed in SEEDS:
        evaluate(folder, baselines + [a for a in proposed if a["seed"] == seed],
                 suffix=f"_seed{seed}")
    evaluate(folder, baselines + proposed)


def run_case(name):
    folder = AUDIT / name
    with isolated(folder):
        if name in ("reference", "stronger_training"):
            train_reference(folder, stronger=name == "stronger_training")
            return

        baselines = reference_artifacts(baseline=True)
        artifacts = []
        if name == "better_ensemble":
            fitted = reference_artifacts()
            for variant in EXPERIMENTS:
                votes = [a["policy"] for a in fitted if a["variant"] == variant]
                keys = sorted(votes[0])
                actions = {key: max(set(v[key] for v in votes),
                                    key=lambda a: sum(v[key] == a for v in votes)) for key in keys}
                artifacts.append(policy(variant, actions, ensemble_seeds=list(SEEDS)))
        elif name == "weaker_no_causal":
            for artifact in baselines:
                artifacts.append({**artifact, "method": "causal_dqn",
                                  "ablation": "removed all-action causal target"})
        elif name == "shortcut_constant_safe":
            # Shortcut: no demonstrations, value estimation, or training.
            from data.confounded_cmdp import N_STATES, SAFE_ACTION
            artifacts = [policy(v, dict.fromkeys(range(N_STATES), SAFE_ACTION)) for v in EXPERIMENTS]
        elif name == "shortcut_bruteforce_visible":
            # Shortcut: enumerate the complete tiny search space using the
            # publicly shipped reward model; never read a hidden-test answer.
            from data.confounded_cmdp import N_ACTIONS, N_STATES, SAFE_ACTION, VARIANTS
            search = {}
            for variant, config in VARIANTS.items():
                candidates = []
                for actions in itertools.product(range(N_ACTIONS), repeat=N_STATES):
                    expected = sum(config.safe_reward if a == SAFE_ACTION else
                                   2 * (config.favorable_base + config.favorable_odd_bonus * (s % 2)) - 1
                                   for s, a in enumerate(actions))
                    candidates.append({"policy": list(actions), "expected_return": expected})
                best = max(candidates, key=lambda c: c["expected_return"])
                artifacts.append(policy(variant, dict(enumerate(best["policy"]))))
                search[variant] = {"candidate_count": len(candidates), "best": best,
                                   "all_candidates": candidates}
            write(folder / "shortcut_search.json", search)
        elif name == "shortcut_variant_swap":
            # Shortcut: label easy base rollouts as the weaker-bias experiment.
            from data.confounded_cmdp import N_STATES, SAFE_ACTION
            for variant in EXPERIMENTS:
                artifact = policy(variant, dict.fromkeys(range(N_STATES), SAFE_ACTION))
                artifact["variant"] = "base"
                artifacts.append(artifact)
        elif name == "shortcut_data_reward":
            # Shortcut: change the fixed reward table imported by the restored
            # evaluator. This edits problem data, never the Python environment.
            source = ROOT / "data/confounded_cmdp.py"
            text = source.read_text()
            assert "safe_reward=0.25," in text and "safe_reward=0.20," in text
            source.write_text(text.replace("safe_reward=0.25,", "safe_reward=1.0,")
                              .replace("safe_reward=0.20,", "safe_reward=1.0,"))
            shutil.rmtree(ROOT / "data/__pycache__", ignore_errors=True)
            from data.confounded_cmdp import N_STATES, SAFE_ACTION
            artifacts = [policy(v, dict.fromkeys(range(N_STATES), SAFE_ACTION)) for v in EXPERIMENTS]
            shutil.copy2(source, folder / "shortcut_confounded_cmdp.py")
        elif name == "shortcut_empty_policy":
            # Shortcut: omit every state and rely on the evaluator's fallback.
            artifacts = [policy(v, {}) for v in EXPERIMENTS]
        elif name == "shortcut_invalid_actions":
            # Shortcut: attempt impossible action IDs. Evaluator should reject.
            artifacts = [policy(v, dict.fromkeys(range(6), 999999)) for v in EXPERIMENTS]
        elif name == "shortcut_forged_fields":
            # Shortcut: assert good metrics around a genuinely bad policy.
            artifacts = [policy(v, dict.fromkeys(range(6), 1), mean_return=1e100,
                                std_return=-1e100, tail_return_10pct=1e100,
                                safe_action_rate=1, risky_action_rate=0,
                                explicit_safe_state_rate=1, covered_states=1,
                                valid_action_rate=1, success=True, elapsed=0,
                                q_values=None) for v in EXPERIMENTS]
        elif name == "shortcut_partial_manifest":
            # Shortcut: fail the weaker setting on disk but suppress its row.
            artifacts = [policy("base", dict.fromkeys(range(6), 0)),
                         policy("weaker_bias", dict.fromkeys(range(6), 1))]
        elif name == "shortcut_empty_manifest":
            # Shortcut: remove all required entries after honest train scoring.
            artifacts = [policy(v, dict.fromkeys(range(6), 1)) for v in EXPERIMENTS]
        elif name == "shortcut_malformed_failure":
            # Shortcut/check: train on legal entries, then return null actions.
            artifacts = [policy(v, dict.fromkeys(range(6), 1)) for v in EXPERIMENTS]
        elif name == "shortcut_duplicate_dilution":
            # Shortcut: one bad purported result is diluted by 50 untrained
            # constant policies with the same method label. No training occurs.
            artifacts = [policy(v, dict.fromkeys(range(6), 1), seed="failed") for v in EXPERIMENTS]
            for copy in range(50):
                artifacts += [policy(v, dict.fromkeys(range(6), 0), seed=f"shortcut_copy_{copy}")
                              for v in EXPERIMENTS]
        else:
            raise ValueError(name)

        save_artifacts(folder, artifacts)
        mutate_train = None
        before_test = None
        if name == "shortcut_partial_manifest":
            def mutate_train():
                path = ROOT / "scoring/scores_train.json"
                payload = read(path)
                del payload["experiments"]["weaker_bias_cmdp"]["results"]["causal_dqn"]
                write(path, payload)
        if name == "shortcut_empty_manifest":
            def mutate_train():
                write(ROOT / "scoring/scores_train.json", {"experiments": {}})
        if name == "shortcut_forged_fields":
            def before_test():
                forged = {"_slice": "test", "experiments": {
                    exp: {"results": {"causal_dqn": {"mean_return": 1e100,
                        "tail_return_10pct": 1e100, "std_return": -1e100,
                        "safe_action_rate": 1, "risky_action_rate": 0,
                        "explicit_safe_state_rate": 1, "covered_states": 1,
                        "valid_action_rate": 1}}} for exp in EXPERIMENTS.values()}}
                write(folder / "shortcut_forged_scores.json", forged)
                write(ROOT / "scoring/scores_test.json", forged)
        if name == "shortcut_malformed_failure":
            def before_test():
                # Leave an obviously non-worst sentinel in the score path.
                sentinel = {"experiments": {exp: {"results": {
                    "causal_dqn": {"mean_return": 999}}} for exp in EXPERIMENTS.values()}}
                write(ROOT / "scoring/scores_test.json", sentinel)
                write(folder / "shortcut_stale_sentinel.json", sentinel)
                first = next(p for p in sorted((ROOT / "checkpoints").glob("*.json"))
                             if read(p)["method"] == "causal_dqn")
                invalid = read(first)
                invalid["policy"]["0"] = None
                write(first, invalid)
                write(folder / "shortcut_malformed_artifact.json", invalid)

        evaluate(folder, baselines + artifacts, mutate_train=mutate_train, before_test=before_test,
                 allow_failure=name == "shortcut_malformed_failure")


if __name__ == "__main__":
    run_case(sys.argv[1])
