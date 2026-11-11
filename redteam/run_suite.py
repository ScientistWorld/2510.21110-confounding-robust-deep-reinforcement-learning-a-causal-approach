"""Run independent audit cases sequentially in one compute submission."""
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    "reference", "better_ensemble", "stronger_training", "weaker_no_causal",
    "shortcut_constant_safe", "shortcut_bruteforce_visible", "shortcut_variant_swap",
    "shortcut_data_reward", "shortcut_empty_policy", "shortcut_invalid_actions",
    "shortcut_forged_fields", "shortcut_partial_manifest", "shortcut_empty_manifest",
    "shortcut_malformed_failure", "shortcut_duplicate_dilution",
]

results = {}
for name in CASES:
    print(f"START {name}", flush=True)
    started = time.monotonic()
    completed = subprocess.run(["bash", f"redteam/{name}/run.sh"], cwd=ROOT)
    results[name] = {"runner_exit_code": completed.returncode,
                     "seconds": time.monotonic() - started}
    (ROOT / "redteam/audit/suite.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"FINISH {name}: {results[name]}", flush=True)
    if name == "reference" and completed.returncode:
        raise SystemExit(completed.returncode)
if any(r["runner_exit_code"] for r in results.values()):
    raise SystemExit(1)
