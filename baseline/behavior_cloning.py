"""Behaviour-cloning baseline: copy the demonstrator's most frequent action.

Not a baseline of the paper; it is included as the simplest learner that uses
the logged actions but ignores rewards. For each observed state it picks the
action logged most often in that state (ties -> lowest action id; states never
visited -> the most frequent action overall). Deterministic, so all runs are
identical.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.cmdp_family import load_dataset


def behavior_policy(inst: dict, n_states: int, n_actions: int) -> list[int]:
    counts = np.zeros((n_states, n_actions))
    np.add.at(counts, (inst["state"], inst["action"]), 1.0)
    fallback = int(counts.sum(0).argmax())
    return [int(row.argmax()) if row.sum() > 0 else fallback for row in counts]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--output-root", default="checkpoints/behavior_cloning")
    args = p.parse_args()
    policies = {}
    for split in args.splits:
        payload = load_dataset(split, args.experiment)
        for inst in payload["instances"]:
            policies[inst["id"]] = behavior_policy(inst, payload["n_states"], payload["n_actions"])
    for k in range(args.runs):
        out = Path(args.output_root) / args.experiment / f"run{k}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"policies": policies}))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
