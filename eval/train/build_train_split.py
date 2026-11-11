"""Build the public training split from the public family generator.

Writes ``data/train/<experiment>.json`` (logged demonstrations) and
``eval/train/envs/<experiment>.json`` (true dynamics used by the train-slice
evaluator). Idempotent: existing files are kept unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.cmdp_family import (
    EXPERIMENTS,
    collect_demonstrations,
    config_dict,
    dataset_header,
    dataset_record,
    env_record,
    generate_instance,
)

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_SEED = 20251021
N_INSTANCES = 32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for exp_index, (experiment, cfg) in enumerate(EXPERIMENTS.items()):
        data_path = ROOT / "data" / "train" / f"{experiment}.json"
        env_path = ROOT / "eval" / "train" / "envs" / f"{experiment}.json"
        if data_path.exists() and env_path.exists() and not args.force:
            print(f"keep {data_path} and {env_path}")
            continue
        records, envs = [], []
        for i in range(N_INSTANCES):
            rng = np.random.default_rng(np.random.SeedSequence([PUBLIC_SEED, exp_index, i]))
            inst = generate_instance(rng, cfg)
            log = collect_demonstrations(inst, cfg, rng)
            iid = f"train_{i:02d}"
            records.append(dataset_record(iid, inst, log))
            envs.append(env_record(iid, inst))
        header = dataset_header(experiment, "train", cfg)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text(json.dumps({**header, "instances": records}, separators=(",", ":")))
        env_path.write_text(json.dumps({**header, "config": config_dict(cfg), "instances": envs},
                                       separators=(",", ":")))
        print(f"wrote {data_path} and {env_path}")


if __name__ == "__main__":
    main()
