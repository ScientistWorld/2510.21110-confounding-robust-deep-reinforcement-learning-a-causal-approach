"""Causal-DQN (Li, Zhang & Bareinboim; Algorithm 2) on the confounded CMDP family.

Faithful to the paper's procedure at reduced scale:

* the demonstrator's logged transitions arrive as a stream and are appended to
  a replay memory in order (Algorithm 2, steps 6-9); the learner never acts;
* after each new transition the learner performs ``updates / transitions``
  minibatch updates on samples from the replay memory (step 10);
* every action gets a target (Eq. 26): the logged action gets the usual
  Double-DQN bootstrapped target, every other action gets the reward floor
  ``a`` plus the discounted value of the worst next state, where the worst
  state is estimated by sampling candidate next states from the replay memory
  and taking the one with the smallest value (Sec. 3);
* the loss is the squared error summed over actions (Eq. 9 / step 12), the
  target network is refreshed every ``target_interval`` updates (step 13).

Independent Q-networks are trained for every instance of every requested
split at once (batched matrix multiplies); instances never share parameters
or data. Output: ``<output-root>/<experiment>/run<k>.json`` with one greedy
action per observed state for every instance, plus the final Q-values.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from data.cmdp_family import REWARD_FLOOR, load_dataset


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_batch(experiment: str, splits: list[str], device: torch.device) -> dict:
    ids, obs, cols = [], [], {k: [] for k in ("state", "action", "reward", "next_state", "done")}
    meta = None
    for split in splits:
        payload = load_dataset(split, experiment)
        meta = meta or payload
        for inst in payload["instances"]:
            ids.append(inst["id"])
            obs.append(inst["obs"])
            for k in cols:
                cols[k].append(inst[k])
    lengths = {len(c) for c in cols["state"]}
    if len(lengths) != 1:
        raise ValueError("all instances must log the same number of transitions")
    batch = {k: torch.as_tensor(np.stack(v), device=device) for k, v in cols.items()}
    batch["state"] = batch["state"].long()
    batch["action"] = batch["action"].long()
    batch["next_state"] = batch["next_state"].long()
    batch["reward"] = batch["reward"].float()
    batch["done"] = batch["done"].float()
    batch["obs"] = torch.as_tensor(np.stack(obs), device=device).float()
    batch["ids"] = ids
    batch["n_states"] = meta["n_states"]
    batch["n_actions"] = meta["n_actions"]
    return batch


class BatchedQNetwork(torch.nn.Module):
    """One independent MLP Q-network per instance: obs -> Q(s, .)."""

    def __init__(self, n_nets: int, in_dim: int, hidden: int, n_actions: int, generator: torch.Generator):
        super().__init__()
        dims = [in_dim, hidden, hidden, n_actions]
        self.weights = torch.nn.ParameterList()
        self.biases = torch.nn.ParameterList()
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            bound = 1.0 / math.sqrt(d_in)  # torch.nn.Linear default initialisation
            w = (torch.rand(n_nets, d_in, d_out, generator=generator) * 2 - 1) * bound
            b = (torch.rand(n_nets, 1, d_out, generator=generator) * 2 - 1) * bound
            self.weights.append(torch.nn.Parameter(w))
            self.biases.append(torch.nn.Parameter(b))

    def forward(self, obs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        h = torch.gather(obs, 1, states[..., None].expand(-1, -1, obs.shape[-1]))
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = torch.baddbmm(b, h, w)
            if i < last:
                h = torch.relu(h)
        return h


def double_dqn_value(q, q_target, obs, states):
    a_star = q(obs, states).argmax(-1, keepdim=True)
    return q_target(obs, states).gather(-1, a_star).squeeze(-1)


def train(args: argparse.Namespace) -> dict:
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    gen_cpu = torch.Generator().manual_seed(args.seed)
    batch = load_batch(args.experiment, args.splits, device)
    obs = batch["obs"]
    N, T = batch["state"].shape
    A = batch["n_actions"]
    q = BatchedQNetwork(N, obs.shape[-1], args.hidden, A, gen_cpu).to(device)
    q_target = BatchedQNetwork(N, obs.shape[-1], args.hidden, A, gen_cpu).to(device)
    q_target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=args.lr)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    rows = torch.arange(N, device=device)[:, None]

    start = min(args.learning_starts, T)
    started = time.time()
    for j in range(args.updates):
        # Replay memory = first `size` logged transitions (streamed in order).
        size = start + (j * (T - start)) // max(args.updates, 1) + 1
        size = min(size, T)
        idx = torch.randint(0, size, (N, args.batch_size), generator=gen, device=device)
        s = batch["state"][rows, idx]
        x = batch["action"][rows, idx]
        y = batch["reward"][rows, idx]
        s2 = batch["next_state"][rows, idx]
        d = batch["done"][rows, idx]
        with torch.no_grad():
            logged = y + args.gamma * (1.0 - d) * double_dqn_value(q, q_target, obs, s2)
            # Worst-case next state: sample candidate next states from the
            # replay memory and keep the one with the smallest value.
            widx = torch.randint(0, size, (N, args.worst_state_samples), generator=gen, device=device)
            cand = batch["next_state"][rows, widx]
            v_worst = double_dqn_value(q, q_target, obs, cand).min(-1).values  # [N]
            pessimistic = args.reward_floor + args.gamma * (1.0 - d) * v_worst[:, None]
            onehot = torch.nn.functional.one_hot(x, A).float()
            targets = onehot * logged[..., None] + (1.0 - onehot) * pessimistic[..., None]
        pred = q(obs, s)
        # Eq. 9: squared error summed over all actions, averaged over the batch;
        # summed over instances (each instance only receives its own gradient).
        loss = ((pred - targets) ** 2).sum(-1).mean(-1).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (j + 1) % args.target_interval == 0:
            q_target.load_state_dict(q.state_dict())
        if args.log_interval and ((j + 1) % args.log_interval == 0 or j == 0):
            print(f"causal_dqn {args.experiment} run{args.run} update {j + 1}/{args.updates} "
                  f"buffer={size} loss/net={loss.item() / N:.4f} {time.time() - started:.0f}s", flush=True)

    with torch.no_grad():
        all_states = torch.arange(batch["n_states"], device=device)[None, :].expand(N, -1)
        qv = q(obs, all_states).cpu().numpy()
    return {
        "policies": {iid: [int(a) for a in qv[i].argmax(-1)] for i, iid in enumerate(batch["ids"])},
        "q_values": {iid: np.round(qv[i], 5).tolist() for i, iid in enumerate(batch["ids"])},
        "config": {k: v for k, v in vars(args).items() if k not in ("output_root",)},
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--run", type=int, default=0, help="run index k; output file is run<k>.json")
    p.add_argument("--seed", type=int, default=None, help="defaults to the run index")
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--output-root", default="checkpoints/causal_dqn")
    p.add_argument("--updates", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--target-interval", type=int, default=500)
    p.add_argument("--learning-starts", type=int, default=500)
    p.add_argument("--worst-state-samples", type=int, default=64)
    p.add_argument("--reward-floor", type=float, default=REWARD_FLOOR)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--log-interval", type=int, default=5000)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.seed is None:
        args.seed = args.run
    artifact = train(args)
    out = Path(args.output_root) / args.experiment / f"run{args.run}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
