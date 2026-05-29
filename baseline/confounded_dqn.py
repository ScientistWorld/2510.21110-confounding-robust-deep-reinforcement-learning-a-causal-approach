"""Standard off-policy DQN baseline trained on the same confounded replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data.confounded_cmdp import N_ACTIONS, N_STATES, VARIANTS, generate_demonstrations, transition_arrays


class QNetwork(nn.Module):
    def __init__(self, n_states: int = N_STATES, n_actions: int = N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, n_actions),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(torch.nn.functional.one_hot(states, N_STATES).float())


def train_confounded_dqn(args: argparse.Namespace) -> dict[str, object]:
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data = transition_arrays(generate_demonstrations(args.demo_episodes, seed=args.seed, variant=args.variant))
    n = len(data["states"])
    q = QNetwork()
    target = QNetwork()
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=args.lr)

    for step in range(args.updates):
        idx = rng.integers(0, n, size=args.batch_size)
        states = torch.as_tensor(data["states"][idx], dtype=torch.long)
        actions = torch.as_tensor(data["actions"][idx], dtype=torch.long)
        rewards = torch.as_tensor(data["rewards"][idx], dtype=torch.float32)
        next_states = torch.as_tensor(data["next_states"][idx], dtype=torch.long)
        dones = torch.as_tensor(data["dones"][idx], dtype=torch.float32)

        with torch.no_grad():
            next_values = target(next_states).max(dim=1).values
            targets = rewards + args.gamma * (1.0 - dones) * next_values

        chosen_q = q(states).gather(1, actions[:, None]).squeeze(1)
        loss = torch.mean((chosen_q - targets) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % args.target_interval == 0:
            target.load_state_dict(q.state_dict())
        if args.log_interval and ((step + 1) % args.log_interval == 0 or step == 0):
            print(
                f"conf_dqn step={step + 1}/{args.updates} loss={loss.item():.6f}",
                flush=True,
            )

    with torch.no_grad():
        all_states = torch.arange(N_STATES, dtype=torch.long)
        q_values = q(all_states).numpy()
    policy = {str(s): int(np.argmax(q_values[s])) for s in range(N_STATES)}
    return {
        "method": "conf_dqn",
        "variant": args.variant,
        "experiment": VARIANTS[args.variant].experiment,
        "seed": args.seed,
        "demo_episodes": args.demo_episodes,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "policy": policy,
        "q_values": q_values.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="checkpoints/conf_dqn_policy.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--demo-episodes", type=int, default=500)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--target-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="base")
    args = parser.parse_args()

    artifact = train_confounded_dqn(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
