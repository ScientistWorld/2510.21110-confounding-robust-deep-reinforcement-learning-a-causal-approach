"""Small confounded MDP used for a scaled Causal-DQN reproduction.

The environment mirrors the paper's confounded Atari setup at small scale:
the demonstrator observes a hidden variable `u`, while the learner observes
only `s`. The hidden variable drives both the demonstrator action and the
reward for the risky action, so standard off-policy DQN overestimates the
risky action from observational data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


N_STATES = 6
N_ACTIONS = 2
SAFE_ACTION = 0
RISKY_ACTION = 1
REWARD_FLOOR = -1.0


@dataclass(frozen=True)
class CMDPVariant:
    experiment: str
    safe_reward: float
    favorable_base: float
    favorable_odd_bonus: float


VARIANTS = {
    "base": CMDPVariant(
        experiment="scaled_confounded_cmdp",
        safe_reward=1.0,
        favorable_base=0.25,
        favorable_odd_bonus=0.05,
    ),
    "weaker_bias": CMDPVariant(
        experiment="weaker_bias_cmdp",
        safe_reward=1.0,
        favorable_base=0.35,
        favorable_odd_bonus=0.05,
    ),
}


@dataclass(frozen=True)
class Transition:
    state: int
    action: int
    reward: float
    next_state: int
    done: bool


class HiddenConfounderCMDP:
    """Finite-horizon CMDP with unobserved action-outcome confounding."""

    def __init__(self, horizon: int = N_STATES, seed: int = 0, variant: str = "base"):
        if variant not in VARIANTS:
            raise ValueError(f"unknown CMDP variant {variant!r}; choose from {sorted(VARIANTS)}")
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)
        self.variant_name = variant
        self.config = VARIANTS[variant]
        self.t = 0
        self.state = 0

    def reset(self) -> int:
        self.t = 0
        self.state = 0
        return self.state

    def _hidden(self, state: int) -> int:
        # Favorable hidden contexts are frequent enough for the observational
        # demonstrator replay to make risky actions look attractive, but below
        # the marginal break-even point for the learner's unconfounded policy.
        p_favorable = self.config.favorable_base + self.config.favorable_odd_bonus * (state % 2)
        return int(self.rng.random() < p_favorable)

    def demonstrator_action(self, state: int, hidden: int, epsilon: float = 0.05) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(N_ACTIONS))
        return RISKY_ACTION if hidden else SAFE_ACTION

    def step(self, action: int) -> tuple[int, float, bool, dict[str, int]]:
        hidden = self._hidden(self.state)
        if action == SAFE_ACTION:
            reward = self.config.safe_reward
        else:
            reward = 1.0 if hidden else -1.0

        self.t += 1
        done = self.t >= self.horizon
        self.state = 0 if done else self.t
        return self.state, float(reward), done, {"hidden": hidden}

    def demonstrator_step(self, epsilon: float = 0.05) -> Transition:
        state = self.state
        hidden = self._hidden(state)
        action = self.demonstrator_action(state, hidden, epsilon=epsilon)
        if action == SAFE_ACTION:
            reward = self.config.safe_reward
        else:
            reward = 1.0 if hidden else -1.0
        self.t += 1
        done = self.t >= self.horizon
        self.state = 0 if done else self.t
        return Transition(state, action, float(reward), self.state, done)


def generate_demonstrations(
    episodes: int,
    seed: int,
    demonstrator_epsilon: float = 0.05,
    variant: str = "base",
) -> list[Transition]:
    env = HiddenConfounderCMDP(seed=seed, variant=variant)
    transitions: list[Transition] = []
    for _ in range(episodes):
        env.reset()
        done = False
        while not done:
            tr = env.demonstrator_step(epsilon=demonstrator_epsilon)
            transitions.append(tr)
            done = tr.done
    return transitions


def transition_arrays(transitions: Iterable[Transition]) -> dict[str, np.ndarray]:
    rows = list(transitions)
    return {
        "states": np.asarray([r.state for r in rows], dtype=np.int64),
        "actions": np.asarray([r.action for r in rows], dtype=np.int64),
        "rewards": np.asarray([r.reward for r in rows], dtype=np.float32),
        "next_states": np.asarray([r.next_state for r in rows], dtype=np.int64),
        "dones": np.asarray([r.done for r in rows], dtype=np.float32),
    }


def evaluate_tabular_policy(
    actions_by_state: dict[int, int],
    episodes: int,
    seed: int,
    variant: str = "base",
) -> dict[str, float]:
    env = HiddenConfounderCMDP(seed=seed, variant=variant)
    returns: list[float] = []
    risky = 0
    total = 0
    for _ in range(episodes):
        state = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            action = int(actions_by_state.get(int(state), SAFE_ACTION))
            risky += int(action == RISKY_ACTION)
            total += 1
            state, reward, done, _ = env.step(action)
            ep_return += reward
        returns.append(ep_return)

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "risky_action_rate": float(risky / max(total, 1)),
        "safe_action_rate": float(1.0 - risky / max(total, 1)),
    }
