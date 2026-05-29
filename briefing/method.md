# Method

The paper proposes Causal-DQN, a DQN variant for confounded off-policy data. It uses experience replay like DQN, but changes the Bellman target.

For each replay transition `(s_i, x_i, y_i, s_{i+1})`, Causal-DQN builds a target for every possible action. For the logged action, it uses the usual one-step DQN target. For actions not taken in the logged transition, it uses the known reward lower bound plus the value of the worst next state. This implements the paper's causal Bellman lower bound and learns a pessimistic policy for the worst CMDP compatible with the observations.

The reproduced implementation is in `method/causal_dqn.py`. It uses a small MLP Q-network, replay samples from the confounded demonstrator, a target network, and the all-action causal target from Algorithm 2 / Eq. 26.

The baseline in `baseline/confounded_dqn.py` is a standard off-policy DQN trained on the same confounded replay, updating only the logged action.

