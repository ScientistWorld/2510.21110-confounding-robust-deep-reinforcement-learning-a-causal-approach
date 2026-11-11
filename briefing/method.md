# Method

The paper proposes Causal-DQN: DQN trained on a confounded demonstrator's transitions, with the causal Bellman lower bound (Prop. 3.1, Eq. 8, Algorithm 2) as its target.

For each replayed transition `(s, x, y, s')`:

- **Logged action `x`:** gets the usual target `y + gamma * max_x' Q(s', x')`, using the Double-DQN form with a target network.
- **Every other action:** gets the pessimistic target `a + gamma * min_s'' max_x' Q(s'', x')`. Here `a = -1` is the reward floor. The worst next state `s''` is estimated by sampling candidate next states from the replay memory and keeping the one with the smallest value.
- **Loss:** the squared error summed over all actions.

The learned greedy policy maximises a lower bound on the interventional value that holds for every environment compatible with the log.

## Implementation

The implementation is `method/causal_dqn.py`. It keeps the paper's procedure and scales its resources down:

- Logged transitions are streamed in order into a replay memory, and updates are interleaved with the stream.
- Batch size 512, Adam with learning rate 5e-4, discount 0.99, Double DQN with a target network refreshed every 500 updates.
- 64 sampled candidates for the worst next state.
- The Q-network is an MLP over the state features: 16 → 128 → 128 → 4.
- 30,000 updates over the 10,000-transition log.
- One independent network per instance, trained in a batch on GPU.

## Baselines

- **`baseline/confounded_dqn.py`:** the paper's Conf. DQN. It uses the same data, network, optimiser and scale, with the standard DQN loss on the logged action only.
- **`baseline/behavior_cloning.py`:** not in the paper. It picks the most frequent logged action per state and serves as a reward-free sanity floor.
