# Constraints

Use the tracked CMDP variants in `data/confounded_cmdp.py` without changing hidden-state probabilities, reward values, horizon, action meanings, variant names, or evaluation seeds.

Evaluation must read policy artifacts and must not import from `method/`. Improvements should come from better handling of confounded off-policy data, not from changing the environment or evaluation protocol.
