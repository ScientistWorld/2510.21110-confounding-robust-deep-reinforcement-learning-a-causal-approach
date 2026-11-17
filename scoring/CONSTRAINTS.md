# Constraints

- Learn only from the logged demonstrations in `data/<split>/<experiment>.json`, plus the documented facts about every instance: the reward range, the termination probability and `logging_epsilon`.
- Do not use the true dynamics of any instance: not `eval/*/envs/`, and not the hidden test side.
- The public generator (`data/cmdp_family.py`) and the train instances may be used to develop, debug and sanity-check a method, and to set a few hyperparameters.
  - Do not fit a model that maps log statistics to the true interventional values of simulated or train instances. That learns this synthetic generator's prior rather than solving the problem, and it is out of scope.
  - A method must produce test policies from the test logs alone.
- Keep the output contract: `checkpoints/<method>/<experiment>/run{0,1,2}.json`, with one legal action per observed state for every train and test instance. The three runs are independent repetitions of the same method.
- Do not change the problem definition, the datasets, `eval/` or the scripts that score. The scorer recomputes everything from the policies, so self-reported numbers are ignored.
- The learner's policy acts on the observed state only. It cannot see the hidden context at execution time.
