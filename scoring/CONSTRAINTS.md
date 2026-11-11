# Constraints

- Learn only from the logged demonstrations in `data/<split>/<experiment>.json`.
- Do not use the true dynamics of any instance: not `eval/*/envs/`, and not the hidden test side.
- The public generator (`data/cmdp_family.py`) and the train instances may be used for development. A method must still produce test policies from the test logs alone.
- Keep the output contract: `checkpoints/<method>/<experiment>/run{0,1,2}.json` with one legal action per observed state for every train and test instance. The three runs are independent repetitions of the same method.
- Do not change the problem definition, the datasets, `eval/` or the scripts that score. The scorer recomputes everything from the policies, so self-reported numbers are ignored.
- The learner's policy acts on the observed state only. It cannot see the hidden context at execution time.
