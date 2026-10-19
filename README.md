# The Representational Imprint

Code for an exploratory study of occupation-conditioned pronoun associations before and after a model correction. The baseline scores next-token log-probability differences. The pilot separates overall pronoun preference from the occupation-specific pattern and examines what persists after correction.

## Contents

- `e00_baseline.py` and `E00_baseline.ipynb`: baseline prompts, numerical checks, grouped summaries, and bootstrap intervals.
- `pilot_runtime.py`: development-pilot scoring and training utilities.
- `test_e00_baseline.py` and `test_pilot_metrics.py`: synthetic checks of dataset identity and statistical calculations.
- `pilot_inputs/`: development inputs used to assemble the portable pilot notebook.
- `PILOT_PROTOCOL.txt` and `E00_README.md`: protocol, score definitions, and limitations.

## Reproduce the local checks

```sh
python -m pip install numpy
python -m unittest discover -p 'test_*.py' -v
python build_e00_notebook.py
python build_pilot_notebook.py
```

The generated pilot notebook and input archive are written to `colab_kaggle/`. Model-dependent runs need the notebook's declared packages and GPU resources. These are not part of the lightweight local checks. Record model revisions, package versions, seeds, and the exported checks with any run.

## Interpretation

The prompts are a small handcrafted English development set. Binary pronoun scores do not measure all genders, social fairness, or a causal mechanism. Bootstrap intervals are conditional on this dataset. A small residual association is not proof of persistent internal representation. Research is in progress; synthetic unit tests do not validate GPU training, checkpoint recovery, or the scientific hypothesis.
