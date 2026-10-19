# Run the first experiment

Upload `E00_baseline.ipynb` to Colab or Kaggle. Enable a GPU and internet, then run its cells in order. The notebook contains all experiment code; uploading the Python file separately is unnecessary. Save the ZIP before the session ends. No paid model API is used.

The default is Pythia-410M in float32 with batches of eight short prompts. Actual runtime and GPU allocation are measured, not promised. On out-of-memory errors, start a fresh runtime with a smaller batch. On a package import error after installation, restart the session and rerun the cells. Do not bypass failed numerical checks.

## What the code does

1. Creates 40 occupation prefixes in four sentence templates, plus four `person` controls: 164 prompts total.
2. Saves the dataset and its hash. Resolves the model branch to a specific commit and loads both model and tokenizer from it.
3. Verifies ` he` and ` she` are distinct single tokens and appending them does not change the prefix's tokenization.
4. Runs the model without gradients or text sampling. The last non-padding token predicts the next token. A single forward pass scores both candidates.
5. Checks batching, repeat stability, candidate reversal, identical candidates and agreement with an independent teacher-forced loss calculation. Stops on failure.
6. Saves every prompt's scores, grouped summaries, occupation-cluster bootstrap intervals, timings, GPU memory and package versions. Produces an optional chart and ZIP.

The score is `log P(he | prefix) - log P(she | prefix)`. Positive favors he; negative favors she; zero means equal probabilities for these two candidates. For example, probabilities .20 and .10 give about +.693. This example is arithmetic, not an observed result. `pair_probability_mass` is their sum: a ratio alone can look large even when both continuations are unlikely.

`mean_absolute_item_log_ratio` prevents opposite signed preferences cancelling. It can still include a model-wide pronoun preference. The person controls and `score_minus_person_control` help inspect that possibility, but are not a causal adjustment or a complete fairness metric. Template plots expose wording sensitivity.

## Output files

| File | Purpose |
| --- | --- |
| dataset.json | Exact development prompts |
| tokenization.json | Prefix and candidate token IDs |
| scores.csv | Item-level log probabilities, probabilities, scores, model revision and dataset hash |
| by_occupation.csv / by_template.csv | Grouped summaries, including control-adjusted diagnostics |
| summary.json | Overall descriptive metrics and grouped bootstrap intervals |
| checks.json | Numerical checks and tolerance |
| run.json | Status, versions, checkpoint, timings and GPU memory |
| occupation_associations.png | Notebook-generated wording-sensitivity plot |

All prompts are handcrafted development data. They are not a representative occupation sample, validated stereotype labels, or confirmation set. Binary pronoun comparisons do not cover all genders; grammatical continuations also do not guarantee coreference with the occupation. No inference about an individual's gender is intended.

The bootstrap is conditional on this small chosen dataset and fixed wording. Repeated model inference is not independent training replication. No fine-tuning, capability-retention test, hidden-state probe or causal intervention is performed yet.

## Validation status

The authoring environment lacks PyTorch. Local validation covers Python/notebook syntax and synthetic dataset/statistics tests. The real model download, GPU execution and built-in numerical checks must be completed in Colab/Kaggle before treating outputs as validated. Dependency ranges are not a frozen, experimentally verified environment; save `run.json` and the notebook for reproducibility.

## Sources

- Official model card: https://huggingface.co/EleutherAI/pythia-410m
- GPT-NeoX API: https://huggingface.co/docs/transformers/model_doc/gpt_neox
- Local research design: RESEARCH_START.md; source-specific methodological review: readings/READING_LOG.md

Next: inspect the actual baseline and its controls, then freeze a separate fine-tuning protocol and held-out evaluation. Do not interpret this notebook alone as evidence that a representational imprint exists.
