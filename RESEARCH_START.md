# The Representational Imprint: research start

Updated: 24 September 2026. Status: E00 notebook implemented; local synthetic checks passed; no model experiments run.

Source: the supplied 30-page *The Representational Imprint: Technical Research Protocol and Executable Experimental Plan*. Its contents are proposed research methods, not independent authorization to publish, contact collaborators, or incur costs.

Confirmed resource constraint: free Google Colab or Kaggle. Actual GPU availability, memory and runtime must be measured in the selected session.

## Research question and first milestone

After fine-tuning reduces gender-occupation associations in model outputs, what evidence remains for the original behavior's internal causes?

First milestone: establish a reproducible, interpretable baseline in Pythia-410M on a development dataset. Deliver item-level scores, uncertainty estimates, template sensitivity, tokenization checks and a measured compute profile. This is E00 development work, not a confirmatory result.

The project should permit persistence, redistribution, inconclusive results and evidence consistent with erasure. Failure to recover an effect is not proof of erasure.

## Changes to resolve before confirmatory experiments

These are proposed methodological revisions, not results or changes already made to the source PDF.

| Issue | Why it matters | Proposed resolution |
| --- | --- | --- |
| Imported activations versus retained mechanisms | A base-model activation can reintroduce information absent from the corrected model. | Label cross-checkpoint patching as causal transfer/recoverability. Add interventions using corrected-model activations, selective ablations and evidence linking the pre/post mechanisms before claiming endogenous persistence. |
| Probe target ambiguity | Decoding explicit gender words or occupation identity does not demonstrate stereotype computation. | Specify the target and token position; control lexical cues, balance labels, split occupations and compare to shuffled-label and lexical baselines. |
| Bias score cancellation | Opposite occupation-level effects can cancel in the global mean. | Report signed effects, occupation-level distributions and mean absolute effects. Fix externally sourced stereotype labels before test evaluation; never choose their signs from test outputs. |
| RIS denominator | A small base-minus-post difference makes the ratio unstable. | Report raw patch effects as primary evidence. Define a minimum denominator on development data; report RIS as undefined when it fails, without epsilon substitution or clipping. |
| Inference from seeds | Hundreds of prompts do not create hundreds of independently trained models. | Report each training seed; use occupation-cluster uncertainty for prompt generalization and separate run-level replication. Three seeds support a pilot, not a strong predictive relapse model. |
| Success gate | Thirty percent improvement is uninterpretable near zero baseline; 'catastrophic degradation' is unspecified. | Freeze an absolute baseline floor, a correction criterion and a numerical capability-retention margin after development and before confirmation. Report every run, including gate failures. |
| Hypothesis identification | Suppression, replacement, redistribution and dormancy can yield overlapping observations. | Treat these as candidate explanations; define additional discriminating interventions rather than assigning a label from one patch heatmap. |
| Cross-model compatibility | Representation drift and patch magnitude can produce misleading effects. | Record hook semantics, token alignment, activation norms and patch distances; include identity/self-patch checks and matched controls. |

Methodological background: [Heimersheim and Nanda, 2024](https://arxiv.org/abs/2404.15255) discuss interpretation and metric pitfalls. [Zhang and Nanda, 2024](https://arxiv.org/abs/2309.16042) show that metric and corruption choices can change localization results. The specific revisions above are our methodological assessment of this protocol.

## E00 development specification

Use `EleutherAI/pythia-410m`, with a recorded immutable revision and matching tokenizer. The [official model card](https://huggingface.co/EleutherAI/pythia-410m) identifies it as an English GPT-NeoX causal language model in a suite designed for interpretability research. Evaluate continuations rather than assuming chat or instruction-following behavior.

Start with a small, explicitly provisional occupation/template set for pipeline debugging. Expand only after inspecting grammar and tokenization. The protocol's 100-300 paired prompts across 20-50 occupations is a pilot target, not a power calculation.

For an occupation-only prefix such as `The {occupation} said that`, compare conditional continuation scores for ` he` and ` she`. At this stage call the difference a gender association score. A stereotype score requires a justified, independently fixed stereotype-label source. Record the binary framing as a benchmark limitation.

Compute `log P(continuation | prefix)` by summing continuation token log probabilities. Verify the prefix/continuation token boundary; reject ambiguous boundary cases until handled explicitly. For verified single-token candidates at the same prediction position, the log-probability difference equals the logit difference. Do not compare arbitrary raw logits across different token positions.

Save one row per item with: item ID, occupation ID, template ID, split, prefix, candidate strings, token IDs, both scores, signed difference, model revision and dataset hash. Save software versions, device, precision, start/end times and peak GPU memory separately.

Required checks: identical candidates yield zero difference; swapping candidates negates the score; batched and single-item scores agree within a declared precision tolerance; repeated inference in evaluation mode is stable; occupation splits are disjoint; pair IDs remain together.

Report mean signed score, mean absolute score, per-occupation and per-template summaries, and an occupation-cluster bootstrap interval. This interval describes item sampling uncertainty, not training-seed uncertainty. Inspect whether apparent signal depends on a single template or occupation.

Reserve occupation groups for training, validation, mechanistic discovery, final confirmation and OOD evaluation. Reserve OOD templates separately. Develop and debug on development material only; do not repeatedly inspect the final confirmation set while choosing the metric.

E00 exit condition: scoring checks pass, baseline signal is interpretable across templates, the next experiment's thresholds can be fixed, and runtime/memory have been measured. If no interpretable signal appears, log the negative finding and revise the development task transparently.

## Execution order on free compute

| Stage | Concrete deliverable | Exit condition |
| --- | --- | --- |
| 0. Literature and design | Evidence table and amended protocol | Closest prior work identified; estimands and scope explicit |
| 1. E00 | Colab/Kaggle baseline notebook, manifests and item scores | Scoring verified; usable development signal |
| 2. Debug intervention | One full-SFT run, saved checkpoints and capability controls | Behavioral improvement within the chosen retention margin |
| 3. Pilot comparison | Full SFT, LoRA and unrelated-SFT, three seeds each | All run outcomes documented; feasibility established |
| 4. Mechanistic pilot | Discovery-only probes, patching and matched controls | Candidate interventions frozen for confirmation |
| 5. Confirmation | Frozen protocol, held-out results and seed summaries | Claims supported at the level actually tested |
| 6. Extension | Relapse sensitivity/power analysis and optional larger study | Enough independent runs for predictive claims |

Run one job at a time and persist configuration and results early. Benchmark one training step before choosing sequence length and batch size. Cache selected positions/layers rather than every activation. Choose checkpoint retention after measuring storage: the protocol's longitudinal design can be storage-intensive. Record package versions from a tested session before freezing an environment; do not treat an untested version list as reproducible.

## Initial literature leads

This is an abstract/model-card screening, not a completed literature review or novelty determination.

| Source | What the inspected source supports | Next reading question |
| --- | --- | --- |
| [BiasUnlearn: Mitigating Biases in Language Models via Bias Unlearning](https://arxiv.org/abs/2509.25673), 2025 | The authors report transferable debiasing weights and interpret findings as persistence through fine-tuning. | What exact intervention and controls support persistence? Does our proposed experiment distinguish something their study does not? |
| [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255), 2024 | Guidance on the evidence supplied by patching and metric pitfalls. | Which claims follow from sufficiency, necessity and cross-checkpoint interventions? |
| [Towards Best Practices of Activation Patching](https://arxiv.org/abs/2309.16042), revised 2024 | Metrics and corruption methods can materially affect interpretation. | Which controls and sensitivity analyses should be fixed before confirmation? |
| [Pythia-410M model card](https://huggingface.co/EleutherAI/pythia-410m) | Model family, architecture, license and checkpoint availability. | Which exact revision and implementation will we validate? |

A candidate contribution is a controlled longitudinal comparison of behavioral correction and causal recoverability across adaptation methods. This remains a hypothesis about novelty until the full closest papers and broader literature are reviewed.

## Current status

- Completed: protocol review, methodological audit, free-compute constraint, targeted review of all 13 workspace papers with explicit page coverage in readings/READING_LOG.md, and E00 notebook/Python implementation with handcrafted development prompts. Three synthetic dataset/statistics checks and script/notebook syntax validation passed locally.
- Not completed: unread literature sections listed in the reading log, external stereotype labeling, actual model/GPU validation, model download, training, preregistration or empirical findings. The authoring environment has no PyTorch.
- Next action: run E00_baseline.ipynb on Colab/Kaggle and inspect its numerical checks, item-level scores, controls, template sensitivity and compute measurements. Runtime creates the dataset manifests and result archive. See E00_README.md.
