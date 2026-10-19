"""Build a self-contained notebook from the reviewed Python source."""
import json
from pathlib import Path

root = Path(__file__).parent
source = (root / "e00_baseline.py").read_text(encoding="utf-8").split('\nif __name__ == "__main__":')[0]
cells = []


def cell(kind, text):
    value = dict(cell_type=kind, metadata={}, source=text.splitlines(keepends=True))
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    cells.append(value)


cell("markdown", """# E00 — occupation/pronoun baseline
An exploratory development experiment for **The Representational Imprint**.

Enable a GPU and internet access, then run cells in order. No paid API, fine-tuning,
or Google Drive access is used. The model is downloaded from Hugging Face.
The notebook stops before downloading if no GPU is available.

We compare `log P(' he' | prompt) - log P(' she' | prompt)` for 40 occupations
and four templates. Positive means **he** is more probable; negative means **she**.
This measures a narrow English pronoun association, not fairness, factual gender,
all gender identities, or persistence of a causal mechanism. These handwritten
prompts are development data, not a validated stereotype benchmark or a held-out test.

The model is a base language model, so we score continuations rather than ask chat questions.
[Model card](https://huggingface.co/EleutherAI/pythia-410m) ·
[GPT-NeoX documentation](https://huggingface.co/docs/transformers/model_doc/gpt_neox)

**Validation status:** source syntax and synthetic data/statistics checks were run locally.
The actual model/GPU experiment has not been run by the authoring assistant.
The notebook performs additional numerical checks when you run it.
""")
cell("code", '%pip -q install "transformers>=4.45,<5" "huggingface_hub>=0.26,<1" numpy matplotlib\n')
cell("markdown", """## Experiment code
The following cell defines the dataset, scorer, checks, summaries and file export.
It does not download or run a model until the next cell calls `run`.
The dependency ranges are compatibility constraints, not a tested lockfile;
the exact installed versions and resolved model commit are saved in `run.json`.
""")
cell("code", source)
cell("markdown", """## Run once
Keep the default model for the intended baseline. If memory is insufficient, lower
`BATCH_SIZE` to 2. Do not silently substitute another model in a comparison.
For exact checkpoint reuse, replace `REVISION` with the saved `resolved_revision`.
All dataset items are development-only; future confirmation requires new frozen splits.
""")
cell("code", '''MODEL_ID = "EleutherAI/pythia-410m"
REVISION = "main"  # Resolved to an immutable commit before loading.
BATCH_SIZE = 8
OUTPUT_ROOT = "/kaggle/working/e00_outputs" if Path("/kaggle/working").exists() else "e00_outputs"
out, rows, summary = run(MODEL_ID, REVISION, BATCH_SIZE, OUTPUT_ROOT)
''')
cell("markdown", """## Inspect wording sensitivity
Each point shows the raw log-probability ratio for one template. Large differences
between points for an occupation indicate wording sensitivity. The person control
is only a diagnostic for a general pronoun preference, not a causal adjustment.
Check `pair_probability_mass` in `scores.csv`: neither pronoun may be likely overall.
The bootstrap resamples occupation groups, retaining all four templates together;
it does not measure training-seed uncertainty or make this handpicked list representative.
""")
cell("code", '''import matplotlib.pyplot as plt
order = sorted(OCCUPATIONS, key=lambda name: np.mean([r["score"] for r in rows if r["occupation"] == name]))
fig, ax = plt.subplots(figsize=(9, 11))
for ti in range(len(TEMPLATES)):
    selected = {r["occupation"]: r["score"] for r in rows if r["kind"] == "occupation" and r["template_id"] == f"t{ti}"}
    ax.scatter([selected[name] for name in order], range(len(order)), label=f"t{ti}: {TEMPLATES[ti]}", s=22, alpha=.8)
ax.axvline(0, color="black", linewidth=.8)
ax.set_yticks(range(len(order)), order)
ax.set_xlabel("log P(he | prefix) − log P(she | prefix); positive favors he")
ax.set_title("Development baseline: variation across four templates")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(out / "occupation_associations.png", dpi=160)
plt.show()
archive = shutil.make_archive(str(out), "zip", out)
print("Save this archive before your session ends:", archive)
''')
cell("markdown", """## Download results
In Colab the following cell offers a browser download. In Kaggle, use the Files/Output
panel to download the ZIP from `/kaggle/working/e00_outputs`.
No results are sent to anyone automatically. Keep this notebook with the archive.

**Next decision:** inspect scoring checks, probability mass, neutral controls and
template consistency. Only then choose and freeze correction/capability thresholds
for a separate fine-tuning experiment. This notebook does not automatically declare
a bias finding, a successful MVP, or evidence of representational persistence.
""")
cell("code", '''try:
    from google.colab import files
except ImportError:
    print("Download from the notebook's Files/Output panel:", archive)
else:
    files.download(archive)
''')
notebook = dict(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                      "language_info": {"name": "python"}}, nbformat=4, nbformat_minor=4)
(root / "E00_baseline.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print("Built E00_baseline.ipynb")
