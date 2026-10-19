"""Build portable notebook + compact inputs. Never accesses confirmation data."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / 'colab_kaggle'
OUT.mkdir(exist_ok=True)
cells = []


def md(text):
    cells.append(dict(cell_type='markdown', metadata={}, source=text))


def code(text):
    ast.parse(text)
    cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=text))


md('''# What Returns After Correction? — E04 development pilot
Works on **Google Colab or Kaggle**, one run at a time. Select a GPU first.

Upload `pilot_inputs.zip` alongside this notebook. This package contains only
development inputs; no confirmation occupations or templates are included.

**Start with cells 1–6 only:** reconstruct E03 seed 2026 and export its comparison.
The original executed E03 notebook is unavailable. This reconstruction uses explicit
AdamW defaults and NumPy shuffling; numerical agreement is not assumed.
Do not start continued training until you have reviewed the reconstruction.
No findings about recovery exist yet. This is not a social-fairness certification.

Colab: Runtime → Change runtime type → GPU. Upload the ZIP with the Files sidebar.
Kaggle: create/import this notebook, enable GPU and Internet, and attach the ZIP
as a **private** input dataset. Do not publish the research data by accident.
Internet is needed initially for packages, the model and WikiText.
''')
md('## 1. Install matching research libraries\nKeep the platform-provided GPU PyTorch. Restart the session if the platform asks, then rerun from this cell. Package differences are recorded.')
code('''import os, sys, subprocess
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'transformers==4.57.6', 'datasets==4.8.5',
                'huggingface-hub==0.36.2', 'matplotlib'], check=True)
import torch
assert torch.cuda.is_available(), 'Enable a GPU runtime before proceeding.'
print(torch.__version__, torch.cuda.get_device_name())
''')
md('''## 2. Locate and unpack the development package
Set `INPUT_ZIP` manually if automatic discovery finds more than one copy.
If Kaggle expanded the ZIP already, point `EXPANDED_BUNDLE` to the folder containing
`pilot_runtime.py` and `checksums.json`. Extraction rejects unsafe paths.
''')
code('''from pathlib import Path
import zipfile
INPUT_ZIP = None  # Example: '/content/pilot_inputs.zip'
EXPANDED_BUNDLE = None
WORK = Path('/kaggle/working' if Path('/kaggle').exists() else '/content')
if EXPANDED_BUNDLE:
    BUNDLE = Path(EXPANDED_BUNDLE)
else:
    if INPUT_ZIP is None:
        candidates = list(WORK.glob('pilot_inputs*.zip'))
        if Path('/kaggle/input').exists():
            candidates += list(Path('/kaggle/input').rglob('pilot_inputs*.zip'))
        assert len(candidates) == 1, 'Set INPUT_ZIP to the single package you uploaded.'
        INPUT_ZIP = candidates[0]
    BUNDLE = WORK / 'pilot_bundle'
    BUNDLE.mkdir(exist_ok=True)
    with zipfile.ZipFile(INPUT_ZIP) as archive:
        for member in archive.infolist():
            target = (BUNDLE / member.filename).resolve()
            assert target.is_relative_to(BUNDLE.resolve()), 'Unsafe archive path'
        archive.extractall(BUNDLE)
sys.path.insert(0, str(BUNDLE))
print('Bundle:', BUNDLE)
''')
md('''## 3. Choose output storage and verify inputs
Keep the same `OUTPUT_DIR` to resume. Use a new versioned directory after changing
code or settings. Colab files under `/content` are temporary. For persistent Colab
storage set `USE_COLAB_DRIVE=True`; mounting requires your normal Google authorization.

Kaggle working files are not automatically durable: save/export them before leaving.
Full-model resume states are several GB; an atomic save temporarily needs two copies.
Start with at least **14 GiB free**. Over the whole project, retained weights can
require more; export completed runs or use larger persistent storage. Measurements
ZIPs deliberately exclude weights. Do not delete checkpoints needed for later work.
''')
code('''USE_COLAB_DRIVE = False
if USE_COLAB_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    OUTPUT_DIR = Path('/content/drive/MyDrive/representational_imprint/e04_v01')
else:
    OUTPUT_DIR = WORK / 'e04_v01'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
import shutil
free_gib = shutil.disk_usage(OUTPUT_DIR).free / 1024**3
assert free_gib >= 14, f'Only {free_gib:.1f} GiB free: allocate storage before running.'
from pilot_runtime import Pilot
pilot = Pilot(BUNDLE, OUTPUT_DIR)
print('Outputs:', OUTPUT_DIR, '| free GiB:', round(free_gib, 1))
''')
md('''## 4. Reconstruct E03 — seed 2026 first
This performs 24 correction updates and evaluates the original and corrected model.
Later repeat with 2027 and 2028. Resumption uses the last saved full checkpoint;
reconstruction interrupted before its final checkpoint restarts the short correction.
The old 0.22% perplexity increase and ~57% spread reduction are comparisons, not
hard-coded expected outputs. Baseline discrepancy >0.01 log-odds stops execution
as a software/input diagnostic, not a scientific acceptance threshold.
''')
code('''SEED = 2026  # Later repeat this cell with 2027, then 2028.
pilot.train(seed=SEED, stage='replicate')
''')
md('## 5. Inspect the reconstruction\nOnly seed 2026 compares the same seed as the old run. Other seeds are new replications. A mismatch must be explained; do not rename it an exact reproduction.')
code('''import json
comparison_path = OUTPUT_DIR / f'seed_{SEED}_replicate_corrected/reconstruction_comparison.json'
print(comparison_path.read_text())
''')
md('''## 6. Export measurements now
Send this ZIP back for review before continued training. It contains no large model
weights. On Kaggle, download it from the Output/Files panel. Preserve checkpoint
files separately if the session will end.
''')
code('''measurement_zip = pilot.export()
if 'google.colab' in sys.modules:
    from google.colab import files
    files.download(str(measurement_zip))
else:
    from IPython.display import FileLink, display
    display(FileLink(str(measurement_zip)))
''')
md('''## 7. After reconstruction review: freeze the continued-training corpus
No GPU training happens here. The pinned WikiText **train** split is filtered at
article level for development occupation aliases and exact test-paragraph overlap.
There must be 2,400 unique 128-token blocks; otherwise preparation stops.
No confirmation items are used for filtering. Filtered text is not guaranteed
socially neutral, and pretraining overlap is not excluded.
''')
code('''ACCEPT_DOCUMENTED_RECONSTRUCTION = False  # Change only after reviewing cells 4–6.
assert ACCEPT_DOCUMENTED_RECONSTRUCTION, 'Review the reconstruction first.'
manifest = pilot.prepare_corpus()
print({k: manifest[k] for k in ['shape', 'blocks_sha256', 'rejected_documents',
                               'selected_he_token_count', 'selected_she_token_count']})
''')
md('''## 8. Disposable runtime and memory check
One eight-microbatch update on a disposable model, then discard all changes.
Its timing excludes evaluation and checkpoint I/O and is not a completion estimate.
If storage or GPU resources are inadequate, stop and revise the documented budget.
''')
code('''assert ACCEPT_DOCUMENTED_RECONSTRUCTION
pilot.smoke_test()
''')
md('''## 9. Run ONE trajectory
Use each of seeds 2026, 2027, 2028 with `corrected` and `base`: six trajectories.
Each corrected seed must have completed cell 4 first. Each pair uses the same
shuffled text blocks. Early and late budgets are checkpoints within the same run.

300 updates, effective batch 8, sequence length 128, FP32, lr 1e-6, fresh AdamW,
dropout disabled. Evaluation at 0/1/5/10/25/50/100/200/300. Full resume state every
25 updates; an interruption can lose up to 24 updates, which are replayed on resume.
Run the same cell/configuration to resume. Completed runs are not retrained.
No automatic six-run loop: inspect cost and available storage between runs.
''')
code('''RUN_SEED = 2026
CONDITION = 'corrected'  # Then 'base'; repeat both for 2027 and 2028.
assert ACCEPT_DOCUMENTED_RECONSTRUCTION
pilot.train(seed=RUN_SEED, stage='continue', condition=CONDITION,
            accept_reconstruction=ACCEPT_DOCUMENTED_RECONSTRUCTION)
''')
md('''## 10. Summarize and export
Plots show each individual run. Paired gaps appear only when both conditions have
the same checkpoint. Missing runs are not imputed. A rising pattern-strength score
is behavioral strengthening, not proof of internal memory or a causal mechanism.
''')
code('''rows = pilot.summarize()
measurement_zip = pilot.export()
print('Share:', measurement_zip)
if 'google.colab' in sys.modules:
    from google.colab import files
    files.download(str(measurement_zip))
else:
    from IPython.display import FileLink, display
    display(FileLink(str(measurement_zip)))
''')

inputs = {path.name: path.read_bytes() for path in (HERE / 'pilot_inputs').iterdir() if path.is_file() and path.name != 'checksums.json'}
inputs['pilot_runtime.py'] = (HERE / 'pilot_runtime.py').read_bytes()
inputs['PILOT_PROTOCOL.txt'] = (HERE / 'PILOT_PROTOCOL.txt').read_bytes()
inputs['PACKAGE_NOTES.txt'] = b'''E04 implementation v0.1. GPU execution is not locally verified.
Reconstruction uses explicit AdamW defaults, eager attention and NumPy shuffling.
Corpus filtering is conservative article-level lexical filtering. Pronoun controls
are a new development instrument, not a validated benchmark. Run one seed-2026
reconstruction and review the exported results before permitting continuation.
Full-state checkpoint save is implemented; an actual GPU interruption/resume test
remains required before relying on long runs. Do not treat CPU unit tests as GPU
validation. Original E03 source notebook and weights were not available here.
'''
inputs['checksums.json'] = json.dumps({name:hashlib.sha256(data).hexdigest() for name,data in inputs.items()}, indent=2).encode()
with zipfile.ZipFile(OUT / 'pilot_inputs.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for name,data in inputs.items():
        z.writestr(name,data)
notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                                         language_info=dict(name='python'), accelerator='GPU'), nbformat=4, nbformat_minor=4)
(OUT / 'E04_Colab_Kaggle.ipynb').write_text(json.dumps(notebook, indent=1), encoding='utf-8')
print(f'Built {len(cells)} cells; {len(inputs)} development-only package files in {OUT}')
