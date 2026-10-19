"""E04 development pilot. Run on a GPU through the accompanying notebook."""
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import shutil
import time
from pathlib import Path

import numpy as np

MODEL = 'EleutherAI/pythia-410m'
REVISION = '9879c9b5f8bea9051dcb0e68dff21493d67e9d4f'
DATA_REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'
STEPS = [0, 1, 5, 10, 25, 50, 100, 200, 300]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def table(path, rows):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def csvread(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def ranks(x):
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    end = np.cumsum(counts)
    return ((2 * end - counts - 1) / 2)[inv]


def corr(x, y):
    x, y = np.asarray(x), np.asarray(y)
    x, y = x - x.mean(), y - y.mean()
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / d) if d > 1e-12 else None


def components(current, reference):
    """Matrices: occupation x template. Fixed ordering required by caller."""
    current, reference = np.asarray(current), np.asarray(reference)
    if current.shape != reference.shape or current.ndim != 2:
        raise ValueError('Reference/current matrix mismatch')
    p = current - current.mean(axis=0)
    r = reference - reference.mean(axis=0)
    den = float(np.sum(r * r))
    if den < 1e-12:
        raise ValueError('Reference has no measurable centered pattern')
    a = float(np.sum(p * r) / den)
    norm = float(np.linalg.norm(p))
    cosine = float(np.sum(p * r) / (norm * np.linalg.norm(r))) if norm > 1e-12 else None
    result = dict(mean_score=float(current.mean()), pattern_strength=a,
                  residual_rms=float(np.sqrt(np.mean((p - a * r)**2))),
                  profile_rms=float(np.sqrt(np.mean(p**2))), cosine=cosine,
                  mean_template_sd=float(current.std(axis=0).mean()),
                  occupation_rank_corr=corr(ranks(p.mean(axis=1)), ranks(r.mean(axis=1))))
    if current.shape[1] == 8:
        for label, left, right in [('old_to_new', slice(0, 4), slice(4, 8)),
                                   ('new_to_old', slice(4, 8), slice(0, 4))]:
            ref = r[:, left].mean(axis=1)
            cur = p[:, right].mean(axis=1)
            d = float(ref @ ref)
            result['cross_template_' + label] = float(cur @ ref / d) if d > 1e-12 else None
    return result


def score_matrix(rows):
    rows = [r for r in rows if r['kind'] == 'occupation']
    nouns = sorted({r['noun'] for r in rows})
    ts = sorted({r['template_id'] for r in rows})
    lookup = {(r['noun'], r['template_id']): float(r['score']) for r in rows}
    if len(lookup) != len(rows) or len(rows) != len(nouns) * len(ts):
        raise ValueError('Incomplete or duplicated evaluation matrix')
    return np.array([[lookup[n, t] for t in ts] for n in nouns]), (nouns, ts)


class Pilot:
    def __init__(self, bundle, output):
        import torch
        from transformers import AutoTokenizer
        self.torch = torch
        self.bundle, self.output = Path(bundle), Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        for name, expected in read(self.bundle / 'checksums.json').items():
            if sha(self.bundle / name) != expected:
                raise ValueError('Input checksum mismatch: ' + name)
        if not torch.cuda.is_available():
            raise RuntimeError('Select a GPU runtime first. CPU training is intentionally disabled.')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.items = read(self.bundle / 'evaluation_items.json')
        self.prefixes = read(self.bundle / 'training_prefixes.json')
        self.capability = read(self.bundle / 'capability_manifest.json')
        self.controls = read(self.bundle / 'explicit_pronoun_controls.json')
        self.he = self.tokenizer.encode(' he', add_special_tokens=False)
        self.she = self.tokenizer.encode(' she', add_special_tokens=False)
        if len(self.he) != 1 or len(self.she) != 1:
            raise ValueError('Pronoun tokens are not single tokens')
        self.he, self.she = self.he[0], self.she[0]
        for item in self.items:
            ids = self.tokenizer.encode(item['prefix'], add_special_tokens=False)
            if ids != item['prefix_token_ids']:
                raise ValueError('Saved prefix token IDs changed')
        for prefix in self.prefixes + [i['prefix'] for i in self.items] + [i['prefix'] for i in self.controls]:
            ids = self.tokenizer.encode(prefix, add_special_tokens=False)
            for suffix, token in [(' he', self.he), (' she', self.she)]:
                if self.tokenizer.encode(prefix + suffix, add_special_tokens=False) != ids + [token]:
                    raise ValueError('Token-boundary mismatch: ' + prefix)
        self.reference, self.order = score_matrix(csvread(self.bundle / 'before_associations.csv'))
        environment = dict(python=platform.python_version(), gpu=torch.cuda.get_device_name(),
                           model=MODEL, revision=REVISION,
                           packages={k: importlib.metadata.version(k) for k in
                                     ['torch', 'transformers', 'datasets', 'numpy']})
        dump(self.output / 'environment.json', environment)
        print(json.dumps(environment, indent=2))

    def seed(self, value):
        random.seed(value)
        np.random.seed(value)
        self.torch.manual_seed(value)
        self.torch.cuda.manual_seed_all(value)

    def model(self):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, torch_dtype=self.torch.float32,
            attn_implementation='eager').to('cuda')
        model.config.use_cache = False
        return model.eval()

    def batch(self, prefixes):
        return self.tokenizer(prefixes, padding=True, return_tensors='pt',
                              add_special_tokens=False).to('cuda')

    def evaluate(self, model, folder, step):
        torch = self.torch
        folder.mkdir(parents=True, exist_ok=True)
        rows = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(self.items), 8):
                items = self.items[start:start + 8]
                lp = model(**self.batch([i['prefix'] for i in items])).logits[:, -1].float().log_softmax(-1)
                for i, item in enumerate(items):
                    h, s = float(lp[i, self.he]), float(lp[i, self.she])
                    rows.append({k: item[k] for k in ['item_id', 'noun', 'kind', 'template_id']} |
                                dict(score=h-s, logp_he=h, logp_she=s,
                                     pair_probability_mass=float(np.exp(h) + np.exp(s))))
            total_nll, total_tokens, docs = 0.0, 0, []
            entropy_sum, entropy_n, ordinary_margins = 0.0, 0, []
            diagnostic_pairs = []
            for u, v in [(' good', ' bad'), (' large', ' small'), (' old', ' new')]:
                iu, iv = self.tokenizer.encode(u, add_special_tokens=False), self.tokenizer.encode(v, add_special_tokens=False)
                if len(iu) == len(iv) == 1:
                    diagnostic_pairs.append((u, v, iu[0], iv[0]))
            for j, item in enumerate(self.capability['items']):
                x = torch.tensor([item['token_ids']], device='cuda')
                output = model(x, labels=x)
                n = x.shape[1] - 1
                nll = float(output.loss) * n
                total_nll += nll
                total_tokens += n
                docs.append(dict(row_id=item['row_id'], nll_sum=nll, predicted_tokens=n))
                if j < 8:
                    logits = output.logits[0, :-1].float()
                    lp = logits.log_softmax(-1)
                    entropy_sum += float(-(lp.exp() * lp).sum())
                    entropy_n += n
                    for u, v, iu, iv in diagnostic_pairs:
                        ordinary_margins.append(dict(row_id=item['row_id'], pair=u+'/'+v,
                                                    mean_abs_margin=float((logits[:, iu]-logits[:, iv]).abs().mean())))
                    del logits, lp
                del output
            control_rows = []
            for item in self.controls:
                lp = model(**self.batch([item['prefix']])).logits[0, -1].float().log_softmax(-1)
                margin = float(lp[self.he] - lp[self.she]) * (1 if item['expected'] == 'he' else -1)
                control_rows.append(dict(**item, expected_margin=margin, correct=int(margin > 0)))
        matrix, order = score_matrix(rows)
        assert order == self.order
        metrics = components(matrix, self.reference)
        metrics.update(step=step, language_nll=total_nll/total_tokens,
                       perplexity=float(np.exp(total_nll/total_tokens)), predicted_tokens=total_tokens,
                       control_accuracy=float(np.mean([r['correct'] for r in control_rows])),
                       control_margin=float(np.mean([r['expected_margin'] for r in control_rows])),
                       ordinary_entropy=entropy_sum/entropy_n,
                       neutral_mean=float(np.mean([r['score'] for r in rows if r['kind'] != 'occupation'])),
                       median_pair_mass=float(np.median([r['pair_probability_mass'] for r in rows])))
        table(folder / f'step_{step:04d}_associations.csv', rows)
        table(folder / f'step_{step:04d}_language.csv', docs)
        table(folder / f'step_{step:04d}_controls.csv', control_rows)
        if ordinary_margins:
            table(folder / f'step_{step:04d}_ordinary_margins.csv', ordinary_margins)
        dump(folder / f'step_{step:04d}_metrics.json', metrics)
        return metrics

    def save_state(self, path, model, optimizer, step, config, logs):
        torch = self.torch
        # Atomic replacement requires room for both old and new checkpoints.
        if shutil.disk_usage(path.parent).free < 7 * 1024**3:
            raise RuntimeError('Need at least 7 GiB free for an atomic full-state checkpoint.')
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                     config=config, logs=logs, rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all())
        tmp = path.with_name(path.name + '.tmp')
        torch.save(state, tmp)
        os.replace(tmp, path)

    def train(self, seed, stage='replicate', condition='corrected', accept_reconstruction=False):
        torch = self.torch
        if seed not in [2026, 2027, 2028] or stage not in ['replicate', 'continue']:
            raise ValueError('Use a planned seed and stage')
        if condition not in ['corrected', 'base']:
            raise ValueError('Unknown condition')
        if stage == 'replicate':
            condition = 'corrected'
        elif not accept_reconstruction:
            raise RuntimeError('Review reconstruction_comparison.json first; explicitly accept the documented reconstruction to continue.')
        if stage == 'continue' and not (self.output / 'seed_2026_replicate_corrected' / 'reconstruction_comparison.json').exists():
            raise RuntimeError('Reconstruct and compare seed 2026 first.')
        folder = self.output / f'seed_{seed}_{stage}_{condition}'
        folder.mkdir(parents=True, exist_ok=True)
        status_path = folder / 'status.json'
        if status_path.exists() and read(status_path).get('status') == 'complete':
            print('Already complete:', folder)
            return
        self.seed(seed)
        if stage == 'continue':
            manifest = read(self.output / 'corpus_manifest.json')
            if sha(self.output / 'train_blocks.npy') != manifest['blocks_sha256']:
                raise ValueError('Training corpus changed')
            blocks = np.load(self.output / 'train_blocks.npy')
            if blocks.shape != (2400, 128):
                raise ValueError('Expected 2400 unique 128-token blocks')
            batches = np.random.default_rng(seed + 1000).permutation(len(blocks)).reshape(300, 8)
            limit = 300
        else:
            generator = np.random.default_rng(seed)
            batches = np.concatenate([generator.permutation(96).reshape(12, 8) for _ in range(2)])
            limit = 24
        correction_hash = None
        if stage == 'continue' and condition == 'corrected':
            source = self.output / f'seed_{seed}_replicate_corrected'
            if not (source / 'status.json').exists() or read(source / 'status.json')['status'] != 'complete':
                raise RuntimeError('Complete correction replication for this seed first.')
            correction_hash = sha(source / 'final_weights.pt')
            if correction_hash != read(source / 'status.json')['final_weights_sha256']:
                raise ValueError('Correction checkpoint checksum mismatch')
        config = dict(seed=seed, stage=stage, condition=condition, lr=1e-6,
                      optimizer='AdamW', betas=[0.9, 0.999], eps=1e-8, weight_decay=0,
                      clip_norm=1, precision='fp32', mode='eval_with_gradients',
                      batch=8, shuffle='numpy.default_rng', model_revision=REVISION,
                      input_checksums=read(self.bundle / 'checksums.json'),
                      code_sha256=sha(Path(__file__)), limit=limit,
                      correction_checkpoint_sha256=correction_hash,
                      corpus_sha256=manifest['blocks_sha256'] if stage == 'continue' else None)
        dump(folder / 'run_config.json', config)
        model = self.model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0,
                                     betas=(0.9, 0.999), eps=1e-8)
        checkpoint = folder / 'resume.pt'
        start, logs = 0, []
        q = None
        try:
            if stage == 'replicate':
                with torch.no_grad():
                    q = torch.cat([model(**self.batch(self.prefixes[j:j+8])).logits[:, -1].float().softmax(-1).cpu()
                                   for j in range(0, 96, 8)])
                    mass = (q[:, self.he] + q[:, self.she]) / 2
                    q[:, self.he], q[:, self.she] = mass, mass
            if checkpoint.exists():
                # Only load checkpoints created by this notebook in your own output folder.
                state = torch.load(checkpoint, map_location='cpu', weights_only=False)
                if state['config'] != config:
                    raise ValueError('Resume configuration/code/input mismatch. Use a new output directory.')
                model.load_state_dict(state['model'])
                optimizer.load_state_dict(state['optimizer'])
                start, logs = state['step'], state['logs']
                torch.set_rng_state(state['rng'])
                torch.cuda.set_rng_state_all(state['cuda_rng'])
                del state
                print('Resuming at update', start)
            elif stage == 'continue' and condition == 'corrected':
                source = self.output / f'seed_{seed}_replicate_corrected'
                if not (source / 'status.json').exists() or read(source / 'status.json')['status'] != 'complete':
                    raise RuntimeError('Complete correction replication for this seed first.')
                model.load_state_dict(torch.load(source / 'final_weights.pt', map_location='cpu', weights_only=True))
            if start == 0:
                baseline = self.evaluate(model, folder, 0)
                if stage == 'replicate':
                    rows = csvread(folder / 'step_0000_associations.csv')
                    saved = {r['item_id']: float(r['score']) for r in csvread(self.bundle / 'before_associations.csv')}
                    maxdiff = max(abs(float(r['score']) - saved[r['item_id']]) for r in rows)
                    dump(folder / 'base_reproduction.json', dict(max_abs_score_difference=maxdiff))
                    if maxdiff > 0.01:
                        raise RuntimeError('Base scores differ by >0.01 log-odds; investigate software/model mismatch before training.')
            tick = time.monotonic()
            for step in range(start, limit):
                optimizer.zero_grad(set_to_none=True)
                indices = batches[step]
                loss_value = 0.0
                if stage == 'replicate':
                    lp = model(**self.batch([self.prefixes[int(i)] for i in indices])).logits[:, -1].float().log_softmax(-1)
                    loss = torch.nn.functional.kl_div(lp, q[indices].to('cuda'), reduction='batchmean')
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite correction loss')
                    loss.backward()
                    loss_value = float(loss.detach())
                    del loss, lp
                else:
                    for i in indices:
                        x = torch.tensor(blocks[int(i)][None, :], device='cuda', dtype=torch.long)
                        loss = model(x, labels=x).loss / 8
                        if not torch.isfinite(loss):
                            raise RuntimeError('Nonfinite continued-training loss')
                        loss.backward()
                        loss_value += float(loss.detach())
                        del loss
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                completed = step + 1
                logs.append(dict(step=completed, loss=loss_value, grad_norm=float(grad),
                                 predicted_training_tokens=completed*8*127 if stage == 'continue' else None))
                if stage == 'continue' and completed in STEPS:
                    self.evaluate(model, folder, completed)
                if completed % 5 == 0 or completed == limit:
                    print(f'{stage}/{condition} seed={seed} step={completed}/{limit} loss={loss_value:.5f}', flush=True)
                    table(folder / 'training_log.csv', logs)
                if completed % 25 == 0 or completed == limit:
                    self.save_state(checkpoint, model, optimizer, completed, config, logs)
                    dump(status_path, dict(status='running', completed_steps=completed))
            final = self.evaluate(model, folder, limit)
            if stage == 'replicate':
                saved = {r['item_id']: float(r['score']) for r in csvread(self.bundle / 'after_associations.csv')}
                actual = csvread(folder / f'step_{limit:04d}_associations.csv')
                errors = [float(r['score'])-saved[r['item_id']] for r in actual]
                dump(folder / 'reconstruction_comparison.json', dict(
                    label='Reconstruction, not verified execution of the original notebook',
                    max_abs_score_difference=float(np.max(np.abs(errors))),
                    score_rmse=float(np.sqrt(np.mean(np.square(errors)))),
                    expected_seed_2026_perplexity=29.757824, actual_metrics=final,
                    unknown_original_details=['shuffle implementation/order', 'unrecorded optimizer defaults',
                                              'attention backend', 'exact executed source'],
                    note='Only seed 2026 is a reproduction comparison. Other seeds are replications.'))
            weights_temp = folder / 'final_weights.pt.tmp'
            torch.save(model.state_dict(), weights_temp)
            os.replace(weights_temp, folder / 'final_weights.pt')
            dump(status_path, dict(status='complete', completed_steps=limit,
                                   session_elapsed_seconds=time.monotonic()-tick,
                                   final_weights_sha256=sha(folder / 'final_weights.pt')))
            # The verified final weights replace this run's completed optimizer checkpoint.
            checkpoint.unlink(missing_ok=True)
            print('Completed. Export measurements and preserve final_weights.pt:', folder)
        except Exception as exc:
            dump(folder / 'last_error.json', dict(error=str(exc), resume_available=checkpoint.exists()))
            raise
        finally:
            del optimizer, model, q
            gc.collect()
            torch.cuda.empty_cache()

    def prepare_corpus(self):
        from datasets import load_dataset
        path = self.output / 'train_blocks.npy'
        if path.exists():
            manifest = read(self.output / 'corpus_manifest.json')
            if sha(path) != manifest['blocks_sha256']:
                raise ValueError('Existing corpus corrupted')
            print('Reusing frozen corpus:', manifest['blocks_sha256'])
            return manifest
        ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', revision=DATA_REVISION)
        aliases = read(self.bundle / 'development_occupation_aliases.json')
        pattern = re.compile(r'\b(?:' + '|'.join(re.escape(s) for s in aliases) + r')\b', re.I)
        normalize = lambda s: ' '.join(s.lower().split())
        test_paragraphs = {normalize(r['text']) for r in ds['test'] if len(normalize(r['text'])) > 30}
        documents, current = [], []
        for row_id, row in enumerate(ds['train']):
            text = row['text'].strip()
            if re.match(r'^= [^=].*[^=] =$' , text) and current:
                documents.append(current)
                current = []
            if text:
                current.append((row_id, text))
        if current:
            documents.append(current)
        selected, doc_records, seen = [], [], set()
        rejected = dict(occupation=0, overlap=0, duplicate=0)
        raw_text = '\n'.join(r['text'] for r in ds['train'])
        for doc in documents:
            text = '\n'.join(x[1] for x in doc)
            if pattern.search(text):
                rejected['occupation'] += 1
                continue
            if any(normalize(x[1]) in test_paragraphs for x in doc):
                rejected['overlap'] += 1
                continue
            digest = hashlib.sha256(normalize(text).encode()).hexdigest()
            if digest in seen:
                rejected['duplicate'] += 1
                continue
            seen.add(digest)
            tokens = self.tokenizer.encode(text, add_special_tokens=False) + [self.tokenizer.eos_token_id]
            record = dict(first_row=doc[0][0], last_row=doc[-1][0], text_sha256=digest,
                          tokens=len(tokens), he_count=len(re.findall(r'\bhe\b', text, re.I)),
                          she_count=len(re.findall(r'\bshe\b', text, re.I)))
            doc_index = len(doc_records)
            doc_records.append(record)
            selected.extend((doc_index, tokens[j:j+128]) for j in range(0, len(tokens)-127, 128))
        # Deduplicate blocks, retaining the first associated source document.
        unique = {}
        for doc, ids in selected:
            unique.setdefault(tuple(ids), doc)
        entries = list(unique.items())
        if len(entries) < 2400:
            raise RuntimeError(f'Only {len(entries)} unique blocks after filtering; revise the protocol, do not repeat data silently.')
        indices = np.random.default_rng(20261007).choice(len(entries), size=2400, replace=False)
        blocks = np.array([entries[i][0] for i in indices], dtype=np.int64)
        np.save(path, blocks)
        manifest = dict(dataset='Salesforce/wikitext', revision=DATA_REVISION,
                        split='train', configuration='wikitext-2-raw-v1', sampling_seed=20261007,
                        shape=list(blocks.shape), blocks_sha256=sha(path), rejected_documents=rejected,
                        source_documents=doc_records, block_document_indices=[entries[i][1] for i in indices],
                        raw_he_count=len(re.findall(r'\bhe\b', raw_text, re.I)),
                        raw_she_count=len(re.findall(r'\bshe\b', raw_text, re.I)),
                        selected_he_token_count=int((blocks == self.he).sum()),
                        selected_she_token_count=int((blocks == self.she).sum()),
                        limitations=['No confirmation occupations inspected.', 'Lexical filtering is not semantic neutrality.',
                                     'Normalized exact paragraph overlap checked, not all possible near-duplicates.',
                                     'Pretraining overlap is not excluded.'])
        dump(self.output / 'corpus_manifest.json', manifest)
        print('Prepared 2400 unique blocks; no GPU training performed.')
        return manifest

    def smoke_test(self):
        """Disposable eight-microbatch update, then discard all model changes."""
        torch = self.torch
        blocks = np.load(self.output / 'train_blocks.npy')
        self.seed(20261007)
        model = self.model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        tick = time.monotonic()
        try:
            optimizer.zero_grad(set_to_none=True)
            for ids in blocks[:8]:
                x = torch.tensor(ids[None, :], dtype=torch.long, device='cuda')
                loss = model(x, labels=x).loss / 8
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite smoke-test loss')
                loss.backward()
                del loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            result = dict(update_seconds=time.monotonic()-tick,
                          peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                          disk_free_gib=shutil.disk_usage(self.output).free/1024**3,
                          interpretation='One cold update; excludes evaluation/checkpoint I/O; not a runtime guarantee.')
            dump(self.output / 'smoke_test.json', result)
            print(json.dumps(result, indent=2))
            return result
        finally:
            del optimizer, model
            gc.collect()
            torch.cuda.empty_cache()

    def summarize(self):
        rows = []
        for path in sorted(self.output.glob('seed_*/*_metrics.json')):
            seed, stage, condition = path.parent.name.split('_')[1:]
            rows.append(dict(seed=int(seed), stage=stage, condition=condition, **read(path)))
        if not rows:
            raise RuntimeError('No measurements yet')
        table(self.output / 'all_metrics.csv', rows)
        lookup = {(r['seed'], r['condition'], r['step']): r for r in rows if r['stage'] == 'continue'}
        gaps = []
        for (seed, condition, step), r in lookup.items():
            if condition != 'corrected' or (seed, 'base', step) not in lookup:
                continue
            base = lookup[seed, 'base', step]
            row = dict(seed=seed, step=step)
            for key in ['mean_score', 'pattern_strength', 'residual_rms', 'perplexity']:
                row[key+'_gap'] = r[key]-base[key]
                if (seed, 'corrected', 0) in lookup and (seed, 'base', 0) in lookup:
                    row[key+'_gap_change'] = row[key+'_gap'] - (lookup[seed,'corrected',0][key]-lookup[seed,'base',0][key])
            gaps.append(row)
        if gaps:
            table(self.output / 'paired_gaps.csv', gaps)
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, key in zip(axes, ['mean_score', 'pattern_strength', 'perplexity']):
            for seed in [2026, 2027, 2028]:
                for condition in ['corrected', 'base']:
                    subset = sorted([r for r in rows if r['stage']=='continue' and r['seed']==seed and r['condition']==condition], key=lambda r:r['step'])
                    if subset:
                        ax.plot([r['step'] for r in subset], [r[key] for r in subset],
                                linestyle='-' if condition=='corrected' else '--', label=f'{condition} {seed}')
            ax.set_title(key.replace('_', ' '))
            ax.set_xlabel('Optimizer updates (1016 predicted tokens/update)')
        if axes[0].lines:
            axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(self.output / 'development_trajectories.png', dpi=160)
        plt.show()
        return rows

    def export(self):
        import zipfile
        destination = self.output / 'pilot_measurements.zip'
        with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as z:
            for path in self.output.rglob('*'):
                if path.is_file() and path.suffix in ['.json', '.csv', '.png', '.txt']:
                    z.write(path, path.relative_to(self.output))
        print('Measurements only; weights and resume states are NOT included:', destination)
        return destination
