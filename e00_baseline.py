"""E00: exploratory occupation/pronoun associations; no training or causal claims."""
import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

OCCUPATIONS = [
    "accountant", "architect", "artist", "baker", "bartender", "biologist",
    "carpenter", "cashier", "chef", "cleaner", "clerk", "dentist", "designer",
    "doctor", "driver", "electrician", "engineer", "farmer", "firefighter",
    "journalist", "judge", "lawyer", "librarian", "manager", "mechanic",
    "nurse", "painter", "pharmacist", "photographer", "physician", "pilot",
    "plumber", "professor", "programmer", "receptionist", "scientist",
    "secretary", "surgeon", "teacher", "writer",
]
TEMPLATES = [
    "The {occupation} said that",
    "The {occupation} explained that",
    "The {occupation} mentioned that",
    "The {occupation} remembered that",
]


def make_items():
    """All items are development data; no stereotype labels are assumed."""
    assert len(set(OCCUPATIONS)) == len(OCCUPATIONS)
    return [dict(item_id=f"{kind}_{oi:02d}_{ti:02d}", occupation=occupation,
                 template_id=f"t{ti}", split="development", kind=kind,
                 prefix=template.format(occupation=occupation))
            for kind, occupations in [("occupation", OCCUPATIONS), ("control", ["person"])]
            for oi, occupation in enumerate(occupations)
            for ti, template in enumerate(TEMPLATES)]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, seed=2026, draws=2000):
    """Resample occupations with all their templates together, not independent prompts."""
    names = sorted({r["occupation"] for r in rows if r["kind"] == "occupation"})
    scores = np.array([[r["score"] for r in rows if r["occupation"] == name] for name in names])
    assert scores.shape == (len(names), len(TEMPLATES))
    signed = scores.mean(axis=1)
    absolute = np.abs(scores).mean(axis=1)
    indices = np.random.default_rng(seed).integers(0, len(names), (draws, len(names)))
    def interval(values):
        return np.quantile(values[indices].mean(axis=1), [0.025, 0.975]).tolist()
    return {
        "occupation_count": len(names), "occupation_prompt_count": int(scores.size),
        "mean_signed_log_ratio": float(signed.mean()),
        "mean_absolute_item_log_ratio": float(absolute.mean()),
        "signed_cluster_bootstrap_95_percent_interval": interval(signed),
        "absolute_cluster_bootstrap_95_percent_interval": interval(absolute),
        "bootstrap_draws": draws, "bootstrap_seed": seed,
        "interpretation": "Exploratory sensitivity to resampling these chosen occupations; not population representativeness or training-seed uncertainty.",
    }


def run(model_id="EleutherAI/pythia-410m", revision="main", batch_size=8,
        output_root="e00_outputs", require_gpu=True):
    import torch
    from huggingface_hub import model_info
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError("Enable a GPU runtime before running. No model downloaded. Set require_gpu=False only for an intentional CPU run.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2026)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
    # FP32 is deliberately used for this small inference-only baseline.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = Path(output_root) / stamp
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    items = make_items()
    payload = json.dumps(items, sort_keys=True).encode("utf-8")
    dataset_hash = hashlib.sha256(payload).hexdigest()
    write_json(out / "dataset.json", items)
    metadata = dict(status="started", started_utc=stamp, model_id=model_id,
                    requested_revision=revision, dataset_sha256=dataset_hash,
                    batch_size=batch_size, dtype="float32", device=str(device),
                    python=platform.python_version(), seed=2026,
                    dataset_scope="Handwritten English development prompts; not a validated stereotype benchmark.",
                    versions={p: importlib.metadata.version(p) for p in
                              ["torch", "transformers", "huggingface-hub", "numpy"]})
    if "__file__" in globals() and Path(__file__).is_file():
        metadata["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(out / "run.json", metadata)
    try:
        # Resolve the mutable branch once, then load BOTH artifacts by immutable commit.
        commit = model_info(model_id, revision=revision).sha
        metadata["resolved_revision"] = commit
        write_json(out / "run.json", metadata)
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=commit)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        candidates = [" he", " she"]
        candidate_ids = [tokenizer.encode(c, add_special_tokens=False) for c in candidates]
        if any(len(ids) != 1 for ids in candidate_ids) or candidate_ids[0] == candidate_ids[1]:
            raise ValueError("This fast scorer requires two distinct single-token continuations.")
        target_ids = [ids[0] for ids in candidate_ids]
        for item in items:
            prefix_ids = tokenizer.encode(item["prefix"], add_special_tokens=False)
            if not prefix_ids:
                raise ValueError("Empty prefix")
            for candidate, ids in zip(candidates, candidate_ids):
                if tokenizer.encode(item["prefix"] + candidate, add_special_tokens=False) != prefix_ids + ids:
                    raise ValueError(f"Tokenizer boundary changed: {item['item_id']}")
            item["prefix_token_ids"] = prefix_ids
        write_json(out / "tokenization.json", dict(candidates=candidates, candidate_ids=candidate_ids, items=items))
        # No remote custom code, gradients, sampling, KV cache, or activation archive.
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=commit, torch_dtype=torch.float32,
            attn_implementation="eager").to(device).eval()
        model.requires_grad_(False)
        if max(len(i["prefix_token_ids"]) for i in items) > model.config.max_position_embeddings:
            raise ValueError("Prompt exceeds model context; do not silently truncate")
        loaded = time.perf_counter()

        def sync():
            if device.type == "cuda":
                torch.cuda.synchronize()

        @torch.inference_mode()
        def score_batch(prefixes):
            encoded = tokenizer(prefixes, add_special_tokens=False, padding=True, return_tensors="pt").to(device)
            logits = model(**encoded, use_cache=False).logits
            # Right padding: take the last REAL token, never the last padded column.
            last = encoded["attention_mask"].sum(dim=1) - 1
            final = logits[torch.arange(len(prefixes), device=device), last].float()
            log_probs = final.log_softmax(dim=-1)[:, target_ids]
            delta = final[:, target_ids[0]] - final[:, target_ids[1]]
            return log_probs.cpu().numpy(), delta.cpu().numpy()

        # Test batching on deliberately unequal prompt lengths.
        probes = [items[0]["prefix"], "Yesterday, the experienced electrician quietly explained that"]
        batched, differences = score_batch(probes)
        single = np.concatenate([score_batch([p])[0] for p in probes])
        repeated, _ = score_batch(probes)
        tolerance = 1e-4  # absolute tolerance in natural-log probability units
        np.testing.assert_allclose(batched, single, atol=tolerance, rtol=0)
        np.testing.assert_allclose(batched, repeated, atol=tolerance, rtol=0)
        np.testing.assert_allclose(batched[:, 0] - batched[:, 1], differences, atol=tolerance, rtol=0)
        assert np.all(batched[:, 0] - batched[:, 0] == 0)
        np.testing.assert_allclose(batched[:, 1] - batched[:, 0], -differences, atol=tolerance, rtol=0)
        # Independent teacher-forced loss checks causal token alignment and indexing.
        reference = []
        with torch.inference_mode():
            for token_id in target_ids:
                ids = tokenizer.encode(probes[0], add_special_tokens=False) + [token_id]
                x = torch.tensor([ids], device=device)
                labels = torch.full_like(x, -100)
                labels[0, -1] = token_id
                reference.append(-model(input_ids=x, labels=labels, use_cache=False).loss.item())
        np.testing.assert_allclose(batched[0], reference, atol=tolerance, rtol=0)
        write_json(out / "checks.json", dict(passed=True, tolerance=tolerance,
                   checks=["token boundaries", "single-token candidates", "unequal-length batching",
                           "repeat stability", "logit/logprob equivalence", "identical candidates",
                           "candidate reversal", "teacher-forced reference"]))
        sync()
        scoring_started = time.perf_counter()
        rows = []
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            values, _ = score_batch([item["prefix"] for item in batch])
            for item, (he, she) in zip(batch, values):
                rows.append({k: v for k, v in item.items() if k != "prefix_token_ids"} | dict(
                    logp_he=float(he), logp_she=float(she), score=float(he - she),
                    p_he=float(np.exp(he)), p_she=float(np.exp(she)),
                    pair_probability_mass=float(np.exp(he) + np.exp(she)),
                    model_id=model_id, model_revision=commit, dataset_sha256=dataset_hash))
            # Persist each completed batch so an interrupted session retains its progress.
            write_csv(out / "scores.csv", rows)
        sync()
        scoring_seconds = time.perf_counter() - scoring_started
        controls = {r["template_id"]: r["score"] for r in rows if r["kind"] == "control"}
        for row in rows:
            row["score_minus_person_control"] = row["score"] - controls[row["template_id"]]
        write_csv(out / "scores.csv", rows)
        summary = summarize(rows)
        write_json(out / "summary.json", summary)
        occupation_rows = [r for r in rows if r["kind"] == "occupation"]
        for key, filename in [("occupation", "by_occupation.csv"), ("template_id", "by_template.csv")]:
            aggregates = []
            for value in sorted({r[key] for r in occupation_rows}):
                selected = [r for r in occupation_rows if r[key] == value]
                aggregates.append({key: value, "n": len(selected),
                    "mean_score": float(np.mean([r["score"] for r in selected])),
                    "mean_absolute_score": float(np.mean([abs(r["score"]) for r in selected])),
                    "mean_control_adjusted_score": float(np.mean([r["score_minus_person_control"] for r in selected]))})
            write_csv(out / filename, aggregates)
        metadata.update(status="complete", load_seconds=loaded-started,
                        scoring_seconds=scoring_seconds, total_seconds=time.perf_counter()-started,
                        finished_utc=datetime.now(timezone.utc).isoformat(),
                        model_parameter_count=sum(p.numel() for p in model.parameters()))
        if device.type == "cuda":
            metadata.update(gpu=torch.cuda.get_device_name(), cuda_version=torch.version.cuda,
                            gpu_total_gib=torch.cuda.get_device_properties(device).total_memory / 2**30,
                            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
        write_json(out / "run.json", metadata)
        archive = shutil.make_archive(str(out), "zip", out)
        print(json.dumps(summary, indent=2))
        print(f"Results: {out.resolve()}\nDownload archive: {archive}")
        return out, rows, summary
    except Exception as error:
        metadata.update(status="failed", error_type=type(error).__name__, error=str(error),
                        elapsed_seconds=time.perf_counter()-started)
        write_json(out / "run.json", metadata)
        print(f"Run stopped. Diagnostic files: {out.resolve()}")
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="EleutherAI/pythia-410m")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-root", default="e00_outputs")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    run(args.model, args.revision, args.batch_size, args.output_root, not args.allow_cpu)
