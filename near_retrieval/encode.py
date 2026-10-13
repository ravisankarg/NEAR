"""Resumable, provenance-checked frozen encoding; no relevance labels are read."""
import argparse
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from .data import read_jsonl, sha256, write_json


def format_text(row, kind, family):
    if family == "embeddinggemma2":
        if kind == "queries":
            return "task: search result | query: " + row["text"]
        return f"title: {row.get('title') or 'none'} | text: {row['text']}"
    return ((row.get("title", "") + " ") if kind == "corpus" else "") + row["text"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--backend", choices=["litert", "sentence-transformers"], default="litert")
    p.add_argument("--family", choices=["embeddinggemma2", "plain"], default="embeddinggemma2")
    p.add_argument("--revision")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--dimension", type=int, default=768)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--offline", action="store_true")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configuration = vars(args).copy()
    configuration = {k: str(v) if isinstance(v, Path) else v for k, v in configuration.items()}
    configuration["source_hashes"] = {k: sha256(args.data / (k + ".jsonl")) for k in ["corpus", "queries"]}
    configuration["model_sha256"] = sha256(args.model) if Path(args.model).is_file() else None
    configuration["encoding_code_sha256"] = sha256(__file__)
    configuration["normalization"] = "L2, float32 stored"
    configuration["overflow"] = "ERROR" if args.backend == "litert" else "TRUNCATE"
    configuration["query_prefix"] = "task: search result | query: " if args.family == "embeddinggemma2" else ""
    manifest_path = args.out / "encoding_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["configuration"] != configuration:
        raise ValueError("Encoding configuration changed: use a new output directory")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "configuration": configuration, "python": platform.python_version(), "completed": {},
        "packages": {n: importlib.metadata.version(n) for n in ["numpy", "datasets"]},
    }
    write_json(manifest_path, manifest)
    if args.backend == "litert":
        import litert_lm
        (args.out / "runtime_cache").mkdir(parents=True, exist_ok=True)
        manifest["packages"]["litert-lm-api"] = importlib.metadata.version("litert-lm-api")
        with litert_lm.ModelInfo(args.model) as info:
            manifest["model_info"] = {"dimension": info.embedding.dimension,
                                      "signature_lengths": info.embedding.signature_selection,
                                      "context_tokens": info.max_context_tokens}
        model = litert_lm.EmbeddingEngine(
            model_path=args.model, backend=litert_lm.Backend.CPU(thread_count=args.threads),
            cache_dir=str((args.out / "runtime_cache").resolve()), max_input_length=args.max_length,
        )
        options = litert_lm.EmbeddingOptions(normalize=True, insert_special_tokens=True,
                    input_overflow_strategy=litert_lm.InputOverflowStrategy.ERROR, output_size=args.dimension)

        def encode(texts):
            # The API batches separate requests; it does not concatenate their text.
            return np.asarray([r.embedding for r in model.compute_embedding_batch(texts, options)], dtype=np.float32)
    else:
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(args.threads)
        kwargs = {"vision_config": None, "audio_config": None} if args.family == "embeddinggemma2" else {}
        model = SentenceTransformer(args.model, revision=args.revision, device=args.device,
                  local_files_only=args.offline, config_kwargs=kwargs,
                  model_kwargs={"torch_dtype": torch.float32})
        model.max_seq_length = args.max_length
        manifest["packages"].update({n: importlib.metadata.version(n) for n in ["torch", "transformers", "sentence-transformers"]})

        def encode(texts):
            return model.encode(texts, prompt="", batch_size=args.batch_size, normalize_embeddings=True,
                                truncate_dim=args.dimension, convert_to_numpy=True, show_progress_bar=False)
    write_json(manifest_path, manifest)
    for kind in ["corpus", "queries"]:
        rows = read_jsonl(args.data / (kind + ".jsonl"))
        ids = [str(r["_id"]) for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate IDs")
        write_json(args.out / (kind + "_ids.json"), ids)
        path = args.out / (kind + ".npy")
        progress_path = args.out / (kind + "_progress.json")
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"rows": 0, "seconds": 0.0}
        if progress["rows"] and not path.exists():
            raise ValueError("Progress exists without embeddings")
        arr = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+", dtype=np.float32,
                                        shape=(len(rows), args.dimension))
        if arr.shape != (len(rows), args.dimension):
            raise ValueError("Embedding shape mismatch")
        for start in range(progress["rows"], len(rows), args.batch_size):
            end = min(len(rows), start + args.batch_size)
            t = time.perf_counter()
            vectors = encode([format_text(r, kind, args.family) for r in rows[start:end]])
            norms = np.linalg.norm(vectors, axis=1)
            if vectors.shape != (end-start, args.dimension) or not np.isfinite(vectors).all() or np.any(norms < 1e-8):
                raise ValueError("Invalid encoder output")
            arr[start:end] = vectors / norms[:, None]
            arr.flush()
            progress = {"rows": end, "seconds": progress["seconds"] + time.perf_counter() - t}
            write_json(progress_path, progress)
            print(json.dumps({"kind": kind, **progress, "total": len(rows)}), flush=True)
        manifest["completed"][kind] = {**progress, "sha256": sha256(path), "shape": list(arr.shape)}
        write_json(manifest_path, manifest)
    if hasattr(model, "close"):
        model.close()


if __name__ == "__main__":
    main()
