"""Development-only selection followed by one frozen held-out comparison."""
import argparse
import csv
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .data import read_jsonl, read_qrels, sha256, write_json
from .metrics import evaluate, paired_comparison, rank
from .near import NearConfig, NearIndex, unit


def select_queries(query_ids, qrels, embeddings):
    lookup = {q: i for i, q in enumerate(query_ids)}
    ids = sorted(qrels)
    if not set(ids).issubset(lookup):
        raise ValueError("Missing query embeddings")
    return ids, np.asarray(embeddings[[lookup[q] for q in ids]])


def search_exact(queries, keys):
    return rank(queries @ keys.T, 100)


def latency(fn, count=100, warmups=5):
    for i in range(warmups):
        fn(i)
    times = []
    for i in range(count):
        start = time.perf_counter()
        fn(i)
        times.append((time.perf_counter() - start) * 1000)
    return {"median_ms": float(np.median(times)), "p95_ms": float(np.quantile(times, .95)),
            "queries": count, "includes_encoder": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--embeddings", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/scifact.json"))
    p.add_argument("--hnsw", action="store_true", help="Additionally audit paper HNSW settings with FAISS")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text())
    encoder = json.loads((args.embeddings / "encoding_manifest.json").read_text())
    for name in ["corpus", "queries"]:
        if name not in encoder["completed"]:
            raise ValueError("Encoding incomplete: " + name)
        if sha256(args.embeddings / (name + ".npy")) != encoder["completed"][name]["sha256"]:
            raise ValueError("Embedding hash mismatch")
        if sha256(args.data / (name + ".jsonl")) != encoder["configuration"]["source_hashes"][name]:
            raise ValueError("Dataset/embedding provenance mismatch")
    corpus_ids = json.loads((args.embeddings / "corpus_ids.json").read_text())
    query_ids = json.loads((args.embeddings / "queries_ids.json").read_text())
    corpus_rows = read_jsonl(args.data / "corpus.jsonl")
    query_rows = read_jsonl(args.data / "queries.jsonl")
    if corpus_ids != [str(r["_id"]) for r in corpus_rows] or query_ids != [str(r["_id"]) for r in query_rows]:
        raise ValueError("ID ordering mismatch")
    corpus = np.load(args.embeddings / "corpus.npy")
    queries = np.load(args.embeddings / "queries.npy")
    if len(corpus) != len(corpus_ids) or len(queries) != len(query_ids):
        raise ValueError("ID/embedding row count mismatch")
    dev_path = args.data / "qrels" / (cfg["selection_split"] + ".tsv")
    dev_qrels = read_qrels(dev_path)
    query_text = {str(r["_id"]): r["text"] for r in query_rows}
    test_path = args.data / "qrels" / (cfg["test_split"] + ".tsv")
    # Access only held-out query IDs/texts for deduplication. Test relevance
    # values are not used here, or by corpus fitting / parameter selection.
    with test_path.open() as f:
        held_out_ids = {str(row["query-id"]) for row in csv.DictReader(f, delimiter="\t")}
    if set(dev_qrels) & held_out_ids:
        raise ValueError("Development/test query ID overlap")
    held_out_text = {query_text[q].strip().casefold() for q in held_out_ids}
    overlap_ids = sorted(q for q in dev_qrels if query_text[q].strip().casefold() in held_out_text)
    if overlap_ids and not cfg.get("exclude_dev_queries_with_exact_test_text", False):
        raise ValueError("Exact query text overlaps across splits; define a deduplication protocol")
    original_dev_count = len(dev_qrels)
    dev_qrels = {q: r for q, r in dev_qrels.items() if q not in overlap_ids}
    write_json(args.out / "split_audit.json", {
        "original_dev_queries": original_dev_count, "excluded_dev_query_ids": overlap_ids,
        "reason": "Exact stripped, case-folded text matches a held-out query",
        "dev_queries_after_exclusion": len(dev_qrels), "test_queries_unchanged": len(held_out_ids),
        "test_relevance_used_for_deduplication": False,
    })
    corpus_set = set(corpus_ids)
    if any(not set(r).issubset(corpus_set) for r in dev_qrels.values()):
        raise ValueError("Development judgments reference missing documents")
    dev_ids, dev_vectors = select_queries(query_ids, dev_qrels, queries)
    settings = NearConfig(**{k: cfg[k] for k in asdict(NearConfig()) if k in cfg}, max_depth=max(cfg["depths"]))
    print("Building corpus-only NEAR index", flush=True)
    start = time.perf_counter()
    index = NearIndex(settings).fit(corpus)
    build_seconds = time.perf_counter() - start
    index.save(args.out / "index")
    dev_baseline, _ = evaluate(search_exact(unit(dev_vectors), unit(corpus)), dev_ids, corpus_ids, dev_qrels)
    candidates = []
    best = None
    for normalized in cfg["normalization"]:
        components = index.score_components(dev_vectors, normalized)
        for depth in cfg["depths"]:
            residual_score = sum(components[1:depth+1]) / depth
            for alpha in cfg["alphas"]:
                selection = {"depth": depth, "normalized": normalized, "alpha": alpha}
                metrics, _ = evaluate(rank(components[0] + alpha * residual_score), dev_ids, corpus_ids, dev_qrels)
                candidate = {**selection, **metrics}
                candidates.append(candidate)
                print(json.dumps({"dev": candidate}), flush=True)
                # Grid order is the predetermined tie breaker; no test metrics enter.
                if best is None or metrics[cfg["selection_metric"]] > best["score"] + 1e-12:
                    best = {"selection": selection, "score": metrics[cfg["selection_metric"]]}
    write_json(args.out / "dev_grid.json", candidates)
    lock = {"selection": best["selection"], "dev_ndcg@10": best["score"], "dev_baseline": dev_baseline,
            "config_sha256": sha256(args.config), "dev_qrels_sha256": sha256(dev_path),
            "embedding_manifest_sha256": sha256(args.embeddings / "encoding_manifest.json"),
            "selection_before_test_evaluation": True, "candidate_count": len(candidates)}
    lock_path = args.out / "selection.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise ValueError("Existing frozen selection differs; use a new output directory")
    write_json(lock_path, lock)
    print("Selection locked; opening test qrels", json.dumps(best), flush=True)
    test_qrels = read_qrels(test_path)
    if set(test_qrels) & set(dev_qrels):
        raise ValueError("Development/test query ID overlap")
    dev_text = {query_text[q].strip().casefold() for q in dev_qrels}
    if any(query_text[q].strip().casefold() in dev_text for q in test_qrels):
        raise ValueError("Exact query text overlaps across splits")
    if any(not set(r).issubset(corpus_set) for r in test_qrels.values()):
        raise ValueError("Test judgments reference missing documents")
    test_ids, test_vectors = select_queries(query_ids, test_qrels, queries)
    selected = best["selection"]
    base_keys = unit(corpus)
    memory_keys = index.memory_keys(**selected)
    query_keys = index.query_keys(test_vectors, **selected)
    baseline_ranking = search_exact(unit(test_vectors), base_keys)
    near_ranking = search_exact(query_keys, memory_keys)
    base_mean, base_per = evaluate(baseline_ranking, test_ids, corpus_ids, test_qrels)
    near_mean, near_per = evaluate(near_ranking, test_ids, corpus_ids, test_qrels)
    components = index.score_components(test_vectors, selected["normalized"])
    explicit = components[0] + selected["alpha"] / selected["depth"] * sum(components[1:selected["depth"]+1])
    concat_scores = query_keys @ memory_keys.T
    max_score_error = float(np.max(np.abs(explicit - concat_scores)))
    np.testing.assert_allclose(explicit, concat_scores, rtol=1e-4, atol=1e-5)
    comparisons = paired_comparison(base_per, near_per, cfg["bootstrap_samples"], cfg["seed"])
    write_json(args.out / "per_query.json", {q: {"baseline": base_per[q], "near": near_per[q]} for q in test_ids})
    for name, ranking, qvec, dvec in [("baseline", baseline_ranking, unit(test_vectors), base_keys),
                                     ("near", near_ranking, query_keys, memory_keys)]:
        with (args.out / (name + ".trec")).open("w") as f:
            for row, qid in enumerate(test_ids):
                scores = qvec[row] @ dvec[ranking[row]].T
                for position, (doc, score) in enumerate(zip(ranking[row], scores), 1):
                    f.write(f"{qid} Q0 {corpus_ids[doc]} {position} {float(score):.9g} {name}\n")
    examples = []
    for qi in sorted(range(len(test_ids)), key=lambda i: near_per[test_ids[i]]["ndcg@10"] - base_per[test_ids[i]]["ndcg@10"]):
        qid = test_ids[qi]
        positives = test_qrels[qid]
        def relevant_ranks(ranking):
            return {corpus_ids[j]: i+1 for i, j in enumerate(ranking) if corpus_ids[j] in positives}
        examples.append({"query_id": qid, "query": query_text[qid],
            "ndcg_delta": near_per[qid]["ndcg@10"] - base_per[qid]["ndcg@10"],
            "relevant_documents": [{"id": d, "title": corpus_rows[corpus_ids.index(d)].get("title", "")} for d in positives],
            "baseline_relevant_ranks_top100": relevant_ranks(baseline_ranking[qi]),
            "near_relevant_ranks_top100": relevant_ranks(near_ranking[qi])})
    write_json(args.out / "examples.json", {"largest_losses": examples[:5], "largest_gains": list(reversed(examples[-5:]))})
    # Measure query transform + exact search on a single query, exclude encoding.
    timings = {
        "baseline": latency(lambda i: search_exact(unit(test_vectors[i % len(test_ids):i % len(test_ids)+1]), base_keys)),
        "near": latency(lambda i: search_exact(index.query_keys(test_vectors[i % len(test_ids):i % len(test_ids)+1], **selected), memory_keys)),
    }
    ann_result = None
    if args.hnsw:
        import faiss
        faiss.omp_set_num_threads(1)
        ann_result = {}
        for name, keys, qvec, exact in [("baseline", base_keys, unit(test_vectors), baseline_ranking),
                                       ("near", memory_keys, query_keys, near_ranking)]:
            ann = faiss.IndexHNSWFlat(keys.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
            ann.hnsw.efConstruction = 200
            ann.hnsw.efSearch = 128
            ann.hnsw.rng = faiss.RandomGenerator(cfg["seed"])
            ann.add(np.ascontiguousarray(keys))
            _, found = ann.search(np.ascontiguousarray(qvec), 100)
            metrics, _ = evaluate(found, test_ids, corpus_ids, test_qrels)
            ann_result[name] = {"metrics": metrics, "exact_top10_overlap": float(np.mean([
                len(set(a[:10]) & set(b[:10])) / 10 for a, b in zip(found, exact)]))}
            faiss.write_index(ann, str(args.out / (name + ".faiss")))
    validation = {"equation4_max_absolute_score_error": max_score_error,
                  "query_id_overlap": 0, "exact_query_text_overlap": 0,
                  "excluded_dev_query_ids": overlap_ids,
                  "retrieval": "exact inner product", "test_selection": False}
    try:
        import pytrec_eval
        runs = {}
        for name in ["baseline", "near"]:
            with (args.out / (name + ".trec")).open() as f:
                runs[name] = pytrec_eval.parse_run(f)
        trec_qrels = {q: {d: int(s) for d, s in r.items()} for q, r in test_qrels.items()}
        reference = pytrec_eval.RelevanceEvaluator(trec_qrels, {"ndcg_cut.10", "recall.10,100"})
        for name, measured in [("baseline", base_per), ("near", near_per)]:
            for qid, row in reference.evaluate(runs[name]).items():
                for ref, ours in [("ndcg_cut_10", "ndcg@10"), ("recall_10", "recall@10"), ("recall_100", "recall@100")]:
                    np.testing.assert_allclose(row[ref], measured[qid][ours], atol=1e-12)
        validation["pytrec_eval"] = "all per-query nDCG@10 and Recall@10/100 match"
    except ImportError:
        validation["pytrec_eval"] = "not installed"
    report = {
        "dataset": cfg["dataset"], "documents": len(corpus_ids), "dev_queries": len(dev_ids), "test_queries": len(test_ids),
        "model": encoder, "selected": selected, "fixed_settings": asdict(settings),
        "baseline": base_mean, "near": near_mean, "comparison": comparisons,
        "build_seconds": build_seconds, "latency": timings,
        "storage": {"baseline_keys_bytes": base_keys.nbytes, "near_keys_bytes": memory_keys.nbytes,
                    "near_dimension": memory_keys.shape[1], "base_dimension": base_keys.shape[1],
                    "reference_table_bytes": sum(l["anchors"].nbytes + l["values"].nbytes for l in index.layers[:selected["depth"]])},
        "validation": validation, "hnsw": ann_result,
        "provenance": {"config_sha256": sha256(args.config), "test_qrels_sha256": sha256(test_path),
                       "selection_sha256": sha256(lock_path), "platform": platform.platform(),
                       "code_sha256": {f.name: sha256(f) for f in Path(__file__).parent.glob("*.py")}},
        "limitations": ["Single standard dataset, not the paper's eight-task LMEB reproduction.",
                        "Quantized LiteRT EmbeddingGemma2 text path; not the full-precision checkpoint.",
                        "Query holdout with a shared corpus, not candidate-bank holdout.",
                        "Bootstrap resamples queries with fixed selected hyperparameters; related claims may be dependent.",
                        "Exact search separates scoring quality from ANN approximation; HNSW results require --hnsw."]}
    write_json(args.out / "results.json", report)
    with (args.out / "metrics.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["metric", "baseline", "near", "delta_pp", "ci95_low_pp", "ci95_high_pp"])
        for metric, row in comparisons.items():
            w.writerow([metric, row["baseline"], row["near"], row["delta_percentage_points"],
                        *[v*100 for v in row["paired_bootstrap_95ci"]]])
    print(json.dumps({"baseline": base_mean, "near": near_mean, "selected": selected}, indent=2), flush=True)


if __name__ == "__main__":
    main()
