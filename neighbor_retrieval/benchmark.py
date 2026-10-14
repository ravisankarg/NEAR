"""Separate development selection and locked evaluation for one-hop methods."""
import argparse
import csv
import itertools
import json
import platform
import time
from pathlib import Path

import numpy as np

from near_retrieval.benchmark import latency, select_queries
from near_retrieval.data import read_jsonl, read_qrels, sha256, write_json
from near_retrieval.metrics import evaluate, rank
from near_retrieval.near import NearIndex, unit
from .one_hop import OneHopGraph, score_method, softmax


def configurations(cfg):
    for family in cfg["family_order"]:
        diffusion = family.endswith("diffusion")
        prefix = "diffusion" if diffusion else "feedback"
        for weight, seeds, temperature in itertools.product(cfg[prefix+"_weights"],
                cfg[prefix+"_seed_counts"], cfg[prefix+"_temperatures"]):
            row = dict(family=family, weight=weight, seeds=seeds, temperature=temperature)
            if not diffusion:
                row["neighbor_fraction"] = cfg["graph_feedback_neighbor_fraction"] if family == "graph_feedback" else 0
            yield row


def better(candidate, incumbent, cfg):
    for key in (cfg["primary_metric"], cfg["tie_break_metric"]):
        if candidate[key] > incumbent[key] + 1e-12:
            return True
        if candidate[key] < incumbent[key] - 1e-12:
            return False
    return False


def provenance(args):
    paths = [args.config, args.embeddings/"encoding_manifest.json", args.data/"qrels/train.tsv",
             args.data/"qrels/test.tsv", args.prior/"selection.json", args.prior/"baseline.trec",
             args.prior/"near.trec"]
    paths += sorted(Path(__file__).parent.glob("*.py"))
    # Core sources and embeddings have separately checked hashes below.
    return {str(p): sha256(p) for p in paths}


def load_inputs(args):
    manifest = json.loads((args.embeddings/"encoding_manifest.json").read_text())
    for name in ("corpus", "queries"):
        if sha256(args.embeddings/(name+".npy")) != manifest["completed"][name]["sha256"]:
            raise ValueError("Cached embedding hash mismatch")
        if sha256(args.data/(name+".jsonl")) != manifest["configuration"]["source_hashes"][name]:
            raise ValueError("Cached text hash mismatch")
    prior = json.loads((args.prior/"results.json").read_text())
    if manifest != prior["model"]:
        raise ValueError("Encoder differs from original experiment")
    for name, digest in prior["provenance"]["code_sha256"].items():
        if sha256(Path("near_retrieval")/name) != digest:
            raise ValueError("Original benchmark source changed: "+name)
    if sha256(args.data/"qrels/test.tsv") != prior["provenance"]["test_qrels_sha256"]:
        raise ValueError("Test judgments differ from original benchmark")
    rows = {n: read_jsonl(args.data/(n+".jsonl")) for n in ("corpus", "queries")}
    ids = {n: json.loads((args.embeddings/(n+"_ids.json")).read_text()) for n in rows}
    for n in rows:
        if ids[n] != [str(r["_id"]) for r in rows[n]]:
            raise ValueError("ID/embedding ordering mismatch")
    corpus = unit(np.load(args.embeddings/"corpus.npy"))
    queries = unit(np.load(args.embeddings/"queries.npy"))
    return corpus, queries, ids, rows, manifest


def dev_split(args, ids, rows, queries):
    dev = read_qrels(args.data/"qrels/train.tsv")
    # Only held-out IDs and texts are used before the lock, as in the original run.
    with (args.data/"qrels/test.tsv").open() as f:
        held_ids = {r["query-id"] for r in csv.DictReader(f, delimiter="\t")}
    text = {r["_id"]: r["text"].strip().casefold() for r in rows["queries"]}
    held_text = {text[q] for q in held_ids}
    excluded = sorted(q for q in dev if text[q] in held_text)
    dev = {q: v for q, v in dev.items() if q not in excluded}
    audit = json.loads((args.prior/"split_audit.json").read_text())
    if excluded != audit["excluded_dev_query_ids"] or len(dev) != 807 or len(held_ids) != 300:
        raise ValueError("Development split differs from prior experiment")
    if set(dev) & held_ids:
        raise ValueError("Development/test ID overlap")
    selected_ids, vectors = select_queries(ids["queries"], dev, queries)
    return selected_ids, vectors, dev


def select(args, cfg):
    if (args.out/"selection.json").exists():
        raise ValueError("Selection already locked; use --phase evaluate or a new output directory")
    corpus, queries, ids, rows, manifest = load_inputs(args)
    dev_ids, dev_q, qrels = dev_split(args, ids, rows, queries)
    started = time.perf_counter()
    graph = OneHopGraph.fit(corpus)
    graph.save(args.graph)
    build_seconds = time.perf_counter()-started
    direct = dev_q @ corpus.T
    order = rank(direct, 100)
    base, _ = evaluate(order, dev_ids, ids["corpus"], qrels)
    best = {}
    selected = {"settings": {"family": "baseline"}, "dev_metrics": base}
    grid = []
    print(json.dumps({"dev_baseline": base, "graph_build_seconds": build_seconds}), flush=True)
    for setting in configurations(cfg):
        scores = score_method(graph, dev_q, corpus, direct, order, setting)
        means, _ = evaluate(rank(scores), dev_ids, ids["corpus"], qrels)
        candidate = {"settings": setting, "dev_metrics": means}
        grid.append(candidate)
        family = setting["family"]
        if family not in best or better(means, best[family]["dev_metrics"], cfg):
            best[family] = candidate
        if better(means, selected["dev_metrics"], cfg):
            selected = candidate
        print(json.dumps({"candidate": len(grid), **setting, "recall@10": means["recall@10"],
                          "ndcg@10": means["ndcg@10"]}), flush=True)
    write_json(args.out/"dev_grid.json", grid)
    stats = {"documents": len(corpus), "outdegree": 1, "hops": 1,
             "self_edges": int(np.sum(graph.neighbor == np.arange(len(corpus)))),
             "mutual_edge_fraction": float(np.mean(graph.neighbor[graph.neighbor] == np.arange(len(corpus)))),
             "destinations_with_votes": int(np.count_nonzero(graph.indegree)),
             "max_indegree": int(graph.indegree.max()),
             "similarity_quantiles": np.quantile(graph.similarity, [0, .1, .5, .9, 1]).tolist(),
             "graph_array_bytes": sum(a.nbytes for a in (graph.neighbor, graph.similarity, graph.indegree)),
             "build_seconds": build_seconds}
    write_json(args.out/"graph_summary.json", stats)
    with (args.out/"graph_edges.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["source_id", "neighbor_id", "cosine", "neighbor_indegree"])
        for i, j in enumerate(graph.neighbor):
            writer.writerow([ids["corpus"][i], ids["corpus"][j], float(graph.similarity[i]), int(graph.indegree[j])])
    lock = {"created_unix": time.time(), "selection_before_new_test_evaluation": True,
            "test_has_been_inspected_in_prior_experiment": True, "dev_queries": len(dev_ids),
            "config": cfg, "provenance": provenance(args), "graph_sha256": sha256(args.graph),
            "baseline": base, "best_positive_per_family": best, "deployment_choice": selected,
            "candidate_count": len(grid), "embedding_sha256": manifest["completed"],
            "dev_query_ids": dev_ids}
    write_json(args.out/"selection.json", lock)
    print("LOCKED: "+json.dumps({"deployment_choice": selected, "families": best}), flush=True)


def paired_stats(base, method, cfg):
    ids = sorted(base)
    rng = np.random.default_rng(cfg["seed"])
    indices = rng.integers(len(ids), size=(cfg["bootstrap_samples"], len(ids)))
    result = {}
    for metric in next(iter(base.values())):
        delta = np.array([method[q][metric]-base[q][metric] for q in ids])
        samples = delta[indices].mean(axis=1)
        result[metric] = {"delta_pp": float(delta.mean()*100),
                         "ci95_pp": (np.quantile(samples, [.025, .975])*100).tolist(),
                         "improved": int(np.sum(delta > 1e-12)), "worsened": int(np.sum(delta < -1e-12)),
                         "unchanged": int(np.sum(np.abs(delta) <= 1e-12))}
        if metric == cfg["primary_metric"]:
            alpha = .05 / len(cfg["family_order"])
            result[metric]["bonferroni_ci_pp"] = (np.quantile(samples, [alpha/2, 1-alpha/2])*100).tolist()
            nz = delta[np.abs(delta) > 1e-12]
            signs = rng.choice([-1., 1.], (cfg["permutation_samples"], len(nz)))
            observed = abs(delta.sum())
            result[metric]["paired_sign_randomization_p"] = float((1+np.count_nonzero(
                np.abs(signs @ nz) >= observed-1e-12))/(len(signs)+1))
    return result


def holm(pvalues):
    ordered = sorted(pvalues, key=pvalues.get)
    result, previous = {}, 0.
    for i, key in enumerate(ordered):
        previous = max(previous, min(1., (len(ordered)-i)*pvalues[key]))
        result[key] = previous
    return result


def candidate_audit(graph, orders, qids, corpus_ids, qrels, counts):
    lookup = {d: i for i, d in enumerate(corpus_ids)}
    result = {}
    per_query = {}
    for n in counts:
        rows = []
        for qid, order in zip(qids, orders):
            positive = {lookup[d] for d, relevance in qrels[qid].items() if relevance > 0}
            seed = set(order[:n])
            expanded = set(graph.expanded(order[:n]))
            hit = len(positive & expanded)
            row = {"query_id": qid, "seed_recall": len(positive & seed)/len(positive),
                   "union_recall": hit/len(positive), "union_size": len(expanded),
                   "equal_count_cosine_recall": len(positive & set(order[:len(expanded)]))/len(positive),
                   "cosine_2n_recall": len(positive & set(order[:2*n]))/len(positive),
                   "union_oracle_recall@10": min(hit, 10)/len(positive),
                   "new_relevant_documents": len((positive & expanded)-seed)}
            rows.append(row)
        result[str(n)] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "query_id"}
        result[str(n)]["queries_with_new_relevant"] = sum(r["new_relevant_documents"] > 0 for r in rows)
        per_query[str(n)] = rows
    return result, per_query


def read_prior_run(path, qids, corpus_ids):
    lookup = {d: i for i, d in enumerate(corpus_ids)}
    runs = {q: [] for q in qids}
    with path.open() as f:
        for line in f:
            q, _, doc, pos, score, _ = line.split()
            runs[q].append((int(pos), lookup[doc], float(score)))
    ordered = [sorted(runs[q]) for q in qids]
    if any(len(r) != 100 for r in ordered):
        raise ValueError("Prior run must have 100 documents per query")
    return np.array([[v[1] for v in r] for r in ordered]), ordered


def make_examples(graph, family, setting, scores, base_scores, base_order, qids, corpus_ids, rows, qrels, base_per, per):
    text = {r["_id"]: r["text"] for r in rows["queries"]}
    title = {r["_id"]: r.get("title", "") for r in rows["corpus"]}
    lookup = {d: i for i, d in enumerate(corpus_ids)}
    all_order = rank(scores, len(corpus_ids))
    records = []
    for qi, qid in enumerate(qids):
        gain = per[qid]["recall@10"]-base_per[qid]["recall@10"]
        ndcg_gain = per[qid]["ndcg@10"]-base_per[qid]["ndcg@10"]
        records.append((gain, ndcg_gain, qid, qi))
    chosen = sorted(set([r[3] for r in sorted(records, reverse=True)[:3]] + [r[3] for r in sorted(records)[:3]]))
    examples = []
    for qi in chosen:
        qid = qids[qi]
        positions = {int(d): i+1 for i, d in enumerate(all_order[qi])}
        old_positions = {int(d): i+1 for i, d in enumerate(rank(base_scores[qi:qi+1], len(corpus_ids))[0])}
        seeds = base_order[qi, :setting["seeds"]]
        relevant = []
        for did in qrels[qid]:
            di = lookup[did]
            parents = seeds[graph.neighbor[seeds] == di]
            relevant.append({"id": did, "title": title[did], "baseline_rank": old_positions[di],
                             "new_rank": positions[di], "original_cosine": float(base_scores[qi, di]),
                             "new_score": float(scores[qi, di]), "indegree": int(graph.indegree[di]),
                             "supporting_seeds": [{"id": corpus_ids[i], "title": title[corpus_ids[i]],
                                 "baseline_rank": old_positions[int(i)], "query_cosine": float(base_scores[qi, i]),
                                 "edge_cosine": float(graph.similarity[i])} for i in parents]})
        example = {"query_id": qid, "query": text[qid], "family": family, "settings": setting,
                   "recall10_delta_pp": (per[qid]["recall@10"]-base_per[qid]["recall@10"])*100,
                   "ndcg10_delta_pp": (per[qid]["ndcg@10"]-base_per[qid]["ndcg@10"])*100,
                   "relevant_documents": relevant}
        if family.endswith("diffusion"):
            p = softmax(base_scores[qi], setting["temperature"])
            g = graph.support(base_scores[qi], seeds, setting["temperature"], int(family == "degree_diffusion"))
            for doc in relevant:
                di = lookup[doc["id"]]
                doc.update(direct_mass=float(p[di]), neighbor_mass=float(g[di]),
                           direct_term=float((1-setting["weight"])*p[di]),
                           neighbor_term=float(setting["weight"]*g[di]))
        examples.append(example)
    return examples


def evaluate_locked(args, cfg):
    lock = json.loads((args.out/"selection.json").read_text())
    if lock["config"] != cfg or lock["provenance"] != provenance(args) or lock["graph_sha256"] != sha256(args.graph):
        raise ValueError("Frozen configuration/code/data/graph changed after development selection")
    if (args.out/"results.json").exists():
        raise ValueError("Evaluation already recorded; use a new experiment directory")
    corpus, queries, ids, rows, manifest = load_inputs(args)
    qrels = read_qrels(args.data/"qrels/test.tsv")
    qids, test_q = select_queries(ids["queries"], qrels, queries)
    graph = OneHopGraph.load(args.graph)
    direct = test_q @ corpus.T
    all_base = rank(direct, len(corpus))
    base_order = all_base[:, :100]
    base_mean, base_per = evaluate(base_order, qids, ids["corpus"], qrels)
    prior_base, _ = read_prior_run(args.prior/"baseline.trec", qids, ids["corpus"])
    np.testing.assert_array_equal(base_order, prior_base)
    near_order, near_run = read_prior_run(args.prior/"near.trec", qids, ids["corpus"])
    near_mean, near_per = evaluate(near_order, qids, ids["corpus"], qrels)
    old_per = json.loads((args.prior/"per_query.json").read_text())
    for q in qids:
        for metric in base_per[q]:
            np.testing.assert_allclose(base_per[q][metric], old_per[q]["baseline"][metric], atol=1e-12)
            np.testing.assert_allclose(near_per[q][metric], old_per[q]["near"][metric], atol=1e-12)
    methods = {"baseline": {"metrics": base_mean}, "near": {"metrics": near_mean, "comparison": paired_stats(base_per, near_per, cfg)}}
    per_query = {q: {"baseline": base_per[q], "near": near_per[q]} for q in qids}
    run_scores = {"baseline": (base_order, np.take_along_axis(direct, base_order, axis=1))}
    run_scores["near"] = (near_order, np.array([[v[2] for v in r] for r in near_run]))
    examples = {}
    for family, chosen in lock["best_positive_per_family"].items():
        setting = chosen["settings"]
        scores = score_method(graph, test_q, corpus, direct, base_order, setting)
        order = rank(scores)
        means, per = evaluate(order, qids, ids["corpus"], qrels)
        methods[family] = {"settings": setting, "dev_metrics": chosen["dev_metrics"], "metrics": means,
                           "comparison": paired_stats(base_per, per, cfg)}
        for q in qids:
            per_query[q][family] = per[q]
        run_scores[family] = (order, np.take_along_axis(scores, order, axis=1))
        examples[family] = make_examples(graph, family, setting, scores, direct, base_order, qids, ids["corpus"], rows, qrels, base_per, per)
        if family.endswith("diffusion"):
            for i in range(len(qids)):
                allowed = set(base_order[i]) | set(graph.neighbor[base_order[i, :setting["seeds"]]])
                assert set(order[i]).issubset(allowed), "Diffusion escaped the one-hop candidate union"
        print(json.dumps({"test_family": family, "metrics": means}), flush=True)
    for name, (order, values) in run_scores.items():
        with (args.out/(name+".trec")).open("w") as f:
            for q, indices, scores in zip(qids, order, values):
                for position, (j, score) in enumerate(zip(indices, scores), 1):
                    f.write(f"{q} Q0 {ids['corpus'][j]} {position} {float(score):.17g} {name}\n")
    # An expansion-only cosine rerank is exactly the original list for k<=N.
    for n in cfg["candidate_audit_seed_counts"]:
        for i in range(len(qids)):
            pool = graph.expanded(all_base[i, :n])
            reordered = pool[np.argsort(-direct[i, pool], kind="stable")]
            np.testing.assert_array_equal(reordered[:n], all_base[i, :n])
    candidate_stats, candidate_rows = candidate_audit(graph, all_base, qids, ids["corpus"], qrels, cfg["candidate_audit_seed_counts"])
    pvals = {family: methods[family]["comparison"][cfg["primary_metric"]]["paired_sign_randomization_p"] for family in cfg["family_order"]}
    for family, corrected in holm(pvals).items():
        methods[family]["comparison"][cfg["primary_metric"]]["holm_p"] = corrected
    timings = {"baseline": latency(lambda i: rank(test_q[i:i+1] @ corpus.T), count=100)}
    prior_settings = json.loads((args.prior/"selection.json").read_text())["selection"]
    near_index = NearIndex.load(args.prior/"index")
    near_keys = near_index.memory_keys(**prior_settings)
    timings["near"] = latency(lambda i: rank(near_index.query_keys(test_q[i:i+1], **prior_settings) @ near_keys.T), count=100)
    for family, chosen in lock["best_positive_per_family"].items():
        def timed(i):
            q = test_q[i:i+1]
            s = q @ corpus.T
            order = rank(s)
            return rank(score_method(graph, q, corpus, s, order, chosen["settings"]))
        timings[family] = latency(timed, count=100)
    validation = {"original_baseline_top100_exact_match": True, "original_metrics_exact_match": True,
                  "expansion_only_topN_exact_match": True, "diffusion_top100_within_one_hop_union": True,
                  "graph_self_edges": int(np.sum(graph.neighbor == np.arange(len(corpus)))),
                  "frozen_selection_sha256": sha256(args.out/"selection.json")}
    try:
        import pytrec_eval
        reference = pytrec_eval.RelevanceEvaluator({q: {d: int(v) for d, v in r.items()} for q, r in qrels.items()}, {"ndcg_cut.10", "recall.10,100"})
        for family in run_scores:
            with (args.out/(family+".trec")).open() as f:
                run = pytrec_eval.parse_run(f)
            for q, row in reference.evaluate(run).items():
                for theirs, ours in [("ndcg_cut_10", "ndcg@10"), ("recall_10", "recall@10"), ("recall_100", "recall@100")]:
                    np.testing.assert_allclose(row[theirs], per_query[q][family][ours], atol=1e-12)
        validation["pytrec_eval"] = "All 6 methods, all 300 queries match nDCG@10 and Recall@10/100"
    except ImportError:
        validation["pytrec_eval"] = "unavailable"
    result = {"experiment": cfg["experiment"], "documents": len(corpus), "dev_queries": 807, "test_queries": len(qids),
              "encoder": {"family": manifest["configuration"]["family"], "model_sha256": manifest["configuration"]["model_sha256"],
                          "corpus_sha256": manifest["completed"]["corpus"]["sha256"], "queries_sha256": manifest["completed"]["queries"]["sha256"]},
              "methods": methods, "deployment_choice": lock["deployment_choice"], "candidate_audit": candidate_stats,
              "latency": timings, "validation": validation, "config": cfg,
              "graph": json.loads((args.out/"graph_summary.json").read_text()),
              "platform": platform.platform(), "numpy": np.__version__,
              "limitations": ["Same previously inspected test set: exploratory follow-up, not independent confirmation.",
                  "One encoder, one dataset, quantized text-only inference and exact search.",
                  "Related queries may share evidence documents; query-level bootstrap/sign tests assume exchangeable queries.",
                  "Relevance is judged evidence, not all semantic relatedness; unjudged documents count as nonrelevant.",
                  "The new methods are tuned for Recall@10; historical NEAR was tuned for nDCG@10.",
                  "Multiple-family correction does not account for prior test reuse or related-query dependence.",
                  "Graph feedback is one-hop expansion followed by a second full-corpus search, unlike bounded diffusion."]}
    write_json(args.out/"results.json", result)
    write_json(args.out/"per_query.json", per_query)
    write_json(args.out/"examples.json", examples)
    write_json(args.out/"candidate_audit_per_query.json", candidate_rows)
    with (args.out/"metrics.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "recall@5", "recall@10", "recall@100", "ndcg@10", "mrr@10", "delta_recall10_pp", "ci95_low_pp", "ci95_high_pp"])
        for family, row in methods.items():
            comparison = row.get("comparison", {}).get("recall@10", {})
            writer.writerow([family, *[row["metrics"][m] for m in ["recall@5", "recall@10", "recall@100", "ndcg@10", "mrr@10"]],
                             comparison.get("delta_pp", 0), *comparison.get("ci95_pp", [0, 0])])
    print("DONE: "+str(args.out/"results.json"), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", required=True, choices=["select", "evaluate"])
    p.add_argument("--data", type=Path, default=Path("data/scifact"))
    p.add_argument("--embeddings", type=Path, default=Path("artifacts/scifact_eg2"))
    p.add_argument("--prior", type=Path, default=Path("results/scifact_eg2"))
    p.add_argument("--config", type=Path, default=Path("configs/one_hop_scifact.json"))
    p.add_argument("--graph", type=Path, default=Path("artifacts/scifact_one_hop/graph.npz"))
    p.add_argument("--out", type=Path, default=Path("results/scifact_one_hop"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text())
    (select if args.phase == "select" else evaluate_locked)(args, cfg)


if __name__ == "__main__":
    main()
