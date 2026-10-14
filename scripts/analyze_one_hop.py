"""Post-evaluation mechanism audit; never selects or changes parameters."""
import json
from pathlib import Path

import numpy as np

from near_retrieval.data import read_jsonl, read_qrels, write_json
from near_retrieval.metrics import rank
from near_retrieval.near import unit
from neighbor_retrieval.one_hop import OneHopGraph, softmax


def main():
    out = Path("results/scifact_one_hop")
    lock = json.loads((out/"selection.json").read_text())
    corpus = unit(np.load("artifacts/scifact_eg2/corpus.npy"))
    queries = unit(np.load("artifacts/scifact_eg2/queries.npy"))
    cids = json.loads(Path("artifacts/scifact_eg2/corpus_ids.json").read_text())
    qids = json.loads(Path("artifacts/scifact_eg2/queries_ids.json").read_text())
    qrows = {r["_id"]: r for r in read_jsonl("data/scifact/queries.jsonl")}
    crows = {r["_id"]: r for r in read_jsonl("data/scifact/corpus.jsonl")}
    qrels = read_qrels("data/scifact/qrels/test.tsv")
    graph = OneHopGraph.load("artifacts/scifact_one_hop/graph.npz")
    settings = lock["best_positive_per_family"]["diffusion"]["settings"]
    traces = []
    label_rates = []
    for qid in sorted(qrels):
        q = queries[qids.index(qid)]
        scores = q @ corpus.T
        original = rank(scores[None], len(cids))[0]
        seeds = original[:settings["seeds"]]
        labels = np.array([cids[i] in qrels[qid] for i in range(len(cids))])
        a = softmax(scores[seeds], settings["temperature"])
        label_rates.append({"query_id": qid, "seed_judged_relevance_mass": float(a @ labels[seeds]),
                            "neighbor_judged_relevance_mass": float(a @ labels[graph.neighbor[seeds]]),
                            "relevant_seeds": int(labels[seeds].sum()),
                            "relevant_seeds_whose_neighbor_is_judged_relevant": int(np.sum(labels[seeds]&labels[graph.neighbor[seeds]]))})
        if qid not in {"303", "834", "1226", "785"}:
            continue
        p = softmax(scores, settings["temperature"])
        g = graph.support(scores, seeds, settings["temperature"])
        mixed = (1-settings["weight"])*p+settings["weight"]*g
        new = rank(mixed[None], len(cids))[0]
        old_position = {int(d): i+1 for i, d in enumerate(original)}
        new_position = {int(d): i+1 for i, d in enumerate(new)}
        selected = sorted(set(original[:12]) | set(new[:12]) | {cids.index(d) for d in qrels[qid]}, key=old_position.get)
        documents = []
        for j in selected:
            parents = seeds[graph.neighbor[seeds] == j]
            documents.append({"id": cids[j], "title": crows[cids[j]].get("title", ""),
                "judged_relevant": bool(labels[j]), "baseline_rank": old_position[j], "diffusion_rank": new_position[j],
                "cosine": float(scores[j]), "direct_mass": float(p[j]), "neighbor_mass": float(g[j]),
                "mixed_score": float(mixed[j]), "indegree": int(graph.indegree[j]),
                "parents": [{"id": cids[i], "title": crows[cids[i]].get("title", ""), "seed_rank": old_position[int(i)],
                             "query_cosine": float(scores[i]), "edge_cosine": float(graph.similarity[i])} for i in parents]})
        traces.append({"query_id": qid, "query": qrows[qid]["text"], "settings": settings, "documents": documents})
    write_json(out/"mechanism_audit.json", {
        "posthoc": True, "used_for_selection": False,
        "trace_choice": "303: expansion finds an out-of-top50 positive; 834: the common Recall@10 loss; 1226: diffusion nDCG gain; 785: prior NEAR example.",
        "traces": traces, "label_transfer_per_query": label_rates,
        "label_transfer_summary": {"mean_seed_judged_relevance_mass": float(np.mean([r["seed_judged_relevance_mass"] for r in label_rates])),
            "mean_neighbor_judged_relevance_mass": float(np.mean([r["neighbor_judged_relevance_mass"] for r in label_rates])),
            "relevant_seed_edges": sum(r["relevant_seeds"] for r in label_rates),
            "relevant_seed_edges_to_judged_relevant": sum(r["relevant_seeds_whose_neighbor_is_judged_relevant"] for r in label_rates)},
        "caveat": "Judged relevance is sparse evidence relevance. Unjudged destinations are not proven semantically irrelevant."})


if __name__ == "__main__":
    main()
