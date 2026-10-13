"""Standard, uncapped BEIR recall; paper capped recall reported separately."""
import numpy as np


def rank(scores, top_k=100):
    return np.argsort(-scores, axis=1, kind="stable")[:, :top_k]


def evaluate(ranking, query_ids, corpus_ids, qrels):
    if len(ranking) != len(query_ids):
        raise ValueError("Query/ranking length mismatch")
    result = {}
    for qid, indices in zip(query_ids, ranking):
        relevant = {d: s for d, s in qrels[qid].items() if s > 0}
        if not relevant:
            raise ValueError("No positive relevance judgment for query " + qid)
        docs = [corpus_ids[i] for i in indices]
        gains = np.asarray([relevant.get(d, 0.0) for d in docs])
        ideal = np.asarray(sorted(relevant.values(), reverse=True))
        row = {}
        for k in [5, 10, 100]:
            top = gains[:k]
            dcg = float(np.sum(top / np.log2(np.arange(len(top)) + 2)))
            ideal_k = ideal[:k]
            idcg = float(np.sum(ideal_k / np.log2(np.arange(len(ideal_k)) + 2)))
            hits = int(np.count_nonzero(top))
            row[f"ndcg@{k}"] = dcg / idcg
            row[f"recall@{k}"] = hits / len(relevant)
        hits10 = int(np.count_nonzero(gains[:10]))
        row["capped_recall@10"] = hits10 / min(10, len(relevant))
        positions = np.flatnonzero(gains[:10])
        row["mrr@10"] = 1.0 / (int(positions[0]) + 1) if len(positions) else 0.0
        row["hit@10"] = float(bool(hits10))
        result[qid] = row
    means = {key: float(np.mean([r[key] for r in result.values()])) for key in next(iter(result.values()))}
    return means, result


def paired_comparison(base, near, samples=10000, seed=42):
    if set(base) != set(near) or not base:
        raise ValueError("Paired metrics require identical, nonempty query sets")
    ids = sorted(base)
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(ids), size=(samples, len(ids)))
    output = {}
    for metric in next(iter(base.values())):
        b = np.array([base[q][metric] for q in ids])
        n = np.array([near[q][metric] for q in ids])
        delta = n - b
        interval = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975])
        output[metric] = {
            "baseline": float(b.mean()), "near": float(n.mean()), "delta": float(delta.mean()),
            "delta_percentage_points": float(delta.mean() * 100),
            "relative_change_percent": float(delta.mean() / b.mean() * 100) if b.mean() else None,
            "paired_bootstrap_95ci": interval.tolist(),
            "improved_queries": int(np.sum(delta > 1e-12)),
            "worsened_queries": int(np.sum(delta < -1e-12)),
            "unchanged_queries": int(np.sum(np.abs(delta) <= 1e-12)),
        }
    return output
