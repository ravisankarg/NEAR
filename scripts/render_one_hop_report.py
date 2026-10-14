"""Build the follow-up report and publication data from locked results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"results/scifact_one_hop"
LABELS = {"baseline": "Cosine baseline", "near": "NEAR (previous)", "diffusion": "One-hop diffusion",
          "degree_diffusion": "Degree-corrected diffusion", "rocchio": "Query feedback control", "graph_feedback": "One-hop query feedback"}
ORDER = list(LABELS)


def main():
    result = json.loads((OUT/"results.json").read_text())
    lock = json.loads((OUT/"selection.json").read_text())
    audit = json.loads((OUT/"mechanism_audit.json").read_text())
    methods = result["methods"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#f5f3ec", "axes.facecolor": "#f5f3ec", "text.color": "#1e302a", "axes.labelcolor": "#1e302a"})
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    labels = [LABELS[f] for f in ORDER[1:]]
    for ax, metric, title in zip(axs, ["recall@10", "ndcg@10"], ["Recall@10 change", "nDCG@10 change"]):
        for i, family in enumerate(ORDER[1:]):
            c = methods[family]["comparison"][metric]
            delta, (low, high) = c["delta_pp"], c["ci95_pp"]
            color = "#224ae8" if family == "near" else "#9b5148"
            ax.errorbar(delta, i, xerr=[[max(0, delta-low)], [max(0, high-delta)]], fmt="o", color=color, capsize=4, markersize=7)
        ax.axvline(0, color="#7d8983", linewidth=1, linestyle="--")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_yticks(np.arange(len(labels)), labels if ax is axs[0] else [])
        ax.invert_yaxis()
        ax.set_xlabel("Percentage points vs cosine · paired 95% interval")
        ax.grid(axis="x", alpha=.16)
    fig.suptitle("One neighbor, one hop: no measured Recall@10 gain", fontsize=16, fontweight="bold")
    fig.savefig(OUT/"comparison.svg")
    fig.savefig(OUT/"comparison.png", dpi=160)
    plt.close(fig)
    (ROOT/"docs/data/one-hop-comparison.svg").write_bytes((OUT/"comparison.svg").read_bytes())
    table = ["| Method | Recall@10 | Δ pp | nDCG@10 | Recall@100 | Median ms |", "|---|---:|---:|---:|---:|---:|"]
    for family in ORDER:
        row = methods[family]
        table.append(f"| {LABELS[family]} | {row['metrics']['recall@10']*100:.2f} | {row.get('comparison',{}).get('recall@10',{}).get('delta_pp',0):+.2f} | {row['metrics']['ndcg@10']*100:.2f} | {row['metrics']['recall@100']*100:.2f} | {result['latency'][family]['median_ms']:.3f} |")
    settings = []
    for family in ORDER[2:]:
        s = methods[family]["settings"]
        settings.append(f"- **{LABELS[family]}:** N={s['seeds']}, temperature={s['temperature']}, weight={s['weight']}" + (f", neighbor fraction={s['neighbor_fraction']}" if 'neighbor_fraction' in s else "") + ".")
    body = "\n".join(table)
    report = f"""# One nearest neighbor, one hop: frozen EG2 follow-up

**The tested methods did not improve Recall@10.** All four development-selected positive-weight variants scored 91.60 versus cosine's 91.93. Each lost query 834 and recovered no query at that cutoff. The development selection rule, which includes the baseline, selected **unchanged cosine** before new test evaluation.

{body}

Scores are ×100. Timings were measured together in this run on CPU, include query transformation and exact search, and exclude encoding. Earlier NEAR latency numbers came from a different timing run and are not reused here. The corpus graph took {result['graph']['build_seconds']:.3f}s to build and occupies {result['graph']['graph_array_bytes']/1024:.2f} KiB in addition to the original 15.18 MiB vectors.

## Protocol and evidence boundary

Exactly the original 5,183 document vectors, 807 development queries after the original duplicate-text exclusion, and 300 test queries. Model and vector hashes are in results.json. The encoder is frozen; graph fitting consumes only corpus vectors. No new model, query embedding, or judgment is inferred from document text. All graph operations follow one original seed-to-neighbor edge, without recursive expansion.

The configuration was written before selection. We evaluated 120 positive-weight candidates on development, then wrote selection.json before opening the new test metrics. Selection maximizes Recall@10, then nDCG@10, then fixed grid order. Baseline development Recall@10 is already 100%; baseline nDCG@10 is 98.6706. The best positive candidates tie recall but lower nDCG. There is consequently very little useful selection signal for recall improvement on this split.

This is an **exploratory follow-up on a previously inspected test set**, not independent confirmation. We did not retune after the new test results. Original NEAR was selected for nDCG@10, whereas this follow-up is recall-first. A fresh dataset or genuinely uninspected queries are required before generalization claims.

## Locked positive configurations

{chr(10).join(settings)}

These are diagnostic positive-weight runs, not the baseline-inclusive deployment choice.

## What the mathematics says

Let q and x_j be unit vectors, s_j=q·x_j, and S_N the exact cosine top N. Build n(i)=argmax_(j≠i) x_i·x_j over the corpus, c_i=x_i·x_n(i), and h_j=number of incoming edges to j. Each document has exactly one outgoing edge. Define C=S_N∪{{n(i):i∈S_N}}.

**Expansion alone.** C contains S_N, so its candidate coverage cannot decrease. But its size can be as large as 2N. For k≤N, sorting C by the unchanged exact cosine produces exactly the original top k, provided ties use the same deterministic ordering. A new score is necessary for fixed-cutoff improvement. This statement concerns exact search; approximate ANN search can behave differently.

**Weighted one-hop evidence.** Define p_j=exp(s_j/T)/Σ_d exp(s_d/T). This is a scoring distribution, not a calibrated probability of relevance. For a seed i, let a_i=exp(s_i/T)/Σ_(u∈S_N) exp(s_u/T). Accumulate b_j=Σ_(i∈S_N:n(i)=j) a_i max(c_i,0)/h_j^γ, with γ=0 for ordinary diffusion and γ=1 for degree correction. Normalize g_j=b_j/Σ_d b_d. If all edge weights vanish, use g=p. The new score is f_j=(1−λ)p_j+λg_j.

This is a two-source mixture: direct query evidence and one-hop support. It stays nonnegative and sums to one. For N≤100, scoring all corpus rows is equivalent at the top 100 to reranking the union of the original top 100 and neighbor(S_N): all other documents receive only the same positive scaling of p. The implementation checks this equivalence on every test query.

The degree divisor estimates a document's corpus-wide popularity. If every source voted equally and all edge cosines were equal, incoming sum would be proportional to h_j; dividing by h_j makes all destinations with h_j>0 equal. It does **not** prove relevance, calibrate probabilities, or guarantee improvement. With only one outgoing edge, normalizing its cosine to sum to one would cancel the edge weight; we retain that weight before the global vote normalization.

For relevant R and competitor C, f_R−f_C=(1−λ)(p_R−p_C)+λ(g_R−g_C). If R initially loses, let Δp=p_C−p_R>0 and Δg=g_R−g_C. Reversal requires Δg>0 and λ>Δp/(Δp+Δg). This is an exact pairwise condition, not a guarantee that R crosses the top-10 boundary against all competitors.

If g_R=0, then for λ<1 the ranking is equivalent to p_j+[λ/(1−λ)]g_j. R receives no increment while all competitors receive nonnegative increments, so R's rank cannot improve. In particular, zero corpus indegree means R cannot benefit from this inbound-vote method at any seed depth. Query 834 is an observed example.

For binary relevance labels y_j, expected judged relevance under the scoring distribution changes by λ Σ_j(g_j−p_j)y_j. It is positive exactly when the neighbor distribution concentrates more mass on judged relevant documents. This identity is **not** a top-k recall theorem. Exact recall change is (|R∩(A_new\\A_old)|−|R∩(A_old\\A_new)|)/|R| for the two returned top-k sets. Gains must exceed displaced relevant documents.

**Query feedback control and graph extension.** Let η_i=η max(c_i,0), v_i=(1−η_i)x_i+η_i x_n(i), and v=Σ_(i∈S_N) a_i v_i. Search the full corpus with q'=unit((1−β)q+βv). η=0 is ordinary positive query feedback; η=0.5 uses one neighbor per seed. This requires a second search, so it is labeled separately from bounded candidate reranking. The normalization of q' does not affect ordering within the query. An original losing pair reverses only when β v·(x_R−x_C)>(1−β)q·(x_C−x_R). Again, graph smoothness is an assumption about relevance, not a theorem.

Why averaging might help under explicit assumptions: in an unnormalized model x_i=μ_i+ε_i and x_n=μ_n+ε_n with independent zero-mean noise of variance E||ε||²=σ², the pair average estimates μ_i with MSE σ²/2+||μ_n−μ_i||²/4. This is below σ² exactly when ||μ_n−μ_i||²<2σ². Correlated errors or a different semantic target can remove the benefit. Frozen neural embeddings need not satisfy this model, and cosine normalization changes its geometry; this is motivation, not a proof of the measured method.

## Candidate coverage: account for the budget

| Initial N | Mean union size | Initial recall | Union recall | Cosine at the same per-query count | Cosine at 2N |
|---:|---:|---:|---:|---:|---:|
"""
    for n, row in sorted(result["candidate_audit"].items(), key=lambda v:int(v[0])):
        report += f"| {n} | {row['union_size']:.2f} | {row['seed_recall']*100:.2f} | {row['union_recall']*100:.2f} | {row['equal_count_cosine_recall']*100:.2f} | {row['cosine_2n_recall']*100:.2f} |\n"
    report += """
The same-count control retrieves exactly |C_q| cosine results separately for each query, not the rounded average count. The union is an unranked candidate pool; its recall must not be reported as Recall@10. At N=10 it adds judged evidence for only queries 303, 913, and 914. At N=100 it adds no judged relevant document. Its N=10 oracle Recall@10 is 92.93: even a perfect reranker restricted to that union cannot exceed this ceiling on these judgments.

## Working cases

- **Potential discovered, insufficient promotion — query 303:** “DMRT1 is a sex-determining gene that is epigenetically regulated by the MHM region.” The seed at rank 3 points to the judged document at cosine rank 53, with document-to-document cosine 0.9256. The one-hop union discovers it. The locked diffusion moves it 53→34, still outside the top ten.
- **Ranking improvement without recall improvement — query 1226:** the TET-protein query's judged document receives four seed votes and moves 5→4. nDCG@10 rises by 4.38 points; Recall@10 stays at 1.
- **The common recall loss — query 834:** “NOX2-independent pathways can generate peroxynitrite by reacting with nitrogen intermediates.” Its judged document has corpus indegree zero and receives no neighbor vote. Other documents are boosted, and it falls 9→11. Recall@10 falls 1→0. Two different competing documents overtake it in diffusion.
- **Prior NEAR example — query 785:** NEAR moved the relevant document 12→9. Ordinary one-hop diffusion moves it 12→13; none of its 50 seeds points to it.

These examples were selected after evaluation to explain mechanisms, never to set weights. See mechanism_audit.json for titles, IDs, parent edges, direct mass, neighbor mass, and scores.

## Why similar documents can be bad votes

277 of the 300 test queries have exactly one judged relevant document. The graph excludes self-edges, so that document's outgoing edge necessarily points to an unjudged document for those queries. Other seeds can still point back to the relevant document, but similarity alone does not ensure they do. Across selected top-50 seeds, only 45 of 331 edges originating at judged relevant seeds point to another judged relevant document. In the softmax-weighted seed diagnostic, mean judged-relevance mass falls from 35.25% on seeds to 6.78% on their neighbors. This is a post-hoc label-transfer diagnostic, not a tuning signal; unjudged does not mean semantically unrelated.

The graph has 5,183 edges, no self-edges, 3,069 destinations with positive indegree, and maximum indegree 8. Its median edge cosine is about 0.909, yet that topical closeness is insufficient for evidence-level relevance. Correcting popularity also did not recover recall here.

## Uncertainty and validation

Each positive method's Recall@10 change is −0.333 points, with paired 95% bootstrap interval [−1.00, 0.00]. The four-family Bonferroni 98.75% interval is [−1.333, 0.00]; Holm-adjusted two-sided paired sign-randomization p is 1.0. There is only one nonzero paired change, so this is not evidence of a reliable general loss either. Query-level resampling assumes exchangeability, while related claims may share evidence. Historical test reuse is not corrected by these intervals.

All 25 Python numerical tests passed. The original baseline top-100 rankings and all original per-query metrics match. pytrec_eval agrees for every query of all six methods on nDCG@10 and Recall@10/100. Graph self-exclusion, one-hop-only behavior, probability conservation, degree correction, exact-cosine expansion invariance, and the pairwise reversal condition have numerical tests. The published result verifier checks saved metrics and traces without re-encoding.

## Reproduce

For this workspace's original matching embeddings and baseline/NEAR index:

```bash
OPENBLAS_NUM_THREADS=4 python -m unittest discover -s tests -v
python -m neighbor_retrieval.benchmark --phase select \\
  --out results/rerun_one_hop --graph artifacts/rerun_one_hop/graph.npz
python -m neighbor_retrieval.benchmark --phase evaluate \\
  --out results/rerun_one_hop --graph artifacts/rerun_one_hop/graph.npz
```

For a fresh clone, first follow the root README to download SciFact and the correct LiteRT model, and run scripts/run_benchmark.sh to create a **matching** baseline/NEAR result, index, and embedding cache. Pass those paths with --prior, --embeddings, and --data to both commands above. The old native index and model weights are not bundled. Selection refuses an existing lock; evaluation refuses changed source/configuration/data hashes and refuses to overwrite existing results. A fresh rerun can differ with runtime or input changes. The frozen published results remain under results/scifact_one_hop.

## Related work and scope

The idea belongs to graph-based retrieval and pseudo-relevance feedback; this implementation does not claim invention of either field or reproduce those papers' full systems.

- [Stanford IR book: pseudo-relevance feedback](https://nlp.stanford.edu/IR-book/html/htmledition/pseudo-relevance-feedback-1.html): useful motivation and the risk of query drift.
- [Zerveas et al., 2023: reciprocal nearest neighbors for dense retrieval](https://arxiv.org/abs/2305.15720): document relationships can support inference-time reranking; their reciprocal-neighbor similarity is different from our strict single-edge mixture.
- [Klein and Wolf, 2021: graph query expansion](https://arxiv.org/abs/2112.02666): learned aggregation over extended image neighborhoods; our experiment is training-free, text-only, and one hop.

The defensible conclusion is narrow: **these locked one-hop variants do not improve recall on these EG2/SciFact embeddings.** That does not prove that all frozen-encoder graph methods fail. The main open requirements are a relevance-preserving neighbor signal and a development set with enough recall errors to evaluate it.
"""
    (OUT/"REPORT.md").write_text(report)
    public = {"methods": methods, "order": ORDER, "labels": LABELS, "candidate_audit": result["candidate_audit"],
              "graph": result["graph"], "latency": result["latency"], "selection": lock["deployment_choice"],
              "examples": audit["traces"], "label_transfer": audit["label_transfer_summary"]}
    (ROOT/"docs/data/one-hop.json").write_text(json.dumps(public, indent=2)+"\n")
    print("Built one-hop report, comparison plot, and site data")


if __name__ == "__main__":
    main()
