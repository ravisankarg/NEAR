"""Render an experiment's measured results; never contains hardcoded scores."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    args = p.parse_args()
    folder = args.results.parent
    data = json.loads(args.results.read_text())
    selected = data["selected"]
    metrics = ["ndcg@10", "recall@10", "recall@100", "mrr@10"]
    labels = ["nDCG@10", "Recall@10", "Recall@100", "MRR@10"]
    base = np.array([data["baseline"][m] * 100 for m in metrics])
    near = np.array([data["near"][m] * 100 for m in metrics])
    fig, (ax, delta_ax) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.4, 1]})
    fig.patch.set_facecolor("#f7f9fc")
    x = np.arange(len(metrics))
    ax.bar(x-.19, base, width=.36, label="Cosine baseline", color="#69778c")
    ax.bar(x+.19, near, width=.36, label="NEAR", color="#087f8c")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 104)
    ax.set_ylabel("Score × 100 (higher is better)")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.14), ncol=2)
    for i, (b, n) in enumerate(zip(base, near)):
        ax.text(i-.19, b+1.3, f"{b:.2f}", ha="center", fontsize=9)
        ax.text(i+.19, n+1.3, f"{n:.2f}", ha="center", fontsize=9)
    deltas = near - base
    intervals = np.array([data["comparison"][m]["paired_bootstrap_95ci"] for m in metrics]) * 100
    delta_ax.errorbar(deltas, x, xerr=np.maximum(0, np.stack([deltas-intervals[:,0], intervals[:,1]-deltas])),
                      fmt="o", color="#087f8c", ecolor="#087f8c", capsize=4)
    delta_ax.axvline(0, color="#7c8798", lw=1, linestyle="--")
    delta_ax.set_yticks(x, labels)
    delta_ax.invert_yaxis()
    delta_ax.set_xlabel("NEAR − baseline (percentage points)")
    delta_ax.set_title("Paired query bootstrap · 95% CI", fontsize=11)
    for axis in [ax, delta_ax]:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y" if axis is ax else "x", alpha=.15)
        axis.set_axisbelow(True)
    fig.suptitle("EmbeddingGemma2 + NEAR on SciFact", fontsize=17, fontweight="bold", x=.08, ha="left")
    fig.text(.08, .91, f"{data['documents']:,} documents · {data['test_queries']} held-out test queries · quantized LiteRT text encoder", color="#465168")
    fig.text(.08, .015, "Exact retrieval. Hyperparameters selected on training queries. One dataset; query-level CIs assume independent queries.", fontsize=9, color="#465168")
    fig.tight_layout(rect=[.015, .05, .995, .87])
    fig.savefig(folder / "comparison.png", dpi=180, facecolor=fig.get_facecolor())
    fig.savefig(folder / "comparison.svg", facecolor=fig.get_facecolor())
    primary = data["comparison"]["ndcg@10"]
    lo, hi = [n*100 for n in primary["paired_bootstrap_95ci"]]
    conclusion = ("improved" if primary["delta"] > 0 else "reduced" if primary["delta"] < 0 else "matched")
    relative = primary["relative_change_percent"]
    relative_text = f"{relative:+.2f}% relative" if relative is not None else "relative change undefined for zero baseline"
    uncertainty = ("The interval includes zero, so this run does not establish a reliable improvement."
                   if lo <= 0 <= hi else "The interval excludes zero under the query-bootstrap assumptions.")
    lines = ["# Measured result: EmbeddingGemma2 + NEAR on SciFact", "",
        f"NEAR **{conclusion} nDCG@10**; change: **{primary['delta_percentage_points']:+.2f} percentage points** "
        f"({relative_text}). The paired 95% bootstrap interval for the change is "
        f"[{lo:+.2f}, {hi:+.2f}] points. {uncertainty}", "",
        "These are locally measured held-out results, not numbers copied from the paper.", "",
        "| Metric | Baseline | NEAR | Change (points) | 95% CI for change |",
        "|---|---:|---:|---:|---:|"]
    for metric, label in zip(metrics, labels):
        row = data["comparison"][metric]
        a, b = [v*100 for v in row["paired_bootstrap_95ci"]]
        lines.append(f"| {label} | {row['baseline']*100:.2f} | {row['near']*100:.2f} | {row['delta_percentage_points']:+.2f} | [{a:+.2f}, {b:+.2f}] |")
    lines += ["", "Scores above are multiplied by 100. Recall is standard uncapped recall; capped Recall@10 is also in results.json.", "",
              "![Measured comparison](comparison.png)", "", "## Protocol", "",
        f"- Corpus: all **{data['documents']:,}** SciFact documents; **{data['dev_queries']}** official training queries for selection; **{data['test_queries']}** official test queries.",
        f"- Cross-split exact-text duplicates removed only from development: {data['validation'].get('excluded_dev_query_ids', [])}. Test queries unchanged. See split_audit.json.",
        "- Encoder: EmbeddingGemma2, text path of the 440M text/vision LiteRT package; 270M text backbone, INT4 QAT weights, 768-dimensional output. No images are used.",
        "- Identical cached embeddings for both methods; no encoder training. Query prefix: `task: search result | query: `; document format: `title: {title} | text: {text}`.",
        "- Up to 8,192 input tokens, automatic signature selection, overflow errors rather than silent truncation. All input items completed successfully.",
        "- NEAR: k=50, at most 16 anchors, 4 routed anchors, beta=10, seed=42. All 30 paper configurations tried only on development queries.",
        f"- Selected: L={selected['depth']}, normalized residuals={selected['normalized']}, alpha={selected['alpha']}. Selection saved before test evaluation.",
        "- Exact inner-product retrieval over the complete corpus. This isolates NEAR scoring from approximate-search errors.",
        f"- Test nDCG@10 improved on {primary['improved_queries']} queries, worsened on {primary['worsened_queries']}, and was unchanged on {primary['unchanged_queries']}.", "",
        "## Cost and validation", "",
        f"- NEAR corpus preprocessing for all {data['fixed_settings']['max_depth']} candidate layers, reused by the development grid: {data['build_seconds']:.2f} seconds.",
        f"- Vector index: {data['storage']['base_dimension']} → {data['storage']['near_dimension']} dimensions; "
        f"{data['storage']['baseline_keys_bytes']/2**20:.2f} → {data['storage']['near_keys_bytes']/2**20:.2f} MiB for float32 keys.",
        f"- Query reference tables: {data['storage']['reference_table_bytes']/1024:.1f} KiB.",
        "- Storage numbers describe the selected float32 search keys and routing tables, excluding research caches and any ANN graph overhead.",
        f"- Median CPU transform + exact retrieval: baseline {data['latency']['baseline']['median_ms']:.2f} ms; NEAR {data['latency']['near']['median_ms']:.2f} ms. Encoder time excluded; these are local microbenchmarks.",
        f"- Concatenated score versus explicit equation: max absolute difference {data['validation']['equation4_max_absolute_score_error']:.3g}.",
        f"- Independent metric verification: {data['validation']['pytrec_eval']}.", "", "## Limits", ""]
    lines += ["- " + limit for limit in data["limitations"]]
    lines += ["- The paper leaves clustering initialization/stopping unspecified. This implementation uses seeded spherical k-means++, three restarts, and at most 100 iterations.",
              "- Pretrained model exposure to this standard benchmark is unknown; this is a same-encoder comparison, not a claim of uncontaminated pretraining.",
              "", "## Inspect and reproduce", "",
              "See [the project README](../../README.md) for exact commands. Inspect [selection.json](selection.json), "
              "[dev_grid.json](dev_grid.json), [results.json](results.json), [per_query.json](per_query.json), and "
              "[examples.json](examples.json). TREC run files retain the top 100 results for every test query.", "",
              "Sources: [NEAR paper](https://arxiv.org/html/2610.13314v1), "
              "[EmbeddingGemma2](https://huggingface.co/google/embeddinggemma-2), "
              "[LiteRT checkpoint](https://huggingface.co/litert-community/embeddinggemma-2-text-vision-440m-litert-lm), "
              "[SciFact mirror](https://huggingface.co/datasets/mteb/scifact).", ""]
    selection_path = folder / "selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        lines += ["## Development result", "",
                  f"The best positive-weight NEAR candidate scored {selection['dev_ndcg@10']*100:.2f} nDCG@10 on development, "
                  f"versus {selection['dev_baseline']['ndcg@10']*100:.2f} for the baseline. "
                  "The published 30-setting grid contains only positive residual weights. The baseline would win a development-only choice between the two methods. "
                  "The development baseline is nearly saturated; this limits how informative that split is for selecting a correction.", ""]
    example_path = folder / "examples.json"
    if example_path.exists():
        examples = json.loads(example_path.read_text())
        lines += ["## Queries that changed most", "",
                  "These examples are selected after evaluation for explanation, not parameter tuning. A missing rank means the judged document was outside the top 100.", ""]
        for label, key in [("Largest gains", "largest_gains"), ("Largest losses", "largest_losses")]:
            lines += [f"### {label}", ""]
            for item in examples[key][:2]:
                lines += [f"**Query {item['query_id']}:** {item['query']}", "",
                          f"nDCG@10 change: {item['ndcg_delta']*100:+.2f} points.", ""]
                for doc in item["relevant_documents"]:
                    before = item["baseline_relevant_ranks_top100"].get(doc["id"], ">100")
                    after = item["near_relevant_ranks_top100"].get(doc["id"], ">100")
                    lines += [f"- Relevant document {doc['id']}: rank **{before} → {after}**. {doc['title']}"]
                lines += [""]
    (folder / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
