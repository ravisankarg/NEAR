# NEAR, under the microscope

[Read the interactive research blog](https://ravisankarg.github.io/NEAR/) · [Measured report](results/scifact_eg2/REPORT.md) · [Query 785 trace](results/scifact_eg2/query_785_trace.json)

An independent implementation of [NEAR: Neighborhood-Aware Residuals for Memory Retrieval](https://arxiv.org/html/2610.13314v1), with a reproducible frozen-encoder comparison on **SciFact** from BEIR.

The requested model is **EmbeddingGemma2**, not EmbeddingGemma-300M. The local experiment uses its INT4 QAT text backbone through the 440M text/vision LiteRT package. Only text inputs are evaluated; the text backbone has 270M parameters and emits 768-dimensional vectors. [Checkpoint details](https://huggingface.co/litert-community/embeddinggemma-2-text-vision-440m-litert-lm).

The completed run is in [results/scifact_eg2/REPORT.md](results/scifact_eg2/REPORT.md), with a chart, raw metrics, paired confidence intervals, per-query changes, and TREC run files.

| Held-out metric (×100) | Cosine baseline | NEAR | Change (points) |
|---|---:|---:|---:|
| nDCG@10 | 83.86 | 84.39 | +0.53 |
| Recall@10 | 91.93 | 92.20 | +0.27 |
| Recall@100 | 99.00 | 98.33 | -0.67 |
| MRR@10 | 81.73 | 82.52 | +0.79 |

**The top-10 gain is small and inconclusive:** the nDCG change's paired 95% interval is **[-0.33, +1.39] points**. The selected L=1, normalized-residual, alpha=0.25 configuration doubles vector-key storage and lowers Recall@100. On development, baseline nDCG@10 was 98.67 versus 98.54 for the best NEAR candidate. This experiment does not establish that NEAR is a better deployment choice for this encoder and dataset.

## What is implemented

`near_retrieval/near.py` contains the corpus-only preprocessing and query transform:

1. Normalize the frozen encoder's vectors and find each document's nearest *other* documents.
2. Average and normalize those neighbors to get a reference direction. Remove the document's projection onto it.
3. Build a small routing table using spherical k-means on the current document states. Each entry stores the mean document reference for its cluster.
4. Route each query through that table and remove its corresponding projection. Repeat with normalized residual states for deeper layers.
5. Concatenate the original and residual views into fixed document keys and query vectors. Their inner product equals the paper's fused score.

For unit input state `h` and unit reference `c`, the residual is `r = h - (h·c)c`. The matching vector is either `r` or `unit(r)`, while the next layer always receives `unit(r)`.

For depth `L`, the score is `s0/tau0 + alpha/L * sum(sl/taul)`. Each `tau` is computed from corpus population covariance, including self-similarity. **Concatenated keys are not L2-normalized.** Normalizing them would change this score.

The implementation uses blockwise exact neighbor search and exact final retrieval for this small corpus. This is the paper's scoring method with exact search, rather than its FAISS HNSW approximation. An optional `--hnsw` audit uses the published HNSW settings when FAISS is installed; that branch is not part of the default measured run.

## Frozen experiment

The complete protocol is in [configs/scifact.json](configs/scifact.json).

| Item | Setting |
|---|---|
| Dataset | Full SciFact corpus, 5,183 documents |
| Selection | 807 of 809 official training queries, used only to select hyperparameters; two exact test-text duplicates excluded |
| Final evaluation | All 300 official test queries |
| Encoder | Identical cached EmbeddingGemma2 LiteRT vectors for both methods |
| Input | Original titles and abstracts; model-card search/document formatting |
| Length | 8,192-token limit; overflow raises an error |
| Main metric | nDCG@10, linear relevance gains |
| Other metrics | Uncapped Recall@5/10/100, capped Recall@10, MRR@10, Hit@10 |
| NEAR grid | L=1/2/3, raw/normalized residuals, alpha=0.25/0.5/1/2/4 |
| Fixed NEAR parameters | k=50, A≤16, R≤4, beta=10, seed=42 |
| Clustering detail | Spherical k-means++, 3 restarts, at most 100 iterations |
| Uncertainty | 10,000 paired query bootstrap samples, fixed selected configuration |

This is a **transfer experiment on a standard retrieval benchmark**, not a reproduction of the paper's eight LMEB tasks. The document bank is shared between selection and test, as in standard SciFact retrieval. Queries are held out; related claims can share relevant documents. No test metric selects parameters, and no encoder training occurs. The pretrained encoder's exposure to benchmark material is unknown.

A text-only split audit, before any retrieval scoring, found that train IDs `871` and `1291` duplicate test IDs `870` and `1292`. The two training queries are excluded from tuning. All 300 test queries remain. The audit is recorded in `split_audit.json`; it uses held-out query IDs/texts, not their relevance judgments.

## Reproduce the benchmark

```bash
git clone https://github.com/ravisankarg/NEAR.git
cd NEAR
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_scifact.py --out data/scifact
# Obtain the LiteRT checkpoint linked above, then:
NEAR_MODEL=/path/to/embeddinggemma-2-text-vision-440m.litertlm bash scripts/run_benchmark.sh
```

The public runner writes new results to `results/rerun_scifact_eg2`, preserving the measured record. Override `NEAR_PYTHON`, `NEAR_DATA`, `NEAR_EMBEDDINGS`, or `NEAR_RESULTS` as needed. The original run used Python 3.10.22, NumPy 2.2.6, datasets 5.0.0, and litert-lm-api 0.18.0 on CPU with eight encoder threads. It used cached MTEB Arrow files exported by `prepare_scifact` in `near_retrieval/data.py`. The fresh-download helper is supplied for reruns and was not used in that measured experiment.

The model and dataset are not bundled. The measured model SHA-256 is `92dcbea108899e5d6e30d919b0744f90d9967e80c67a4ab5503ac16d54f62eb0`. Compare downloaded model and source hashes against `results.json`; different exports or models can produce different results. `--revision` pins the download helper to a dataset commit when available.

Encoding is checkpointed after each batch. It refuses to reuse an embedding cache if the model, source text, encoding code, or configuration changes. Use a fresh output directory for a different model or protocol. A full rerun requires the compatible native LiteRT runtime; the published scores can be inspected without it.

To use the full-precision checkpoint instead, install `requirements-optional.txt` and use a runtime that supports `google/embeddinggemma-2`:

```bash
python -m near_retrieval.encode --backend sentence-transformers \
  --model google/embeddinggemma-2 --family embeddinggemma2 \
  --data data/scifact --out artifacts/scifact_eg2_full_precision \
  --max-length 8192 --dimension 768 --device cpu
```

That adapter follows the official model-card prefixes, float32 precision, and text-only configuration. It is provided for reruns and has not been validated in this restricted environment. Results from different encoder backends must be reported separately.

## Files and validation

- `near_retrieval/near.py`: NEAR math, routing tables, index save/load.
- `near_retrieval/encode.py`: frozen encoders with resumable, hashed caches.
- `near_retrieval/benchmark.py`: 30-candidate development sweep, frozen selection, test comparison, bootstrap, timings, and exports.
- `near_retrieval/metrics.py`: retrieval metrics, including the distinction between capped and ordinary recall.
- `tests/test_near.py`: numerical invariants, degeneracy handling, serialization, and hand-calculated metrics.
- `results/scifact_eg2/`: reviewable report and machine-readable evidence after completion.

```bash
OPENBLAS_NUM_THREADS=2 python \
  -m unittest discover -s tests -v
```

The benchmark also checks concatenated versus explicit scores on the actual test embeddings and cross-checks every query's nDCG@10 and Recall@10/100 against `pytrec_eval` when available. Model and dataset hashes, package versions, code hashes, and the selected configuration accompany the results.

Primary sources: [NEAR](https://arxiv.org/html/2610.13314v1), [EmbeddingGemma2](https://huggingface.co/google/embeddinggemma-2), [LiteRT checkpoint](https://huggingface.co/litert-community/embeddinggemma-2-text-vision-440m-litert-lm), [SciFact mirror](https://huggingface.co/datasets/mteb/scifact), [BEIR](https://github.com/beir-cellar/beir).

## Use the index in another retriever

```python
import numpy as np
from near_retrieval.near import NearIndex

# X: corpus embeddings; Q: query embeddings from the same frozen encoder.
index = NearIndex().fit(X)
# Choose these settings on development queries, as benchmark.py does.
settings = dict(depth=1, normalized=False, alpha=0.5)
keys = index.memory_keys(**settings)        # build once per corpus
query_vectors = index.query_keys(Q, **settings)
scores = query_vectors @ keys.T            # or one inner-product index search
top10 = np.argsort(-scores, axis=1)[:, :10]
```

Changing the corpus requires rebuilding its neighborhoods, reference tables, and score scales. This implementation does not claim an incremental-update procedure.

## Mathematics and interactive blog

The [blog](https://ravisankarg.github.io/NEAR/) explains the shared-reference cosine identity, the exact condition for a relevant document to overtake a competitor, the general separate-reference expansion, and why more residual layers need not improve separation. Its toy vectors are explicitly illustrative. The query slider recomputes a measured document pair from saved scores; it does not rerun full retrieval at each alpha.

```bash
# Standard library only: consistency checks, not a fresh retrieval evaluation.
python scripts/verify_published_results.py
# Node only: numerical checks for the explanatory labs.
node tests/test_blog.mjs
# Serve the buildless blog.
python3 -m http.server 8000 --directory docs
```

The site is served from `docs/` by `.github/workflows/pages.yml`. It uses local assets, no analytics, no external fonts, and no model calls in the browser. Original benchmark core files retain the code hashes recorded in the experiment. Public provenance copies replace two machine-specific absolute paths with descriptive values and omit the full third-party abstract from the query trace; `publication_manifest.json` records original and published file hashes. Numerical values, document IDs/titles, model hashes, and code hashes are unchanged.

The authors of the NEAR paper, the encoder, and SciFact are credited in the report and blog. This is an independent implementation and analysis. Third-party research and dataset text remain under their respective terms.
