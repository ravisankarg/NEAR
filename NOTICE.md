# Attribution and scope

This repository is an independent implementation and experiment by Ravi Sankar Guntur. It is not the authors' official NEAR implementation and is not affiliated with the authors or model providers.

- Method: Han Luo and colleagues, **NEAR: Neighborhood-Aware Residuals for Memory Retrieval**, [arXiv:2610.13314v1](https://arxiv.org/html/2610.13314v1).
- Encoder: [EmbeddingGemma2](https://huggingface.co/google/embeddinggemma-2); measured checkpoint from [LiteRT Community](https://huggingface.co/litert-community/embeddinggemma-2-text-vision-440m-litert-lm).
- Dataset: [SciFact](https://github.com/allenai/scifact), evaluated in its [BEIR](https://github.com/beir-cellar/beir) / [MTEB](https://huggingface.co/datasets/mteb/scifact) retrieval form.

Original benchmark and blog code are provided under the MIT license. Third-party papers, query/document quotations, datasets, model weights, and runtime packages remain subject to their respective terms; this license does not relicense them. Model weights, the full corpus, cached embeddings, and vendor packages are not distributed here.

Examples selected after evaluation illustrate the mechanism; they were not used for hyperparameter selection. The mathematical counterexamples are constructed illustrations, not decoded model dimensions or additional benchmark measurements.
