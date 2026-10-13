"""Independent NEAR implementation, equations 1–4 and Appendix C.

Only corpus vectors enter fit(). Labels and query vectors cannot affect the
neighborhoods, anchors, reference tables, or bank-specific score scales.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def validate_vectors(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or not x.shape[0] or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty, finite 2D embedding array")
    if np.any(np.linalg.norm(x, axis=1) < 1e-12):
        raise ValueError("Original embeddings must be nonzero")
    return unit(x)


def score_scale(x):
    """sqrt(mean_i Var_j[x_i dot x_j]), population variance including self.

    E_i[x_i^T Cov(x_j) x_i] = trace(E[xx^T] Cov(x)). O(d²) storage.
    Float64 moments avoid cancellation when vectors are almost identical.
    """
    x = np.asarray(x, dtype=np.float64)
    moment = (x.T @ x) / len(x)
    mean = x.mean(axis=0)
    covariance = moment - np.outer(mean, mean)
    return max(1e-6, float(np.sqrt(max(0.0, np.sum(moment * covariance)))))


def local_references(states, k=50, block_size=256):
    n = len(states)
    if n < 2 or k < 1:
        raise ValueError("NEAR requires at least two memories and k >= 1")
    refs = np.empty_like(states)
    neighbors = np.empty((n, min(k, n-1)), dtype=np.int64)
    for start in range(0, n, block_size):
        end = min(n, start + block_size)
        scores = states[start:end] @ states.T
        scores[np.arange(end-start), np.arange(start, end)] = -np.inf
        # Stable ordering resolves exact ties by original corpus row index.
        ix = np.argsort(-scores, axis=1, kind="stable")[:, :min(k, n-1)]
        neighbors[start:end] = ix
        refs[start:end] = unit(states[ix].mean(axis=1))
    return refs, neighbors


def spherical_kmeans(x, n_clusters, seed=42, max_iter=100, n_init=3):
    """Cosine assignment + normalized centroid update; deterministic restarts.

    The paper does not specify initialization or stopping details. We use
    spherical k-means++ initialization, then retain highest cosine objective.
    Empty clusters keep a direction but get no member-reference contribution.
    """
    n_clusters = min(n_clusters, len(x))
    if n_clusters < 1 or max_iter < 1 or n_init < 1:
        raise ValueError("Positive clustering settings required")
    best = None
    for run in range(n_init):
        rng = np.random.default_rng(seed + run)
        chosen = [int(rng.integers(len(x)))]
        distance = np.maximum(0.0, 1.0 - x @ x[chosen[0]])
        while len(chosen) < n_clusters:
            distance[chosen] = 0
            if float(distance.sum()) <= 1e-12:
                candidate = next(i for i in range(len(x)) if i not in chosen)
            else:
                candidate = int(rng.choice(len(x), p=distance / distance.sum()))
            chosen.append(candidate)
            distance = np.minimum(distance, np.maximum(0.0, 1.0 - x @ x[candidate]))
        centers = x[chosen].copy()
        previous = None
        for iteration in range(max_iter):
            assignment = np.argmax(x @ centers.T, axis=1)
            if previous is not None and np.array_equal(previous, assignment):
                break
            previous = assignment.copy()
            for j in range(n_clusters):
                members = x[assignment == j]
                if len(members):
                    centers[j] = unit(members.mean(axis=0))
        assignment = np.argmax(x @ centers.T, axis=1)
        objective = float(np.sum(x * centers[assignment]))
        if best is None or objective > best[0]:
            best = (objective, centers.copy(), assignment.copy(), iteration + 1)
    return best[1], best[2], {"objective": best[0], "iterations": best[3], "restarts": n_init}


@dataclass(frozen=True)
class NearConfig:
    k: int = 50
    anchors: int = 16
    routing: int = 4
    beta: float = 10.0
    max_depth: int = 3
    seed: int = 42
    kmeans_iterations: int = 100
    kmeans_restarts: int = 3


class NearIndex:
    def __init__(self, config=NearConfig()):
        if config.max_depth < 1 or config.routing < 1 or config.beta < 0:
            raise ValueError("Invalid NEAR settings")
        self.config = config

    def fit(self, corpus):
        self.original = validate_vectors(corpus)
        self.layers = []
        self.tau0 = score_scale(self.original)
        state = self.original
        for layer in range(self.config.max_depth):
            refs, neighbors = local_references(state, self.config.k)
            anchors, assignment, km_info = spherical_kmeans(
                state, self.config.anchors, seed=self.config.seed + layer,
                max_iter=self.config.kmeans_iterations, n_init=self.config.kmeans_restarts)
            # Drop empty table entries: their reference mean is undefined.
            occupied = np.unique(assignment)
            values = np.stack([unit(refs[assignment == h].mean(axis=0)) for h in occupied])
            residual = state - np.sum(state * refs, axis=1, keepdims=True) * refs
            normalized = unit(residual)
            self.layers.append({
                "anchors": anchors[occupied], "values": values, "residual": residual,
                "normalized": normalized, "refs": refs, "neighbors": neighbors,
                "tau_raw": score_scale(residual), "tau_normalized": score_scale(normalized),
                "kmeans": km_info,
            })
            state = normalized
        return self

    def query_views(self, queries, depth=None, normalized=False):
        depth = len(self.layers) if depth is None else depth
        self._validate_selection(depth, 0)
        state = validate_vectors(queries)
        if state.shape[1] != self.original.shape[1]:
            raise ValueError("Query/corpus dimension mismatch")
        views = [state]
        for layer in self.layers[:depth]:
            similarities = state @ layer["anchors"].T
            ix = np.argsort(-similarities, axis=1, kind="stable")[:, :self.config.routing]
            logits = self.config.beta * np.take_along_axis(similarities, ix, axis=1)
            weights = np.exp(logits - logits.max(axis=1, keepdims=True))
            weights /= weights.sum(axis=1, keepdims=True)
            reference = unit(np.sum(weights[:, :, None] * layer["values"][ix], axis=1))
            residual = state - np.sum(state * reference, axis=1, keepdims=True) * reference
            state = unit(residual)
            views.append(state if normalized else residual)
        return views

    def _validate_selection(self, depth, alpha):
        if not 1 <= depth <= len(self.layers) or not np.isfinite(alpha) or alpha < 0:
            raise ValueError("Invalid depth or residual weight")

    def memory_keys(self, depth=1, normalized=False, alpha=1.0):
        self._validate_selection(depth, alpha)
        name = "normalized" if normalized else "residual"
        weight = np.sqrt(alpha / depth)
        # DO NOT normalize concatenated keys: that would change Eq. 3.
        return np.ascontiguousarray(np.concatenate(
            [self.original] + [weight * l[name] for l in self.layers[:depth]], axis=1), dtype=np.float32)

    def query_keys(self, queries, depth=1, normalized=False, alpha=1.0):
        self._validate_selection(depth, alpha)
        views = self.query_views(queries, depth, normalized)
        tau_name = "tau_normalized" if normalized else "tau_raw"
        weight = np.sqrt(alpha / depth)
        return np.ascontiguousarray(np.concatenate(
            [views[0] / self.tau0] + [weight * v / l[tau_name]
                for v, l in zip(views[1:], self.layers[:depth])], axis=1), dtype=np.float32)

    def score_components(self, queries, normalized=False):
        views = self.query_views(queries, normalized=normalized)
        name, tau_name = ("normalized", "tau_normalized") if normalized else ("residual", "tau_raw")
        return [views[0] @ self.original.T / self.tau0] + [
            q @ layer[name].T / layer[tau_name] for q, layer in zip(views[1:], self.layers)]

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {"original": self.original}
        metadata = {"config": asdict(self.config), "tau0": self.tau0, "layers": []}
        for i, layer in enumerate(self.layers):
            metadata["layers"].append({k: v for k, v in layer.items() if not isinstance(v, np.ndarray)})
            arrays.update({f"layer{i}_{k}": v for k, v in layer.items() if isinstance(v, np.ndarray)})
        np.savez_compressed(directory / "index.npz", **arrays)
        (directory / "index.json").write_text(json.dumps(metadata, indent=2) + "\n")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        meta = json.loads((directory / "index.json").read_text())
        result = cls(NearConfig(**meta["config"]))
        result.tau0 = meta["tau0"]
        with np.load(directory / "index.npz", allow_pickle=False) as arrays:
            result.original = arrays["original"]
            result.layers = []
            for i, values in enumerate(meta["layers"]):
                prefix = f"layer{i}_"
                values.update({k[len(prefix):]: arrays[k] for k in arrays.files if k.startswith(prefix)})
                result.layers.append(values)
        return result
