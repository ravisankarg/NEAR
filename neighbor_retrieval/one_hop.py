"""One outgoing edge per document; no recursive query expansion.

Softmax scores are a modeling distribution, not calibrated relevance probabilities.
All graph statistics use corpus vectors alone, with no queries or relevance labels.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from near_retrieval.near import unit, validate_vectors


def softmax(scores, temperature):
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("Temperature must be positive and finite")
    scores = np.asarray(scores, dtype=np.float64)
    weights = np.exp((scores - scores.max(axis=-1, keepdims=True)) / temperature)
    return weights / weights.sum(axis=-1, keepdims=True)


@dataclass
class OneHopGraph:
    neighbor: np.ndarray
    similarity: np.ndarray
    indegree: np.ndarray

    @classmethod
    def fit(cls, vectors, block_size=256):
        x = validate_vectors(vectors)
        if len(x) < 2:
            raise ValueError("At least two documents are required")
        neighbor = np.empty(len(x), dtype=np.int64)
        similarity = np.empty(len(x), dtype=np.float32)
        for start in range(0, len(x), block_size):
            end = min(start + block_size, len(x))
            scores = x[start:end] @ x.T
            scores[np.arange(end-start), np.arange(start, end)] = -np.inf
            # argmax breaks exact ties by original corpus row order.
            nn = np.argmax(scores, axis=1)
            neighbor[start:end] = nn
            similarity[start:end] = scores[np.arange(end-start), nn]
        return cls(neighbor, similarity, np.bincount(neighbor, minlength=len(x)))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, neighbor=self.neighbor, similarity=self.similarity, indegree=self.indegree)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as state:
            return cls(*(state[k] for k in ("neighbor", "similarity", "indegree")))

    def expanded(self, seeds):
        seeds = np.asarray(seeds, dtype=np.int64)
        return np.unique(np.concatenate((seeds, self.neighbor[seeds])))

    def support(self, scores, seeds, temperature, degree_power=0):
        """Normalized weighted votes on the destinations of exactly one hop.

        g_j proportional to sum_{i in seeds: n(i)=j}
          exp(s_i/T) max(c_i,0) / indegree(j)^degree_power.
        With degree_power=1, an equal vote from every corpus document gives a
        constant score for every destination with positive indegree (if c_i equal).
        Zero total positive-edge weight falls back to direct-query softmax.
        """
        scores = np.asarray(scores)
        seeds = np.asarray(seeds, dtype=np.int64)
        if degree_power < 0:
            raise ValueError("Degree power cannot be negative")
        dest = self.neighbor[seeds]
        weights = softmax(scores[seeds], temperature)
        weights *= np.maximum(self.similarity[seeds], 0)
        weights /= np.maximum(self.indegree[dest], 1) ** degree_power
        votes = np.bincount(dest, weights=weights, minlength=len(self.neighbor))
        return votes / votes.sum() if votes.sum() > 0 else softmax(scores, temperature)

    def diffuse(self, scores, seeds, temperature, weight, degree_power=0):
        if not 0 <= weight <= 1:
            raise ValueError("Mixture weight must lie in [0,1]")
        p = softmax(scores, temperature)
        if weight == 0:
            return p
        g = self.support(scores, seeds, temperature, degree_power)
        return (1-weight)*p + weight*g

    def feedback_query(self, query, vectors, scores, seeds, temperature, weight, neighbor_fraction):
        """Convex query feedback; neighbor_fraction=0 is the Rocchio control.

        v_i = (1-eta_i)x_i + eta_i x_n(i), eta_i = fraction*max(c_i,0).
        q_new = unit((1-weight)q + weight*sum_i a_i v_i).
        Only original seeds vote: expanded documents never become seeds.
        """
        if not 0 <= weight <= 1 or not 0 <= neighbor_fraction <= 1:
            raise ValueError("Feedback weights must lie in [0,1]")
        if weight == 0:
            return unit(query)
        a = softmax(np.asarray(scores)[seeds], temperature)
        eta = neighbor_fraction * np.maximum(self.similarity[seeds], 0)
        prototype = ((1-eta)[:, None]*vectors[seeds] + eta[:, None]*vectors[self.neighbor[seeds]])
        adjusted = (1-weight)*query + weight*(a @ prototype)
        return unit(adjusted if np.linalg.norm(adjusted) > 1e-12 else query)


def score_method(graph, queries, corpus, baseline_scores, baseline_order, settings):
    """Score all documents; diffusion top-100 equals ranking its bounded union.

    All configured seed counts are <=100. Documents outside baseline top-100
    and neighbor(seeds) cannot enter diffusion top-100: they receive only the
    same positive rescaling of their original softmax score.
    """
    if settings["family"] == "baseline":
        return baseline_scores
    seeds = baseline_order[:, :settings["seeds"]]
    if settings["family"] in {"diffusion", "degree_diffusion"}:
        return np.stack([graph.diffuse(s, ix, settings["temperature"], settings["weight"],
                                     int(settings["family"] == "degree_diffusion"))
                         for s, ix in zip(baseline_scores, seeds)])
    if settings["family"] not in {"rocchio", "graph_feedback"}:
        raise ValueError("Unknown family")
    updated = np.stack([graph.feedback_query(q, corpus, s, ix, settings["temperature"],
                                           settings["weight"], settings["neighbor_fraction"])
                        for q, s, ix in zip(queries, baseline_scores, seeds)])
    return updated @ corpus.T
