"""Verify published evidence and site links without model, NumPy, or network access.

This checks consistency of recorded results; it does not independently rerun
retrieval or recompute relevance judgments. Full metrics need SciFact qrels.
"""
import hashlib
import json
import math
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, unquote

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/scifact_eg2'


def read(name):
    return json.loads((RESULTS / name).read_text())


def close(a, b, tolerance=1e-10):
    assert math.isclose(a, b, abs_tol=tolerance, rel_tol=tolerance), (a, b)


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.references = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            assert attrs['id'] not in self.ids, ('duplicate id', attrs['id'])
            self.ids.add(attrs['id'])
        for name in ('href', 'src'):
            if name in attrs:
                self.references.append(attrs[name])


def main():
    result, per_query, trace = read('results.json'), read('per_query.json'), read('query_785_trace.json')
    assert len(per_query) == result['test_queries'] == 300
    for metric, comparison in result['comparison'].items():
        baseline = [row['baseline'][metric] for row in per_query.values()]
        near = [row['near'][metric] for row in per_query.values()]
        delta = [b - a for a, b in zip(baseline, near)]
        close(sum(baseline)/len(baseline), comparison['baseline'])
        close(sum(near)/len(near), comparison['near'])
        close(sum(delta)/len(delta), comparison['delta'])
        close(comparison['delta']*100, comparison['delta_percentage_points'])
        assert sum(d > 1e-12 for d in delta) == comparison['improved_queries']
        assert sum(d < -1e-12 for d in delta) == comparison['worsened_queries']
        assert sum(abs(d) <= 1e-12 for d in delta) == comparison['unchanged_queries']
    gains = {q: row['near']['recall@10']-row['baseline']['recall@10'] for q, row in per_query.items()}
    best = max(gains.values())
    assert sorted(q for q, d in gains.items() if math.isclose(d, best)) == sorted(trace['tied_max_recall10_gain_query_ids'])
    for method in ('baseline', 'near'):
        runs = defaultdict(list)
        for line in (RESULTS / f'{method}.trec').read_text().splitlines():
            qid, _, docid, rank, score, _ = line.split()
            runs[qid].append((int(rank), docid, float(score)))
        assert set(runs) == set(per_query)
        assert all(len(row) == 100 and [x[0] for x in row] == list(range(1, 101)) for row in runs.values())
        for document in trace['documents']:
            rank = next(x[0] for x in runs[trace['query_id']] if x[1] == document['document_id'])
            assert rank == document[f'{method}_rank']
    for name, digest in result['provenance']['code_sha256'].items():
        assert hashlib.sha256((ROOT / 'near_retrieval' / name).read_bytes()).hexdigest() == digest, name
    assert hashlib.sha256((ROOT / 'configs/scifact.json').read_bytes()).hexdigest() == result['provenance']['config_sha256']
    for document in trace['documents']:
        calculated = document['baseline_cosine']/trace['tau_original'] + trace['alpha']*document['residual_cosine']/trace['tau_normalized_residual']
        close(calculated, document['near_score'], 1e-5)
    page_data = json.loads((ROOT / 'docs/data/experiment.json').read_text())
    assert page_data['comparison'] == result['comparison']
    assert page_data['query']['documents'] == trace['documents']
    assert page_data['selected'] == result['selected']
    page = Page()
    page.feed((ROOT / 'docs/index.html').read_text())
    for reference in page.references:
        url = urlsplit(reference)
        if url.scheme or url.netloc:
            continue
        if url.path:
            assert (ROOT / 'docs' / unquote(url.path)).is_file(), reference
        elif url.fragment:
            assert url.fragment in page.ids, reference
    print('PASS: 300 per-query records, aggregate deltas, both TREC runs, maximum recall ties, query ranks, score arithmetic, original code hashes, site data, and local links.')


if __name__ == '__main__':
    main()
