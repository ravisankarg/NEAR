"""Standard-library consistency audit of the frozen one-hop experiment.

This verifies recorded evidence, not a new retrieval or metric evaluation.
The measured run independently cross-checked all methods with pytrec_eval.
"""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"results/scifact_one_hop"


def read(name):
    return json.loads((OUT/name).read_text())


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.links = set(), []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            assert attrs["id"] not in self.ids, attrs["id"]
            self.ids.add(attrs["id"])
        self.links.extend(attrs[key] for key in ("src", "href") if key in attrs)


def main():
    result, per, lock = read("results.json"), read("per_query.json"), read("selection.json")
    site = json.loads((ROOT/"docs/data/one-hop.json").read_text())
    assert len(per) == result["test_queries"] == 300
    assert lock["candidate_count"] == len(read("dev_grid.json")) == 120
    assert lock["deployment_choice"]["settings"]["family"] == "baseline"
    close(lock["baseline"]["recall@10"], 1)
    assert hashlib.sha256((OUT/"selection.json").read_bytes()).hexdigest() == result["validation"]["frozen_selection_sha256"]
    for path, expected in lock["provenance"].items():
        if path.startswith("neighbor_retrieval/") or path.startswith("configs/"):
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected, path
    for family, row in result["methods"].items():
        for metric, expected in row["metrics"].items():
            actual = sum(v[family][metric] for v in per.values())/len(per)
            close(actual, expected)
        for metric, comparison in row.get("comparison", {}).items():
            delta = [v[family][metric]-v["baseline"][metric] for v in per.values()]
            close(sum(delta)/len(delta)*100, comparison["delta_pp"])
            assert sum(d>1e-12 for d in delta) == comparison["improved"]
            assert sum(d<-1e-12 for d in delta) == comparison["worsened"]
            assert sum(abs(d)<=1e-12 for d in delta) == comparison["unchanged"]
        assert site["methods"][family] == row
    prior = json.loads((ROOT/"results/scifact_eg2/per_query.json").read_text())
    for q, row in per.items():
        assert row["baseline"] == prior[q]["baseline"]
        assert row["near"] == prior[q]["near"]
    runs = {}
    for family in result["methods"]:
        by_query = defaultdict(dict)
        with (OUT/(family+".trec")).open() as f:
            for line in f:
                q, _, doc, pos, score, tag = line.split()
                assert doc not in by_query[q]
                assert tag == family and math.isfinite(float(score))
                by_query[q][doc] = int(pos)
        assert set(by_query) == set(per)
        for row in by_query.values():
            assert sorted(row.values()) == list(range(1, 101))
        runs[family] = by_query
    with (OUT/"graph_edges.csv").open() as f:
        edges = list(csv.DictReader(f))
    assert len(edges) == 5183
    assert len({r["source_id"] for r in edges}) == len(edges)
    indegree = Counter(r["neighbor_id"] for r in edges)
    for edge in edges:
        assert edge["source_id"] != edge["neighbor_id"]
        assert int(edge["neighbor_indegree"]) == indegree[edge["neighbor_id"]]
        assert -1.00001 <= float(edge["cosine"]) <= 1.00001
    assert len(indegree) == result["graph"]["destinations_with_votes"]
    assert max(indegree.values()) == result["graph"]["max_indegree"]
    candidates = read("candidate_audit_per_query.json")
    for n, rows in candidates.items():
        assert len(rows) == 300
        for row in rows:
            assert int(n) <= row["union_size"] <= 2*int(n)
            assert row["union_recall"] >= row["seed_recall"]
        for key, value in result["candidate_audit"][n].items():
            if key == "queries_with_new_relevant":
                assert sum(r["new_relevant_documents"]>0 for r in rows) == value
            else:
                close(sum(r[key] for r in rows)/300, value)
    audit = read("mechanism_audit.json")
    assert audit["posthoc"] and not audit["used_for_selection"]
    for trace in audit["traces"]:
        q = trace["query_id"]
        weight = trace["settings"]["weight"]
        for d in trace["documents"]:
            if d["baseline_rank"] <= 100:
                assert runs["baseline"][q][d["id"]] == d["baseline_rank"]
            if d["diffusion_rank"] <= 100:
                assert runs["diffusion"][q][d["id"]] == d["diffusion_rank"]
            close(d["mixed_score"], (1-weight)*d["direct_mass"]+weight*d["neighbor_mass"])
    assert site["examples"] == audit["traces"]
    assert site["candidate_audit"] == result["candidate_audit"]
    assert site["selection"] == lock["deployment_choice"]
    page = Page()
    page.feed((ROOT/"docs/one-hop.html").read_text())
    for link in page.links:
        url = urlsplit(link)
        if url.scheme or url.netloc:
            continue
        if url.path:
            assert (ROOT/"docs"/url.path).is_file(), link
        elif url.fragment:
            assert url.fragment in page.ids, link
    print("PASS: all 6 methods, 300 queries, 5,183 graph edges, frozen selection, traces, candidate budgets, and page assets agree")


if __name__ == "__main__":
    main()
