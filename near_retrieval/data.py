"""Canonical BEIR data and the cached MTEB SciFact mirror."""
import csv
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def prepare_scifact(cache_root, destination):
    from datasets import Dataset
    source = Path(cache_root)
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": "mteb/scifact", "source_files": {}, "output_files": {}}
    for name, config in [("corpus", "corpus"), ("queries", "queries"),
                         ("train", "default"), ("test", "default")]:
        matches = list((source / config).rglob(f"scifact-{name}.arrow"))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one original {name} arrow file: {matches}")
        src = matches[0]
        rows = Dataset.from_file(str(src))
        manifest["source_files"][name] = {"path": str(src), "sha256": sha256(src), "rows": len(rows)}
        if name in {"corpus", "queries"}:
            out = dest / f"{name}.jsonl"
            with out.open("w") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            out = dest / "qrels" / f"{name}.tsv"
            out.parent.mkdir(exist_ok=True)
            with out.open("w") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(["query-id", "corpus-id", "score"])
                writer.writerows((str(r["query-id"]), str(r["corpus-id"]), r["score"]) for r in rows)
        manifest["output_files"][str(out.relative_to(dest))] = sha256(out)
    write_json(dest / "manifest.json", manifest)
    return manifest


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def read_qrels(path):
    result = {}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            score = float(row["score"])
            if score > 0:
                result.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = score
    return result
