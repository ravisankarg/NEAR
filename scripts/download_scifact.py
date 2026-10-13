"""Download the MTEB SciFact mirror into the benchmark's BEIR-style layout."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    from datasets import load_dataset
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('data/scifact'))
    parser.add_argument('--revision', help='Optional Hugging Face dataset commit')
    args = parser.parse_args()
    expected = ['corpus.jsonl', 'queries.jsonl', 'qrels/train.tsv', 'qrels/test.tsv']
    if any((args.out / f).exists() for f in expected):
        raise SystemExit('Destination contains data. Choose a new --out directory to preserve provenance.')
    manifest = {'source': 'mteb/scifact', 'revision_requested': args.revision, 'files': {}}
    for kind, config, split in [('corpus', 'corpus', 'corpus'), ('queries', 'queries', 'queries'),
                                ('train', 'default', 'train'), ('test', 'default', 'test')]:
        rows = load_dataset('mteb/scifact', config, split=split, revision=args.revision)
        relative = f'{kind}.jsonl' if kind in ('corpus', 'queries') else f'qrels/{kind}.tsv'
        path = args.out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        with temporary.open('w', encoding='utf-8', newline='') as handle:
            if relative.endswith('.jsonl'):
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            else:
                writer = csv.writer(handle, delimiter='\t')
                writer.writerow(['query-id', 'corpus-id', 'score'])
                for row in rows:
                    writer.writerow([str(row['query-id']), str(row['corpus-id']), row['score']])
        temporary.replace(path)
        manifest['files'][relative] = {'rows': len(rows), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        print(f'{relative}: {len(rows)} rows')
    (args.out / 'download_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
