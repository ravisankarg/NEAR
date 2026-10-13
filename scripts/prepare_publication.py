"""Stage the reviewed source and evidence, excluding model/data/cache files.

This does not publish anything. Original benchmark files remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit('Use a new empty staging directory.')
    out.mkdir(parents=True, exist_ok=True)
    names = ['README.md', '.gitignore', '.gitattributes', 'LICENSE', 'NOTICE.md', 'requirements.txt', 'requirements-optional.txt']
    files = [root / name for name in names]
    for folder in ['near_retrieval', 'tests', 'scripts', 'configs', 'docs', '.github', 'results/scifact_eg2']:
        files += [p for p in (root / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'index' not in p.parts and p.suffix not in {'.pyc', '.faiss'}]
    # Cache-recovery and the historical local runner are environment-specific.
    files = [p for p in files if p.name not in {'recover_cached_wheels.py', 'run_local.sh'}]
    manifest = {'purpose': 'Public source and measured evidence; machine-specific paths redacted only where recorded below.', 'files': {}}
    for source in sorted(files):
        relative = str(source.relative_to(root))
        raw = source.read_bytes()
        published = raw
        redacted = []
        if relative == 'results/scifact_eg2/results.json':
            data = json.loads(raw)
            data['model']['configuration']['model'] = Path(data['model']['configuration']['model']).name
            published = (json.dumps(data, indent=2, sort_keys=True) + '\n').encode()
            redacted = ['model.configuration.model: absolute local path replaced by model filename']
        elif relative == 'results/scifact_eg2/runtime_provenance.json':
            data = json.loads(raw)
            data['cached_wheel_path'] = 'local pip HTTP cache (machine-specific path omitted)'
            published = (json.dumps(data, indent=2, sort_keys=True) + '\n').encode()
            redacted = ['cached_wheel_path: local path omitted; wheel hash retained']
        elif relative == 'results/scifact_eg2/query_785_trace.json':
            data = json.loads(raw)
            data['relevant_document'].pop('text', None)
            data['relevant_document']['source'] = 'SciFact corpus, document 12471115; full third-party abstract omitted from public copy'
            published = (json.dumps(data, indent=2, sort_keys=True) + '\n').encode()
            redacted = ['relevant_document.text: full third-party abstract omitted; document ID, title, scores and ranks retained']
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(published)
        manifest['files'][relative] = {'original_sha256': digest(raw), 'published_sha256': digest(published), 'redactions': redacted}
    (out / 'publication_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Staged {len(files)} files at {out}. Model, corpus, embeddings, index, vendor cache, and local previews excluded.')


if __name__ == '__main__':
    main()
