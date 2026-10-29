"""Acquire the pinned Mem-GF dependency and verify every original byte."""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def acquire(output, source=None):
    manifest = json.loads((ROOT / 'configs/reference.json').read_text(encoding='utf-8'))
    snapshot = manifest['author_snapshot']
    output = Path(output)
    for name, info in snapshot['files'].items():
        target = output / name
        if target.exists():
            data = target.read_bytes()
        elif source is not None:
            data = (Path(source) / name).read_bytes()
        else:
            request = urllib.request.Request(info['url'], headers={'User-Agent': 'memgf-reproduction/0.1.0'})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
        if len(data) != info['bytes'] or hashlib.sha256(data).hexdigest() != info['sha256']:
            raise ValueError(f'{name}: source differs from the pinned commit; no file was overwritten')
        if not target.exists():
            output.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    return {'commit': snapshot['commit'], 'verified_files': len(snapshot['files']), 'output': str(output)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'external/memgf')
    parser.add_argument('--source', type=Path, help='copy an existing pinned snapshot instead of downloading')
    args = parser.parse_args()
    print(json.dumps(acquire(args.output, args.source)))
