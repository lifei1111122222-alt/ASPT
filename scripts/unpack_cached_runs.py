#!/usr/bin/env python3
"""Verify published checksums and restore cached experiment paths (no API calls)."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def digest(data):
    return hashlib.sha256(data).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dest', type=Path, default=ROOT)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    dest = args.dest.resolve()
    manifest = json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
    for item in manifest['inputs'] + manifest['additional_controls'] + [manifest['recorded_requests']]:
        assert digest((ROOT/item['path']).read_bytes()) == item['sha256'], item['path']
    count = 0
    for archive in manifest['archives']:
        path = ROOT/archive['path']
        assert digest(path.read_bytes()) == archive['sha256'], archive['path']
        expected = {x['path']: x for x in archive['members']}
        with zipfile.ZipFile(path) as z:
            assert len(z.namelist()) == len(expected), archive['path']
            assert set(z.namelist()) == set(expected), archive['path']
            for name in z.namelist():
                rel = PurePosixPath(name)
                if rel.is_absolute() or '..' in rel.parts or '\\' in name or ':' in name:
                    raise ValueError('Unsafe archive member: '+name)
                payload = z.read(name)
                assert digest(payload) == expected[name]['sha256'], name
                target = dest.joinpath(*rel.parts).resolve()
                if dest not in target.parents:
                    raise ValueError('Target outside extraction directory: '+name)
                if not args.verify_only:
                    if target.exists() and target.read_bytes() != payload:
                        raise FileExistsError('Refusing to replace a different local file: '+str(target))
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        target.write_bytes(payload)
                count += 1
        print(('Verified ' if args.verify_only else 'Restored ')+archive['path'])
    print(f'Checksums verified for {count} archived files and all benchmark/control inputs.')

if __name__ == '__main__':
    main()
