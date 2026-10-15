"""Content identity: find Hub uploads that mirror text already in the catalog.

Names and links miss re-uploads (``mteb/belebele``, ``someone/imdb-sentiment``). The published
Jev ``full`` config holds sampled rows of every implemented text source, so hashed text segments
from it identify a candidate whose viewer preview reuses those texts, whatever its name.
"""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

JEV = 'tasksource/tasksource-jev-typed-decisions'
MIN_CHARS = 32  # shorter segments (labels, names, numbers) recur across unrelated datasets
PREFIX = re.compile(r'^[\w ()-]{1,30}:\s+')


def segments(text):
    """Normalized hashed text segments: line-level, prompt prefixes removed, truncated."""
    for line in str(text).split('\n'):
        line = re.sub(r'[^\w]', '', PREFIX.sub('', line.strip()).casefold())
        if len(line) >= MIN_CHARS:
            yield int.from_bytes(hashlib.blake2b(line[:200].encode(), digest_size=8).digest(), 'little') >> 1


def row_segments(row):
    values = row.values() if isinstance(row, dict) else row
    out = set()
    for value in values:
        if isinstance(value, str):
            out.update(segments(value))
        elif isinstance(value, (list, dict)):
            out.update(row_segments(value if isinstance(value, dict) else {i: v for i, v in enumerate(value)}))
    return out


def build(output, revision=None):
    """Index state/options segments of Jev full; returns the npz path."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi().dataset_info(JEV, revision=revision)
    files = sorted(s.rfilename for s in info.siblings if s.rfilename.startswith('full/') and s.rfilename.endswith('.parquet'))
    hashes, owners, sources = [], [], {}
    for name in files:
        table = pq.read_table(hf_hub_download(JEV, name, repo_type='dataset', revision=info.sha), columns=['state', 'options', 'source'])
        for state, options, source in zip(*(table.column(c).to_pylist() for c in ('state', 'options', 'source'))):
            found = set(segments(state))
            for option in (json.loads(options) if isinstance(options, str) else options or []):
                found.update(segments(option))
            owner = sources.setdefault(source, len(sources))
            hashes.extend(found); owners.extend([owner] * len(found))
        print('Indexed', name, len(hashes), flush=True)
    hashes, owners = np.array(hashes, dtype=np.int64), np.array(owners, dtype=np.int32)
    order = np.argsort(hashes, kind='stable')
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, hashes=hashes[order], owners=owners[order],
             sources=np.array(list(sources)), meta=np.array(json.dumps(dict(dataset=JEV, revision=info.sha, files=files))))
    return path


class Index:
    def __init__(self, path):
        data = np.load(path)
        self.hashes, self.owners, self.sources = data['hashes'], data['owners'], list(data['sources'])
        self.meta = json.loads(str(data['meta']))

    def owners_of(self, values):
        values = np.array(sorted(values), dtype=np.int64)
        left = np.searchsorted(self.hashes, values, 'left'); right = np.searchsorted(self.hashes, values, 'right')
        found = Counter()
        for a, b in zip(left, right):
            for owner in set(self.owners[a:b].tolist()):
                found[self.sources[owner]] += 1
        return found

    def mirrors(self, rows, min_hits=2):
        """Catalog sources sharing at least ``min_hits`` distinct text segments with ``rows``."""
        values = set().union(*map(row_segments, rows)) if rows else set()
        return {source: hits for source, hits in self.owners_of(values).most_common() if hits >= min_hits}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('build/discovery/index/jev-segments.npz'))
    parser.add_argument('--revision')
    print(build(parser.parse_args().output, parser.parse_args().revision))


def _names(feature):
    if isinstance(feature, list):
        return _names(feature[0]) if feature else None
    if isinstance(feature, dict):
        return feature.get('names') if feature.get('_type') == 'ClassLabel' else _names(feature.get('feature'))


def signature(info):
    """Dataset-level identity from viewer ``/info``, per configuration.

    Re-uploads keep row counts and label names even when the previewed rows were never sampled
    into Jev. Keys pair two split sizes of one configuration, or a split size with that
    configuration's labels; a label vocabulary alone counts only when large. Round counts and tiny
    splits are too common to identify a dataset.
    """
    keys = set()
    for config in (info.get('dataset_info') or {}).values():
        sizes = sorted({n for split in (config.get('splits') or {}).values()
                        if (n := split.get('num_examples') or 0) >= 300 and n % 500})
        keys.update(f'sizes:{a}|{b}' for i, a in enumerate(sizes) for b in sizes[i + 1:])
        for feature in (config.get('features') or {}).values():
            names = _names(feature)
            if names and len(names) >= 2:
                vocabulary = '|'.join(sorted(str(n).casefold() for n in names))
                keys.update(f'sized_labels:{n}:{vocabulary}' for n in sizes)
                if len(names) >= 8:
                    keys.add('labels:' + vocabulary)
    return sorted(keys)


class DatasetIndex:
    """Signature keys of catalog Hub repositories; label sets shared by several sources are generic."""

    def __init__(self, path):
        self.keys = json.loads(Path(path).read_text())
        self.generic = {k for k, v in self.keys.items() if k.startswith('labels:') and len(v) >= 3}  # e.g. shared tag sets

    @staticmethod
    def build(repos, cache, path):
        from concurrent.futures import ThreadPoolExecutor
        from urllib.parse import urlencode
        from .link import cached_json
        def fetch(repo):
            try:
                return repo, signature(cached_json('https://datasets-server.huggingface.co/info?' + urlencode(dict(dataset=repo)), cache)['data'])
            except (OSError, ValueError):
                return repo, []
        keys = {}
        with ThreadPoolExecutor(4) as pool:
            for repo, found in pool.map(fetch, sorted(repos)):
                for key in found:
                    keys.setdefault(key, []).append(repo)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(keys, sort_keys=True))
        return DatasetIndex(path)

    def matches(self, keys):
        """Catalog repos sharing a key; label vocabularies shared by several catalog sources are generic."""
        hits = {}
        for key in keys:
            if key not in self.generic:
                for repo in self.keys.get(key, []):
                    hits.setdefault(repo, []).append(key)
        return hits
