"""Small shared helpers for procedural MC exports; no model-generated labels."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import tempfile
from urllib.request import urlopen
import zipfile

from datasets import Features, Image, IterableDataset, IterableDatasetDict, Sequence, Value

from ..vision import _materialize


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:8], 'big')


def checkout(repo, revision):
    """Cache and import immutable generator code; dependencies are installed separately."""
    root = Path(os.environ.get('TASKSOURCE_GENERATOR_CACHE',tempfile.gettempdir())) / 'tasksource-generators' / revision
    archive = root.with_suffix('.zip')
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(urlopen(f'https://codeload.github.com/{repo}/zip/{revision}', timeout=120).read())
    if not (root / '.complete').exists():
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                parts = Path(member.filename).parts[1:]
                if not parts or '..' in parts or member.is_dir():continue
                path = root.joinpath(*parts);path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(source.read(member))
        (root / '.complete').write_text(revision)
    return root, hashlib.sha256(archive.read_bytes()).hexdigest()


def verified_choices(gold, donors, verify, seed, max_choices=4):
    """Reject equivalent answers using the current instance's native verifier."""
    if not 2 <= max_choices <= 26:raise ValueError('Expected 2–26 choices')
    if verify(gold) != 1:
        raise ValueError('Native verifier rejects its own gold answer')
    unique = {gold: None}
    donors = list(donors);random.Random(seed).shuffle(donors)
    for answer, identity in donors:
        if answer.strip().casefold() in {a.strip().casefold() for a in unique}:continue
        if verify(answer) == 0:unique[answer] = identity
        if len(unique) == max_choices:break
    if len(unique) < 2:
        raise ValueError('No verifier-rejected same-task distractors')
    choices = list(unique);random.Random(seed).shuffle(choices)
    scores = [verify(answer) for answer in choices]
    if scores != [int(answer == gold) for answer in choices]:
        raise ValueError('Choices do not have exactly one accepted answer')
    return choices, choices.index(gold), dict(donor_ids=[unique[a] for a in choices], verifier_scores=scores)


def write_static(name, splits, report, builder):
    """Group identical encoded images and prevent generated-image split overlap."""
    seen, grouped, excluded = {}, {}, []
    for split in ('test', 'validation', 'train'):
        if split not in splits:continue
        groups = {}
        for row in splits[split]:
            digest = hashlib.sha256(row['image']).hexdigest()
            if digest in seen and seen[digest] != split:
                excluded.append(dict(id=row['id'],reason='image_in_heldout_split'));continue
            seen[digest] = split
            group = groups.setdefault(digest,dict(images=[dict(bytes=row['image'],path=None)],image_group_id=digest,qa=[]))
            group['qa'].append(dict(inputs=row['inputs'],choices_list=row['choices_list'],labels=row['labels'],
                metadata=json.dumps(dict(id=row['id'],**row['metadata']),sort_keys=True,ensure_ascii=False)))
        if groups:grouped[split] = list(groups.values())
    features = Features(dict(images=Sequence(Image(decode=False)),image_group_id=Value('string'),qa=[dict(
        inputs=Value('string'),choices_list=Sequence(Value('string')),labels=Value('int64'),metadata=Value('string'))]))
    dataset = IterableDatasetDict({split:IterableDataset.from_generator(lambda rows:iter(rows),
        gen_kwargs=dict(rows=rows),features=features) for split,rows in grouped.items()})
    result = _materialize(dataset,name,report['source'],report['revision'],excluded,builder,
                          report['license'],report['license_url'])
    path = Path('build') / (name+'-release') / 'provenance.json'
    written = json.loads(path.read_text());written.update(report)
    written['mirror'] = 'tasksource/'+name
    written['scripts_sha256'] = {p.name:file_sha256(p) for p in (Path(__file__),Path(inspect.getsourcefile(builder)))}
    written['parquet_sha256'] = {p.name:file_sha256(p) for p in path.parent.glob('*.parquet')}
    written['image_overlap_excluded'] = len(excluded)
    path.write_text(json.dumps(written,indent=2)+'\n')
    return result
