"""Join the released UK-election target annotations to tweets and native splits."""
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import tarfile
from urllib.request import urlopen

from .relevance import text_key, write_mirror

UK_ELECTION = 'https://doi.org/10.6084/m9.figshare.4479563.v1'
UK_FILES = {
    'tweets.tar.gz': (7217933, 'f2edb60f988485688d811ce26044da00'),
    'annotations.tar.gz': (7217924, '4f6cb1f219a3789f90eb0d9060c3d31d'),
    'train_id.txt': (7217930, 'cb012947ed123b198bc8ff63ddcafaa2'),
    'test_id.txt': (7217927, 'a05a040dea03a77873601b9b36d1a645'),
}


def uk_targets(tweet, annotation):
    """Keep native entity/keyword targets and human-added targets, not overall topics."""
    for entity in tweet['entities']:
        label = annotation['items'][str(entity['id'])]
        start, target = entity['offset'], entity['entity']
        reason = ('does_not_apply' if label == 'doesnotapply' else
                  'span_mismatch' if tweet['content'][start:start+len(target)].casefold() != target.casefold() else None)
        yield target, label, dict(target_id=str(entity['id']), offset=start, target_type=entity['type']), reason
    for target, label in dict(annotation['additional_items']).items():
        yield target, label, dict(target_type='human_added'), None


def uk_election():
    """Human target-specific sentiment from the EACL 2017 UK-election corpus.

    Joins original tweet text, final annotations and official train/test IDs.
    Released split IDs omit a constant 500000000000000000; a complete bijection
    to all 4,077 tweet IDs is required before joining. Entity/keyword targets and
    human-added targets are retained; overall topic judgments, doesnotapply and
    invalid character spans are excluded. No synthetic labels or holdouts.
    Annotations: CC BY 4.0; original tweet content remains under Twitter's terms.
    Original distribution: https://doi.org/10.6084/m9.figshare.4479563.v1.
    """
    directory = Path('build/source-archives/uk-election-v1');directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, (identity, md5) in UK_FILES.items():
        path = directory/name
        if not path.exists():path.write_bytes(urlopen(f'https://ndownloader.figshare.com/files/{identity}', timeout=60).read())
        raw = path.read_bytes()
        if hashlib.md5(raw).hexdigest() != md5:raise ValueError(f'UK-election source checksum mismatch: {name}')
        files[name] = raw
    def archive(name):
        with tarfile.open(fileobj=io.BytesIO(files[name])) as tar:
            return {row['tweet_id']: row for member in tar if member.isfile()
                    for row in [json.load(tar.extractfile(member))]}
    tweets, annotations = archive('tweets.tar.gz'), archive('annotations.tar.gz')
    ids = {split: [str(int(i)+500000000000000000) for i in files[split+'_id.txt'].decode().splitlines()]
           for split in ('train', 'test')}
    combined = ids['train']+ids['test']
    if len(combined) != len(set(combined)) or set(combined) != set(tweets) or set(tweets) != set(annotations):
        raise ValueError('UK-election split IDs do not bijectively cover the released tweets')
    splits, excluded = {}, Counter()
    for split, identities in ids.items():
        splits[split] = []
        for identity in identities:
            tweet, annotation = tweets[identity], annotations[identity]
            for index, (target, label, metadata, reason) in enumerate(uk_targets(tweet, annotation)):
                if reason:excluded[reason] += 1;continue
                if label not in ('negative', 'neutral', 'positive'):raise ValueError(f'Unknown UK-election label: {label}')
                splits[split].append(dict(sentence1=tweet['content'], sentence2=target, labels=label,
                    metadata=json.dumps(dict(id=f'{identity}:{index}', tweet_id=identity,
                        split_group_id=text_key(tweet['content']), annotator=annotation['username'], **metadata))))
    report = dict(source_url=UK_ELECTION, revision='figshare-v1', license='cc-by-4.0',
        license_scope='Annotations CC BY 4.0; Twitter retains tweet-content rights',
        split_id_offset=500000000000000000, excluded_by_reason=dict(excluded),
        source_file_sha256={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()},
        conversion_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return write_mirror('uk-election', splits, 'string', report)
