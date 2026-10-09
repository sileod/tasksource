"""Text-only Mind2Web preparation: native actions and DOM candidates, no screenshots."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

from datasets import Dataset, DatasetDict, Features, Sequence, Value
from huggingface_hub import snapshot_download

SOURCE = 'osunlp/Mind2Web'
REVISION = '17ece8eb89862368edc0cc806acee6fca5163474'


def page_html(action):
    # Remove indentation between tags; retain all native DOM content and node IDs.
    return re.sub(r'>\s+<', '><', action['cleaned_html']).strip()


def page_key(action):
    return hashlib.sha256(page_html(action).encode()).hexdigest()


def trajectory_splits(trajectories, seed=42):
    """80/10/10 hash partitions of trajectories connected by identical DOM snapshots."""
    parents, pages = {}, {}
    def root(key):
        parents.setdefault(key, key)
        if parents[key] != key:
            parents[key] = root(parents[key])
        return parents[key]
    for trajectory in trajectories:
        identity = trajectory['annotation_id']
        root(identity)
        for action in trajectory['actions']:
            key = page_key(action)
            if key in pages:
                a, b = root(identity), root(pages[key])
                parents[max(a, b)] = min(a, b)
            pages[key] = identity
    groups = {identity: root(identity) for identity in parents}
    splits = {}
    for identity, group in groups.items():
        value = int(hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest(), 16) % 100
        splits[identity] = 'train' if value < 80 else 'validation' if value < 90 else 'test'
    return splits, groups


def mind2web_record(trajectory, index, elements=False, seed=42):
    """Gold flags/arguments stay out of inputs; alternative positives are never negatives."""
    action = trajectory['actions'][index]
    operation = action['operation']['op']
    if operation not in ('CLICK', 'TYPE', 'SELECT'):
        return None, 'unsupported_operation'
    html = page_html(action)
    if not html or not trajectory['confirmed_task'].strip():
        return None, 'missing_context'
    history = trajectory['action_reprs'][:index]
    inputs = (f"Task: {trajectory['confirmed_task']}\nPrevious actions:\n" + '\n'.join(history)
              + '\nDOM:\n' + html)
    identity = f"{trajectory['annotation_id']}:{action['action_uid']}"
    metadata = dict(id=identity, source_row=identity, trajectory_id=trajectory['annotation_id'],
        action_uid=action['action_uid'], action_index=index, website=trajectory['website'],
        domain=trajectory['domain'], source_dataset=SOURCE, source_revision=REVISION,
        source_split='train', input_view='dom', page_group_id=page_key(action))
    if not elements:
        return dict(inputs=inputs, labels=operation, metadata=json.dumps(metadata, sort_keys=True)), None
    positives = action['pos_candidates']
    original = [candidate for candidate in positives if candidate.get('is_original_target') is True]
    if len(original) != 1:
        return None, 'ambiguous_or_missing_original_target'
    positive_ids = {str(candidate['backend_node_id']) for candidate in positives}
    nodes = set(re.findall(r'backend_node_id=["\']([^"\']+)["\']', html))
    def describe(candidate):
        node = str(candidate['backend_node_id'])
        if node not in nodes:
            return None
        attributes = candidate['attributes']
        attributes = json.loads(attributes) if isinstance(attributes, str) else attributes
        visible = {key: attributes[key] for key in
                   ('aria_label', 'aria-label', 'placeholder', 'title', 'role', 'type', 'id', 'text')
                   if attributes.get(key)}
        return node, json.dumps(dict(node=node, tag=candidate['tag'], attributes=visible),
                                ensure_ascii=False, sort_keys=True)
    gold = describe(original[0])
    if gold is None:
        return None, 'target_missing_from_dom'
    negatives = {}
    for candidate in action['neg_candidates']:
        if str(candidate['backend_node_id']) not in positive_ids:
            option = describe(candidate)
            if option:
                negatives[option[0]] = option
    # Four native candidates match the normal text MC template; never invent negatives.
    if len(negatives) < 3:
        return None, 'fewer_than_three_native_negatives'
    rng = random.Random(f'{seed}:{identity}')
    choices = [gold] + rng.sample(sorted(negatives.values()), 3)
    rng.shuffle(choices)
    metadata.update(candidate_ids=[node for node, _ in choices], target_id=gold[0],
                    candidate_sampling_seed=seed, native_negative_count=len(negatives))
    return dict(inputs=inputs + f'\nOperation: {operation}', choices_list=[text for _, text in choices],
                labels=choices.index(gold), metadata=json.dumps(metadata, sort_keys=True)), None


def mind2web_dom(seed=42):
    """Mind2Web public training trajectories as compact HTML-only actions and native MCQs.

    Dataset annotations are CC BY 4.0: https://huggingface.co/datasets/osunlp/Mind2Web
    No official benchmark test files are downloaded or republished. Derived 80/10/10
    train/validation/test partitions group whole trajectories and identical cleaned DOM
    snapshots. They are internal holdouts, not the official Mind2Web benchmark splits.
    Only past actions enter context. Four-way MC uses the original target and three
    deterministically sampled native negatives, excluding all alternative positives.
    """
    root = Path(snapshot_download(SOURCE, repo_type='dataset', revision=REVISION,
                                 allow_patterns=['data/train/*.json']))
    paths = sorted(root.glob('data/train/*.json'))
    def trajectories():
        for path in paths:
            yield from json.loads(path.read_text())
    splits, groups = trajectory_splits(trajectories(), seed)
    output = Path('build/mind2web-dom-release')
    output.mkdir(parents=True, exist_ok=True)
    features = Features(dict(inputs=Value('string'), labels=Value('string'), metadata=Value('string')))
    result, counts, exclusions = {}, Counter(), []
    for view in ('action', 'dom-element'):
        schema = features.copy()
        if view == 'dom-element':
            schema.update(labels=Value('int64'), choices_list=Sequence(Value('string')))
        def rows():
            for trajectory in trajectories():
                for index in range(len(trajectory['actions'])):
                    record, reason = mind2web_record(trajectory, index, view == 'dom-element', seed)
                    split = splits[trajectory['annotation_id']]
                    if reason:
                        counts[f'{view}/{split}/excluded/{reason}'] += 1
                        exclusions.append(dict(view=view, trajectory_id=trajectory['annotation_id'],
                            action_uid=trajectory['actions'][index]['action_uid'], reason=reason))
                        continue
                    metadata = json.loads(record['metadata'])
                    metadata.update(split_group_id=groups[trajectory['annotation_id']],
                                    split_policy='public-train-connected-trajectories-80/10/10-v1', split_seed=seed)
                    record['metadata'] = json.dumps(metadata, sort_keys=True)
                    record['_split'] = split
                    counts[f'{view}/{split}/included'] += 1
                    yield record
        full_schema = Features({**schema, '_split': Value('string')})
        data = Dataset.from_generator(rows, features=full_schema, cache_dir=str(output / 'cache'))
        result[view] = DatasetDict({split: data.filter(lambda row: row['_split'] == split).remove_columns('_split')
                                  for split in ('train', 'validation', 'test')})
        for split, dataset in result[view].items():
            dataset.to_parquet(str(output / f'{view}-{split}.parquet'))
    report = dict(source=SOURCE, revision=REVISION, seed=seed, counts=dict(counts),
        split_policy='public-train-connected-trajectories-80/10/10-v1',
        trajectory_splits=splits, trajectory_groups=groups,
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        conversion_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        license='cc-by-4.0', official_benchmark_tests_consumed=False)
    (output / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'excluded-questions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in exclusions))
    return result
