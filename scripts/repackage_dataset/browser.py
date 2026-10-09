"""Prepare archive-only HTML supervision; no browser rendering or synthetic negatives."""
from collections import Counter
import csv
import hashlib
import heapq
from html.parser import HTMLParser
import io
import inspect
import json
from pathlib import Path
import re
import zipfile

from datasets import Dataset, DatasetDict, Features, Sequence, Value
from huggingface_hub import hf_hub_download

from .vision import _materialize

WEB_SRC = 'X-LANCE/WebSRC_v1.0'
WEB_SRC_REVISION = '7aa0bc6efc7ef43f68c192e2091108541acbaf1a'


class NativeDOM(HTMLParser):
    """Read native tid/text evidence without rewriting the original HTML."""
    def __init__(self, source):
        super().__init__(convert_charrefs=True)
        self.stack, self.nodes = [], {}
        self.feed(source)
        if self.stack:
            raise ValueError('Unclosed native HTML')
        for node in self.nodes.values():
            node['text'] = re.sub(r'\s+', ' ', ''.join(node['parts'])).strip()
            del node['parts']

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        identity = attrs.get('tid')
        if identity is None or identity in self.nodes:
            raise ValueError('Missing or duplicate native tid')
        self.nodes[identity] = {'tag': tag, 'parts': [], 'children': []}
        if self.stack:
            self.nodes[self.stack[-1]]['children'].append(identity)
        self.stack.append(identity)
        if tag in {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}:
            self.stack.pop()

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if self.stack and self.nodes[self.stack[-1]]['tag'] == tag:
            self.stack.pop()

    def handle_endtag(self, tag):
        if not self.stack or self.nodes[self.stack[-1]]['tag'] != tag:
            raise ValueError('Unbalanced native HTML')
        self.stack.pop()

    def handle_data(self, data):
        for identity in self.stack:
            self.nodes[identity]['parts'].append(data)


def websrc_record(qa, html, nodes, metadata, max_candidates=128):
    """Native boolean or deepest answer-containing DOM element; validate span evidence."""
    target = qa['element_id']
    metadata = {**metadata, 'id': qa['id'], 'source_answer': qa['answer'],
                'answer_start': int(qa['answer_start']), 'input_view': 'dom'}
    inputs = f"HTML:\n{html}\nQuestion: {qa['question']}"
    if target == '-1':
        answer = qa['answer'].casefold()
        if answer not in ('yes', 'no') or int(qa['answer_start']) != int(answer == 'yes'):
            return None, 'invalid_boolean'
        return ('yesno', {'inputs': inputs, 'labels': answer,
                         'metadata': json.dumps(metadata, sort_keys=True)}), None
    if nodes is None or target not in nodes:
        return None, 'invalid_dom_target'
    answer = qa['answer']
    start = int(qa['answer_start'])
    if not answer.strip() or start < 0 or nodes[target]['text'][start:start+len(answer)] != answer:
        return None, 'span_mismatch'
    # All text-bearing native elements, independent of question and answer.
    ids = [identity for identity, node in nodes.items() if node['text']]
    if not 2 <= len(ids) <= max_candidates or target not in ids:
        return None, 'candidate_count'
    if any(answer in nodes[child]['text'] for child in nodes[target]['children']):
        return None, 'non_deepest_target'
    metadata.update(target_id=target, candidate_ids=ids)
    choices = [f"tid={i} <{nodes[i]['tag']}> {nodes[i]['text']}" for i in ids]
    return ('element', {'inputs': inputs, 'choices_list': choices, 'labels': ids.index(target),
                        'metadata': json.dumps(metadata, sort_keys=True)}), None


def websrc(max_rows=5000, max_rows_eval=100, seed=42):
    """WebSRC HTML-only native yes/no and DOM-element supervision (CC BY 4.0).

    Original HTML is preserved. Native website-level train/dev splits are retained;
    no test labels are consumed or evaluation splits fabricated. Element choices
    are all nonempty native DOM elements, capped at 128 by excluding larger pages.
    Span offsets and deepest-element evidence are checked before inclusion. No
    generated distractors. Deterministic bounded samples cover each eligible source.
    """
    path = hf_hub_download(WEB_SRC, 'WebSRC_v1.0_train+dev.zip', repo_type='dataset',
        revision=WEB_SRC_REVISION, local_dir='build/browser-research/websrc-native')
    pools = {view: {s: [] for s in ('train', 'validation')} for view in ('yesno', 'element')}
    counts, exclusions = Counter(), []
    with zipfile.ZipFile(path) as archive:
        sites = list(csv.DictReader(io.StringIO(archive.read('release/dataset_split.csv').decode())))
        assert len({(s['domain'], s['website']) for s in sites}) == len(sites)
        for site in sites:
            split = {'train': 'train', 'dev': 'validation'}[site['split']]
            prefix = f"release/{site['domain']}/{int(site['website']):02d}"
            rows = csv.DictReader(io.StringIO(archive.read(prefix + '/dataset.csv').decode('utf-8-sig')))
            page_cache = {}
            for qa in rows:
                page = qa['id'][2:9]
                if page not in page_cache:
                    html = archive.read(f'{prefix}/processed_data/{page}.html').decode('utf-8-sig')
                    try:
                        nodes = NativeDOM(html).nodes
                    except ValueError:
                        nodes = None
                    page_cache[page] = html, nodes
                html, nodes = page_cache[page]
                metadata = {'source_dataset': WEB_SRC, 'source_revision': WEB_SRC_REVISION,
                            'website': prefix, 'page_group_id': prefix + '/' + page}
                record, reason = websrc_record(qa, html, nodes, metadata)
                if reason:
                    counts[f'{split}/excluded/{reason}'] += 1
                    exclusions.append({'id': qa['id'], 'split': split, 'reason': reason})
                    continue
                view, row = record
                counts[f'{split}/{view}/eligible'] += 1
                limit = max_rows if split == 'train' else max_rows_eval
                priority = -int.from_bytes(hashlib.sha256(f"{seed}:{qa['id']}".encode()).digest())
                pool = pools[view][split]
                item = (priority, qa['id'], row)
                if len(pool) < limit:
                    heapq.heappush(pool, item)
                elif item > pool[0]:
                    heapq.heapreplace(pool, item)
    result, reports = {}, {}
    for view, splits in pools.items():
        features = Features({'inputs': Value('string'), 'labels': Value('string' if view == 'yesno' else 'int64'),
                             'metadata': Value('string')})
        if view == 'element':
            features['choices_list'] = Sequence(Value('string'))
        data = DatasetDict({split: Dataset.from_list([r for _, _, r in sorted(rows)], features=features)
                            for split, rows in splits.items()})
        result[view] = _materialize(data, 'websrc-' + view, WEB_SRC, WEB_SRC_REVISION, exclusions,
            websrc, 'cc-by-4.0', f'https://huggingface.co/datasets/{WEB_SRC}/blob/{WEB_SRC_REVISION}/README.md')
        reports[view] = json.loads(Path(f'build/websrc-{view}-release/provenance.json').read_text())
        reports[view].update(selection={'seed': seed, 'max_rows': max_rows, 'max_rows_eval': max_rows_eval},
            source_counts=dict(counts), max_candidates=128,
            parser_sha256=hashlib.sha256(inspect.getsource(NativeDOM).encode()).hexdigest(),
            record_sha256=hashlib.sha256(inspect.getsource(websrc_record).encode()).hexdigest())
    output = Path('build/websrc-release')
    output.mkdir(exist_ok=True)
    (output / 'provenance.json').write_text(json.dumps(reports, indent=2) + '\n')
    (output / 'excluded-questions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in exclusions))
    return result
