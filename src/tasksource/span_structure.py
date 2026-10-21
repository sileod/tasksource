"""Procedural span views of human English structural annotations.

All offsets refer to the displayed text. Token corpora use tokens joined with a single space;
UD uses its native sentence text and rejects sentences that cannot be aligned exactly.
"""
import json
import re
import zlib
from collections import Counter

# Subtypes (e.g. nsubj:pass, obl:agent) retain their base UD relation.
UD_FIELDS = {'nsubj': 'subject', 'obj': 'object', 'iobj': 'indirect object',
             'obl': 'oblique modifier', 'ccomp': 'clausal complement', 'xcomp': 'open clausal complement'}


def token_offsets(tokens):
    text = ' '.join(tokens)
    starts, pos = [], 0
    for token in tokens:
        starts.append(pos)
        pos += len(token) + 1
    return text, starts


def token_span(tokens, starts, begin, end):
    if not 0 <= begin < end <= len(tokens):
        raise ValueError('Invalid token span')
    return starts[begin], starts[end - 1] + len(tokens[end - 1])


def unique_anchor(text, start, end):
    """Unique literal occurrence, including overlapping occurrences (conservative)."""
    anchor = text[start:end]
    return bool(anchor.strip()) and [m.start() for m in re.finditer(f'(?={re.escape(anchor)})', text)] == [start]


def cap_null_groups(dataset, max_null_share=0.4):
    """Per split, keep at most ``max_null_share`` of sentence groups (``metadata['group']``) whose every field is
    null, choosing them by ``crc32(group)``; groups with any span are all kept. Mirrors keep every sentence; this
    task-level cap stops always-asked fields (negation cues of sentences without negation) from earning mostly
    nulls."""
    def capped(split):
        groups = [json.loads(m)['group'] for m in split['metadata']]
        positive = {g for g, a in zip(groups, split['answers']) if a['text']}
        nulls = sorted(set(groups) - positive, key=lambda g: zlib.crc32(g.encode()))
        keep = positive | set(nulls[:int(max_null_share / (1 - max_null_share) * len(positive))])
        return split.select([i for i, g in enumerate(groups) if g in keep])
    return type(dataset)({name: capped(split) for name, split in dataset.items()})


def record(text, question, spans, field, group, **metadata):
    spans = sorted(set(spans))
    if any(not 0 <= a < b <= len(text) or not text[a:b].strip() for a, b in spans):
        raise ValueError('Invalid character span')
    return dict(context=text, question=question,
        answers={'text': [text[a:b] for a, b in spans], 'answer_start': [a for a, b in spans]},
        metadata=json.dumps(dict(field=field, group=group, type='list', **metadata), ensure_ascii=False))


def structure_spans(dataset, mapper):
    """Flatten structural records through the existing shared QA mapper, with split drop counts."""
    from .extractive_tasks import _span_records
    def convert(rows, split):
        counts = Counter()
        def tracked(row):
            counts['input_rows'] += 1
            outputs = list(mapper(row, counts))
            counts['output_fields'] += len(outputs)
            counts['null_fields'] += sum(not r['answers']['text'] for r in outputs)
            yield from outputs
        result = _span_records(type(dataset)({split: rows}), tracked, load_from_cache_file=False)[split]
        print(f'{mapper.__name__} {split}: {json.dumps(dict(counts), sort_keys=True)}', flush=True)
        return result
    return type(dataset)({split: convert(rows, split) for split, rows in dataset.items()})


def ud_records(row, counts):
    tokens, text = row['tokens'], row['text']
    starts, pos = [], 0
    for token in tokens:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if not token or not text.startswith(token, pos):
            counts['unaligned_sentences'] += 1
            return
        starts.append(pos)
        pos += len(token)
    if text[pos:].strip():
        counts['unaligned_sentences'] += 1
        return
    heads = [int(h) - 1 for h in row['head']]
    if len(heads) != len(tokens) or any(h < -1 or h >= len(tokens) for h in heads):
        raise ValueError('Invalid UD heads')
    children = {i: [] for i in range(len(tokens))}
    for i, h in enumerate(heads):
        if h >= 0:
            children[h].append(i)
    relations = [r.split(':', 1)[0] for r in row['deprel']]
    candidates = [i for i in children if any(relations[c] in UD_FIELDS for c in children[i])
                  and unique_anchor(text, starts[i], starts[i] + len(tokens[i]))]
    if not candidates:
        counts['no_unambiguous_head'] += 1
        return
    # One head per sentence, verbs first, then earliest position; six fields fit the default cap.
    head = min(candidates, key=lambda i: (row['upos'][i] != 'VERB', i))
    # The relations the head has, plus one it lacks (an earned null), picked by sentence id.
    present = [r for r in UD_FIELDS if any(relations[c] == r for c in children[head])]
    absent = [r for r in UD_FIELDS if r not in present]
    asked = present + absent[zlib.crc32(row['sent_id'].encode()) % len(absent):][:1] if absent else present
    for relation, name in ((r, UD_FIELDS[r]) for r in UD_FIELDS if r in asked):
        spans, bad = [], False
        for child in children[head]:
            if relations[child] != relation:
                continue
            visited, pending = set(), [child]
            while pending:
                i = pending.pop()
                if i in visited:
                    raise ValueError('Cycle in UD tree')
                visited.add(i)
                pending.extend(children[i])
            ordered = sorted(visited)
            if ordered != list(range(ordered[0], ordered[-1] + 1)):
                counts['nonprojective_subtrees'] += 1
                bad = True
            else:
                spans.append((starts[ordered[0]], starts[ordered[-1]] + len(tokens[ordered[-1]])))
        if bad:  # A partial answer would misrepresent "every"; never turn a rejected positive into null.
            counts['nonprojective_fields_dropped'] += 1
            continue
        yield record(text, f"Every {name} of {tokens[head]!r}, including each dependent's full subtree, as written.",
                     spans, name.replace(' ', '_'), row['sent_id'], id=row['sent_id'], head_index=head,
                     relation=relation)


def preco_records(row, counts):
    sentences = row['sentences']
    tokens = [t for sentence in sentences for t in sentence]
    text, starts = token_offsets(tokens)
    offsets, pos = [], 0
    for sentence in sentences:
        offsets.append(pos)
        pos += len(sentence)
    for index, cluster in enumerate(row['mention_clusters']):
        spans = []
        for sent, begin, end in cluster:
            if not 0 <= sent < len(sentences) or not 0 <= begin < end <= len(sentences[sent]):
                raise ValueError('Invalid PreCo mention')
            spans.append(token_span(tokens, starts, offsets[sent] + begin, offsets[sent] + end))
        spans = sorted(set(spans))
        if len(spans) < 2:
            counts['singleton_clusters'] += 1
            continue
        a, b = spans[0]
        if not unique_anchor(text, a, b):
            counts['ambiguous_clusters'] += 1
            continue
        yield record(text, f'Every mention of the entity first mentioned as {text[a:b]!r}, including that first mention.',
                     spans, f'entity_{index}', row['id'], id=f'{row["id"]}:{index}')


def redocred_records(row, counts):
    from .tasks import DOCRED_RELATIONS
    sentences = row['sents']
    tokens = [t for sentence in sentences for t in sentence]
    text, starts = token_offsets(tokens)
    offsets, pos = [], 0
    for sentence in sentences:
        offsets.append(pos)
        pos += len(sentence)
    entities = []
    for mentions in row['vertexSet']:
        spans = []
        for m in mentions:
            sent, (begin, end) = m['sent_id'], m['pos']
            if not 0 <= sent < len(sentences) or not 0 <= begin < end <= len(sentences[sent]):
                raise ValueError('Invalid Re-DocRED mention')
            spans.append(token_span(tokens, starts, offsets[sent] + begin, offsets[sent] + end))
        entities.append(sorted(set(spans)))
    grouped = {}
    for label in row['labels']:
        h, t, relation = label['h'], label['t'], label['r']
        if relation not in DOCRED_RELATIONS:
            raise ValueError(f'Unknown relation: {relation}')
        grouped.setdefault((h, relation), set()).add(t)
    for (head, relation), tails in sorted(grouped.items()):
        # Anchor must belong to this entity alone and have one literal occurrence in the text.
        anchors = [(a, b) for a, b in entities[head] if unique_anchor(text, a, b)
                   and not any((a, b) in spans for i, spans in enumerate(entities) if i != head)]
        if not anchors:
            counts['ambiguous_relation_fields'] += 1
            continue
        a, b = anchors[0]
        # One entity = its earliest annotated mention, rather than all aliases as separate entities.
        spans = [entities[t][0] for t in sorted(tails)]
        name = DOCRED_RELATIONS[relation]
        yield record(text, f'Every entity linked from {text[a:b]!r} by the relation {name!r}. '
                     'Return the first mention of each target entity in document order.', spans,
                     f'entity_{head}_{relation}', row['title'], id=f'{row["title"]}:{head}:{relation}',
                     head=head, relation=relation, tail_entities=sorted(tails))
