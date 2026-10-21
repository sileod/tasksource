"""Extractive question answering catalog (``list_tasks(extractive=True)``).

Every task is SQuAD-formatted for ``AutoModelForQuestionAnswering``: ``context``, ``question`` and
``answers`` = ``{'text': [...], 'answer_start': [...]}``, character offsets checked against the context
(see ``preprocess.ExtractiveQA``). Empty answers mean the context does not answer the question.
Sources have human-selected answers or procedural QA over human-verified facts (TimeQA).
MuSiQue and DROP answers are located verbatim; unverified distant supervision is left out.
"""
import re

from datasets import Value

from .preprocess import ExtractiveQA
from .metadata.source_revisions import SOURCE_REVISIONS
from .span_structure import cap_null_groups, structure_spans, ud_records, preco_records, redocred_records

# Hub revisions; script-based datasets are read from their parquet conversion commit.
REVISIONS = {
    'google-research-datasets/tydiqa': 'da78f23f9119363459acbaf46bf89426ff26c259',
    'deepset/covid_qa_deepset': 'be51d91ad5d090f207cc864d0497bf800a6716f7',
    'allenai/ropes': 'd59f1e2ee2b423d7c6ba71edd47fceb4158b07dd',
    'alzoubi36/policy_qa': 'edbe3906a01da6c27ccfdd0e3029a3d4601c69f5',
    'hugosousa/TimeQA': '1db7060e4ba5934c25025f107cbf3017e13ef1f6',
    'next-tat/TAT-QA': 'c96247f5077eac447f63527fd3dcfdc58bb56d6a',
    'megagonlabs/subjqa': '60308110d69c81899c19a5297cfc943d615a11d9',

    'MahmoudAbdelWahed/QAConvSQuAD': '53fe038f448caab5b6e6363f82bd9706963a4f69',
    'tasksource/molweni': 'f2d24c023fa066b2972dcf827e648208350ad7f5',
    'tasksource/friendsqa': 'bc4adc08c0bf71b2ce7cbfb36002d3cabf403a6a',
    'ibm-research/duorc': '2de12436b8945030c283bf4af83925d60efe10c6',
    'stanfordnlp/concurrentqa': '2cc9db602b10a561026c905403a641422f066369',
    'WorkInTheDark/FairytaleQA': 'fad16faf261459cc55d6d516b6fb3e68c0ddd5a6',
    'rajpurkar/squad_v2': '3ffb306f725f7d2ce8394bc1873b24868140c412',
    'UCLNLP/adversarial_qa': 'c2d5f738db1ad21a4126a144dfbb00cb51e0a4a9',
    'mrqa-workshop/mrqa': 'f3178d9888471dfb2b67c93de14f0ddf499a8d9f',
    'allenai/quoref': '41fed5bc359f81c2f28259e6572cd1f9aab26e29',
    'bdsaglam/musique': '4413770150c3e120306e7cff6e1ec6364a6ae006',
    'ucinlp/drop': '95cda593fae71b60b5b19f82de3fcf3298c1239c',
    # Script-only dataset; read from its Hub parquet-conversion commit, the 'multidoc2dial'
    # config's own files (the convert branch's merged default config mixes document_domain and
    # dialogue_domain schemas and fails to load).
    'IBM/multidoc2dial': 'd79fa05a7bc89f2fae00b8ac6d82593771b74c96',
    # Author upload (Ildikó Pilán) of the TAB v1.0 standoff JSON.
    'ildpil/text-anonymization-benchmark': '1f22ce098996054689b5b331fb20b3a74a78e1c5',
    # Repackaged by scripts/repackage_dataset/web_extraction.py, not yet published: pin the Hub
    # commit (HfApi().dataset_info(repo).sha) once uploaded.
    'tasksource/websrc-spans': 'ed1886369b0d0516930c11d51cfad3d363caff6c',
    'tasksource/swde': '2a754b98a6b62a3d6d835455f90fec6d95cc8ef9',
    'universal-dependencies/universal_dependencies': 'c3fed78df6b9f8cd27ed24cc7142e9eccf1e6703',
    'coref-data/preco_raw': '02c8ffedb9d0b49168809a572ff553c2f6f9f7ca',
    'tonytan48/Re-DocRED': 'e0ab3489edfe72c968261bffed5243b6fefddd22',

    # data-only mirrors from scripts/repackage_dataset/span_structure.py and sherlock_spans.py
    'tasksource/qasrl-bank-spans': 'cdef9250905c169a1bb8e2a2752dd906988352fa',
    'tasksource/bioscope-spans': '042825641b80ee236d1df7be6e260cbbf573b875',
    'tasksource/sherlock-negation-spans': 'f7e28a5e6776d6ec568521a15d7f62ef7d9bb5f3',
}


def _kwargs(repo):
    return {'revision': REVISIONS[repo]}


squad_v2 = ExtractiveQA(dataset_name='rajpurkar/squad_v2', task_id='squad-v2',
    metadata=lambda r: {'id': r['id'], 'title': r['title']}, load_dataset_kwargs=_kwargs('rajpurkar/squad_v2'))

# The test split's answers are hidden.
adversarial_qa = ExtractiveQA(dataset_name='UCLNLP/adversarial_qa', config_name='adversarialQA',
    task_id='adversarial-qa', splits=('train', 'validation', None),
    metadata=lambda r: {'id': r['id'], 'title': r['title'], 'model_in_the_loop': r['metadata']['model_in_the_loop']},
    load_dataset_kwargs=_kwargs('UCLNLP/adversarial_qa'))


def _mrqa_answers(row):
    """MRQA character spans are end-inclusive; each located occurrence is one gold span."""
    text, starts = [], []
    for spans in row['detected_answers']['char_spans']:
        for start, end in zip(spans['start'], spans['end']):
            text.append(row['context'][start:end + 1]); starts.append(start)
    return {'text': text, 'answer_start': starts}


def _mrqa(subset, task_id):
    # MRQA's test split holds other (out-of-domain) datasets; its validation is this subset's own dev set.
    return ExtractiveQA(dataset_name='mrqa-workshop/mrqa', task_id=task_id, splits=('train', 'validation', None),
        pre_process=lambda d: d.filter(lambda s: s == subset, input_columns='subset'),
        answers=_mrqa_answers, metadata=lambda r: {'id': r['qid'], 'mrqa_subset': r['subset']},
        load_dataset_kwargs=_kwargs('mrqa-workshop/mrqa'))


# Real Google queries over Wikipedia, short answers (MRQA's paragraph-level version).
natural_questions = _mrqa('NaturalQuestionsShort', 'natural-questions')
# Questions written from headlines, answered by other crowdworkers in CNN articles.
newsqa = _mrqa('NewsQA', 'newsqa')
# Multi-hop questions over two Wikipedia paragraphs plus distractors.
hotpotqa = _mrqa('HotpotQA', 'hotpotqa')

# 2.5% of Quoref's answer starts are one character early or late.
quoref = ExtractiveQA(dataset_name='allenai/quoref', task_id='quoref', complete_splits=False, offset_tolerance=1,
    metadata=lambda r: {'id': r['id'], 'title': r['title'], 'url': r['url']},
    load_dataset_kwargs=_kwargs('allenai/quoref'))


def _musique(dataset):
    """All 20 paragraphs (supporting and distractor) as the context; the answer is located in the
    paragraph supporting the last hop. The mirror's default config joins the answerable file with
    MuSiQue-full, whose unanswerable contrast questions (a supporting paragraph removed) become empty
    answers; the answerable copies of MuSiQue-full repeat the answerable file and are dropped."""
    def row(r):
        parts, offset, start = [], 0, -1
        last = r['question_decomposition'][-1]['paragraph_support_idx']
        for p in r['paragraphs']:
            block = f"{p['title']}\n{p['paragraph_text']}"
            if r['answerable'] and p['idx'] == last:
                found = p['paragraph_text'].find(r['answer'])
                start = offset + len(p['title']) + 1 + found if found >= 0 else -1
            parts.append(block); offset += len(block) + 2
        located = start >= 0 or not r['answerable']
        return {'context': '\n\n'.join(parts), 'located': located,
                'answers': {'text': [r['answer']] if start >= 0 else [], 'answer_start': [start] if start >= 0 else []}}
    def first(rows):
        seen = set()
        def keep(id, answerable, question):
            key = (id, answerable, question)
            new = key not in seen
            seen.add(key)
            return new
        return rows.filter(keep, input_columns=['id', 'answerable', 'question'])
    dataset = type(dataset)({split: first(rows) for split, rows in dataset.items()})
    dataset = dataset.map(row, remove_columns=['paragraphs'])
    return dataset.filter(lambda located: located, input_columns='located')


musique = ExtractiveQA(dataset_name='bdsaglam/musique', task_id='musique',
    complete_splits=False, pre_process=_musique, max_mismatch=0.0,
    metadata=lambda r: {'id': r['id'], 'answerable': r['answerable'], 'answer_aliases': r['answer_aliases']},
    load_dataset_kwargs=_kwargs('bdsaglam/musique'))


def _drop_spans(dataset):
    """DROP questions whose gold is one passage span located once, verbatim; number and date answers
    are computed, not extracted, and stay out of this SQuAD-format view. Validation lists every
    annotator's answer: kept when they all give the same span."""
    def row(r):
        spans, types = r['answers_spans']['spans'], r['answers_spans']['types']
        ok = len(set(spans)) == 1 and set(types) == {'span'} and r['passage'].count(spans[0]) == 1
        start = r['passage'].find(spans[0]) if ok else -1
        return {'context': r['passage'], 'answers': {'text': [spans[0]] if ok else [], 'answer_start': [start] if ok else []},
                'single_span': ok}
    dataset = dataset.map(row)
    return dataset.filter(lambda ok: ok, input_columns='single_span')


drop_spans = ExtractiveQA(dataset_name='ucinlp/drop', task_id='drop/span', complete_splits=False,
    pre_process=_drop_spans, metadata=lambda r: {'id': r['query_id'], 'section_id': r['section_id']},
    load_dataset_kwargs=_kwargs('ucinlp/drop'))


def _located_answers(context, texts):
    """Keep verbatim human answers; unmatched answers are never turned into null gold."""
    texts = list(dict.fromkeys(t for t in texts if t and t in context))
    return {'text': texts, 'answer_start': [context.index(t) for t in texts]}


def _duorc_spans(dataset):
    def row(r):
        answers = _located_answers(r['plot'], r['answers'])
        return {'span_answers': answers, 'located': r['no_answer'] or bool(answers['text'])}
    return dataset.map(row).filter(lambda ok: ok, input_columns='located')


def _duorc(config):
    return ExtractiveQA(dataset_name='ibm-research/duorc', config_name=config,
        task_id=f'duorc/{config}', context='plot', answers='span_answers',
        complete_splits=False, pre_process=_duorc_spans,
        metadata=lambda r: {'id': r['question_id'], 'plot_id': r['plot_id'], 'title': r['title'],
                            'no_answer': r['no_answer']},
        load_dataset_kwargs=_kwargs('ibm-research/duorc'))


duorc_self = _duorc('SelfRC')
duorc_paraphrase = _duorc('ParaphraseRC')


def _concurrentqa_spans(dataset):
    def row(r):
        context = '\n\n'.join(p['title'] + '\n' + ' '.join(p['sents']) for p in r['sp'])
        answers = _located_answers(context, [r['answer']])
        return {'context': context, 'answers': answers, 'located': bool(answers['text'])}
    return dataset.map(row).filter(lambda ok: ok, input_columns='located')


concurrentqa = ExtractiveQA(dataset_name='stanfordnlp/concurrentqa', task_id='concurrentqa',
    complete_splits=False, pre_process=_concurrentqa_spans,
    metadata=lambda r: {'id': r['_id'], 'domain': r['domain'], 'type': r['type']},
    load_dataset_kwargs=_kwargs('stanfordnlp/concurrentqa'))


def _fairytaleqa_spans(dataset):
    def row(r):
        # Each reference has its own explicit/implicit judgment; do not borrow an implicit reference.
        texts = [r[a] for a, flag in [('answer1', 'ex-or-im'), ('answer2', 'ex-or-im2')]
                 if r[flag] == 'explicit']
        answers = _located_answers(r['story_section'], texts)
        return {'answers': answers, 'located': bool(answers['text'])}
    return dataset.map(row).filter(lambda ok: ok, input_columns='located')


fairytaleqa = ExtractiveQA(dataset_name='WorkInTheDark/FairytaleQA', task_id='fairytaleqa/explicit',
    context='story_section', complete_splits=False, pre_process=_fairytaleqa_spans,
    metadata=lambda r: {'story_name': r['story_name'], 'attribute': r['attribute'],
                        'local_or_summary': r['local-or-sum']},
    load_dataset_kwargs=_kwargs('WorkInTheDark/FairytaleQA'))


# Additional dialogue, contract and scientific-paper span views. Native splits are preserved.
def _span_records(dataset, records, load_from_cache_file=True):
    from datasets import DatasetDict, Features, List
    features = Features({'context': Value('string'), 'question': Value('string'),
        'answers': {'text': List(Value('string')), 'answer_start': List(Value('int32'))},
        'metadata': Value('string')})
    def flatten(batch):
        rows = [out for values in zip(*batch.values())
                for out in records(dict(zip(batch, values)))]
        return {key: [r[key] for r in rows] for key in features}
    return DatasetDict({s: rows.map(flatten, batched=True, batch_size=16,
        remove_columns=rows.column_names, features=features, load_from_cache_file=load_from_cache_file)
        for s, rows in dataset.items()})


def _span_record(context, question, answers, **metadata):
    import json
    return dict(context=context, question=question, answers=answers,
                metadata=json.dumps(metadata, ensure_ascii=False))


def _quac_spans(row):
    history = []
    context = row['context'].removesuffix('CANNOTANSWER').rstrip()
    for i, question in enumerate(row['questions']):
        text = row['orig_answers']['texts'][i]
        start = row['orig_answers']['answer_starts'][i]
        answers = {'text': [] if text == 'CANNOTANSWER' else [text],
                   'answer_start': [] if text == 'CANNOTANSWER' else [start]}
        # History contains only earlier turns, never the current answer.
        yield _span_record(context, ('Previous:\n' + '\n'.join(history) + '\n' if history else '')
            + 'Question: ' + question, answers, id=row['turn_ids'][i], dialogue_id=row['dialogue_id'])
        history.append(f'Q: {question}\nA: {text}')


quac_spans = ExtractiveQA(dataset_name='allenai/quac', config_name='plain_text', task_id='quac/spans',
    complete_splits=False, pre_process=lambda d: _span_records(d, _quac_spans), metadata='metadata',
    load_dataset_kwargs={'revision': SOURCE_REVISIONS['allenai/quac'], 'name': 'default', 'data_dir': '{config_name}'})


def _contract_windows(context, answers, limit=4000):
    """Half-open windows, preferably ending at paragraph/sentence boundaries.

    Never cut a gold span: move the boundary to its start. A span longer than the
    limit cannot fit, so its intersecting windows are omitted rather than mislabeled null.
    """
    spans = [(s, s + len(t)) for s, t in zip(answers['answer_start'], answers['text'])]
    start = 0
    while start < len(context):
        end = min(start + limit, len(context))
        if end < len(context):
            boundaries = [m.end() + start for m in re.finditer(r'\n\s*\n|[.!?]\s+', context[start:end])]
            if boundaries and boundaries[-1] > start + limit // 2:
                end = boundaries[-1]
            changed = True
            while changed:
                changed = False
                for a, b in spans:
                    if start < a < end < b:
                        end = a
                        changed = True
        if not any(a < end and b > start and not (start <= a and b <= end) for a, b in spans):
            yield start, end
        start = end


def _cuad_contract(rows):
    import hashlib
    all_answers = {'text': [], 'answer_start': []}
    context = rows[0]['context']
    for r in rows:
        all_answers['text'].extend(r['answers']['text'])
        all_answers['answer_start'].extend(r['answers']['answer_start'])
    windows = list(_contract_windows(context, all_answers))
    positive, negative = [], []
    for r in rows:
        # Reject malformed original positives before creating any null windows.
        if any(context[s:s + len(t)] != t for s, t in zip(r['answers']['answer_start'], r['answers']['text'])):
            continue
        for start, end in windows:
            pairs = [(t, s - start) for t, s in zip(r['answers']['text'], r['answers']['answer_start'])
                     if start <= s and s + len(t) <= end]
            out = _span_record(context[start:end], r['question'],
                {'text': [t for t, s in pairs], 'answer_start': [s for t, s in pairs]},
                id=r['id'], title=r['title'], window_start=start, window_end=end)
            (positive if pairs else negative).append(out)
    # Stable hash ordering is independent of Python hash seeds and source row order.
    negative.sort(key=lambda r: hashlib.sha256(('cuad-0:' + r['metadata']).encode()).digest())
    yield from positive
    yield from negative[:len(positive)]


def _cuad_spans(dataset):
    import hashlib
    from datasets import Dataset, DatasetDict
    def generate(rows):
        groups = {}
        for i, r in enumerate(rows):
            key = (r['title'], hashlib.sha256(r['context'].encode()).hexdigest())
            groups.setdefault(key, []).append(i)
        for indexes in groups.values():
            yield from _cuad_contract(list(rows.select(indexes)))
    return DatasetDict({s: Dataset.from_generator(generate, gen_kwargs={'rows': rows})
                        for s, rows in dataset.items()})


cuad_spans = ExtractiveQA(dataset_name='theatticusproject/cuad-qa', task_id='cuad/spans',
    complete_splits=False, pre_process=_cuad_spans, metadata='metadata', max_mismatch=0,
    load_dataset_kwargs={'revision': SOURCE_REVISIONS['theatticusproject/cuad-qa'], 'name': 'default', 'data_dir': 'default'})


# TAB entity types: field name and what to extract (definitions of the TAB annotation guidelines).
TAB_TYPES = {
    'PERSON': ('person', "person's name (full name, surname, initials, nickname or alias)"),
    'CODE': ('code', 'identification number or code (application, case, ID, phone or registration number)'),
    'LOC': ('location', 'place or location (address, city, region, country, named infrastructure)'),
    'ORG': ('organization', 'organization (company, court, public institution, school, hospital, prison, NGO)'),
    'DEM': ('demographic_attribute', 'demographic attribute or role (nationality, ethnicity, religion, age, '
            'profession, family relation)'),
    'DATETIME': ('date_time', 'date, time or duration'),
    'QUANTITY': ('quantity', 'meaningful quantity (amount of money, percentage, count)'),
    'MISC': ('other_personal_information', 'other personal detail (illness, offence, event, vehicle, publication)'),
}


def _tab_masked(annotations):
    """Mentions of one document to mask, agreed by most of its annotators, and the number of
    minority regions left out.

    Masked (DIRECT or QUASI) mentions of all annotators are grouped into regions of overlapping
    spans. A region is masked when more than half of the annotators mask something in it; its
    mentions are the reading (set of spans and types) most of them gave, ties going to the annotator
    listed first. Every kept span is thus one annotator's exact span. With one annotator, this is
    that annotator's masked mentions."""
    n = len(annotations)
    mentions = sorted({(m['start_offset'], m['end_offset'], m['entity_type'], m['identifier_type'], i)
                       for i, a in enumerate(annotations) for m in a['entity_mentions']
                       if m['identifier_type'] in ('DIRECT', 'QUASI')})
    regions, end = [], -1
    for m in mentions:
        if m[0] < end:
            regions[-1].append(m); end = max(end, m[1])
        else:
            regions.append([m]); end = m[1]
    kept, minority = [], 0
    for region in regions:
        readings = {}
        for start, stop, kind, _, i in region:
            readings.setdefault(i, set()).add((start, stop, kind))
        if 2 * len(readings) <= n:
            minority += 1
            continue
        counts = {}
        for i in sorted(readings):
            counts.setdefault(frozenset(readings[i]), []).append(i)
        reading, voters = max(counts.items(), key=lambda item: (len(item[1]), -item[1][0]))
        for start, stop, kind in sorted(reading):
            levels = [level for s, e, k, level, i in region if (s, e, k) == (start, stop, kind) and i in voters]
            kept.append((start, stop, kind, max(sorted(set(levels)), key=levels.count)))
    return kept, minority


def _tab_records(annotations):
    """One record per window of a document and TAB entity type: the mentions of that type to mask."""
    first = annotations[0]
    text, person = first['text'], first['meta']['applicant']
    if any(a['text'] != text for a in annotations):
        raise ValueError(f"TAB {first['doc_id']}: annotators saw different texts")
    masked, minority = _tab_masked(annotations)
    spans = {'text': [text[s:e] for s, e, *_ in masked], 'answer_start': [s for s, *_ in masked]}
    for start, end in _contract_windows(text, spans):
        for kind, (field, description) in TAB_TYPES.items():
            inside = [(s - start, e - start, level) for s, e, k, level in masked
                      if k == kind and start <= s and e <= end]
            yield _span_record(text[start:end],
                f'Every {description} that must be masked to anonymise {person} (a direct or quasi-identifier), as written.',
                {'text': [text[start + s:start + e] for s, e, _ in inside], 'answer_start': [s for s, _, _ in inside]},
                id=f"{first['doc_id']}:{start}:{kind}", field=field, entity_type=kind,
                identifier_types=[level for _, _, level in inside],
                doc_id=first['doc_id'], task=first['task'], annotators=[a['annotator_id'] for a in annotations],
                minority_regions_dropped=minority, window_start=start, window_end=end)


def _tab_pii(dataset):
    from datasets import Dataset, DatasetDict
    def generate(rows):
        documents = {}
        for i, doc in enumerate(rows['doc_id']):
            documents.setdefault(doc, []).append(i)
        for indexes in documents.values():
            yield from _tab_records([rows[i] for i in indexes])
    return DatasetDict({s: Dataset.from_generator(generate, gen_kwargs={'rows': rows})
                        for s, rows in dataset.items()})


# Text Anonymization Benchmark: ECHR judgments in which annotators marked every personal-information
# mention and whether it must be masked to anonymise the applicant (DIRECT and QUASI identifiers are).
# One record per document window and entity type (a list field); several annotators: _tab_masked.
tab_pii = ExtractiveQA(dataset_name='ildpil/text-anonymization-benchmark', task_id='tab/pii',
    complete_splits=False, pre_process=_tab_pii, metadata='metadata', max_mismatch=0,
    load_dataset_kwargs={**_kwargs('ildpil/text-anonymization-benchmark'),
        'data_files': {s: f'echr_{n}.zip' for s, n in [('train', 'train'), ('validation', 'dev'), ('test', 'test')]}})


def _word_bounded(text, start, end):
    """The span neither starts nor ends inside an alphanumeric word."""
    return not (start > 0 and text[start - 1].isalnum() and text[start].isalnum()) and \
        not (end < len(text) and text[end].isalnum() and text[end - 1].isalnum())


PAVE_FIELDS = 8  # fields per extraction row (jev.extraction.MAX_FIELDS)


def _pave_records(line):
    """One WDC PAVE offer (a JSON line): one record per attribute of its category. Values were annotated verbatim in the title (pid 0) or the description (pid 1); each
    occurrence there is an alternative gold span, and the annotated variants of a value are
    alternatives too. "n/a" (no value in the offer) is a native null. Values not found verbatim are
    dropped, and an attribute none of whose values is found is left out, never turned into a null."""
    import json
    row = json.loads(line)
    title, description = row['input_title'], row['input_description']
    prefix = 'Title: '
    context = f'{prefix}{title}\nDescription: {description}'
    offsets = (len(prefix), len(prefix) + len(title) + len('\nDescription: '))
    sources, records = (title, description), []
    for attribute in sorted(row['target_scores']):
        values = row['target_scores'][attribute]
        texts, starts = [], []
        for value, info in ({} if 'n/a' in values else values).items():
            for pid in sorted(set(info['pid'])):
                found = [m.start() for m in re.finditer(re.escape(value), sources[pid])] if value.strip() else []
                # Occurrences inside a longer word ("Light" in "Lightweight") only when there is no other.
                whole = [s for s in found if _word_bounded(sources[pid], s, s + len(value))]
                found = whole or found
                texts += [value] * len(found); starts += [offsets[pid] + s for s in found]
        if 'n/a' not in values and not texts:
            continue
        name = re.sub(r'\W+', '_', attribute.replace('(s)', 's').lower()).strip('_')
        records.append((name, attribute, {'text': texts, 'answer_start': starts}, sorted(v for v in values if v != 'n/a')))
    # Categories have 8 to 20 attributes: dealt round-robin into the fewest groups of at most
    # PAVE_FIELDS, so that one offer's rows have balanced, mixed fields.
    groups = -(-len(records) // PAVE_FIELDS)
    for i, (name, attribute, answers, annotated) in sorted(enumerate(records), key=lambda r: (r[0] % groups, r[0])):
        yield _span_record(context, f'{attribute} of the product, as written in the offer.', answers,
            id=f"{row['id']}:{name}", field=name, group=i % groups, attribute=attribute, offer_id=row['id'],
            cluster_id=row['cluster_id'], category=row['category'], url=row['url'], annotated_values=annotated)


def _pave(dataset):
    return _span_records(dataset, lambda r: _pave_records(r['text']) if r['text'].strip() else [])


# WDC PAVE: product offers (title and description) from the Web Data Commons product corpus, with every
# attribute of the offer's category annotated by hand with its verbatim value(s) or n/a. The authors'
# extraction benchmark (all attributes, values not normalized) with its full training and test splits.
_PAVE = ('https://raw.githubusercontent.com/wbsg-uni-mannheim/wdc-pave/'
         '66a4cfd75087353d05471ae452a1a4a1fc9075af/data/processed_datasets/wdc_with_all_attributes/')
wdc_pave = ExtractiveQA(dataset_name='text', task_id='wdc-pave', complete_splits=False, splits=('train', None, 'test'),
    pre_process=_pave, metadata='metadata', max_mismatch=0,
    load_dataset_kwargs={'data_files': {'train': _PAVE + 'train_large.jsonl', 'test': _PAVE + 'test.jsonl'}})


def _qasper_spans(row):
    context = row['title'] + '\n' + row['abstract'] + '\n' + '\n'.join(
        p for section in row['full_text']['paragraphs'] for p in section)
    for question, identity, annotations in zip(row['qas']['question'], row['qas']['question_id'], row['qas']['answers']):
        # Each annotator is kept separately; multiple extractive spans are all gold.
        for annotation_id, a in zip(annotations['annotation_id'], annotations['answer']):
            texts = a['extractive_spans']
            if a['yes_no'] is not None or a['free_form_answer']:
                continue
            if a['unanswerable']:
                answers = {'text': [], 'answer_start': []}
            elif texts and all(t and t in context for t in texts):
                answers = _located_answers(context, texts)
            else:
                continue
            yield _span_record(context, question, answers, id=identity, paper_id=row['id'], annotation_id=annotation_id)


qasper_spans = ExtractiveQA(dataset_name='allenai/qasper', config_name='qasper', task_id='qasper/spans',
    complete_splits=False, pre_process=lambda d: _span_records(d, _qasper_spans), metadata='metadata', max_mismatch=0,
    load_dataset_kwargs={'revision': SOURCE_REVISIONS['allenai/qasper'], 'name': 'default', 'data_dir': '{config_name}'})


def _verbatim_spans(dataset):
    """Retain exact offsets; uniquely relocate bad starts without changing gold text.

    Unlocatable positive rows are dropped, never converted to native null answers.
    """
    def row(r):
        text, starts = r['answers']['text'], r['answers']['answer_start']
        fixed = []
        for t, start in zip(text, starts):
            if t and start >= 0 and r['context'][start:start + len(t)] == t:
                fixed.append(start)
            elif t and r['context'].count(t) == 1:
                fixed.append(r['context'].index(t))
            else:
                return {'span_answers': {'text': [], 'answer_start': []}, 'located': False}
        return {'span_answers': {'text': text, 'answer_start': fixed}, 'located': len(text) == len(starts)}
    return dataset.map(row).filter(lambda ok: ok, input_columns='located')


tydiqa_english = ExtractiveQA(dataset_name='google-research-datasets/tydiqa', config_name='secondary_task',
    task_id='tydiqa/english', complete_splits=False,
    pre_process=lambda d: _verbatim_spans(d.filter(
        lambda identity: identity.startswith('english-'), input_columns='id')), answers='span_answers', max_mismatch=0,
    metadata=lambda r: {'id': r['id'], 'title': r['title']},
    load_dataset_kwargs=_kwargs('google-research-datasets/tydiqa'))

covidqa = ExtractiveQA(dataset_name='deepset/covid_qa_deepset', task_id='covid-qa',
    complete_splits=False, splits=('train', None, None), pre_process=_verbatim_spans,
    answers='span_answers', max_mismatch=0,
    metadata=lambda r: {'id': r['id'], 'document_id': r['document_id']},
    load_dataset_kwargs=_kwargs('deepset/covid_qa_deepset'))

policyqa = ExtractiveQA(dataset_name='alzoubi36/policy_qa', task_id='policyqa', complete_splits=False,
    metadata=lambda r: {'id': r['id'], 'title': r['title']},
    load_dataset_kwargs=_kwargs('alzoubi36/policy_qa'))


def _ropes_spans(dataset):
    def row(r):
        context = r['background'] + '\n\n' + r['situation']
        answers = _located_answers(context, r['answers']['text'])
        return {'context': context, 'answers': answers, 'located': bool(answers['text'])}
    return dataset.map(row).filter(lambda ok: ok, input_columns='located')


ropes = ExtractiveQA(dataset_name='allenai/ropes', task_id='ropes', complete_splits=False,
    splits=('train', 'validation', None),
    pre_process=_ropes_spans, max_mismatch=0, metadata=lambda r: {'id': r['id']},
    load_dataset_kwargs=_kwargs('allenai/ropes'))


def _subjqa(config):
    return ExtractiveQA(dataset_name='megagonlabs/subjqa', config_name=config,
        task_id=f'subjqa/{config}', complete_splits=False,
        pre_process=_verbatim_spans, answers='span_answers', max_mismatch=0,
        metadata=lambda r: {'id': r['id'], 'review_id': r['review_id'], 'domain': r['domain']},
        load_dataset_kwargs={**_kwargs('megagonlabs/subjqa'), 'name': 'default', 'data_dir': '{config_name}'})


subjqa_books = _subjqa('books')
subjqa_electronics = _subjqa('electronics')
subjqa_grocery = _subjqa('grocery')
subjqa_movies = _subjqa('movies')
subjqa_tripadvisor = _subjqa('tripadvisor')



def _tatqa_spans(row):
    # Text-only, one-span gold: no table lookup, arithmetic or multi-span answers.
    context = '\n\n'.join(p['text'] for p in sorted(row['paragraphs'], key=lambda p: p['order']))
    for q in row['questions']:
        if q.get('answer_type') != 'span' or q.get('answer_from') != 'text':
            continue
        texts = q['answer']
        if not isinstance(texts, list) or len(texts) != 1 or not texts[0] or texts[0] not in context:
            continue
        yield _span_record(context, q['question'], _located_answers(context, texts), id=q['uid'])


tatqa_spans = ExtractiveQA(dataset_name='next-tat/TAT-QA', task_id='tatqa/text-span',
    complete_splits=False, splits=('train', 'validation', None),
    pre_process=lambda d: _span_records(d, _tatqa_spans), metadata='metadata', max_mismatch=0,
    load_dataset_kwargs={**_kwargs('next-tat/TAT-QA'), 'data_files':
        {'train': 'tatqa_dataset_train.json', 'validation': 'tatqa_dataset_dev.json'}})



def _timeqa_spans(dataset):
    def row(r):
        answers = _located_answers(r['context'], r['targets'])
        return {'answers': answers, 'located': bool(answers['text'])}
    return dataset.map(row).filter(lambda ok: ok, input_columns='located')


timeqa = ExtractiveQA(dataset_name='hugosousa/TimeQA', task_id='timeqa', complete_splits=False,
    pre_process=_timeqa_spans, max_mismatch=0,
    metadata=lambda r: {'id': r['idx'], 'level': r['level']},
    load_dataset_kwargs=_kwargs('hugosousa/TimeQA'))


# QAConv (emails, court and NPR/CNN interview transcripts, BC3 threads) in SQuAD format;
# unanswerables (an empty string at offset 0) become empty answers. Media transcripts are
# research-only: non-commercial.
def _qaconv_answers(row):
    pairs = [(t, s) for t, s in zip(row['answers']['text'], row['answers']['answer_start']) if t.strip()]
    return {'text': [t for t, _ in pairs], 'answer_start': [s for _, s in pairs]}


qaconv = ExtractiveQA(dataset_name='MahmoudAbdelWahed/QAConvSQuAD', task_id='qaconv', complete_splits=False,
    answers=_qaconv_answers,
    metadata=lambda r: {'id': r['id'], 'title': r['title']}, load_dataset_kwargs=_kwargs('MahmoudAbdelWahed/QAConvSQuAD'))


# Molweni and FriendsQA have no usable Hub upload: scripts/repackage_dataset/dialogue.py republishes
# Molweni's native SQuAD-format spans and relocates FriendsQA's utterance/token-span answers to
# character offsets in a flattened "Speaker: utterance" context, as tasksource mirrors.
molweni = ExtractiveQA(dataset_name='tasksource/molweni', task_id='molweni', max_mismatch=0, complete_splits=False,
    metadata='metadata', load_dataset_kwargs=_kwargs('tasksource/molweni'))

friendsqa = ExtractiveQA(dataset_name='tasksource/friendsqa', task_id='friendsqa', max_mismatch=0, complete_splits=False,
    metadata='metadata', load_dataset_kwargs=_kwargs('tasksource/friendsqa'))


# IBM/multidoc2dial is a script-only loader; its 'multidoc2dial' config (the paper's grounding
# reading-comprehension task) is read from the Hub's own parquet conversion of that one config,
# already SQuAD-formatted (question = dialogue history joined with [SEP], context = the grounding
# document, answers = the verbatim grounded span). Native train/validation/test kept as published.
def _multidoc2dial_kwargs():
    revision = REVISIONS['IBM/multidoc2dial']
    return {'data_files': {split: f'hf://datasets/IBM/multidoc2dial@{revision}/multidoc2dial/{split}/0000.parquet'
                            for split in ('train', 'validation', 'test')}}


multidoc2dial = ExtractiveQA(dataset_name='parquet', task_id='multidoc2dial', complete_splits=False,
    metadata=lambda r: {'id': r['id'], 'title': r['title'], 'domain': r['domain']},
    load_dataset_kwargs=_multidoc2dial_kwargs())


# Web pages as visible text (scripts/repackage_dataset/web_extraction.py: decoded entities, collapsed
# whitespace, one line per block element, ' | ' between table cells); spans are verified on that text.

def _websrc_page(row):
    """WebSRC span questions of one page (yes/no questions are not spans and stay out). Long pages are
    windowed without cutting a gold span; WebSRC has no unanswerable questions, so a question is asked
    only in the window that holds its answer."""
    context, questions = row['context'], row['questions']
    gold = {'text': [t for q in questions for t in q['answers']['text']],
            'answer_start': [s for q in questions for s in q['answers']['answer_start']]}
    for start, end in _contract_windows(context, gold):
        for q in questions:
            pairs = [(t, s - start) for t, s in zip(q['answers']['text'], q['answers']['answer_start'])
                     if start <= s and s + len(t) <= end]
            if pairs:
                yield _span_record(context[start:end], q['question'],
                    {'text': [t for t, _ in pairs], 'answer_start': [s for _, s in pairs]},
                    id=q['id'], page=row['id'], window_start=start)


# Crowd-written questions over 6.4k web page segments (Chen et al., 2021); native website-disjoint
# train/dev/test, the test answers released by the authors.
websrc_spans = ExtractiveQA(dataset_name='tasksource/websrc-spans', task_id='websrc/spans', complete_splits=False,
    pre_process=lambda d: _span_records(d, _websrc_page), metadata='metadata', max_mismatch=0,
    load_dataset_kwargs=_kwargs('tasksource/websrc-spans'))


# SWDE attributes: (question, type). ``text`` attributes have one value or alternative surface forms
# ("Free Style" / "Free Style (2009)"), every occurrence of which is gold; ``list`` attributes hold
# several items (co-authors, genres, address parts, phone numbers), each listed once.
SWDE_ATTRIBUTES = {
    'auto': {'model': ('What car model is the page about?', 'text'),
             'price': ('What is the price of the car?', 'text'),
             'engine': ('Every engine of the car, as written (each distinct text once).', 'list'),
             'fuel_economy': ('Every fuel economy figure of the car, as written (each distinct text once).', 'list')},
    'book': {'title': ('What is the title of the book?', 'text'),
             'author': ('Every author of the book, as written (each distinct text once).', 'list'),
             'isbn_13': ('What is the ISBN-13 of the book?', 'text'),
             'publisher': ('Who is the publisher of the book?', 'text'),
             'publication_date': ('When was the book published?', 'text')},
    'camera': {'model': ('What camera model is the page about?', 'text'),
               'price': ('What is the price of the camera?', 'text'),
               'manufacturer': ('Who is the manufacturer of the camera?', 'text')},
    'job': {'title': ('What is the title of the job?', 'text'),
            'company': ('Which company is hiring?', 'text'),
            'location': ('Where is the job located?', 'text'),
            'date_posted': ('When was the job posted?', 'text')},
    'movie': {'title': ('What is the title of the movie?', 'text'),
              'director': ('Every director of the movie, as written (each distinct text once).', 'list'),
              'genre': ('Every genre of the movie, as written (each distinct text once).', 'list'),
              'mpaa_rating': ('What is the MPAA rating of the movie?', 'text')},
    'nbaplayer': {'name': ('What is the name of the player?', 'text'),
                  'team': ('Which team does the player play for?', 'text'),
                  'height': ('What is the height of the player?', 'text'),
                  'weight': ('What is the weight of the player?', 'text')},
    'restaurant': {'name': ('What is the name of the restaurant?', 'text'),
                   'address': ('Every part of the restaurant address, as written (each distinct text once).', 'list'),
                   'phone': ('What is the phone number of the restaurant?', 'text'),
                   'cuisine': ('Every cuisine of the restaurant, as written (each distinct text once).', 'list')},
    'university': {'name': ('What is the name of the university?', 'text'),
                   'phone': ('Every phone number of the university, as written (each distinct text once).', 'list'),
                   'website': ('What is the website of the university?', 'text'),
                   'type': ('What type of institution is the university (public or private)?', 'text')},
}


def _swde_page(vertical):
    """One question per attribute of the vertical: located values, or none for a labelled null."""
    def records(row):
        import json
        fields = json.loads(row['fields'])
        for attribute, (question, kind) in SWDE_ATTRIBUTES[vertical].items():
            if attribute not in fields:  # a value that could not be located: not asked, never null
                continue
            occurrences = fields[attribute]
            if kind == 'list':
                spans = [min(found) for found in occurrences]
            else:
                spans = sorted({tuple(o) for found in occurrences for o in found})
            yield _span_record(row['context'], question,
                {'text': [t for _, t in spans], 'answer_start': [s for s, _ in spans]},
                id=f"{row['id']}/{attribute}", field=attribute, type=kind)
    return records


def _swde(vertical):
    return ExtractiveQA(dataset_name='tasksource/swde', config_name=vertical, task_id=f'swde/{vertical}',
        complete_splits=False, pre_process=lambda d: _span_records(d, _swde_page(vertical)),
        metadata='metadata', max_mismatch=0, load_dataset_kwargs=_kwargs('tasksource/swde'))


# Product, book, job, movie, player, restaurant, car and university pages of 80 websites (Hao et al.,
# 2011), attribute values labelled by per-site handcrafted rules; website-disjoint splits by hash.
swde_auto = _swde('auto')
swde_book = _swde('book')
swde_camera = _swde('camera')
swde_job = _swde('job')
swde_movie = _swde('movie')
swde_nbaplayer = _swde('nbaplayer')
swde_restaurant = _swde('restaurant')
swde_university = _swde('university')


# English structural annotations (span_structure): dependency subtrees, coreference clusters, relation targets.


def _ud(config):
    return ExtractiveQA(dataset_name='universal-dependencies/universal_dependencies', config_name=f'en_{config}',
        task_id=f'ud/{config}', complete_splits=False, splits=('train', 'dev', 'test'),
        pre_process=lambda d: structure_spans(d, ud_records), metadata='metadata', max_mismatch=0,
        load_dataset_kwargs=_kwargs('universal-dependencies/universal_dependencies'))


ud_ewt = _ud('ewt')
ud_gum = _ud('gum')

preco_mentions = ExtractiveQA(dataset_name='coref-data/preco_raw', task_id='preco/mentions',
    complete_splits=False, splits=('train', 'validation', None),
    pre_process=lambda d: structure_spans(d, preco_records), metadata='metadata', max_mismatch=0,
    load_dataset_kwargs=_kwargs('coref-data/preco_raw'))

redocred_entities = ExtractiveQA(dataset_name='tonytan48/Re-DocRED', task_id='redocred/entities',
    complete_splits=False, pre_process=lambda d: structure_spans(d, redocred_records),
    metadata='metadata', max_mismatch=0, load_dataset_kwargs=_kwargs('tonytan48/Re-DocRED'))


# QA-SRL Bank 2.0, BioScope and *SEM 2012 ConanDoyle-neg have no usable Hub upload (script-only loaders, missing
# data files or an under-sized test upload): tasksource mirrors, repackaged by the scripts above.

# QA-SRL Bank 2.0 "orig" partition: >=2 independent human worker judgments with a strict valid
# majority, one modal complete reading per predicate question. Native train/validation/test.
qasrl_orig = ExtractiveQA(dataset_name='tasksource/qasrl-bank-spans', task_id='qasrl/orig',
    complete_splits=False, metadata='metadata', max_mismatch=0,
    load_dataset_kwargs=_kwargs('tasksource/qasrl-bank-spans'))

# BioScope public abstracts: no native split, so a deterministic document-disjoint 80/10/10
# split by crc32(DocID). Expert negation/speculation cue and scope annotations.
bioscope_abstracts = ExtractiveQA(dataset_name='tasksource/bioscope-spans', config_name='abstracts',
    task_id='bioscope/abstracts', complete_splits=False, metadata='metadata', pre_process=cap_null_groups, max_mismatch=0,
    load_dataset_kwargs=_kwargs('tasksource/bioscope-spans'))

# BioScope public full papers: only 9 documents, too few for a representative split; train-only,
# matching BigBio (which assigns this corpus no evaluation split either).
bioscope_papers = ExtractiveQA(dataset_name='tasksource/bioscope-spans', config_name='papers',
    task_id='bioscope/papers', complete_splits=False, splits=('train', None, None), pre_process=cap_null_groups,
    metadata='metadata', max_mismatch=0,
    load_dataset_kwargs=_kwargs('tasksource/bioscope-spans'))

# *SEM 2012 ConanDoyle-neg (Conan Doyle "Sherlock" stories): human negation cue and scope gold,
# including sub-token affix cues and discontinuous scopes; native story train/validation/test
# partitions (Baskervilles / Wisteria / Cardboard+Circle).
sherlock_scopes = ExtractiveQA(dataset_name='tasksource/sherlock-negation-spans', task_id='sherlock/scopes',
    complete_splits=False, metadata='metadata', pre_process=cap_null_groups, max_mismatch=0,
    load_dataset_kwargs=_kwargs('tasksource/sherlock-negation-spans'))
