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
    'ibm-research/duorc': '2de12436b8945030c283bf4af83925d60efe10c6',
    'stanfordnlp/concurrentqa': '2cc9db602b10a561026c905403a641422f066369',
    'WorkInTheDark/FairytaleQA': 'fad16faf261459cc55d6d516b6fb3e68c0ddd5a6',
    'rajpurkar/squad_v2': '3ffb306f725f7d2ce8394bc1873b24868140c412',
    'UCLNLP/adversarial_qa': 'c2d5f738db1ad21a4126a144dfbb00cb51e0a4a9',
    'mrqa-workshop/mrqa': 'f3178d9888471dfb2b67c93de14f0ddf499a8d9f',
    'allenai/quoref': '41fed5bc359f81c2f28259e6572cd1f9aab26e29',
    'bdsaglam/musique': '4413770150c3e120306e7cff6e1ec6364a6ae006',
    'ucinlp/drop': '95cda593fae71b60b5b19f82de3fcf3298c1239c',
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
def _span_records(dataset, records):
    from datasets import DatasetDict, Features, List
    features = Features({'context': Value('string'), 'question': Value('string'),
        'answers': {'text': List(Value('string')), 'answer_start': List(Value('int32'))},
        'metadata': Value('string')})
    def flatten(batch):
        rows = [out for values in zip(*batch.values())
                for out in records(dict(zip(batch, values)))]
        return {key: [r[key] for r in rows] for key in features}
    return DatasetDict({s: rows.map(flatten, batched=True, batch_size=16,
        remove_columns=rows.column_names, features=features) for s, rows in dataset.items()})


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
