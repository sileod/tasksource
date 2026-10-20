"""Extractive question answering catalog (``list_tasks(extractive=True)``).

Every task is SQuAD-formatted for ``AutoModelForQuestionAnswering``: ``context``, ``question`` and
``answers`` = ``{'text': [...], 'answer_start': [...]}``, character offsets checked against the context
(see ``preprocess.ExtractiveQA``). Empty answers mean the context does not answer the question.
Sources have human-written questions and human-selected answers (MuSiQue and DROP: human-written,
located verbatim in the passage); distantly supervised sets (TriviaQA, SearchQA) are left out.
"""
import re

from datasets import Value

from .preprocess import ExtractiveQA

# Hub revisions; script-based datasets are read from their parquet conversion commit.
REVISIONS = {
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
