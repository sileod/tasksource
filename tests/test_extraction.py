import json

import pytest
from datasets import Dataset, DatasetDict

from tasksource.jev.extraction import check, entity_spans, ner_rows, qa_rows, readable
from tasksource.preprocess import ExtractiveQA, answer_spans


def qa(context, question, texts, starts, id="q"):
    return {"context": context, "question": question, "answers": {"text": texts, "answer_start": starts},
            "metadata": json.dumps({"id": id})}


def test_answer_spans_tolerance():
    context = "Paris is in France."
    assert answer_spans(["France"], [12], context)["answer_start"] == [12]
    assert answer_spans(["France"], [11], context) is None
    assert answer_spans(["France"], [11], context, tolerance=1)["answer_start"] == [12]


def test_extractive_qa_drops_mismatches():
    rows = [qa("Paris is in France.", "Where?", ["France"], [12])] * 200 + [qa("Paris.", "What?", ["Lyon"], [0])]
    data = Dataset.from_list([{"c": r["context"], "q": r["question"], "a": r["answers"]} for r in rows])
    task = ExtractiveQA(context="c", query="q", answers="a", dataset_name="x", splits=("train", None, None))
    out = task(DatasetDict(train=data))
    assert sum(map(len, out.values())) == 200 and out["train"][0]["question"] == "Where?"


def test_qa_rows_group_by_context():
    rows = [qa("Paris is in France.", "Where is Paris?", ["France", "France"], [12, 12], "a"),
            qa("Paris is in France.", "Which city?", [], [], "b")]
    [row] = qa_rows(rows, "s", "train", kind="list")
    assert [f["type"] for f in row["fields"]] == ["list", "list"]
    assert row["answers"] == [{"null": False, "values": [{"text": "France", "start": 12, "end": 18}]},
                              {"null": True, "values": []}]
    assert json.loads(row["metadata"])["question_ids"] == ["a", "b"]
    assert len(list(qa_rows(rows, "s", "train", max_fields=1))) == 2


def test_check_rejects_misaligned_span():
    row = {"id": "x", "state": "abc", "fields": [{"name": "a", "type": "text", "instruction": "?"}],
           "answers": [{"null": False, "values": [{"text": "b", "start": 0, "end": 1}]}]}
    with pytest.raises(ValueError):
        check(row)


def test_entity_spans_bio_and_io():
    names = ["O", "B-PER", "I-PER", "B-LOC"]
    text, spans = entity_spans(["John", "Smith", "visited", "Rome"], [1, 2, 0, 3], names)
    assert [text[s:e] for s, e in spans["PER"]] == ["John Smith"]
    assert [text[s:e] for s, e in spans["LOC"]] == ["Rome"]
    text, spans = entity_spans(["New", "York", "Times"], ["location-GPE", "location-GPE", "organization-media/newspaper"], None)
    assert [text[s:e] for s, e in spans["location-GPE"]] == ["New York"]


def test_ner_rows_large_ontology():
    names = ["O"] + [f"B-T{i}" for i in range(20)]
    rows = [{"tokens": ["a", "b"], "labels": [1, 0]}, {"tokens": ["c"], "labels": [0]}]
    [row] = ner_rows(rows, "s", "train", names)  # the row without entities is skipped
    assert sum(not a["null"] for a in row["answers"]) == 1 and len(row["fields"]) == 3


def test_readable():
    assert readable("other-biologything") == "biological entity"
    assert readable("person-artist/author") == "artist or author (person)"
    assert readable("event-other") == "other event"
    assert readable("NORP") == "nationality, religious or political group"
    assert readable("Language") == "programming language"
    assert readable("Keyboard_IP") == "keyboard input"
    assert readable("VIOLATED ON") == "party affected by the violation"
    assert readable("TARGET") == "opinion target"
    assert readable("Peop") == "person"
    assert readable("creative_work") == "creative work"


def test_root_card():
    from scripts.build_jev_extraction import root_card
    card = "---\ntags:\n- jev\n---\n| config | x |\n|---|---|\n| `vision` | images |\n\n" \
           'vision = load_dataset("tasksource/tasksource-jev-typed-decisions", "vision")\n'
    out = root_card(card)
    assert "- clairic" in out and "| `extraction` |" in out and 'extraction = load_dataset' in out
    assert root_card(out) == out


def test_duorc_unlocated_answer_is_not_null_gold():
    from tasksource.extractive_tasks import _duorc_spans
    rows = Dataset.from_list([
        {'plot': 'Alice met Bob.', 'answers': ['Alice', 'a paraphrase'], 'no_answer': False},
        {'plot': 'Alice met Bob.', 'answers': ['a paraphrase'], 'no_answer': False},
        {'plot': 'Alice met Bob.', 'answers': [], 'no_answer': True},
    ])
    out = _duorc_spans(DatasetDict(train=rows))['train']
    assert len(out) == 2
    assert out[0]['span_answers'] == {'text': ['Alice'], 'answer_start': [0]}
    assert out[1]['span_answers'] == {'text': [], 'answer_start': []}


def test_fairytaleqa_only_explicit_verbatim_references():
    from tasksource.extractive_tasks import _fairytaleqa_spans
    rows = Dataset.from_list([
        {'story_section': 'Alice met Bob.', 'answer1': 'Alice', 'ex-or-im': 'implicit',
         'answer2': 'Bob', 'ex-or-im2': 'explicit'},
        {'story_section': 'Alice met Bob.', 'answer1': 'a person', 'ex-or-im': 'explicit',
         'answer2': 'Alice', 'ex-or-im2': 'implicit'},
    ])
    out = _fairytaleqa_spans(DatasetDict(train=rows))['train']
    assert len(out) == 1
    assert out[0]['answers'] == {'text': ['Bob'], 'answer_start': [10]}


def test_finer_ord_drops_null_token_sentence():
    from tasksource.tasks import finer_ord
    rows = Dataset.from_list([
        {'tokens': ['Alice', 'works'], 'tags': [1, 0]},
        {'tokens': ['A', None, 'bank'], 'tags': [0, 0, 0]},
    ])
    out = finer_ord.pre_process(DatasetDict(train=rows, test=rows.select([0])))
    assert len(out['train']) == len(out['test']) == 1
    assert out['train'][0]['tokens'] == ['Alice', 'works']
    assert out['train'].features['tags'].feature.names[1] == 'B-PER'


def test_conll04_entities_only_and_adjacent_mentions():
    from tasksource.tasks import _conll04_bio
    rows = Dataset.from_list([{
        'tokens': ['New', 'York', 'Paris', 'reports'],
        'entities': [{'start': 0, 'end': 2, 'type': 'Loc'},
                     {'start': 2, 'end': 3, 'type': 'Loc'}],
        'relations': [{'head': 0, 'tail': 1, 'type': 'irrelevant'}],
    }])
    out = _conll04_bio(DatasetDict(train=rows))['train']
    names = out.features['ner_tags'].feature.names
    assert [names[tag] for tag in out[0]['ner_tags']] == ['B-Loc', 'I-Loc', 'B-Loc', 'O']
    text, spans = entity_spans(out[0]['tokens'], out[0]['ner_tags'], names)
    assert [text[s:e] for s, e in spans['Loc']] == ['New York', 'Paris']
    bad = rows.map(lambda r: {'entities': r['entities'] + [{'start': 1, 'end': 3, 'type': 'Org'}]})
    with pytest.raises(ValueError, match='overlaps'):
        _conll04_bio(DatasetDict(train=bad))


def test_tweetner_native_files_and_tag_ids():
    from tasksource.tasks import tweetner7, _TWEETNER7_TAGS, _MIT_MOVIE_TRIVIA_TAGS, _MIT_RESTAURANT_TAGS
    assert {split: path.rsplit('/', 1)[1] for split, path in
            tweetner7.load_dataset_kwargs['data_files'].items()} == {
        'train': '2020.train.json', 'validation': '2020.dev.json', 'test': '2020.test.json'}
    assert _TWEETNER7_TAGS[0] == 'B-corporation' and _TWEETNER7_TAGS[14] == 'O'
    assert _MIT_MOVIE_TRIVIA_TAGS[15:17] == ['I-Genre', 'I-Year']
    assert _MIT_RESTAURANT_TAGS[15:17] == ['I-Price', 'I-Cuisine']


def test_character_bio_preserves_subword_boundaries_and_adjacent_entities():
    from tasksource.tasks import _character_bio
    row = _character_bio('antiHER2+ drug', [(4, 8, 'gene'), (8, 9, 'marker'), (10, 14, 'chemical')])
    assert row['tokens'] == ['anti', 'HER2', '+', 'drug']
    assert row['ner_tags'] == ['O', 'B-gene', 'B-marker', 'B-chemical']
    assert row['original_offsets'] == [[0, 4], [4, 8], [8, 9], [10, 14]]
    assert row['dropped_overlapping'] == 0


def test_character_bio_drops_every_member_of_overlaps_and_deduplicates():
    from tasksource.tasks import _character_bio
    row = _character_bio('abc def ghi', [(0, 7, 'outer'), (0, 3, 'inner'), (8, 11, 'ok'), (8, 11, 'ok')])
    assert row['dropped_overlapping'] == 2
    assert row['ner_tags'] == ['O', 'O', 'B-ok']
    with pytest.raises(ValueError, match='Invalid character'):
        _character_bio('abc', [(0, 4, 'bad')])


def test_bigbio_discontinuous_and_overlapping_counts():
    from tasksource.tasks import _bigbio_entity_row
    row = {'document_id': 'doc', 'passages': [
        {'text': ['abc'], 'offsets': [[0, 3]]}, {'text': ['def ghi'], 'offsets': [[4, 11]]}],
        'entities': [
            {'text': ['abc', 'ghi'], 'offsets': [[0, 3], [8, 11]], 'type': 'discontinuous'},
            {'text': ['abc def'], 'offsets': [[0, 7]], 'type': 'outer'},
            {'text': ['def'], 'offsets': [[4, 7]], 'type': 'inner'},
            {'text': ['ghi'], 'offsets': [[8, 11]], 'type': 'ok'}]}
    out = _bigbio_entity_row(row)
    assert out['dropped_discontinuous'] == 1
    assert out['dropped_overlapping'] == 2
    assert out['ner_tags'] == ['O', 'O', 'B-ok']
    assert json.loads(out['metadata'])['document_id'] == 'doc'
    row['entities'][-1]['text'] = ['wrong']
    with pytest.raises(ValueError, match='entity text mismatch'):
        _bigbio_entity_row(row)


def test_buster_excludes_silver_and_preserves_gold_token_rows():
    from tasksource.tasks import _buster_gold
    ds = DatasetDict({f'FOLD_{i}': Dataset.from_list([
        {'tokens': [f'doc{i}', 'company'], 'labels': ['O', 'B-Parties.BUYING_COMPANY']}]) for i in range(1, 6)})
    ds['SILVER'] = Dataset.from_list([{'tokens': ['silver'], 'labels': ['B-fake']}])
    out = _buster_gold(ds)
    assert [r['tokens'][0] for r in out['train']] == ['doc1', 'doc2', 'doc3']
    assert out['validation'][0]['tokens'][0] == 'doc4'
    assert out['test'][0]['tokens'][0] == 'doc5'
    assert 'B-fake' not in out['train'].features['ner_tags'].feature.names


def test_skillspan_independent_layers_preserve_overlapping_annotations():
    from tasksource.tasks import _single_type_bio
    ds = DatasetDict(train=Dataset.from_list([{'tokens': ['SQL', 'skills'],
        'tags_skill': ['B', 'I'], 'tags_knowledge': ['B', 'O']}]))
    skills = _single_type_bio(ds, 'tags_skill', 'skill')
    knowledge = _single_type_bio(ds, 'tags_knowledge', 'knowledge')
    assert skills['train'].features['ner_tags'].feature.int2str(skills['train'][0]['ner_tags']) == ['B-skill', 'I-skill']
    assert knowledge['train'].features['ner_tags'].feature.int2str(knowledge['train'][0]['ner_tags']) == ['B-knowledge', 'O']


def test_entity_spans_rejects_truncated_or_shifted_tag_sequences():
    with pytest.raises(ValueError, match='Token and label counts differ'):
        entity_spans(['a', 'b'], [0, 1, 0], ['O', 'B-chemical'])


def test_scierc_inclusive_document_global_offsets():
    from tasksource.tasks import _scierc_entity_row
    out = _scierc_entity_row({'doc_key': 'd', 'sentences': [['first', '.'], ['second', 'method']],
        'ner': [[[0, 0, 'Task']], [[2, 3, 'Method']]]})
    assert out['ner_tags'] == ['B-Task', 'O', 'B-Method', 'I-Method']
    assert json.loads(out['metadata'])['dropped_overlapping'] == 0


def test_toxic_span_alignment_and_separate_mentions():
    from tasksource.tasks import _toxic_span_row
    row = _toxic_span_row({'text': 'bad awful! ok', 'spans': repr(list(range(9)))})
    assert row['span_aligned']
    assert row['tokens'] == ['bad', 'awful', '!', 'ok']
    assert row['ner_tags'] == ['B-toxic', 'I-toxic', 'O', 'O']
    row = _toxic_span_row({'text': 'bad awful', 'spans': '[0, 1, 2, 4, 5, 6, 7, 8]'})
    assert row['ner_tags'] == ['B-toxic', 'B-toxic']
    assert not _toxic_span_row({'text': 'awful', 'spans': '[1, 2, 3, 4]'})['span_aligned']
    assert _toxic_span_row({'text': 'fine', 'spans': '[]'})['ner_tags'] == ['O']
    with pytest.raises(ValueError, match='Invalid toxic'):
        _toxic_span_row({'text': 'fine', 'spans': '[9]'})


def test_laptop_aspects_use_gold_offsets_without_polarity():
    from tasksource.tasks import _laptop_aspect_row
    row = {'sentenceId': 'a', 'text': 'good battery life', 'aspectTerms': [
        {'term': 'battery life', 'from': '5', 'to': '17', 'polarity': 'positive'}]}
    assert _laptop_aspect_row(row)['ner_tags'] == ['O', 'B-aspect', 'I-aspect']
    whitespace = {'sentenceId': 'b', 'text': 'good battery ', 'aspectTerms': [
        {'term': 'battery ', 'from': '5', 'to': '13'}]}
    assert _laptop_aspect_row(whitespace)['ner_tags'] == ['O', 'B-aspect']
    row['aspectTerms'][0]['from'] = '4'
    with pytest.raises(ValueError, match='aspect text mismatch'):
        _laptop_aspect_row(row)


def test_quac_history_excludes_current_answer_and_retains_null():
    from tasksource.extractive_tasks import _quac_spans
    rows = list(_quac_spans({'context': 'Alice met Bob. CANNOTANSWER',
        'questions': ['Who met Bob?', 'Where?'], 'orig_answers':
        {'texts': ['Alice', 'CANNOTANSWER'], 'answer_starts': [0, 15]},
        'turn_ids': ['a', 'b'], 'dialogue_id': 'd'}))
    assert rows[0]['question'] == 'Question: Who met Bob?'
    assert rows[1]['question'] == 'Previous:\nQ: Who met Bob?\nA: Alice\nQuestion: Where?'
    assert rows[1]['answers'] == {'text': [], 'answer_start': []}
    assert rows[0]['context'] == 'Alice met Bob.'


def test_cuad_windows_preserve_spans_and_balance_per_contract():
    from tasksource.extractive_tasks import _cuad_contract, _contract_windows
    context = 'a. ' * 1400 + 'Alice Bob' + ' z. ' * 1400
    start = context.index('Alice')
    answers = {'text': ['Alice Bob'], 'answer_start': [start]}
    windows = list(_contract_windows(context, answers))
    assert all(0 < b - a <= 4000 for a, b in windows)
    assert sum(a <= start and start + 9 <= b for a, b in windows) == 1
    rows = [{'id': 'a', 'title': 'contract', 'context': context, 'question': 'Who?', 'answers': answers},
            {'id': 'b', 'title': 'contract', 'context': context, 'question': 'Where?',
             'answers': {'text': [], 'answer_start': []}}]
    out = list(_cuad_contract(rows))
    assert out == list(_cuad_contract(list(reversed(rows))))
    assert sum(bool(r['answers']['text']) for r in out) == 1
    assert len(out) == 2
    for r in out:
        for t, s in zip(r['answers']['text'], r['answers']['answer_start']):
            assert r['context'][s:s + len(t)] == t
    # An overlong answer's intersecting chunks must never become false nulls.
    long = {'text': [context], 'answer_start': [0]}
    assert list(_contract_windows(context, long)) == []


def test_ropes_unlocated_positive_dropped_not_null():
    from tasksource.extractive_tasks import _ropes_spans
    d = DatasetDict(train=Dataset.from_list([
        {'background': 'Alice', 'situation': 'Bob', 'question': 'Alice or Eve?', 'answers': {'text': ['Alice']}},
        {'background': 'Alice', 'situation': 'Bob', 'question': 'Alice or Eve?', 'answers': {'text': ['Eve']}},
    ]))
    out = _ropes_spans(d)['train']
    assert len(out) == 1 and out[0]['answers'] == {'text': ['Alice'], 'answer_start': [0]}


def test_qasper_requires_every_span_verbatim_and_keeps_native_null():
    from tasksource.extractive_tasks import _qasper_spans
    def answer(spans, null=False):
        return dict(extractive_spans=spans, unanswerable=null, yes_no=None, free_form_answer='')
    row = dict(id='p', title='Paper', abstract='Alice', full_text={'paragraphs': [['Bob']]},
        qas={'question': ['Who?'], 'question_id': ['q'], 'answers': [{
            'annotation_id': ['a', 'b', 'c'], 'answer': [answer(['Alice', 'Bob']),
                answer(['Alice', 'missing']), answer([], True)]}]})
    rows = list(_qasper_spans(row))
    assert len(rows) == 2
    assert rows[0]['answers']['text'] == ['Alice', 'Bob']
    assert rows[1]['answers']['text'] == []
    from scripts.build_jev_extraction import LIST_QA
    assert {'cuad/spans', 'qasper/spans'} <= LIST_QA


def test_unique_verbatim_relocation_and_native_null():
    from tasksource.extractive_tasks import _verbatim_spans
    d = DatasetDict(train=Dataset.from_list([
        {'context': 'Alice Bob', 'answers': {'text': ['Bob'], 'answer_start': [7]}},
        {'context': 'Bob Bob', 'answers': {'text': ['Bob'], 'answer_start': [1]}},
        {'context': 'Alice', 'answers': {'text': ['Bob'], 'answer_start': [0]}},
        {'context': 'Alice', 'answers': {'text': [], 'answer_start': []}},
    ]))
    rows = _verbatim_spans(d)['train']
    assert len(rows) == 2
    assert rows[0]['span_answers'] == {'text': ['Bob'], 'answer_start': [6]}
    assert rows[1]['span_answers'] == {'text': [], 'answer_start': []}


def test_tatqa_only_single_text_spans():
    from tasksource.extractive_tasks import _tatqa_spans
    row = {'paragraphs': [{'order': 2, 'text': 'Bob'}, {'order': 1, 'text': 'Alice'}],
           'questions': [{'uid': str(i), 'question': 'Who?', 'answer': answer,
                          'answer_type': kind, 'answer_from': source}
                         for i, (answer, kind, source) in enumerate([
                             (['Alice'], 'span', 'text'), (['Bob'], 'span', 'table'),
                             (['Alice', 'Bob'], 'multi-span', 'text'), (['Eve'], 'span', 'text')])]}
    [out] = list(_tatqa_spans(row))
    assert out['context'] == 'Alice\n\nBob'
    assert out['answers'] == {'text': ['Alice'], 'answer_start': [0]}


def test_x3_subjqa_nested_annotations_become_canonical_answers():
    from tasksource.extractive_tasks import subjqa_books
    row = {'context': 'Alice', 'question': 'Who?', 'id': 'a', 'review_id': 'r', 'domain': 'books',
           'answers': {'text': ['Alice'], 'answer_start': [0], 'answer_subj_level': [1],
                       'ans_subj_score': [0.0], 'is_ans_subjective': [False]}}
    out = subjqa_books(DatasetDict(train=Dataset.from_list([row])))['train']
    assert out[0]['answers'] == {'text': ['Alice'], 'answer_start': [0]}


def test_extractive_empty_filtered_evaluation_split():
    task = ExtractiveQA(dataset_name='x', complete_splits=False)
    rows = Dataset.from_list([{'context': 'Alice', 'question': 'Who?',
                              'answers': {'text': ['Alice'], 'answer_start': [0]}}])
    out = task(DatasetDict(train=rows, validation=rows.select([])))
    assert len(out['train']) == 1 and len(out['validation']) == 0
    assert out['train'].features == out['validation'].features


def test_without_excess_empty_keeps_order_and_share():
    from scripts.build_jev_extraction import without_excess_empty
    full = {"answers": [{"null": False, "values": [{"text": "a", "start": 0, "end": 1}]}]}
    empty = {"answers": [{"null": True, "values": []}]}
    rows = [dict(full, i=i) if i % 5 == 0 else dict(empty, i=i) for i in range(50)]
    kept, dropped = without_excess_empty(rows, 0.2, 0, "x")
    assert sum(not r["answers"][0]["null"] for r in kept) == 10 and len(kept) == 12 and dropped == 38
    assert [r["i"] for r in kept] == sorted(r["i"] for r in kept)


@pytest.mark.parametrize('groups', [[[2]], [[-1]], [[0], [0]]])
def test_indexed_annotations_reject_invalid_or_overlapping_tokens(groups):
    from tasksource.tasks import _indexed_span_tags
    with pytest.raises(ValueError):
        _indexed_span_tags(['one', 'two'], groups, 'error')


def test_indexed_annotations_split_discontinuous_groups():
    from tasksource.tasks import _indexed_span_tags
    assert _indexed_span_tags(['a', 'b', 'c', 'd'], [[0, 1, 3]], 'x') == ['B-x', 'I-x', 'O', 'B-x']


def test_vuamc_drops_empty_words_after_labeling():
    from tasksource.tasks import _vuamc_row
    row = _vuamc_row({'words': ['', 'flows', 'down'], 'met_type': [{'type': 'mrw/met', 'word_indices': [1]}]})
    assert row == {'tokens': ['flows', 'down'], 'ner_tags': ['B-metaphor', 'O']}


def test_noncommercial_sources_are_flagged():
    from tasksource import task_licenses
    from tasksource.licenses import card_license_use
    assert set(task_licenses(['vuamc', 'fce/errors']).license_use) == {'non-commercial'}
    assert set(task_licenses(['qaconv'], extractive=True).license_use) == {'non-commercial'}
    assert card_license_use('BNC non-commercial terms') == 'non-commercial'
    assert card_license_use('apache-2.0') == 'commercial'


def test_qaconv_empty_answer_is_unanswerable():
    from tasksource.extractive_tasks import _qaconv_answers
    assert _qaconv_answers({'answers': {'text': [''], 'answer_start': [0]}}) == {'text': [], 'answer_start': []}
    assert _qaconv_answers({'answers': {'text': ['x'], 'answer_start': [3]}}) == {'text': ['x'], 'answer_start': [3]}
