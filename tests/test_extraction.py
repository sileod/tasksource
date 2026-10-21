import hashlib
import json
import re

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


def test_collection_budget_is_shared_equally():
    from scripts.build_jev_extraction import train_limits
    limits = train_limits(["squad-v2", "swde/auto", "swde/book", "swde/job", "swde/movie", "procedural/a",
                           "procedural/b"], max_rows=10_000, max_collection_rows=20_000)
    assert limits["squad-v2"] == 10_000 and limits["swde/auto"] == 5_000
    assert limits["procedural/a"] == 10_000  # explicit larger budget, still capped per source
    assert train_limits(["inspec/keyphrases"], 10_000, 20_000)["inspec/keyphrases"] == 1_000


def test_multilingual_peppering_does_not_shrink_english_siblings():
    """uner/en_ewt and tydiqa/english share a collection prefix with their non-English additions
    (uner/pt_bosque, uner/zh_gsd; tydiqa/arabic, .../bengali, .../finnish, .../korean, .../russian,
    .../swahili); COLLECTION_ROWS["uner"]/["tydiqa"] must be raised enough that dividing by the larger
    source count still leaves the English member at its un-peppered max_rows cap."""
    from scripts.build_jev_extraction import train_limits
    limits = train_limits(["uner/en_ewt", "uner/pt_bosque", "uner/zh_gsd"], max_rows=10_000, max_collection_rows=20_000)
    assert limits["uner/en_ewt"] == 10_000
    assert limits["uner/pt_bosque"] == 1_600 and limits["uner/zh_gsd"] == 1_600
    tydiqa_tasks = ["tydiqa/english", "tydiqa/arabic", "tydiqa/bengali", "tydiqa/finnish",
                    "tydiqa/korean", "tydiqa/russian", "tydiqa/swahili"]
    limits = train_limits(tydiqa_tasks, max_rows=10_000, max_collection_rows=20_000)
    assert limits["tydiqa/english"] == 10_000
    assert limits["tydiqa/swahili"] == 800
    assert all(limits[t] == 1_600 for t in tydiqa_tasks if t not in ("tydiqa/english", "tydiqa/swahili"))


def test_massive_slots_tags_align_with_class_names():
    from tasksource.multilingual_tasks import _massive_slots
    rows = Dataset.from_list([{"annot_utt": "weck mich am [date : freitag] um [time : neun uhr] auf"}])
    out = _massive_slots(DatasetDict(train=rows))["train"]
    names = out.features["ner_tags"].feature.names
    tags = [names[i] for i in out[0]["ner_tags"]]
    assert tags == ["O", "O", "O", "B-date", "O", "B-time", "I-time", "O"]


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


def _friendsqa_context(speakers_per_utterance, utterances):
    lines, offsets, pos = [], [], 0
    for speakers, utterance in zip(speakers_per_utterance, utterances):
        prefix = ', '.join(speakers) + ': '
        line = prefix + utterance
        offsets.append({'prefix_start': pos, 'text_start': pos + len(prefix)})
        lines.append(line)
        pos += len(line) + 1
    return '\n'.join(lines), offsets


def test_friendsqa_locate_token_span():
    from scripts.repackage_dataset.dialogue import _friendsqa_locate
    utterances = [{'utterance': "I 'm not giving up the apartment ."}]
    context, offsets = _friendsqa_context([['Chandler Bing']], [u['utterance'] for u in utterances])
    answer = {'utterance_id': 0, 'inner_start': 3, 'inner_end': 4, 'is_speaker': False,
              'answer_text': 'giving up'}
    start = _friendsqa_locate(context, utterances, offsets, answer)
    assert start is not None and context[start:start + len('giving up')] == 'giving up'


def test_friendsqa_locate_speaker_name_prefix_of_multi_speaker_line():
    from scripts.repackage_dataset.dialogue import _friendsqa_locate
    utterances = [{'utterance': 'Breathe .'}]
    context, offsets = _friendsqa_context([['Monica Geller', 'Rachel Green']], [u['utterance'] for u in utterances])
    answer = {'utterance_id': 0, 'inner_start': -1, 'inner_end': -1, 'is_speaker': True,
              'answer_text': 'Monica Geller'}
    start = _friendsqa_locate(context, utterances, offsets, answer)
    assert start == 0 and context[start:start + len('Monica Geller')] == 'Monica Geller'
    # The gold answer is only ever the credited speaker, but the search still locates a
    # different speaker's name verbatim within the same "Speaker, Speaker: " prefix slot.
    other = _friendsqa_locate(context, utterances, offsets, {**answer, 'answer_text': 'Rachel Green'})
    assert context[other:other + len('Rachel Green')] == 'Rachel Green'
    # A name that never appears in the prefix at all is rejected.
    assert _friendsqa_locate(context, utterances, offsets, {**answer, 'answer_text': 'Phoebe Buffay'}) is None


def test_friendsqa_locate_rejects_misaligned_token_offsets():
    from scripts.repackage_dataset.dialogue import _friendsqa_locate
    utterances = [{'utterance': '[ Scene : a room . ]'}]
    context, offsets = _friendsqa_context([['#NOTE#']], [u['utterance'] for u in utterances])
    # A genuinely bad source annotation (the span text does not match the claimed token offset).
    answer = {'utterance_id': 0, 'inner_start': 5, 'inner_end': 6, 'is_speaker': False,
              'answer_text': 'some other text'}
    assert _friendsqa_locate(context, utterances, offsets, answer) is None


def test_friendsqa_locate_rejects_out_of_range_utterance_id():
    from scripts.repackage_dataset.dialogue import _friendsqa_locate
    utterances = [{'utterance': 'hello .'}]
    context, offsets = _friendsqa_context([['A']], [u['utterance'] for u in utterances])
    answer = {'utterance_id': 7, 'inner_start': 0, 'inner_end': 0, 'is_speaker': False, 'answer_text': 'hello'}
    assert _friendsqa_locate(context, utterances, offsets, answer) is None




def test_maven_token_row_bio_and_overlap_drop():
    from tasksource.tasks import _maven_token_row
    text = "A huge earthquake and tsunami hit the coast."
    row = {'input': text, 'output': {'json_structures': {
        'Catastrophe': [{'trigger': {'text': 'earthquake', 'start': 7, 'end': 17}, 'arguments': []},
                        {'trigger': {'text': 'tsunami', 'start': 22, 'end': 29}, 'arguments': []}],
        'Impact': [{'trigger': {'text': 'huge earthquake', 'start': 2, 'end': 17}, 'arguments': []}],
    }}}
    out = _maven_token_row(row)
    assert out['aligned'] is True
    assert out['tokens'][:6] == ['A', 'huge', 'earthquake', 'and', 'tsunami', 'hit']
    # 'earthquake' overlaps the earlier-sorted (lower start) 'huge earthquake' trigger and is
    # dropped; 'huge earthquake' (Impact) and 'tsunami' (Catastrophe) are both kept.
    assert out['ner_tags'][:5] == ['O', 'B-Impact', 'I-Impact', 'O', 'B-Catastrophe']


def test_maven_token_row_misaligned_span_is_unaligned():
    from tasksource.tasks import _maven_token_row
    text = "Earthquakezone hit Java."
    row = {'input': text, 'output': {'json_structures': {
        'Catastrophe': [{'trigger': {'text': 'quake', 'start': 5, 'end': 10}, 'arguments': []}],
    }}}
    out = _maven_token_row(row)
    assert out == {'tokens': [], 'ner_tags': [], 'aligned': False}


def test_slurp_token_row_contiguous_and_cross_type_overlap():
    from tasksource.tasks import _slurp_token_row
    row = {'metadata': json.dumps({'tokens': ['set', 'an', 'alarm', 'for', 'six', 'am'],
        'entities': [{'span': [4, 5], 'type': 'time'}, {'span': [1, 3], 'type': 'device_type'},
                     {'span': [4], 'type': 'date'}]})}
    out = _slurp_token_row(row)
    assert out['tokens'] == ['set', 'an', 'alarm', 'for', 'six', 'am']
    assert out['ner_tags'] == ['O', 'O', 'O', 'O', 'B-time', 'I-time']  # the non-contiguous
    # 'device_type' span and the overlapping 'date' span (same token as 'time') are both dropped.


def test_ebm_nlp_tags_binarizes_subtype_codes():
    from tasksource.tasks import _ebm_nlp_tags
    row = {'Tokens': ['99', 'patients', 'with', 'H.', 'pylori', 'infection'],
           'Participants_Labels': [0, 1, 0, 2, 2, 2]}
    out = _ebm_nlp_tags('Participants_Labels', 'participant description')(row)
    assert out['ner_tags'] == ['O', 'B-participant description', 'O',
                                'B-participant description', 'I-participant description', 'I-participant description']


def _tab_doc(text, annotators):
    def mention(start, end, kind, level):
        return dict(start_offset=start, end_offset=end, entity_type=kind, identifier_type=level,
                    span_text=text[start:end])
    return [dict(doc_id='d', text=text, task='Task: anonymise Ann Lee', meta={'applicant': 'Ann Lee'},
                 annotator_id=f'a{i}', entity_mentions=[mention(*m) for m in mentions])
            for i, mentions in enumerate(annotators)]


def test_tab_majority_regions_keep_exact_annotator_spans():
    from tasksource.extractive_tasks import _tab_masked
    ann, mrs_ann = (4, 11, 'PERSON', 'DIRECT'), (0, 11, 'PERSON', 'DIRECT')
    oslo, year = (21, 25, 'LOC', 'QUASI'), (32, 36, 'DATETIME', 'QUASI')
    # Three annotators: two read 'Ann Lee', one 'Mrs Ann Lee' (one region, the majority reading wins);
    # Oslo masked by one only (minority region, dropped); 2001 masked by two, NO_MASK for the third.
    docs = _tab_doc('Mrs Ann Lee lives in Oslo since 2001.',
                    [[ann, oslo, year], [mrs_ann, year], [ann, (32, 36, 'DATETIME', 'NO_MASK')]])
    assert _tab_masked(docs) == ([ann, year], 1)
    assert _tab_masked(docs[:1]) == ([ann, oslo, year], 0)  # one annotator: all their masked mentions
    # Two annotators: both must mask the region; tied readings go to the first annotator.
    assert _tab_masked(docs[:2]) == ([ann, year], 1)


def test_tab_records_are_verbatim_windows_with_named_fields():
    from tasksource.extractive_tasks import TAB_TYPES, _tab_records
    text = ('The applicant was born in 1970. ' * 150) + 'Ann Lee lives in Oslo.' + (' Nothing here.' * 150)
    start = text.index('Ann Lee')
    docs = _tab_doc(text, [[(start, start + 7, 'PERSON', 'DIRECT'), (start + 17, start + 21, 'LOC', 'QUASI'),
                            (26, 30, 'DATETIME', 'QUASI'), (start + 17, start + 21, 'ORG', 'NO_MASK')]])
    records = list(_tab_records(docs))
    windows = {json.loads(r['metadata'])['window_start'] for r in records}
    assert len(windows) > 1 and len(records) == len(TAB_TYPES) * len(windows)
    rows = list(qa_rows(records, 'tab/pii', 'train', kind='list'))
    assert len(rows) == len(windows)
    assert [f['name'] for f in rows[0]['fields']] == [name for name, _ in TAB_TYPES.values()]
    assert 'anonymise Ann Lee' in rows[0]['fields'][0]['instruction']
    found = {f['name']: [v['text'] for v in a['values']] for r in rows
             for f, a in zip(r['fields'], r['answers']) if not a['null']}
    assert found == {'person': ['Ann Lee'], 'location': ['Oslo'], 'date_time': ['1970']}  # NO_MASK ORG left out


def test_pave_values_located_verbatim_with_native_nulls():
    from tasksource.extractive_tasks import _pave_records
    line = json.dumps({'id': 1, 'cluster_id': 2.0, 'category': 'Office Products', 'url': 'x.com',
        'input_title': 'Avery Light Tags, Lightweight', 'input_description': 'Tags by Avery. Color: Manila',
        'target_scores': {'Brand': {'Avery': {'pid': [0, 1], 'score': 1}},
                          'Color(s)': {'Manila': {'pid': [1], 'score': 1}, 'manilla': {'pid': [1], 'score': 1}},
                          'Weight': {'Light': {'pid': [0], 'score': 1}},
                          'Width': {'n/a': 1}, 'Gauge': {'thin': {'pid': [0], 'score': 1}}}})
    rows = {json.loads(r['metadata'])['field']: r for r in _pave_records(line)}
    assert set(rows) == {'brand', 'colors', 'weight', 'width'}  # Gauge: value not in the offer, left out
    context = rows['brand']['context']
    for r in rows.values():
        assert all(context[s:s + len(t)] == t for t, s in zip(r['answers']['text'], r['answers']['answer_start']))
    assert len(rows['brand']['answers']['text']) == 2  # one occurrence in the title, one in the description
    assert rows['colors']['answers']['text'] == ['Manila']  # unlocated variant dropped
    assert rows['weight']['answers']['answer_start'] == [context.index('Light ')]  # not inside 'Lightweight'
    assert rows['width']['answers']['text'] == []  # native n/a
    assert rows['colors']['question'] == 'Color(s) of the product, as written in the offer.'
    # 20 attributes: three balanced rows, not 8 + 8 + 4.
    many = json.loads(line)
    many['target_scores'] = {f'Attribute {i:02d}': {'n/a': 1} for i in range(20)}
    out = list(qa_rows(list(_pave_records(json.dumps(many))), 'wdc-pave', 'train'))
    assert [len(r['fields']) for r in out] == [7, 7, 6]
    assert out[0]['fields'][1]['name'] == 'attribute_03'


def _ud_sentence(tokens, heads, relations, upos=None, text=None):
    return dict(sent_id='s', text=text or ' '.join(tokens), tokens=tokens, head=heads,
                deprel=relations, upos=upos or ['VERB'] * len(tokens))


def test_ud_full_subtree_null_subtypes_and_unique_head():
    from collections import Counter
    from tasksource.span_structure import ud_records
    row = _ud_sentence(['The', 'deal', 'was', 'signed', 'by', 'Ann', '.'],
                       [2, 4, 4, 0, 6, 4, 4],
                       ['det', 'nsubj:pass', 'aux:pass', 'root', 'case', 'obl:agent', 'punct'])
    counts = Counter()
    records = list(ud_records(row, counts))
    fields = {json.loads(r['metadata'])['field']: r for r in records}
    assert fields['subject']['answers']['text'] == ['The deal']
    assert fields['oblique_modifier']['answers']['text'] == ['by Ann']
    nulls = [f for f, r in fields.items() if not r['answers']['text']]
    assert len(fields) == 3 and len(nulls) == 1 and nulls[0] not in ('subject', 'oblique_modifier')
    [out] = list(qa_rows(records, 'ud/ewt', 'train'))
    check(out)
    assert all(f['type'] == 'list' for f in out['fields'])
    assert "'signed'" in out['fields'][0]['instruction']
    repeated = _ud_sentence(['Ann', 'signed', 'and', 'signed'], [2, 0, 4, 2], ['nsubj', 'root', 'cc', 'conj'])
    assert list(ud_records(repeated, counts)) == []
    assert counts['no_unambiguous_head'] == 1


def test_ud_nonprojective_answer_drops_entire_field_never_becomes_null():
    from collections import Counter
    from tasksource.span_structure import ud_records
    # Object subtree {0, 2} is interrupted by the head at 1.
    row = _ud_sentence(['which', 'saw', 'dog'], [3, 0, 2], ['det', 'root', 'obj'])
    counts = Counter()
    fields = {json.loads(r['metadata'])['field'] for r in ud_records(row, counts)}
    assert 'object' not in fields
    assert counts['nonprojective_subtrees'] == counts['nonprojective_fields_dropped'] == 1
    counts = Counter()
    assert list(ud_records({**row, 'text': 'different'}, counts)) == []
    assert counts['unaligned_sentences'] == 1


def test_preco_first_mention_sorted_multi_mention_and_ambiguous_anchor():
    from collections import Counter
    from tasksource.span_structure import preco_records
    row = dict(id='p', sentences=[['Alice', 'saw', 'her', 'dog', '.'], ['She', 'smiled', '.']],
               mention_clusters=[[[1, 0, 1], [0, 0, 1]], [[0, 2, 3]], [[0, 3, 4]]])
    counts = Counter()
    [r] = list(preco_records(row, counts))
    assert r['answers']['text'] == ['Alice', 'She']
    assert "'Alice'" in r['question']
    assert counts['singleton_clusters'] == 2
    [out] = list(qa_rows([r], 'preco/mentions', 'train'))
    check(out)
    row['sentences'][1][0] = 'Alice'
    assert list(preco_records(row, counts)) == []
    assert counts['ambiguous_clusters'] == 1


def test_redocred_relation_direction_aggregates_entities_and_first_mentions():
    from collections import Counter
    from tasksource.span_structure import redocred_records
    def mention(sent, a, b):
        return dict(sent_id=sent, pos=[a, b], name='ignored: native tokens determine text')
    row = dict(title='d', sents=[['Ann', 'has', 'Bob', 'and', 'Cal', '.'], ['Bob', 'smiles', '.']],
        vertexSet=[[mention(0, 0, 1)], [mention(1, 0, 1), mention(0, 2, 3)], [mention(0, 4, 5)]],
        labels=[dict(h=0, t=1, r='P40'), dict(h=0, t=2, r='P40'), dict(h=0, t=1, r='P40')])
    [r] = list(redocred_records(row, Counter()))
    assert r['answers']['text'] == ['Bob', 'Cal']
    assert "'Ann'" in r['question'] and "'child'" in r['question']
    assert json.loads(r['metadata'])['tail_entities'] == [1, 2]
    [out] = list(qa_rows([r], 'redocred/entities', 'train'))
    check(out)
    row['sents'][1][0] = 'Ann'
    counts = Counter()
    assert list(redocred_records(row, counts)) == []
    assert counts['ambiguous_relation_fields'] == 1


def test_structural_tasks_pins_native_splits_licenses_and_collection_budgets():
    from tasksource import task_licenses
    from tasksource.extractive_tasks import ud_ewt, ud_gum, preco_mentions, redocred_entities
    from tasksource.tasks import chunking
    from scripts.build_jev_extraction import LIST_QA, NER, train_limits
    tasks = [ud_ewt, ud_gum, preco_mentions, redocred_entities]
    assert {t.task_id for t in tasks} <= LIST_QA
    for task in tasks + [chunking]:
        assert not task.complete_splits
        assert re.fullmatch('[0-9a-f]{40}', task.load_dataset_kwargs['revision'])
    assert ud_ewt.splits == ('train', 'dev', 'test')
    assert preco_mentions.splits == ('train', 'validation', None)
    licenses = task_licenses([t.task_id for t in tasks], extractive=True).set_index('id')
    assert licenses.loc['ud/ewt', 'license_use'] == 'commercial'
    assert licenses.loc['ud/gum', 'license_use'] == 'non-commercial'
    assert licenses.loc['preco/mentions', 'license_use'] == 'non-commercial'
    assert 'conll2000' in NER
    assert train_limits(['ud/ewt', 'ud/gum'], 10_000, 10_000)['ud/ewt'] == 5000


def test_span_structure_mirrors_are_registered_and_pinned():
    """QA-SRL Bank 2.0, BioScope and Sherlock/ConanDoyle-neg load from pinned tasksource mirrors
    (scripts/repackage_dataset/span_structure.py, sherlock_spans.py)."""
    from tasksource import task_licenses
    from tasksource.extractive_tasks import qasrl_orig, bioscope_abstracts, bioscope_papers, sherlock_scopes
    from scripts.build_jev_extraction import LIST_QA
    tasks = [qasrl_orig, bioscope_abstracts, bioscope_papers, sherlock_scopes]
    assert {t.task_id for t in tasks} <= LIST_QA
    for task in tasks:
        assert not task.complete_splits
        assert len(task.load_dataset_kwargs['revision']) == 40
    assert qasrl_orig.dataset_name == 'tasksource/qasrl-bank-spans'
    assert bioscope_abstracts.dataset_name == bioscope_papers.dataset_name == 'tasksource/bioscope-spans'
    assert bioscope_abstracts.config_name == 'abstracts' and bioscope_papers.config_name == 'papers'
    assert bioscope_papers.splits == ('train', None, None)  # only 9 documents: too few to split
    assert sherlock_scopes.dataset_name == 'tasksource/sherlock-negation-spans'
    licenses = task_licenses([t.task_id for t in tasks], extractive=True).set_index('id')
    assert licenses.loc['qasrl/orig', 'license_use'] == 'unspecified'
    assert licenses.loc['bioscope/abstracts', 'license_use'] == 'commercial'
    assert licenses.loc['bioscope/papers', 'license_use'] == 'commercial'
    assert licenses.loc['sherlock/scopes', 'license_use'] == 'unspecified'


def test_span_structure_mirror_schema_smoke_loads_through_extractive_qa():
    """The mirrors' raw parquet schema (context/question/answers/metadata) is exactly the
    ExtractiveQA schema already used by qasrl_orig/bioscope_abstracts/bioscope_papers/sherlock_scopes
    (no pre_process): run real conversion output for each source through the same preprocessing
    class, offline, standing in for the Hub load until a maintainer pins the mirror revision."""
    from collections import Counter
    from scripts.repackage_dataset.span_structure import qasrl_records, bioscope_records
    from scripts.repackage_dataset.sherlock_spans import sherlock_records
    from xml.etree import ElementTree as ET

    def judgment(worker, valid, spans):
        return dict(sourceId=worker, isValid=valid, spans=spans)
    label = dict(answerJudgments=[judgment('turk-a', True, [[0, 1]]), judgment('turk-b', True, [[0, 1]]),
                                   judgment('turk-c', True, [[0, 1]])])
    qasrl_row = dict(sentenceId='s', sentenceTokens=['Ann', 'ran', 'fast', '.'],
                      verbEntries={'1': dict(verbIndex=1, questionLabels={'Who ran?': label})})
    qasrl_rows = list(qasrl_records(qasrl_row, Counter()))

    bioscope_xml = ('<sentence id="s">He <cue type="negation" ref="a">never</cue> '
                     '<xcope id="a">saw it</xcope>.</sentence>')
    bioscope_rows = list(bioscope_records(ET.fromstring(bioscope_xml), 'g', Counter()))

    sherlock_lines = ['\t'.join(r) for r in [
        ['story', '1', '0', 'He', 'he', 'PRP', '*', '_', 'He', '_', '_', 'He', '_'],
        ['story', '1', '1', 'is', 'be', 'VB', '*', '_', 'is', '_', '_', 'is', '_'],
        ['story', '1', '2', 'not', 'not', 'RB', '*', 'not', '_', '_', '_', '_', '_'],
        ['story', '1', '3', 'unhappy', 'unhappy', 'JJ', '*', '_', 'unhappy', '_', 'un', 'happy', '_']]]
    sherlock_rows = list(sherlock_records(sherlock_lines, Counter()))

    for name, rows in [('qasrl', qasrl_rows), ('bioscope', bioscope_rows), ('sherlock', sherlock_rows)]:
        assert rows, name
        data = Dataset.from_list(rows)
        task = ExtractiveQA(context='context', query='question', answers='answers', metadata='metadata',
                             dataset_name='x', splits=('train', None, None), complete_splits=False, max_mismatch=0)
        out = task(DatasetDict(train=data))['train']
        assert len(out) == len(rows), name
        for row in out:
            for text, start in zip(row['answers']['text'], row['answers']['answer_start']):
                assert row['context'][start:start + len(text)] == text


def test_qasrl_requires_independent_human_validation_and_selects_complete_reading():
    from collections import Counter
    from scripts.repackage_dataset.span_structure import qasrl_records
    def judgment(worker, valid, spans):
        return dict(sourceId=worker, isValid=valid, spans=spans)
    label = dict(answerJudgments=[judgment('turk-a', True, [[0, 1], [2, 3]]),
        judgment('turk-b', True, [[0, 1]]), judgment('model-x', True, [[1, 2]])])
    row = dict(sentenceId='s', sentenceTokens=['Ann', 'and', 'Bob', 'ran'],
               verbEntries={'3': dict(verbIndex=3, questionLabels={'Who ran?': label})})
    counts = Counter()
    [r] = list(qasrl_records(row, counts))
    # Tie retains one real complete set, never the union of incompatible validator readings.
    assert r['answers']['text'] == ['Ann']
    assert json.loads(r['metadata'])['valid_judgments'] == 2
    label['answerJudgments'] = [judgment('turk-a', True, [[0, 1]]), judgment('turk-a-eval', True, [[0, 1]]),
                                 judgment('model-x', True, [[0, 1]])]
    assert list(qasrl_records(row, counts)) == []
    label['answerJudgments'] = [judgment('turk-a', True, [[0, 1]]), judgment('turk-b', True, [[2, 3]]),
                                 judgment('turk-c', False, []), judgment('turk-d', False, [])]
    assert list(qasrl_records(row, counts)) == []
    assert counts['unvalidated_questions'] == 2


def test_bioscope_nested_offsets_and_discontinuous_scope_keep_cue_identity():
    from collections import Counter
    from xml.etree import ElementTree as ET
    from scripts.repackage_dataset.span_structure import bioscope_records
    xml = '<sentence id="s"><xcope id="a">He <cue type="negation" ref="a">never</cue> '
    xml += '<xcope id="b"><cue type="speculation" ref="b">may</cue> go</xcope></xcope>'
    xml += ', but <xcope id="a">she can</xcope>.</sentence>'
    records = list(bioscope_records(ET.fromstring(xml), 's', Counter()))
    fields = {json.loads(r['metadata'])['field']: r for r in records}
    assert fields['negation_cues']['answers']['text'] == ['never']
    assert fields['speculation_cues']['answers']['text'] == ['may']
    assert fields['negation_scope_0']['answers']['text'] == ['He never may go', 'she can']
    for row in qa_rows(records, 'bioscope/abstracts', 'train'):
        check(row)
    empty = list(bioscope_records(ET.fromstring('<sentence>No scope.</sentence>'), 'n', Counter()))
    assert all(r['answers']['text'] == [] for r in empty) and len(empty) == 2


def test_sherlock_separate_cues_affixes_and_discontinuous_scopes():
    from collections import Counter
    from scripts.repackage_dataset.sherlock_spans import sherlock_records
    # Two independent scopes; affix cue "un" and its complement "happy" share a token.
    rows = [
        ['story', '1', '0', 'He', 'he', 'PRP', '*', '_', 'He', '_', '_', 'He', '_'],
        ['story', '1', '1', 'is', 'be', 'VB', '*', '_', 'is', '_', '_', 'is', '_'],
        ['story', '1', '2', 'not', 'not', 'RB', '*', 'not', '_', '_', '_', '_', '_'],
        ['story', '1', '3', 'unhappy', 'unhappy', 'JJ', '*', '_', 'unhappy', '_', 'un', 'happy', '_'],
    ]
    records = list(sherlock_records(['\t'.join(r) for r in rows], Counter()))
    fields = {json.loads(r['metadata'])['field']: r for r in records}
    assert fields['negation_cues']['answers']['text'] == ['not', 'un']
    assert fields['negation_scope_0']['answers']['text'] == ['He is', 'unhappy']
    assert fields['negation_scope_1']['answers']['text'] == ['He is', 'happy']
    for row in qa_rows(records, 'sherlock/scopes', 'train'):
        check(row)
    rows[3][10] = 'not-in-token'
    counts = Counter()
    result = list(sherlock_records(['\t'.join(r) for r in rows], counts))
    assert len(result) == 1  # the other scope survives; the all-cues field is dropped
    assert counts['unlocated_cue_fields'] == 1


def test_multispanqa_bio_tokens_become_character_spans():
    from tasksource.extractive_tasks import _multispanqa_row
    row = dict(id='q1', type='HUM', structure='Conjunction', num_span=2,
               question=['who', 'wrote', 'it'],
               context=['Mike', 'Reid', 'and', 'Allen', 'Shamblin', 'wrote', 'it', '.'],
               label=['B', 'I', 'O', 'B', 'I', 'O', 'O', 'O'])
    out = _multispanqa_row(row)
    assert out['context'] == 'Mike Reid and Allen Shamblin wrote it .'
    assert out['question'] == 'who wrote it'
    assert out['answers']['text'] == ['Mike Reid', 'Allen Shamblin']
    for text, start in zip(out['answers']['text'], out['answers']['answer_start']):
        assert out['context'][start:start + len(text)] == text
    assert json.loads(out['metadata']) == {'id': 'q1', 'answer_type': 'HUM', 'structure': 'Conjunction', 'num_span': 2}


def _event_row(text, structures):
    return {'input': text, 'output': {'json_structures': structures}}


def _trigger(start, end, text):
    return {'start': start, 'end': end, 'text': text}


def _argument(role, start, end, text):
    return {'role': role, 'start': start, 'end': end, 'text': text}


def test_event_role_schema_and_present_plus_one_absent_role():
    from tasksource.extractive_tasks import _event_role_schema, _event_role_records
    from datasets import Dataset, DatasetDict
    text = 'Ann killed Bob in Paris with a knife.'
    row = _event_row(text, {'Life:Die': [{'trigger': _trigger(4, 10, 'killed'),
        'arguments': [_argument('killer', 0, 3, 'Ann'), _argument('victim', 11, 14, 'Bob'),
                      _argument('place', 18, 23, 'Paris')]}]})
    other = _event_row(text, {'Life:Die': [{'trigger': _trigger(4, 10, 'killed'),
        'arguments': [_argument('instrument', 29, 37, 'a knife')]}]})
    schema = _event_role_schema(DatasetDict(train=Dataset.from_list([row, other])))
    assert schema['Life:Die'] == ['instrument', 'killer', 'place', 'victim']
    records = list(_event_role_records(schema)(row))
    fields = {json.loads(r['metadata'])['field']: r for r in records}
    assert fields['killer']['answers']['text'] == ['Ann']
    assert fields['victim']['answers']['text'] == ['Bob']
    assert fields['place']['answers']['text'] == ['Paris']
    # Present roles are always asked; absent roles (here just "instrument") earn at most one null field.
    assert set(fields) == {'killer', 'victim', 'place', 'instrument'}
    assert fields['instrument']['answers']['text'] == []
    # The trigger is named with its offsets, so a repeated trigger word is not ambiguous.
    assert "'killed' at characters [4, 10)" in fields['killer']['question']
    for r in records:
        for text_, start in zip(r['answers']['text'], r['answers']['answer_start']):
            assert r['context'][start:start + len(text_)] == text_
        assert json.loads(r['metadata'])['group'] == json.loads(records[0]['metadata'])['group']


def test_event_role_records_rejects_mismatched_trigger_or_argument():
    from tasksource.extractive_tasks import _event_role_records
    text = 'Ann killed Bob.'
    bad_trigger = _event_row(text, {'Life:Die': [{'trigger': _trigger(4, 10, 'kills'), 'arguments': []}]})
    with pytest.raises(ValueError):
        list(_event_role_records({})(bad_trigger))
    bad_argument = _event_row(text, {'Life:Die': [{'trigger': _trigger(4, 10, 'killed'),
        'arguments': [_argument('victim', 11, 14, 'Bo')]}]})
    with pytest.raises(ValueError):
        list(_event_role_records({})(bad_argument))


def test_english_extraction_round_2026_10_21_pins_and_licenses():
    """MultiSpanQA, RAMS and CASIE event-argument sources added in the 2026-10-21 English round."""
    from tasksource import task_licenses
    from tasksource.extractive_tasks import multispanqa, rams_roles, casie_roles
    from scripts.build_jev_extraction import LIST_QA
    assert {'multispanqa', 'rams/role', 'casie/role'} <= LIST_QA
    assert not multispanqa.complete_splits
    assert multispanqa.load_dataset_kwargs['data_files']['test'].endswith('test.json')
    for task in (rams_roles, casie_roles):
        assert not task.complete_splits
        assert re.fullmatch('[0-9a-f]{40}', task.load_dataset_kwargs['revision'])
    licenses = task_licenses(['multispanqa', 'rams/role', 'casie/role'], extractive=True).set_index('id')
    assert licenses.loc['multispanqa', 'license_use'] == 'commercial'
    assert licenses.loc['rams/role', 'license_use'] == 'commercial'
    assert licenses.loc['casie/role', 'license_use'] == 'unspecified'


def test_cap_null_groups_keeps_every_positive_sentence():
    import json
    from datasets import Dataset, DatasetDict
    from tasksource.span_structure import cap_null_groups
    rows = [dict(answers={'text': ['not'] if g < 10 else [], 'answer_start': [0] if g < 10 else []},
                 metadata=json.dumps({'group': f's{g}'})) for g in range(40) for _ in range(2)]
    capped = cap_null_groups(DatasetDict(train=Dataset.from_list(rows)))['train']
    groups = {json.loads(m)['group'] for m in capped['metadata']}
    assert {f's{g}' for g in range(10)} <= groups and len(groups) == 10 + 6  # 6 null groups: 40% of 16
    assert len(capped) == 2 * len(groups)  # a group is kept or dropped whole


def test_phee_records_identify_events_by_trigger_offsets():
    """PHEE event-argument extraction (src/tasksource/extractive_tasks.py): spans are verbatim at their
    offsets, discontinuous mentions (several (text, start) pairs for one argument) become separate
    contiguous spans, Treatment/Subject sub-arguments are asked only when their parent argument is
    present, and every question names its event's trigger with offsets, so two events of one sentence are
    distinct and recoverable from the text."""
    import json
    from tasksource.extractive_tasks import _phee_records
    from tasksource.jev.extraction import qa_rows, check

    context = ('Flecainide-induced interaction with paroxetine caused confusion in the patient '
               'who had previously tolerated flecainide alone.')

    def at(needle, occurrence=1):
        start = -1
        for _ in range(occurrence):
            start = context.find(needle, start + 1)
        assert start >= 0, needle
        return start

    def mkfield(words, occurrence=1):
        """``words`` is one mention's pieces (several for a discontinuous mention), each located at its
        own occurrence in ``context``."""
        pairs = words if isinstance(words[0], tuple) else [(words, occurrence)]
        return {'text': [[t] for t, _ in pairs], 'start': [[at(t, n)] for t, n in pairs], 'entity_id': ['T']}

    row = {
        'id': 'doc1_1', 'context': context, 'is_mult_event': True,
        'annotations': [{'events': [
            {'event_id': 'E1', 'event_type': 'Adverse_event',
             'Trigger': mkfield('caused'),
             'Treatment': {**mkfield('Flecainide'),
                           'Drug': mkfield([('Flecainide', 1), ('paroxetine', 1), ('flecainide', 1)])},
             'Effect': mkfield('confusion'),
             'Subject': {**mkfield('the patient'), 'Age': mkfield('who')}},
            {'event_id': 'E2', 'event_type': 'Adverse_event',
             'Trigger': mkfield('interaction'),
             'Treatment': {**mkfield('paroxetine')}},
        ]}]}
    line = json.dumps(row)
    records = list(_phee_records(line))
    assert records  # non-empty
    for record in records:
        for text, start in zip(record['answers']['text'], record['answers']['answer_start']):
            assert context[start:start + len(text)] == text

    metas = [json.loads(r['metadata']) for r in records]
    # E1 has a Subject (so subject sub-fields are asked); E2 has no Subject (none asked).
    assert any(m['group'] == 'doc1_1:E1' and m['field'] == 'subject_age' for m in metas)
    assert not any(m['group'] == 'doc1_1:E2' and m['field'].startswith('subject_') for m in metas)
    # Discontinuous Treatment.Drug mentions (Flecainide/paroxetine/flecainide) all kept as separate spans.
    drug = next(r for r, m in zip(records, metas) if m['field'] == 'treatment_drug' and 'E1' in m['group'])
    assert sorted(drug['answers']['text']) == ['Flecainide', 'flecainide', 'paroxetine']

    effect = {m['group']: r['question'] for r, m in zip(records, metas) if m['field'] == 'effect'}
    assert f"'caused' at characters [{at('caused')}, {at('caused') + 6})" in effect['doc1_1:E1']
    assert f"'interaction' at characters [{at('interaction')}, {at('interaction') + 11})" in effect['doc1_1:E2']

    for grouped in qa_rows(records, 'phee', 'train', 8, kind='list'):
        check(grouped)


def _write_abstrct_fixture(root, text, spans):
    """One brat .txt/.ann abstract pair; ``spans`` is ``[(label, start, end)]``, offsets taken from
    ``text`` itself so the fixture is always verbatim by construction."""
    (root.parent / f'{root.name}.txt').write_text(text)
    lines = [f'T{i}\t{label} {s} {e}\t{text[s:e]}\n' for i, (label, s, e) in enumerate(spans, 1)]
    (root.parent / f'{root.name}.ann').write_text(''.join(lines))


def test_abstrct_spans_repackage_is_document_level_and_verbatim(tmp_path):
    """AbstRCT (Mayer, Cabrio and Villata, ECAI 2020): major-claim/claim/premise spans over the full
    abstract (not per-sentence BIO), repackaged from the authors' brat release, 2026-10-21 rework.
    Runs scripts/repackage_dataset/abstrct_spans.py end to end against a small fixture archive built
    to look like the real GitLab release, so the task can be validated before the mirror is published."""
    import zipfile
    from scripts.repackage_dataset.abstrct_spans import convert

    revision = 'deadbeef' * 5
    root = tmp_path / 'src' / f'abstrct-{revision}' / 'AbstRCT_corpus' / 'data'

    t1 = "This drug improves outcomes. It also reduces pain in patients."
    (root / 'train' / 'neoplasm_train').mkdir(parents=True)
    _write_abstrct_fixture(root / 'train' / 'neoplasm_train' / '1', t1,
        [('Claim', 0, t1.index('.') + 1), ('Premise', t1.index('It also'), len(t1))])

    t2 = "No major claim is stated here either way at all."
    (root / 'dev' / 'neoplasm_dev').mkdir(parents=True)
    _write_abstrct_fixture(root / 'dev' / 'neoplasm_dev' / '2', t2, [])

    t3 = "A wide ranging major claim opens this abstract for review purposes."
    (root / 'test' / 'neoplasm_test').mkdir(parents=True)
    _write_abstrct_fixture(root / 'test' / 'neoplasm_test' / '3', t3, [('MajorClaim', 0, len(t3))])

    t4 = "Glaucoma abstract text for cross domain testing purposes only."
    (root / 'test' / 'glaucoma_test').mkdir(parents=True)
    _write_abstrct_fixture(root / 'test' / 'glaucoma_test' / '4', t4, [])

    t5 = "Mixed domain abstract text for cross domain testing purposes."
    (root / 'test' / 'mixed_test').mkdir(parents=True)
    _write_abstrct_fixture(root / 'test' / 'mixed_test' / '5', t5, [])

    license_dir = tmp_path / 'src' / f'abstrct-{revision}'
    (license_dir / 'LICENSE').write_text(
        'Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License:\n'
        'https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode\n')

    archive = tmp_path / 'abstrct_fixture.zip'
    with zipfile.ZipFile(archive, 'w') as zf:
        for path in (tmp_path / 'src').rglob('*'):
            if path.is_file():
                zf.write(path, path.relative_to(tmp_path / 'src'))
    sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()

    output = tmp_path / 'out'
    convert(archive, output, revision=revision, sha256=sha256)

    from datasets import load_dataset
    neoplasm_train = load_dataset('parquet', data_files=str(output / 'train.parquet'))['train']
    by_field = {json.loads(r['metadata'])['field']: r for r in neoplasm_train}
    assert by_field['claim']['answers']['text'] == ['This drug improves outcomes.']
    assert by_field['premise']['answers']['text'] == ['It also reduces pain in patients.']
    assert by_field['major_claim']['answers']['text'] == []  # no MajorClaim in this abstract
    for r in neoplasm_train:
        for text, start in zip(r['answers']['text'], r['answers']['answer_start']):
            assert r['context'][start:start + len(text)] == text  # verbatim at its offset
        assert json.loads(r['metadata'])['group'] == 'neoplasm:1'  # document-level group, not sentence

    neoplasm_dev = load_dataset('parquet', data_files=str(output / 'validation.parquet'))['train']
    assert all(not r['answers']['text'] for r in neoplasm_dev)  # abstract 2 has no annotated spans

    neoplasm_test = load_dataset('parquet', data_files=str(output / 'test.parquet'))['train']
    major_claim = next(r for r in neoplasm_test if json.loads(r['metadata'])['field'] == 'major_claim')
    assert major_claim['answers']['text'] == [t3]

    glaucoma_test = load_dataset('parquet', data_files=str(output / 'glaucoma' / 'test.parquet'))['train']
    mixed_test = load_dataset('parquet', data_files=str(output / 'mixed' / 'test.parquet'))['train']
    assert len(glaucoma_test) and len(mixed_test)  # test-only cross-domain configs are produced too

    provenance = json.loads((output / 'provenance.json').read_text())
    assert provenance['license'] == 'cc-by-nc-sa-4.0'
    assert provenance['rows']['neoplasm/train'] == 3  # 3 fields (major_claim, claim, premise) x 1 abstract
    assert (output / 'README.md').exists()


def test_abstrct_neoplasm_points_at_the_mirror_and_is_list_qa():
    from tasksource.extractive_tasks import abstrct_neoplasm
    from scripts.build_jev_extraction import LIST_QA

    assert abstrct_neoplasm.dataset_name == 'tasksource/abstrct-spans'
    assert abstrct_neoplasm.config_name == 'neoplasm'
    assert not abstrct_neoplasm.complete_splits
    assert 'abstrct/neoplasm' in LIST_QA
    # The source no longer exists as a per-sentence token-labeling source.
    import tasksource.tasks as tasks_module
    assert not hasattr(tasks_module, 'abstrct_neoplasm')


def test_gum_segmentation_windows_at_edu_boundaries_and_drops_reddit():
    """GUM EDU segmentation (src/tasksource/extractive_tasks.py): a window boundary always falls on an
    EDU boundary (never cuts a span), and Reddit documents (redistributed as underscored placeholders,
    not real text) are dropped."""
    from tasksource.extractive_tasks import (_gum_segmentation_docs, _gum_segmentation_windows,
        _gum_segmentation_records)

    lines = [
        '# newdoc_id = GUM_test_doc',
        '1\tHello\t_\t_\t_\t_\t_\t_\t_\tBeginSeg=Yes',
        '2\tworld\t_\t_\t_\t_\t_\t_\t_\t_',
        '3\t.\t_\t_\t_\t_\t_\t_\t_\t_',
        '4\tBye\t_\t_\t_\t_\t_\t_\t_\tBeginSeg=Yes',
        '5\tnow\t_\t_\t_\t_\t_\t_\t_\t_',
        '# newdoc_id = GUM_reddit_thread',
        '1\t___\t_\t_\t_\t_\t_\t_\t_\tBeginSeg=Yes',
        '2\t__\t_\t_\t_\t_\t_\t_\t_\t_',
    ]
    docs = _gum_segmentation_docs(lines)
    assert [d[0] for d in docs] == ['GUM_test_doc']  # reddit doc dropped

    doc_id, tokens, begins = docs[0]
    windows = list(_gum_segmentation_windows(doc_id, tokens, begins, window=2))
    # Window boundaries land on EDU starts, never mid-EDU: 2nd window starts at the 2nd BeginSeg token.
    assert [w[1] for w in windows] == [['Hello', 'world', '.'], ['Bye', 'now']]

    # At the module's default window (larger than this toy document), the whole document is one window.
    records = list(_gum_segmentation_records(lines))
    assert len(records) == 1
    record = records[0]
    for text, start in zip(record['answers']['text'], record['answers']['answer_start']):
        assert record['context'][start:start + len(text)] == text
    assert record['answers']['text'] == ['Hello world .', 'Bye now']


def test_gum_segmentation_writes_multiword_tokens_once():
    from tasksource.extractive_tasks import _gum_segmentation_docs
    row = lambda i, t, seg=False: f"{i}\t{t}\t_\t_\t_\t_\t_\t_\t_\t{'BeginSeg=Yes' if seg else '_'}"
    lines = ['# newdoc_id = GUM_news_x', row(1, 'We', True), row('2-3', 'don’t'), row(2, 'do'), row(3, 'n’t'),
             row(4, 'go'), row('5-6', 'cannot'), row(5, 'can'), row(6, 'not', True), row(7, 'stop')]
    [(_, tokens, begins)] = _gum_segmentation_docs(lines)
    # Surface form when no EDU starts inside; parts kept when one does.
    assert tokens == ['We', 'don’t', 'go', 'can', 'not', 'stop']
    assert begins == [True, False, False, False, True, False]
