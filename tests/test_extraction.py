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
