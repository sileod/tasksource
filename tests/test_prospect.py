import pytest
"""Prospect search: metadata prior, label detection and catalog identity fingerprints."""
import json

from tasksource.task_discovery.fingerprint import DatasetIndex, row_segments, segments, signature
from tasksource.task_discovery.prospect import discrete, label_columns, prior


def hub_row(name, downloads=0, likes=0, papers=(), modified='2026-01-01'):
    return dict(name=name, papers=list(papers),
                hub_metadata=dict(downloads=downloads, likes=likes, last_modified=modified))


def test_prior_prefers_used_sources_and_penalizes_bulk_slop():
    authors = {'bulk': [0] * 30, 'lab': [500, 900]}
    used, _ = prior(hub_row('lab/argument-quality', 900, 12, ['arxiv:1']), authors)
    bulk, flags = prior(hub_row('bulk/sentiment-synthetic-v0.3', 3), authors)
    assert used > bulk and set(flags) == {'slop_name', 'bulk_author'}


def test_label_columns_read_nested_class_labels_and_names():
    features = {'tokens': [{'dtype': 'string', '_type': 'Value'}],
                'ner_tags': [{'_type': 'ClassLabel', 'names': ['O', 'B-PER']}],
                'etpc_label': {'dtype': 'int64', '_type': 'Value'},
                'text': {'dtype': 'string', '_type': 'Value'}}
    assert label_columns(features) == ['ner_tags', 'etpc_label']


def test_segments_ignore_prompt_prefixes_and_short_text():
    passage = 'The committee approved the proposal after a lengthy debate on funding.'
    assert set(segments('text_A: ' + passage)) == set(segments(passage))
    assert not list(segments('positive'))
    assert row_segments({'text': passage, 'label': 1, 'choices': [passage]}) == set(segments(passage))


def test_signature_matches_reuploads_but_not_coincidences(tmp_path):
    info = lambda configs: {'dataset_info': {f'c{i}': {
        'splits': {f's{j}': {'num_examples': n} for j, n in enumerate(rows)},
        'features': {'label': {'_type': 'ClassLabel', 'names': names}}} for i, (rows, names) in enumerate(configs)}}
    emotions = [f'e{i}' for i in range(10)]
    catalog = {'org/sst': signature(info([((8544, 2210), ['negative', 'positive'])])),
               'org/treebanks': signature(info([((1234, 321), ['a', 'b']), ((4321, 777), ['a', 'b'])])),
               'org/emotions': signature(info([((43410, 5426), emotions)]))}
    keys = {}
    for repo, found in catalog.items():
        for key in found:
            keys.setdefault(key, []).append(repo)
    (tmp_path / 'index.json').write_text(json.dumps(keys))
    index = DatasetIndex(tmp_path / 'index.json')
    assert set(index.matches(signature(info([((8544, 2210), ['x', 'y'])])))) == {'org/sst'}  # same split sizes
    assert set(index.matches(signature(info([((8544,), ['negative', 'positive'])])))) == {'org/sst'}  # size + labels
    assert not index.matches(signature(info([((1234, 777), ['x', 'y'])])))  # sizes from different configs
    assert set(index.matches(signature(info([((999,), emotions)])))) == {'org/emotions'}  # large vocabulary
    assert not index.matches(signature(info([((999,), ['negative', 'positive'])])))  # small vocabulary alone


def test_discrete_rejects_free_text_and_continuous_scores():
    preview = [dict(answer=f'Answer number {i} is long enough', score=i / 7, rating=i % 5, choices=['a', 'b', 'c'],
                    stance='pro' if i % 2 else 'con') for i in range(40)]
    assert [n for n in ('answer', 'score', 'rating', 'choices', 'stance') if discrete(n, {}, preview)] == ['rating', 'choices', 'stance']


def test_web_agent_known_pile_grows(monkeypatch, tmp_path):
    from tasksource.task_discovery import match, web_agent
    monkeypatch.setattr(match, 'catalog_index', lambda: ({'snli': {'snli'}}, {'stanfordnlp/snli': {'snli'}}))
    monkeypatch.setattr(match, 'previous_decisions', lambda: [dict(name='Some-Parked', huggingface=['org/some-parked'])])
    (tmp_path / 'earlier.jsonl').write_text(json.dumps(dict(name='lab/found-before')) + '\n')
    known = web_agent.Known([tmp_path / 'earlier.jsonl'])
    known.add('https://huggingface.co/datasets/lab/new-gui', 'found earlier')
    assert known.check(['https://huggingface.co/datasets/stanfordnlp/snli', 'SNLI', 'other/some-parked',
                        'lab/found-before', 'lab/new-gui', 'lab/unseen']) == {
        'https://huggingface.co/datasets/stanfordnlp/snli': 'in tasksource', 'SNLI': 'in tasksource',
        'other/some-parked': 'reviewed and declined or parked', 'lab/found-before': 'found earlier',
        'lab/new-gui': 'found earlier', 'lab/unseen': 'new'}


def test_llm_endpoint_is_standard_openai_env(monkeypatch):
    from tasksource import llm
    monkeypatch.setenv('OPENAI_MODEL', 'some-model')
    assert llm.litlm_model() == 'direct/openai/some-model'
    assert llm.litlm_model('openrouter/x/y') == 'openrouter/x/y'
    monkeypatch.delenv('OPENAI_MODEL')
    with pytest.raises(SystemExit):
        llm.model()


def test_train_doubt_gate():
    from tasksource.task_discovery.prospect import train_doubted
    def review(no, yes):
        answers = dict(train=dict(probabilities=dict(no=no, yes=yes, unknown=1 - no - yes)))
        return dict(reviews=[dict(reviewer='jev_review', data=dict(answers=answers))])
    assert train_doubted(review(0.36, 0.03)) and not train_doubted(review(0.1, 0.4))


def test_media_by_task_category():
    from tasksource.task_discovery.prospect import is_media
    assert is_media(dict(modalities=['text'], hub_tags=['task_categories:visual-question-answering']))
    assert is_media(dict(modalities=['image']))
    assert not is_media(dict(modalities=['text'], hub_tags=['task_categories:text-classification']))
