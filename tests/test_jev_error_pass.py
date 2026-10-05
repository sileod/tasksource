import json

from scripts.jev_error_pass import parse


def rows():
    return [{'id': str(i), 'example_id': str(i), 'source': 'sample', 'kind': 'choice',
             'state': 'text', 'question': 'Which?', 'options': ['a', 'b'], 'target': [1, 0]}
            for i in range(2)]


def test_parses_multiline_objects_and_inline_arrays():
    items = [{'i': 1, 'verdict': 'ok', 'reason': 'fine'},
             {'i': 2, 'verdict': 'malformed', 'reason': 'missing'}]
    for text in [json.dumps(items), '\n'.join(json.dumps(item, indent=2) for item in items)]:
        assert set(parse(text, rows())) == {'0', '1'}


def test_rejects_out_of_bounds_and_boolean_indices():
    for index in [0, -1, 3, True, 1.5]:
        assert not parse(json.dumps({'i': index, 'verdict': 'wrong'}), rows())


def test_replay_skips_already_completed_rows():
    text = json.dumps([{'i': 1, 'verdict': 'ok'}, {'i': 2, 'verdict': 'ok'}])
    assert set(parse(text, [None, rows()[1]])) == {'1'}


def test_task_review_recovers_recast_ids_only_from_matching_direct_source(tmp_path):
    import csv
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scripts.review_jev_filtering import prepare

    source = 'civil_comments/severe_toxicity_share'
    record = {'source': source, 'id': 'original-source:train:1', 'example_id': 'one', 'verdict': 'ok'}
    verdicts = tmp_path / 'verdicts.jsonl'
    verdicts.write_text(json.dumps(record) + '\n')
    shards = tmp_path / 'shards'
    shards.mkdir()
    pq.write_table(pa.Table.from_pylist([
        {'id': record['id'], 'source': 'other-task', 'variant': 'direct', 'state': 'wrong source'},
        {'id': record['id'], 'source': source, 'variant': 'label-check', 'state': 'wrong variant'},
        {'id': record['id'], 'source': source, 'variant': 'direct', 'state': 'correct source'},
    ]), shards / 'train-civil-comments-severe-toxicity-share-hash.parquet')
    output = tmp_path / 'review'
    prepare(verdicts, output, [source], shards)
    recovered = json.loads((output / 'examples.jsonl').read_text())
    assert recovered['row']['state'] == 'correct source'
    assert json.loads((output / 'sampling.json').read_text())['missing_source_text'] == []
    assert next(csv.DictReader((output / 'per-task.csv').open()))['review_status'] == 'pending'
    assert not (output / 'bad-examples.csv').exists()
