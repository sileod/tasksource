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
