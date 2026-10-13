import json

from scripts.append_jev_vision import request_key, select_additions


def row(identity, source='vision/clevr/count', image=None, options=None):
    return dict(id=identity, source=source, state=identity, question='Count?', kind='choice',
                options=options or ['zero', 'one'], target=[1., 0.],
                metadata=json.dumps({'image_ids': [image or identity]}))


def test_append_preserves_budget_and_round_robins_family_views():
    old = [row('old')]
    candidates = [row('a'), row('b'), row('c', 'vision/clevr/yesno'), row('d', 'vision/rico-widget/grid7')]
    selected, dropped = select_additions(candidates, old, [], 3)
    assert selected == [0, 2, 3]
    assert dropped == {'family_cap': 1}
    assert old == [row('old')]


def test_append_rejects_permutations_evaluation_images_and_rendered_originals():
    old, evaluation = row('old'), row('eval', image='held-out')
    duplicate = dict(old, id='new-id', options=['one', 'zero'], target=[0., 1.])
    assert request_key(old) == request_key(duplicate)
    overlap = row('overlap', image='held-out')
    augmented = row('marked', image='rendered')
    augmented['metadata'] = json.dumps({'image_ids': ['rendered'],
        'augmentation': {'original_image_sha256': 'held-out'}})
    restricted = row('restricted')
    restricted['metadata'] = json.dumps({'image_ids': ['restricted'], 'source_redistribution': 'restricted'})
    selected, dropped = select_additions([duplicate, overlap, augmented, restricted, row('new')], [old], [evaluation], 20)
    assert selected == [4]
    assert dropped == {'existing_or_duplicate_request': 1, 'published_evaluation_image': 2,
                       'restricted_redistribution': 1}


def test_explicit_batch_is_not_published_after_a_source_failure(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pytest
    import scripts.append_jev_vision as publisher
    (tmp_path/'baseline.json').write_text(json.dumps(dict(cap=20,seed=0,train=[],evaluation=[],parquets={})))
    monkeypatch.setattr(publisher,'HfApi',lambda:object())
    monkeypatch.setattr(publisher,'HfFileSystem',lambda:object())
    def build(args):
        if args.tasks==['bad']:raise ValueError('unavailable')
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'sources.yaml').write_text('sources:\n  vision/good: {}\n')
        (args.output/'build-manifest.json').write_text('{}')
        return {'train':[]}
    monkeypatch.setattr(publisher,'build_vision',build)
    args=SimpleNamespace(output=tmp_path,cap=20,seed=0,tasks=['good','bad'],push=True)
    with pytest.raises(ValueError,match='publication stopped'):
        publisher.run(args)
    assert json.loads((tmp_path/'progress.json').read_text())['vision/bad']['status']=='failed'
