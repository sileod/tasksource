import hashlib
import json

from scripts.repackage_dataset import text
from tasksource.jev.recast import recast_jev
from tasksource.tasks import discosense


def test_discosense_mirror_and_recast_preserve_connective_gold_and_native_splits(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows = {}
    paths, hashes = {}, {}
    for split in ('train', 'test'):
        row = dict(idx=0, context=f'{split}: It rained.', marker='as a result', label=2,
                   option_0='The ground was dry.', option_1='It had not rained.',
                   option_2='The ground became wet.', option_3='It was sunny all day.')
        rows[split] = [row]
        raw = json.dumps(rows[split]).encode()
        filename = f'discosense_{split}.json'
        paths[filename] = tmp_path/filename
        paths[filename].write_bytes(raw)
        hashes[split] = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(text, 'DISCOSENSE_HASHES', hashes)
    monkeypatch.setattr(text, 'hf_hub_download', lambda repo, filename, **kwargs: paths[filename])
    source = text.discosense()
    assert set(source) == {'train', 'test'}
    for split in source:
        assert source[split].to_list() == rows[split]
    canonical = discosense(source, gold_first=False, max_options=None)
    assert set(canonical) == {'train', 'test'}
    assert canonical['train'][0]['inputs'] == 'train: It rained.\nas a result'
    assert canonical['train'][0]['choice2'] == 'The ground became wet.'
    assert canonical['train'][0]['labels'] == 2
    recast = recast_jev(canonical, task='discosense', question=discosense.question)
    row = recast['train'][0]
    assert 'as a result' in row['state'] and 'discourse connective' in row['instructions']
    assert row['answer'] == row['criteria'][row['label']] == 'The ground became wet.'
    report = json.loads((tmp_path/'build/discosense-release/provenance.json').read_text())
    assert report['splits'] == {'train': 1, 'test': 1}
    assert report['context_marker_overlap'] == 0
