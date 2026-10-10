import json
from types import SimpleNamespace

from datasets import Dataset
from huggingface_hub import DatasetCard
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from scripts import append_jev_browser as release


def test_incremental_browser_patch_keeps_unrelated_configs_and_native_ids(tmp_path, monkeypatch):
    base = tmp_path/'base'
    base.mkdir()
    schema = pa.schema([(key, pa.list_(pa.float64()) if key == 'target' else
                         pa.list_(pa.string()) if key == 'options' else pa.string()) for key in
        ('state','kind','id','options','target','question','source','variant','split','group_id',
         'question_id','example_id','license','license_use')])
    pq.write_table(pa.Table.from_pylist([],schema=schema),base/'schema.parquet')
    infos = [dict(config_name=c, splits=[dict(name=s,num_examples=10,num_bytes=100) for s in
        ('train','validation','test')],dataset_size=300,download_size=30)
        for c in ('default','full','filtered-full','vision')]
    card = DatasetCard('---\n'+yaml.safe_dump({'dataset_info':infos})+'---\nOriginal card.\n')
    card.save(base/'README.md')
    stats = {s:dict(rows=10,sources={'old':10},formats={'Classification':10},families={'old':10},
                   kinds={'choice':10},variants={'direct':10},license_use={'commercial':10})
             for s in ('train','validation','test')}
    (base/'release-audit.json').write_text(json.dumps(dict(splits=stats,mix={s:dict(rows=10,sources=1) for s in stats})))
    (base/'sources.yaml').write_text(yaml.safe_dump(dict(sources={},datasets=[])))
    fake = SimpleNamespace(sha='parent',siblings=[SimpleNamespace(rfilename='vision/old.parquet',blob_id='preserved')])
    monkeypatch.setattr(release,'HfApi',lambda:SimpleNamespace(dataset_info=lambda *a,**kw:fake))
    monkeypatch.setattr(release,'hf_hub_download',lambda repo,name,**kw:str(base/('schema.parquet' if name.endswith('.parquet') else name)))
    monkeypatch.setattr(release,'task_provenance',lambda task:dict(dataset='native/source',revision='pinned',config=task.split('/')[-1]))
    monkeypatch.setattr(release,'source_licenses',lambda tasks:{task:dict(license='cc-by-nc-sa-4.0' if task.startswith('weblinx/') else 'cc-by-4.0',license_use='non-commercial' if task.startswith('weblinx/') else 'commercial') for task in tasks})
    def load(task, **kwargs):
        native_splits = [*stats, *(['test_geo'] if task.startswith('weblinx/') else [])]
        return {split:Dataset.from_list([dict(state=f'{task} {split}',criteria=['one','two'],label=1,
            instructions='Choose.',metadata=json.dumps(dict(id=f'original-{split}',source_row=f'original-{split}',
                trajectory_id=f'trajectory-{split}',page_group_id=f'page-{split}')))]) for split in native_splits}
    monkeypatch.setattr(release,'load_task',load)
    output = tmp_path/'patch'
    manifest, additions = release.build(output)
    updated = DatasetCard.load(str(output/'README.md')).data.dataset_info
    assert [i['splits'][0]['num_examples'] for i in updated] == [16,16,10,10]
    assert manifest['preserved_parquets'] == {'vision/old.parquet':'preserved'}
    assert manifest['splits']['train']['rows'] == 6
    assert manifest['splits']['test']['rows'] == 8
    assert not any(name.startswith(('vision/','filtered-full/')) for name,_ in additions)
    table = pq.read_table(next(output.glob('train-*.parquet')))
    rows = table.to_pylist()
    assert len({r['id'] for r in rows}) == 6
    assert all('original-train' in r['id'] for r in rows)
    assert all(r['license_use']=='non-commercial' for r in rows if r['source'].startswith('weblinx/'))
    assert next(r for r in rows if r['source']=='mind2web/action')['group_id'] == next(r for r in rows if r['source']=='mind2web/dom-element')['group_id']
    assert table.schema.equals(schema)
    audit = json.loads((output/'release-audit.json').read_text())
    assert sum(audit['splits']['train']['license_use'].values()) == 16
    # The same publisher preserves soft numeric targets and checks source groups.
    def scored(task, **kwargs):
        return {s:Dataset.from_list([dict(state=s,kind='score',criteria=['0','2','4'],
            target=[0.5,0.5,0.0],instructions='Predict click score.',
            metadata=json.dumps(dict(id=s,split_group_id=s)))]) for s in stats}
    monkeypatch.setattr(release,'load_task',scored)
    numeric = tmp_path/'numeric'
    release.build(numeric,tasks=['scirepeval/search'],description='Numeric scores.')
    numeric_rows=pq.read_table(next(numeric.glob('train-*.parquet'))).to_pylist()
    assert numeric_rows[0]['target']==[0.5,0.5,0.0]
    updated_audit=json.loads((numeric/'release-audit.json').read_text())
    assert updated_audit['splits']['train']['kinds']['score']==1
    def query_leaking(task, **kwargs):
        return {s:d.map(lambda row: {'metadata':json.dumps(dict(id=s,split_group_id='shared-query'))})
                for s,d in scored(task).items()}
    monkeypatch.setattr(release,'load_task',query_leaking)
    with pytest.raises(AssertionError,match='Source group leakage'):
        release.build(tmp_path/'query-leak',tasks=['scirepeval/search'])
    # A source trajectory shared between train and test must block publication.
    def leaking(task,**kwargs):
        data=load(task,**kwargs)
        return {s:d.map(lambda row: {'metadata':json.dumps(dict(id=s,source_row=s,trajectory_id='shared',page_group_id=s))}) for s,d in data.items()}
    monkeypatch.setattr(release,'load_task',leaking)
    with pytest.raises(AssertionError,match='Trajectory leakage'):
        release.build(tmp_path/'leaking')
