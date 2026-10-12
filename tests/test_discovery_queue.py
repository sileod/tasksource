import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from urllib.error import HTTPError

from tasksource.task_discovery import scheduled as queue


def test_queue_resumes_in_order_and_deduplicates_prior_and_earlier_stages(tmp_path,monkeypatch):
    monkeypatch.setattr(queue,'STAGES',{'classification':('text-classification',),'zero-shot-classification':('zero-shot-classification',)})
    prior=tmp_path/'prior.jsonl';queue.write_rows(prior,[dict(name='Prior')])
    output=tmp_path/'queue';calls=[];fetches=[];fail=[2]
    monkeypatch.setattr(queue.time,'sleep',lambda delay:None)
    monkeypatch.setattr(queue,'catalog_index',lambda:({'known':{'known/task'}},{}))
    monkeypatch.setattr(queue,'catalog_references',lambda:{})
    monkeypatch.setattr(queue,'previous_decisions',lambda:[])
    monkeypatch.setattr(queue,'queued_sources',lambda:{})
    def retrieve(cache,tasks,**kwargs):
        fetches.append(tasks)
        assert kwargs['limit'] is None and kwargs['languages']==['en']
        names=['Known','Prior','A','Shared'] if tasks[0]=='text-classification' else ['Shared','B']
        return [dict(name=n) for n in names],dict(exhaustive=True)
    monkeypatch.setattr(queue,'retrieve',retrieve)
    def control(models,path):queue.save_json(path,dict(passed=True));return models
    monkeypatch.setattr(queue,'control',control)
    def run(command,**kwargs):
        rows=queue.read_rows(Path(command[4]));calls.append(rows[0]['name'])
        if rows[0]['name']=='Shared' and fail[0]:fail[0]-=1;return SimpleNamespace(returncode=1)
        batch=Path(command[command.index('--output')+1])
        queue.save_json(batch/'manifest.json',dict(selected=len(rows)))
        queue.write_rows(batch/'handoff.jsonl',rows)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(queue.subprocess,'run',run)
    monkeypatch.setattr(sys,'argv',['queue','--output',str(output),'--exclude',str(prior),'--batch-size','1'])
    queue.main()
    assert json.loads((output/'status.json').read_text())['status']=='batch_failed'
    queue.main()
    assert json.loads((output/'status.json').read_text())['status']=='complete'
    assert calls==['A','Shared','Shared','Shared','B']
    assert fetches==[('text-classification',),('zero-shot-classification',)]
    assert [r['name'] for r in queue.read_rows(output/'handoff.jsonl')]==['A','Shared','B']
    queue.main()
    assert calls==['A','Shared','Shared','Shared','B']
    report=(output/'report.md').read_text()
    assert report.index('| classification |')<report.index('| zero-shot-classification |')


def test_reviewer_control_quarantines_collapsed_question_answers(tmp_path,monkeypatch):
    monkeypatch.setattr(queue,'configure_key',lambda:None)
    def complete(states,model,**kwargs):
        assert kwargs['caching'] is False
        labels={'train':'yes','resources':'no','annotators':'unknown'}
        if model=='broken':labels={q:'yes' for q in labels}
        answers={q:{'probabilities':{label:0.99}} for q,label in labels.items()}
        return [SimpleNamespace(failed=False,data={'answers':answers},model_used=model)]
    monkeypatch.setitem(sys.modules,'litlm',SimpleNamespace(complete=complete))
    monkeypatch.setitem(sys.modules,'litlm_cli',SimpleNamespace(_record=lambda r:dict(model=r.model_used,data=r.data)))
    path=tmp_path/'control.json'
    assert queue.control(['broken','good'],path)==['good']
    data=json.loads(path.read_text())
    assert [r['passed'] for r in data['results']]==[False,True]


def test_viewer_429_retry_is_bounded_cached_and_paced(tmp_path,monkeypatch):
    from importlib import import_module
    linker=import_module('tasksource.task_discovery.link')
    calls=[];pauses=[]
    monkeypatch.setattr(linker,'_VIEWER_LAST',0)
    monkeypatch.setattr(linker.time,'sleep',pauses.append)
    def fetch(request,**kwargs):
        calls.append(request.full_url)
        if len(calls)==1:raise HTTPError(request.full_url,429,'Rate limited',{'Retry-After':'2'},None)
        return io.BytesIO(b'{"splits":[]}')
    monkeypatch.setattr(linker,'urlopen',fetch)
    url='https://datasets-server.huggingface.co/splits?dataset=org/data'
    result=linker.cached_json(url,tmp_path)
    assert len(calls)==2 and 2 in pauses
    assert linker.cached_json(url,tmp_path,offline=True)==result and len(calls)==2
    assert not list(tmp_path.glob('*.tmp'))
