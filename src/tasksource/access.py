from .preprocess import Preprocessing, MultipleChoiceFields, add_question
from .jev.options import JEV_MAX_MC_OPTIONS
import re
from urllib.parse import unquote
import numpy as np
import pandas as pd
from . import tasks, recast as recast_module
from .metadata import dataset_rank
from .metadata.originals import ORIGINALS
from .metadata.canonical import CANONICAL
from datasets import load_dataset, Dataset, DatasetDict, IterableDatasetDict
import funcy as fc
import os
import copy
from sorcery import dict_of
from functools import cache
import random


class lazy_mtasks:
    def __getattr__(self, name):
        from . import multilingual_tasks
        return getattr(multilingual_tasks, name)

    def __dir__(self):
        from . import multilingual_tasks
        return dir(multilingual_tasks)
lmtasks=lazy_mtasks()

def parse_var_name(s):
    config_name,task_name = None,None
    if '__' in s and '___' not in s: # dataset__task
        dataset_name, task_name = s.split('__') 
    elif '__' not in s.replace('___','') and '___' in s: #dataset___config
        dataset_name, config_name = s.split('___') 
    elif  '___' in s and '__' in s.split('___')[1]: #dataset___config__task
        dataset_name, config_task=s.split('___')
        config_name,task_name = config_task.split('__')
    else: # dataset 
        dataset_name = s
    return dataset_name,config_name,task_name

def pretty_name(x):
    dn = x.dataset_name.split("/")[-1]   
    cn = x.config_name if x.config_name else ""
    tn = x.task_name if x.task_name else ""
    return f"{dn}/{cn}/{tn}".replace('//','/').rstrip('/')

def list_tasks(tasks_path=f'{os.path.dirname(__file__)}/tasks.py', multilingual=False, instruct=False, excluded=()):
    """The task catalog as a DataFrame; ``excluded`` holds substrings of task ids to leave out.

    Each call returns a fresh copy, so callers may edit it without affecting later calls."""
    return _list_tasks(tasks_path, multilingual, instruct, tuple(excluded)).copy()

@cache
def _list_tasks(tasks_path, multilingual, instruct, excluded):
    if multilingual:
        tasks_path=tasks_path.replace('/tasks.py','/multilingual_tasks.py')
    task_order = open(tasks_path).readlines()
    task_order = [x.split('=')[0].rstrip() for x in task_order if '=' in x]
    task_order = [x for x in task_order if x.isidentifier()]
    task_order = fc.flip(dict(enumerate(task_order)))

    l = []
    _tasks = (lmtasks if multilingual else tasks)

    for key in dir(_tasks):
        if key not in task_order:
            continue
        value=getattr(_tasks, key)
        if isinstance(value,Preprocessing):
            dataset_name, config_name, task_name = parse_var_name(key)
            dataset_name = (value.dataset_name if value.dataset_name else dataset_name)
            config_name = (value.config_name if value.config_name else config_name)
            hasattr(value,key)
            l+=[{'dataset_name': dataset_name,
                 'config_name' : config_name,
                 'task_name': task_name,
                 'preprocessing_name': key,
                'task_type': value.__class__.__name__,'mapping': value,
                'rank':task_order.get(key,None)}]   
    df=pd.DataFrame(l).explode('config_name')
    df=df.astype(object).where(df.notna(), None)  # keep None: pandas 3 string columns and explode turn it into NaN
    df = df.sort_values('rank').reset_index(drop=True)
    df['id'] = df.apply(
        lambda x: x.mapping.task_id.format(config_name=x.config_name)
        if x.mapping.task_id and "{config_name}" in x.mapping.task_id
        else (x.mapping.task_id or pretty_name(x)), axis=1)
    df.insert(0, 'id', df.pop('id'))
    df['dataset_name'] = df.dataset_name.map(lambda n: CANONICAL.get(n, n))  # after the ids, which keep the short names
    del df['rank']
    if instruct:
        df=df[df.id.map(lambda x: not any(a in x for a in recast_module.improper_labels))]
    df=df[df.id.map(lambda x: not any(a in x for a in excluded))]  # excluded holds substrings of task ids
    return df

#task_df =list_tasks()
#mtask_df =list_tasks(multilingual=True)

RAW_BUILDERS = {"csv", "json", "parquet", "text"}

def hub_datasets(task_ids=None, multilingual=None):
    """Hub dataset ids behind tasks, for the ``datasets:`` field of a model card.

    Lists the repo each task loads from, repos read through hf:// data files,
    and the originals of tasksource copies and mirrors (metadata/originals.py).
    ``task_ids`` defaults to every task; ``multilingual=None`` searches both lists.
    """
    frames = [list_tasks(multilingual=m) for m in ([False, True] if multilingual is None else [multilingual])]
    df = pd.concat(frames)
    if task_ids is not None:
        missing = set(task_ids) - set(df.id)
        if missing:
            raise KeyError(f"unknown tasks: {sorted(missing)}")
        df = df[df.id.isin(task_ids)]
    repos = set()
    for row in df.itertuples():
        loaded = set(re.findall(r"hf://datasets/([\w.-]+/[\w.-]+)", str(row.mapping.load_dataset_kwargs)))
        if row.dataset_name not in RAW_BUILDERS:
            loaded.add(row.dataset_name)
        repos.update(loaded)
        for key in loaded | {row.id}:
            repos.update(ORIGINALS.get(key, []))
    return sorted(repos)

def task_provenance(task_id, multilingual=False):
    """Where one task's data comes from: the loading repo and config, repos read
    through hf:// data files, and the originals of tasksource copies and mirrors."""
    df = list_tasks(multilingual=multilingual)
    row = df[df.id == task_id]
    if row.empty:
        raise KeyError(f"unknown task: {task_id}")
    row = next(row.itertuples())
    kwargs = row.mapping.load_dataset_kwargs or {}
    # hf:// data files may pin a ref: hf://datasets/owner/name@refs%2Fconvert%2Fparquet/...
    refs = {repo: unquote(ref) or None for repo, ref in
            re.findall(r"hf://datasets/([\w.-]+/[\w.-]+)(?:@([^/\s'\"]+))?", str(kwargs))}
    files = sorted(refs)
    dataset = None if row.dataset_name in RAW_BUILDERS else row.dataset_name
    originals = sorted({o for key in {dataset, row.id, *files} if key for o in ORIGINALS.get(key, [])} - {dataset})
    info = {"dataset": dataset, "config": row.config_name or None, "revision": kwargs.get("revision"),
            "data_files_from": files, "data_file_revisions": {repo: ref for repo, ref in refs.items() if ref},
            "originals": originals}
    return {k: v for k, v in info.items() if v}

def _format_loader_kwargs(value, **context):
    """Resolve per-config placeholders in generic-builder data_files settings."""
    if isinstance(value, str):
        return value.format(**context)
    if isinstance(value, list):
        return [_format_loader_kwargs(v, **context) for v in value]
    if isinstance(value, tuple):
        return tuple(_format_loader_kwargs(v, **context) for v in value)
    if isinstance(value, dict):
        return {k: _format_loader_kwargs(v, **context) for k, v in value.items()}
    return value

_HF_URL = re.compile(r"hf://datasets/([\w.-]+/[\w.-]+)(?:@[^/\s'\"]+)?/")

def pin_hf_urls(value, pins):
    """Point ``hf://datasets/owner/repo[@ref]/...`` data-file URLs at the commits in ``pins`` ({repo: sha})."""
    if isinstance(value, str):
        return _HF_URL.sub(lambda m: f"hf://datasets/{m.group(1)}@{pins[m.group(1)]}/"
                           if pins.get(m.group(1)) else m.group(0), value)
    if isinstance(value, (list, tuple)):
        return type(value)(pin_hf_urls(v, pins) for v in value)
    if isinstance(value, dict):
        return {k: pin_hf_urls(v, pins) for k, v in value.items()}
    return value

def load_preprocessing(tasks=tasks, **kwargs):
    df = list_tasks(multilingual=tasks==lmtasks)
    matches = df[np.logical_and.reduce([df[k] == v for k, v in kwargs.items()] + [np.ones(len(df), bool)])]
    if matches.empty:
        raise KeyError(f"unknown task: {kwargs}")
    if len(matches) > 1:
        raise ValueError(f"{kwargs} matches {len(matches)} tasks ({', '.join(matches.id[:5])}...); pass a task id")
    y = matches.iloc[0]
    preprocessing= copy.copy(getattr(tasks, y.preprocessing_name))
    for c in 'dataset_name','config_name':
        if not isinstance(getattr(preprocessing,c), str):
             setattr(preprocessing,c,getattr(y,c))
    preprocessing.dataset_name = CANONICAL.get(preprocessing.dataset_name, preprocessing.dataset_name)
    return preprocessing

def load_task(id=None, dataset_name=None,config_name=None,task_name=None,preprocessing_name=None,
         max_rows=None, max_rows_eval=None, multilingual=False, instruct=False,
         recast=None, prompted=False, seed=0, data_file_pins=None, **load_dataset_kwargs):
    """Load a standardized task.

    ``data_file_pins`` ({repo: commit}) pins the task's ``hf://`` data files, as
    ``revision`` pins its Hub dataset.

    ``prompted=True`` appends the annotation's ``question`` to the inputs;
    otherwise the instruct and Jev recasts use it as their instruction. The
    question is also available as ``dataset.question``.
    """
    query = dict_of(id, dataset_name, config_name, task_name,preprocessing_name)
    query = {k:v for k,v in query.items() if v}
    if 'dataset_name' in query:
        query['dataset_name'] = CANONICAL.get(query['dataset_name'], query['dataset_name'])
    _tasks = (lmtasks if multilingual else tasks)
    preprocessing = load_preprocessing(_tasks, **query)

    # All tasksource datasets are now data-only (Parquet); no loading script,
    # so trust_remote_code is unnecessary (and rejected by datasets>=3 for scripts).
    load_dataset_kwargs.pop("trust_remote_code", None)

    source_kwargs = _format_loader_kwargs(
        copy.deepcopy(preprocessing.load_dataset_kwargs),
        config_name=preprocessing.config_name or "",
    )
    source_kwargs = pin_hf_urls(source_kwargs, data_file_pins or {})
    source_kwargs.update(load_dataset_kwargs)
    source_config = preprocessing.config_name
    if preprocessing.dataset_name in {"csv", "json", "text", "parquet"}:
        source_config = None
    dataset = load_dataset(
        preprocessing.dataset_name, source_config, **source_kwargs
    )
    pre_processed = False
    if isinstance(dataset, IterableDatasetDict):
        # Keep bounded streaming sources (e.g. multilingual sentiment pools)
        # bounded before materializing them. Apply source filtering first, then
        # deterministically shuffle a finite buffer and take the same limits
        # used by ordinary Tasksource sampling.
        dataset = preprocessing.pre_process(dataset)
        pre_processed = True
        materialized = {}
        for split, rows in dataset.items():
            limit = max_rows if split == "train" else max_rows_eval
            if limit:
                rows = rows.shuffle(seed=seed, buffer_size=max(10_000, limit)).take(limit)
            materialized[split] = Dataset.from_list(list(rows))
        dataset = DatasetDict(materialized)
    options = {}
    if recast == "jev" and isinstance(preprocessing, MultipleChoiceFields):
        # Jev permutes criteria itself and needs every source option.
        options = dict(gold_first=False, max_options=JEV_MAX_MC_OPTIONS)
    dataset= preprocessing(dataset,max_rows, max_rows_eval, seed=seed, pre_processed=pre_processed, **options)
    question = getattr(preprocessing, "question", None)
    if prompted:
        dataset = add_question(dataset, question)
    dataset.task_type = preprocessing.__class__.__name__
    dataset.question = question
    if instruct and recast not in (None, "instruct"):
        raise ValueError("Use either instruct=True or recast=..., not both")
    recast = "instruct" if instruct else recast
    if recast == "instruct":
        dataset = recast_module.recast_instruct(dataset, question=None if prompted else question, seed=seed)
    elif recast == "jev":
        source_id = id or preprocessing_name or preprocessing.dataset_name
        if not (id or preprocessing_name) and preprocessing.config_name:
            source_id = f"{source_id}/{preprocessing.config_name}"
        dataset = recast_module.recast_jev(dataset, task=source_id, question=None if prompted else question,
                                           ordinal=getattr(preprocessing, "ordinal", False))
    elif recast is not None:
        raise ValueError(f"Unknown recast format: {recast!r}")
    return dataset
