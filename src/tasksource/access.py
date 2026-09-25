from .preprocess import Preprocessing, MultipleChoiceFields, SoftLabeling, add_question
from .jev.options import JEV_MAX_MC_OPTIONS
import re
from urllib.parse import unquote
import numpy as np
import pandas as pd
from . import tasks, recast as recast_module
from .metadata import dataset_rank
from .metadata.originals import ORIGINALS
from .metadata.canonical import CANONICAL
from .licenses import LICENSE_USES, fetch_card_licenses, source_license
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

def list_tasks(tasks_path=f'{os.path.dirname(__file__)}/tasks.py', multilingual=False, instruct=False, excluded=(),
               soft=False, min_annotators=None, license_use=None):
    """The task catalog as a DataFrame; ``excluded`` holds substrings of task ids to leave out.

    ``soft=False`` lists tasks with hard labels, SoftLabeling annotations by their
    hard view (``task_type`` Classification or MultipleChoice). ``soft=True`` prefers
    soft labels: SoftLabeling annotations are listed as such, including those with
    soft labels only, and the hard tasks they replace are left out. The
    ``soft_labels`` column says whether a task has a soft view. With
    ``min_annotators``, vote shares recorded from fewer annotators count as hard:
    listed by their hard view if they have one, and replacing nothing.

    ``license_use`` (commercial, non-commercial, unspecified, or several) keeps only
    tasks with that use and adds ``license`` and ``license_use`` columns; see
    ``task_licenses``.

    Each call returns a fresh copy, so callers may edit it without affecting later calls."""
    df = _list_tasks(tasks_path, multilingual, instruct, tuple(excluded), soft, min_annotators)
    if license_use is not None:
        uses = {license_use} if isinstance(license_use, str) else set(license_use)
        if uses - set(LICENSE_USES):
            raise ValueError(f"license_use must be among {LICENSE_USES}")
        licenses = _task_licenses(df, multilingual)
        df = df.assign(license=[x["license"] for x in licenses], license_use=[x["license_use"] for x in licenses])
        df = df[df.license_use.isin(uses)]
    return df.copy()

@cache
def _list_tasks(tasks_path, multilingual, instruct, excluded, soft=False, min_annotators=None):
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
            task_type = value.__class__.__name__
            if isinstance(value, SoftLabeling):
                use_soft = soft and value.enough_annotators(min_annotators)
                task_type = "SoftLabeling" if use_soft else value.hard_type
                if task_type is None:  # soft labels only
                    continue
            dataset_name, config_name, task_name = parse_var_name(key)
            dataset_name = (value.dataset_name if value.dataset_name else dataset_name)
            config_name = (value.config_name if value.config_name else config_name)
            hasattr(value,key)
            l+=[{'dataset_name': dataset_name,
                 'config_name' : config_name,
                 'task_name': task_name,
                 'preprocessing_name': key,
                'task_type': task_type,'mapping': value,
                'soft_labels': isinstance(value, SoftLabeling),
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
    if soft:  # a soft view supersedes the hard tasks it names
        replaced = {r for m, t in zip(df.mapping, df.task_type) if t == "SoftLabeling" for r in m.replaces}
        df = df[~df.id.isin(replaced)]
    if instruct:  # improper labels (regression targets ...) are fine as soft labels
        df=df[(df.task_type == "SoftLabeling") | df.id.map(lambda x: not any(a in x for a in recast_module.improper_labels))]
    df=df[df.id.map(lambda x: not any(a in x for a in excluded))]  # excluded holds substrings of task ids
    return df

#task_df =list_tasks()
#mtask_df =list_tasks(multilingual=True)

RAW_BUILDERS = {"csv", "json", "parquet", "text"}

def _task_repos(mapping, dataset_name, task_id):
    """Repos a task loads (its dataset, hf:// data files) and the originals behind them."""
    loaded = set(re.findall(r"hf://datasets/([\w.-]+/[\w.-]+)", str(mapping.load_dataset_kwargs)))
    if dataset_name not in RAW_BUILDERS:
        loaded.add(dataset_name)
    return sorted(loaded | {o for key in loaded | {task_id} for o in ORIGINALS.get(key, [])})

def _task_licenses(df, multilingual, cards=None):
    return [source_license(("multilingual/" if multilingual else "") + i, _task_repos(m, d, i), cards)
            for i, m, d in zip(df.id, df.mapping, df.dataset_name)]

def task_licenses(task_ids=None, multilingual=False, fresh=False):
    """License of each task: ``license``, ``license_use`` and the sources they come from.

    ``license`` lists the license of the Hub card of each repo the task loads (and of the
    original behind a tasksource copy) and the licenses recorded by the Data Provenance
    Initiative, marked ``(DPI)``. ``license_use`` is ``non-commercial`` if any is
    non-commercial or academic-only, else ``commercial`` if one allows commercial use,
    else ``unspecified``. Cards come from a checked-in snapshot; ``fresh=True`` reads
    the current cards from the Hub. A best-effort filter, not legal advice."""
    df = _every_task(multilingual=multilingual)
    if task_ids is not None:
        missing = set(task_ids) - set(df.id)
        if missing:
            raise KeyError(f"unknown tasks: {sorted(missing)}")
        df = df[df.id.isin(task_ids)]
    cards = fetch_card_licenses({r for row in df.itertuples() for r in _task_repos(row.mapping, row.dataset_name, row.id)}) \
        if fresh else None
    licenses = _task_licenses(df, multilingual, cards)
    return pd.DataFrame({"id": list(df.id), "dataset_name": list(df.dataset_name),
                         "license": [x["license"] for x in licenses],
                         "license_use": [x["license_use"] for x in licenses],
                         "card_licenses": [x.get("card_licenses", {}) for x in licenses],
                         "dpi_licenses": [x.get("dpi_licenses", []) for x in licenses]})

def _every_task(multilingual=False):
    """Hard and soft views together: every task id, each once."""
    return pd.concat([list_tasks(multilingual=multilingual, soft=s) for s in (True, False)]).drop_duplicates("id")

def hub_datasets(task_ids=None, multilingual=None):
    """Hub dataset ids behind tasks, for the ``datasets:`` field of a model card.

    Lists the repo each task loads from, repos read through hf:// data files,
    and the originals of tasksource copies and mirrors (metadata/originals.py).
    ``task_ids`` defaults to every task; ``multilingual=None`` searches both lists.
    """
    frames = [_every_task(multilingual=m) for m in ([False, True] if multilingual is None else [multilingual])]
    df = pd.concat(frames)
    if task_ids is not None:
        missing = set(task_ids) - set(df.id)
        if missing:
            raise KeyError(f"unknown tasks: {sorted(missing)}")
        df = df[df.id.isin(task_ids)]
    return sorted({repo for row in df.itertuples() for repo in _task_repos(row.mapping, row.dataset_name, row.id)})

def task_provenance(task_id, multilingual=False):
    """Where one task's data comes from: the loading repo and config, repos read
    through hf:// data files, and the originals of tasksource copies and mirrors."""
    df = _every_task(multilingual=multilingual)
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
    urls = sorted({u for u in re.findall(r"https?://[^\s'\"]+", str(kwargs))})  # non-Hub files: no revision to pin
    info = {"dataset": dataset, "config": row.config_name or None, "revision": kwargs.get("revision"),
            "data_files_from": files, "data_file_revisions": {repo: ref for repo, ref in refs.items() if ref},
            "source_urls": urls, "originals": originals}
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
    df = _every_task(multilingual=tasks==lmtasks)
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
    if isinstance(preprocessing.question, dict):
        preprocessing.question = preprocessing.question.get(preprocessing.config_name)
    preprocessing.dataset_name = CANONICAL.get(preprocessing.dataset_name, preprocessing.dataset_name)
    return preprocessing

def load_task(id=None, dataset_name=None,config_name=None,task_name=None,preprocessing_name=None,
         max_rows=None, max_rows_eval=None, multilingual=False, instruct=False,
         recast=None, prompted=False, seed=0, data_file_pins=None, soft=None, min_annotators=None,
         **load_dataset_kwargs):
    """Load a standardized task.

    ``data_file_pins`` ({repo: commit}) pins the task's ``hf://`` data files, as
    ``revision`` pins its Hub dataset.

    ``soft`` picks the view of a SoftLabeling annotation: its distributions
    (``soft=True``, the default for the Jev recast) or its majority labels;
    ``min_annotators`` drops soft vote targets from fewer annotators.

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
    soft_labels = isinstance(preprocessing, SoftLabeling) and (recast == "jev" if soft is None else soft)
    if isinstance(preprocessing, SoftLabeling):
        kind, row_options = preprocessing.kind, preprocessing.per_row_options
        preprocessing = preprocessing.view(soft_labels, min_annotators)
    elif soft:
        raise ValueError(f"{id or query} has no soft labels")

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
    dataset.task_type = "SoftLabeling" if soft_labels else preprocessing.__class__.__name__
    dataset.question = question
    if instruct and recast not in (None, "instruct"):
        raise ValueError("Use either instruct=True or recast=..., not both")
    recast = "instruct" if instruct else recast
    if soft_labels and recast not in (None, "jev"):
        raise ValueError("soft labels recast only to jev")
    if recast == "instruct":
        dataset = recast_module.recast_instruct(dataset, question=None if prompted else question, seed=seed)
    elif recast == "jev":
        source_id = id or preprocessing_name or preprocessing.dataset_name
        if not (id or preprocessing_name) and preprocessing.config_name:
            source_id = f"{source_id}/{preprocessing.config_name}"
        dataset = recast_module.recast_jev(dataset, task=source_id, question=None if prompted else question,
                                           ordinal=getattr(preprocessing, "ordinal", False),
                                           kind=kind if soft_labels else None,
                                           row_options=soft_labels and row_options,
                                           group="/".join(filter(None, [preprocessing.dataset_name,
                                                                        preprocessing.config_name])))
    elif recast is not None:
        raise ValueError(f"Unknown recast format: {recast!r}")
    return dataset
