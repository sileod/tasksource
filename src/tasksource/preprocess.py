from collections.abc import Iterable
from dotwiz import DotWiz
from dataclasses import dataclass, field
import dataclasses
from typing import Union
import itertools
import re
import funcy as fc
import exrex 
import magicattr 
import numpy as np
import copy
import datasets
import time

MAX_MC_OPTIONS = 4

def get_column_names(dataset):
    cn = dataset.column_names
    if type(cn)==dict:
        return set(fc.flatten(cn.values()))
    else:
        return set(cn)


def sample_dataset(dataset,n=10000, n_eval=1000,seed=0):
    for k in dataset:
        n_k=(n if k=='train' else n_eval)
        if n_k and len(dataset[k])>n_k:
            dataset[k]=dataset[k].train_test_split(train_size=n_k,seed=seed)['train']
    return dataset

class Preprocessing(DotWiz):
    default_splits = ('train','validation','test')
    _instances = []

    def __post_init__(self):
        Preprocessing._instances+=[self]

    @staticmethod
    def __map_to_target(x,fn=lambda x:None, target=None):
        x[target]=fn(x)
        return x
        
    def on_sampled(self, dataset):
        """Work deferred until rows are split and sampled (the soft view's distributions)."""
        return dataset

    def load(self):
        return self(datasets.load_dataset(
            self.dataset_name, self.config_name, **self.load_dataset_kwargs))

    def __call__(self,dataset, max_rows=None, max_rows_eval=None,seed=0, pre_processed=False):
        """``pre_processed=True`` skips ``pre_process`` (already applied, e.g. before streaming sampling)."""
        if not pre_processed:
            dataset = self.pre_process(dataset)

        # manage splits
        for k,v in zip(self.default_splits, self.splits):
            if v and k!=v:
                dataset[k]=dataset[v]
                del dataset[v]
            if k in dataset and not v: # obfuscated label
                del dataset[k]
        dataset = fix_splits(dataset)

        for k in list(dataset.keys()):
            if k not in self.default_splits:
                del dataset[k]
        dataset = sample_dataset(dataset, max_rows, max_rows_eval,seed=seed)
        dataset = self.on_sampled(dataset)
        
        # field annotated with a string
        substitutions = {v:k for k,v in self.to_dict().items()
            if (k and k not in {'splits','dataset_name','config_name','task_id','load_dataset_kwargs','question'}
            and type(v)==str and k!=v)}

        dataset=dataset.remove_columns([c for c in substitutions.values() if c in dataset['train'].features and c not in substitutions])
        dataset=dataset.rename_columns(substitutions)

        # field annotated with a function                                
        for k in self.to_dict().keys():
            v=getattr(self, k)
            if callable(v) and k not in {"post_process","pre_process","load","question"}:
                dataset=dataset.map(self.__map_to_target,
                                    fn_kwargs={'fn':v,'target':k})

        dataset=dataset.remove_columns(  # question is metadata, never a column
            get_column_names(dataset)-(set(self.to_dict().keys())-{'question'}))
        dataset = fix_labels(dataset)
        if self.label_values:
            dataset = cast_explicit_label_values(dataset, self.label_values)
        dataset = fix_splits(dataset) # again: label mapping changed
        dataset = self.post_process(dataset)
        return dataset


@dataclass
class cat(Preprocessing):
    fields:Union[str,list]=None
    separator:str=' '
        
    def __call__(self, example=None):
        values = [example[f] for f in self.fields]
        if all(isinstance(v, (list, tuple)) for v in values):  # batched
            return [self.separator.join(str(v) for v in row if v is not None).strip() for row in zip(*values)]
        return self.separator.join(str(v) for v in values if v is not None).strip()


def pretty(f):
    class pretty_f(DotWiz):
        def __init__(self,*args):
            self.__f_arg = f(*args)
            for a in args:
                setattr(self,'value',a)
                
        def __call__(self, *args,**kwargs):
            return self.__f_arg(*args,**kwargs)

        def __repr__(self):
            return f"{self.__f_arg.__qualname__ .split('.')[0]}({self.value})"
    return pretty_f

class dotgetter:
    def __init__(self, path=''):
        self.path=path

    def __bool__(self):
        return bool(self.path)

    def __getattr__(self, k):
        return self.__class__(f'{self.path}.{k}'.lstrip('.'))
    
    def __getitem__(self, i):
        return self.__class__(f'{self.path}[{i}]')

    def __call__(self, example=None):
        return magicattr.get(DotWiz(example), self.path)

    def __hash__(self):
        return hash(self.path)


@dataclass
class ClassificationFields(Preprocessing):
    sentence1:str='sentence1'
    sentence2:str='sentence2'
    labels:str='labels'

@dataclass
class Seq2SeqLMFields(Preprocessing):
    prompt:str='prompt'
    output:str='output'

@dataclass
class TokenClassificationFields(Preprocessing):
    tokens:str='tokens'
    labels:str='labels'
        
@dataclass
class MultipleChoiceFields(Preprocessing):
    inputs:str='input'
    choices:Iterable=tuple()
    labels:str='labels'
    choices_list:str=None
    def __post_init__(self):
        for i, c in enumerate(self.choices):
            setattr(self,f'choice{i}',c)
        delattr(self,'choices')
        if not self.choices_list:
            delattr(self,'choices_list')
    
    def __call__(self,dataset, *args, gold_first=True, max_options=MAX_MC_OPTIONS, **kwargs):
        """``gold_first=False`` keeps source option order and, with
        ``max_options=None``, every option (padding short rows with None)."""
        dataset = super().__call__(dataset, *args, **kwargs)
        if self.choices_list:
            dataset = dataset.filter(lambda x: 1<len(x['choices_list']))
            lengths = [len(x) for k in dataset for x in dataset[k]['choices_list']]
            if gold_first:
                n_options = min(MAX_MC_OPTIONS,min(lengths))
                dataset = dataset.map(self.flatten_choice_list, fn_kwargs={'n_options':n_options})
            else:
                n_options = max(lengths) if max_options is None else min(max_options, max(lengths))
                features = dataset['train'].features.copy()  # explicit: a batch of padding alone would type as null
                del features['choices_list']
                features.update({f'choice{i}': datasets.Value('string') for i in range(n_options)})
                dataset = dataset.map(self.ordered_choice_list, fn_kwargs={'n_options':n_options},
                                      remove_columns=['choices_list'], features=features)
        elif gold_first:
            dataset = dataset.map(self.sample_choices, fn_kwargs={'n_options':MAX_MC_OPTIONS})
        elif max_options is not None:
            dataset = dataset.map(self.ordered_sample_choices, fn_kwargs={'n_options':max_options})
        return dataset

    @staticmethod
    def _ordered_subset(choices, label, n_options):
        """Keep the gold answer and the first negatives, in source order."""
        if len(choices) <= n_options:
            return list(choices), label
        if not 0 <= label < len(choices):
            return list(choices[:n_options]), label
        negatives = [i for i in range(len(choices)) if i != label][:n_options-1]
        kept = sorted([label, *negatives])
        return [choices[i] for i in kept], kept.index(label)

    @staticmethod
    def ordered_choice_list(x, n_options=None):
        choices, x['labels'] = MultipleChoiceFields._ordered_subset(
            x['choices_list'], x['labels'], n_options)
        for i in range(n_options):
            x[f'choice{i}'] = choices[i] if i < len(choices) else None
        del x['choices_list']
        return x

    @staticmethod
    def ordered_sample_choices(x, n_options=None):
        names = [c for c in x if 'choice' in c]
        choices, x['labels'] = MultipleChoiceFields._ordered_subset(
            [x[c] for c in names], x['labels'], n_options)
        for c in names:
            del x[c]
        for i,o in enumerate(choices):
            x[f'choice{i}']=o
        return x

    @staticmethod
    def shuffled_with_gold(choices, label, n_options=None):
        """Keep the gold option and the first negatives in source order, then permute them so the
        gold slot carries no signal: deterministic, seeded by the options' text. Options that refer
        to each other by position ("all of the above") keep the source order."""
        from .jev.options import choice_permutation  # at call time: jev imports this module
        kept_indices = sorted([label, *[i for i in range(len(choices)) if i != label][:(n_options or len(choices)) - 1]])
        kept, gold = [choices[i] for i in kept_indices], kept_indices.index(label)
        order = choice_permutation(kept, "\x1f".join(map(str, kept)))
        if order is None:  # options refer to each other by position: keep the source order
            return kept, gold
        return [kept[i] for i in order], order.index(gold)

    @staticmethod
    def flatten_choice_list(x, n_options=None):
        x['choices_list'], x['labels'] = MultipleChoiceFields.shuffled_with_gold(
            x['choices_list'], x['labels'], n_options)
        for i,o in enumerate(x['choices_list']):
            x[f'choice{i}']=o
        del x['choices_list']
        return x

    @staticmethod
    def sample_choices(x, n_options=None):
        names = sorted((c for c in x if re.fullmatch(r'choice\d+', c)), key=lambda c: int(c[6:]))  # choice10 after choice2
        choices = [x[c] for c in names]
        # also when nothing is truncated: many annotations put the gold answer first (labels=constant(0))
        if not 0 <= x['labels'] < len(choices):
            return x
        choices_list, x['labels'] = MultipleChoiceFields.shuffled_with_gold(
            choices, x['labels'], n_options if MAX_MC_OPTIONS else None)
        for c in list(x):
            if 'choice' in c:
                del x[c]
        for i,o in enumerate(choices_list):
            x[f'choice{i}']=o
        return x

@dataclass
class SharedFields:
    splits:list=Preprocessing.default_splits
    dataset_name:str = None
    config_name:str = None
    task_id:str = None
    load_dataset_kwargs:dict = field(default_factory=dict)
    label_values:dict = field(default_factory=dict)
    pre_process: callable = fc.identity
    post_process: callable = fc.identity
    # The task's question when the inputs alone do not say what to predict
    # ("Is this search query a well-formed question?"). It is metadata, not a
    # column: raw loading leaves the inputs untouched, load_task(prompted=True)
    # appends it, and the instruct and Jev recasts use it as their instruction.
    # Questions that vary per row belong in sentence2 or the MC inputs instead.
    question: str = None
    ordinal: bool = False  # labels are an ordered scale, listed in order (Jev asks part as score)
    #language:str="en"
    

@dataclass
class Classification(SharedFields, ClassificationFields): pass

@dataclass
class MultipleChoice(SharedFields, MultipleChoiceFields): pass

@dataclass
class TokenClassification(SharedFields, TokenClassificationFields): pass

@dataclass
class Seq2SeqLM(SharedFields, Seq2SeqLMFields): pass


SOFT_KINDS = ("noul", "score", "choice")
_TOLERANCE = 1e-9  # shares like 2/3 or 1 - 0.8 must pass their own threshold


def soft_target(value, kind, n_options=None, low=0, high=1, step=1):
    """A raw label as a probability distribution, or None when missing or out of range.

    ``value`` is vote counts (a list, normalized), a fraction or mean rating
    (``noul``: rescaled from ``[low, high]`` to one probability), or a level
    (``score``/``choice``: one-hot at ``(value - low) / step``)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (list, tuple)):
        total = sum(value)
        return [v / total for v in value] if total else None
    if kind == "noul":
        return [(value - low) / (high - low)] if low <= value <= high else None
    index = round((value - low) / step)
    return [float(i == index) for i in range(n_options)] if 0 <= index < n_options else None


@dataclass
class SoftLabelingFields(Preprocessing):
    sentence1:str='sentence1'
    sentence2:str='sentence2'
    labels:str='labels'
    options:str='options'

@dataclass
class SoftLabelingView(SharedFields, SoftLabelingFields):
    """The soft view of a SoftLabeling annotation, as loaded: ``labels`` is a distribution over ``options``."""


@dataclass
class SoftLabelingSpec(Preprocessing):
    sentence1:object='sentence1'
    sentence2:object=None
    labels:object='labels'
    kind:str='choice'
    options:object=None
    low:float=0
    high:float=1
    step:float=1
    hard:float=None
    soft_question:str=None
    replaces:tuple=()
    annotators:int=None
    aggregation:str='votes'
    count:object=None
    regression:bool=False


@dataclass
class SoftLabeling(SharedFields, SoftLabelingSpec):
    """An annotation whose label is a distribution: annotator votes, rater shares, survey counts.

    Loaded with ``soft=True``, rows have ``sentence1``/``sentence2``, ``labels``
    (probabilities) and ``options`` (their names; empty for ``noul``). ``kind`` says
    what the distribution is over:

    - ``noul``: one probability, e.g. the share of raters who said yes;
    - ``score``: ordered levels (``options``), e.g. votes over a rating scale;
    - ``choice``: named options, fixed (a list) or per row (a column or function).

    With ``hard`` (the least agreement kept, above 0.5), the annotation also has a
    hard view, loaded by default: the majority label, on rows where at least that
    share agrees. ``noul`` then needs ``options=[no, yes]``; per-row options give a
    MultipleChoice task and fixed ones a Classification. Without ``hard`` the
    labels are only meaningful as distributions, and ``list_tasks`` lists the task
    only with ``soft=True``. ``soft_question`` asks for the distribution when the
    hard ``question`` asks for the majority. ``replaces`` names hard task ids that
    the soft view supersedes (``list_tasks(soft=True)`` leaves them out).

    ``aggregation`` says what the distribution summarizes: ``votes`` (the share
    of annotators choosing each option) or ``mean`` (the mean of graded ratings,
    e.g. similarity or probability judgements). ``annotators`` records how many
    annotators judged a typical item. Vote shares from a few annotators are
    coarse (one of three is 0.33), so ``min_annotators`` (in ``load_task`` and
    ``list_tasks``) keeps soft vote targets only from enough annotators: per row
    when ``count`` (a column or function) or vote counts give it, else by the
    recorded ``annotators``. ``regression=True`` keeps the raw value as the
    default (regression) view of a mean rating that has no majority label.
    """

    def __post_init__(self):
        super().__post_init__()
        if self.kind not in SOFT_KINDS:
            raise ValueError(f"kind must be one of {SOFT_KINDS}, got {self.kind!r}")
        if self.hard is not None and not 0.5 < self.hard <= 1:
            raise ValueError("hard is the least majority share kept, in (0.5, 1]")
        if self.hard is not None and self.kind == "noul" and len(self.options or ()) != 2:
            raise ValueError("a noul hard view needs options=[no, yes]")
        if self.kind != "choice" and not isinstance(self.options, (list, tuple, type(None))):
            raise ValueError("only choice options may vary per row")
        if self.aggregation not in ("votes", "mean"):
            raise ValueError("aggregation is 'votes' or 'mean'")
        if self.regression and (self.hard is not None or self.kind != "noul"):
            raise ValueError("a regression view is for noul mean ratings without a hard view")

    @property
    def per_row_options(self):
        return self.kind == "choice" and not isinstance(self.options, (list, tuple))

    @property
    def hard_type(self):
        """The task type of the default (hard or regression) view, or None for soft labels only."""
        if self.regression:
            return "Classification"
        if self.hard is None:
            return None
        return "MultipleChoice" if self.per_row_options else "Classification"

    def enough_annotators(self, min_annotators):
        """Whether the recorded annotator count supports soft targets (mean ratings always do)."""
        return not min_annotators or self.aggregation == "mean" or (self.annotators or 0) >= min_annotators

    @staticmethod
    def _read(field, x):
        return x[field] if isinstance(field, str) else field(x)

    def _soft(self, x):
        if self.kind == "noul":
            options = []
        else:
            options = [str(o) for o in (self._read(self.options, x) if self.per_row_options else self.options)]
        value = self._read(self.labels, x)
        target = soft_target(value, self.kind, len(options), self.low, self.high, self.step)
        if target is not None and self.kind != "noul" and len(target) != len(options):
            target = None
        if self.count is not None:
            count = self._read(self.count, x)
        elif isinstance(value, (list, tuple)):
            count = sum(value)
        else:
            count = self.annotators
        return {"_soft": target or [], "_options": options, "_count": -1 if count is None else count}

    def _prepare(self, dataset, min_annotators=None, drop_unlabeled=True):
        """Each row's distribution. Rows voted on by fewer than ``min_annotators`` are dropped;
        unlabeled rows too, unless ``drop_unlabeled=False`` (the soft view drops them after
        splitting, so sibling annotations of one source split alike)."""
        few = lambda x: bool(min_annotators) and self.aggregation == "votes" and 0 <= x["_count"] < min_annotators
        keep = lambda x: not few(x) and (len(x["_soft"]) > 0 or not drop_unlabeled)
        def add(rows):
            if not hasattr(rows, "features") or rows.features is None:  # streaming
                return rows.map(self._soft).filter(keep)
            features = rows.features.copy()
            features["_soft"] = datasets.List(datasets.Value("float64"))
            features["_options"] = datasets.List(datasets.Value("string"))
            features["_count"] = datasets.Value("float64")
            def batch(columns):
                soft = [self._soft(dict(zip(columns, values))) for values in zip(*columns.values())]
                return {**columns, **{k: [row[k] for row in soft] for k in ("_soft", "_options", "_count")}}
            return rows.map(batch, batched=True, features=features).filter(keep)
        return type(dataset)({split: add(rows) for split, rows in dataset.items()})

    def _agrees(self, x):
        target = x["_soft"]
        top = max(target[0], 1 - target[0]) if self.kind == "noul" else max(target)
        return top >= self.hard - _TOLERANCE

    def view(self, soft=True, min_annotators=None):
        """The loadable task: SoftLabelingView, or the hard Classification/MultipleChoice
        (the regression Classification for ``regression=True``)."""
        shared = {f.name: getattr(self, f.name) for f in dataclasses.fields(SharedFields)}
        if soft:
            shared["question"] = self.soft_question or self.question
            post_process = self.post_process
            shared["post_process"] = lambda dataset: post_process(dataset.filter(lambda x: len(x["labels"]) > 0))
            spec = self

            class View(SoftLabelingView):
                def on_sampled(self, dataset):  # distributions of the sampled rows only
                    return spec._prepare(dataset, min_annotators, drop_unlabeled=False)

            return View(sentence1=self.sentence1, sentence2=self.sentence2 or "sentence2",
                        labels="_soft", options="_options", **shared)
        if self.regression:
            return Classification(sentence1=self.sentence1, sentence2=self.sentence2 or "sentence2",
                                  labels=self.labels, **shared)
        if self.hard is None:
            raise ValueError("this annotation has soft labels only; load it with soft=True")
        pre_process = self.pre_process
        shared["pre_process"] = lambda dataset: self._prepare(pre_process(dataset)).filter(self._agrees)
        if self.kind == "noul":
            label = lambda x: int(x["_soft"][0] >= self.hard - _TOLERANCE)
        else:
            label = lambda x: int(np.argmax(x["_soft"]))
        if self.per_row_options:
            return MultipleChoice(inputs=self.sentence1, choices_list="_options", labels=label, **shared)
        shared["label_values"] = dict(enumerate(self.options))
        return Classification(sentence1=self.sentence1, sentence2=self.sentence2 or "sentence2",
                              labels=label, **shared)

    def load(self, soft=False, min_annotators=None):
        return self.view(soft, min_annotators).load()

    def __call__(self, dataset, *args, soft=False, min_annotators=None, **kwargs):
        return self.view(soft, min_annotators)(dataset, *args, **kwargs)

get=dotgetter()
constant = pretty(fc.constantly)
regen = lambda x: list(exrex.generate(x))

def name(label_name, classes):
    return lambda x:classes[x[label_name]]

def add_question(dataset, question):
    """Append a task question to the inputs (``load_task(prompted=True)``)."""
    if not question:
        return dataset
    features = dataset["train"].features
    if "inputs" in features:  # MultipleChoice
        field = "inputs"
    elif "sentence1" in features:  # Classification: the question pairs with the text
        field = "sentence2"
    else:  # TokenClassification labels each token; there is no text field to extend
        return dataset
    def prompt(x):
        text = x.get(field)
        return {field: f"{text}\n\n{question}" if text else question}
    return dataset.map(prompt)

def fix_splits(dataset):

    if len(dataset)==1 and "train" not in dataset:
        k = list(dataset)[0]
        dataset['train'] = copy.deepcopy(dataset[k])
        del dataset[k]

    if 'auxiliary_train' in dataset:
        del dataset['auxiliary_train']
    
    if 'test' in dataset and 'labels' in dataset['test'].features: # manage obfuscated labels
        test_labels = set(fc.flatten(dataset['test']['labels']))
        # one placeholder value (-1, None, or a value train never uses); a real single-class test set is kept
        train_labels = set(fc.flatten(dataset['train']['labels'])) if 'train' in dataset and 'labels' in dataset['train'].features else set()
        if len(test_labels)==1 and (test_labels & {-1, None} or not test_labels & train_labels):
            del dataset['test']

    if 'validation' in dataset and 'train' not in dataset:
        train_validation = dataset['validation'].train_test_split(0.5, seed=0)
        dataset['train'] = train_validation['train']
        dataset['validation']=train_validation['test']
    
    if 'validation' in dataset and 'test' not in dataset:
        validation_test = dataset['validation'].train_test_split(0.5, seed=0)
        dataset['validation'] = validation_test['train']
        dataset['test']=validation_test['test']

    if 'train' in dataset and 'validation' not in dataset:
        train_val = dataset['train'].train_test_split(train_size=0.90, seed=0)
        dataset['train'] = train_val['train']
        dataset['validation']=train_val['test']

    if 'test' in dataset and 'validation' not in dataset:
        validation_test = dataset['test'].train_test_split(0.5, seed=0)
        dataset['validation'] = validation_test['train']
        dataset['test']=validation_test['test']

    if 'validation' not in dataset and 'test' not in dataset:
        train_val_test = dataset["train"].train_test_split(train_size=0.90, seed=0)
        val_test = train_val_test["test"].train_test_split(0.5, seed=0)
        dataset["train"] = train_val_test["train"]
        dataset["validation"] = val_test["train"]
        dataset["test"] = val_test["test"]
        
    return dataset 

def fix_labels(dataset, label_key='labels'):
    if type(dataset['train'][label_key][0]) in [int,list,float]:
        return dataset
    if type(dataset['train'][label_key][0])==bool:  # names must be strings: "False", "True"
        features=dataset['train'].features.copy(); features[label_key]=datasets.Value('string')
        dataset=dataset.map(lambda x:{label_key:str(x[label_key])}, features=features)
    labels=set(fc.flatten(list(dataset[k][label_key]) for k in dataset))  # a label seen only in eval splits must not crash
    if set(labels)=={'entailment','neutral','contradiction'}:
        order=lambda x:dict(fc.flip(enumerate(['entailment','neutral','contradiction']))).get(x,x)
    else:
        order=str
    labels=sorted(labels, key=order)
    dataset=dataset.cast_column(label_key, datasets.ClassLabel(names=labels))
    return dataset


def cast_explicit_label_values(dataset, value_to_name):
    """Attach a verified ontology to numeric labels without guessing their order."""
    values = list(value_to_name)
    names = list(value_to_name.values())
    if not values or len(set(names)) != len(names):
        raise ValueError("Explicit label values must have distinct readable names")
    for split, rows in dataset.items():
        unknown = set(rows.unique("labels")) - set(values)
        if unknown:
            raise ValueError(f"Unmapped labels in {split}: {sorted(unknown, key=str)}")
    if values != list(range(len(values))):
        indices = {value: index for index, value in enumerate(values)}
        dataset = dataset.map(lambda row: {"labels": indices[row["labels"]]})
    return dataset.cast_column("labels", datasets.ClassLabel(names=names))

def concatenate_dataset_dict(l):
    """Concatenate a list of DatastDict objects sharing same splits and columns."""
    keys=l[0].keys()
    return datasets.DatasetDict({k: datasets.concatenate_datasets([x[k] for x in l]) for k in keys})
