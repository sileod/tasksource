from collections.abc import Iterable
from dotwiz import DotWiz
from dataclasses import dataclass
from typing import Union
import itertools
import funcy as fc
import exrex 
import magicattr 
import numpy as np
import copy
import datasets

def get_column_names(dataset):
    cn = dataset.column_names
    if type(cn)==dict:
        return set(fc.flatten(cn.values()))
    else:
        return set(cn)


def sample_dataset(dataset,n=10000, n_eval=1000):
    for k in dataset:
        n_k=(n if k=='train' else n_eval)
        if n_k and len(dataset[k])>n_k:
            dataset[k]=dataset[k].train_test_split(train_size=n_k)['train']
    return dataset

class Preprocessing(DotWiz):
    default_splits = ('train','validation','test')
    @staticmethod
    def __map_to_target(x,fn=lambda x:None, target=None):
        x[target]=fn(x)
        return x

    def load(self):
        return self(datasets.load_dataset(self.dataset_name,self.config_name))

    def __call__(self,dataset, max_rows=None, max_rows_eval=None):
        dataset = self.pre_process(dataset)
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
        dataset = sample_dataset(dataset, max_rows, max_rows_eval)
        
        dataset=dataset.rename_columns({v:k for k,v in self.to_dict().items()
                                        if (k and k not in {'splits','dataset_name','config_name'} 
                                        and type(v)==str and k!=v)})
        for k in self.to_dict().keys():
            v=getattr(self, k)
            if callable(v) and k not in {"post_process","pre_process","load"}:
                dataset=dataset.map(self.__map_to_target,
                                    fn_kwargs={'fn':v,'target':k})

        dataset=dataset.remove_columns(
            get_column_names(dataset)-set(self.to_dict().keys()))
        dataset = fix_labels(dataset)
        dataset = self.post_process(dataset)
        return dataset


@dataclass
class cat(Preprocessing):
    fields:Union[str,list]=None
    separator:str=' '
        
    def __call__(self, example=None):
        y=[np.char.array(example[f]) + sep 
                for f,sep in zip(self.fields[::-1],itertools.repeat(self.separator))]
        y=list(sum(*y))
        if len(y)==1:
            y=y[0]
        return y


def pretty(f):
    class pretty_f(DotWiz):
        def __init__(self,*args):
            self.__f_arg = f(*args)
            for a in args:
                setattr(self,'value',a)
                
        def __call__(self, *args,**kwargs):
            return self.__f_arg(*args,**kwargs)
            return 
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
    
    def __call__(self,dataset, *args, **kwargs):
        dataset = super().__call__(dataset, *args, **kwargs)
        if self.choices_list:
            dataset = dataset.filter(lambda x: 1<len(x['choices_list']))
            n_options = min([len(x) for k in dataset for x in dataset[k]['choices_list']])
            n_options = min(4,n_options)
            dataset = dataset.map(self.flatten, fn_kwargs={'n_options':n_options})
        return dataset

    @staticmethod
    def flatten(x, n_options=None):
        n_neg = n_options-1 if n_options else None
        choices = x['choices_list']
        label=x['labels']
        neg = choices[:label] + choices[label+1:]
        pos = choices[label]
        x['labels']=0
        x['choices_list']=[pos]+neg[:n_neg]
        for i,o in enumerate(x['choices_list']):
            x[f'choice{i}']=o
        del x['choices_list']
        return x

@dataclass
class SharedFields:
    splits:list=Preprocessing.default_splits
    dataset_name:str = None
    config_name:str = None
    pre_process: callable = lambda x:x
    post_process: callable = lambda x:x
    #language:str="en"

@dataclass
class Classification(SharedFields, ClassificationFields): pass

@dataclass
class MultipleChoice(SharedFields, MultipleChoiceFields): pass

@dataclass
class TokenClassification(SharedFields, TokenClassificationFields): pass
        
get=dotgetter()
constant = pretty(fc.constantly)
regen = lambda x: list(exrex.generate(x))


def fix_splits(dataset):

    if len(dataset)==1 and "train" not in dataset:
        k = list(dataset)[0]
        dataset['train'] = copy.deepcopy(dataset[k])
        del dataset[k]

    if 'auxiliary_train' in dataset:
        del dataset['auxiliary_train']
    
    if 'test' in dataset:
        if 'labels' in dataset['test'].features:
            if len(set(fc.flatten(dataset['test'].to_dict()['labels'])))==1:
                 # obfuscated label
                del dataset['test']

    if 'validation' in dataset and 'train' not in dataset:
        train_validation = dataset['validation'].train_test_split(0.5, seed=0)
        dataset['train'] = train_validation['train']
        dataset['validation']=train_validation['test']
    
    if 'validation' in dataset and 'test' not in dataset:
        validation_test = dataset['validation'].train_test_split(0.5, seed=0)
        dataset['validation'] = validation_test['train']
        dataset['test']=validation_test['test']

    if 'test' in dataset and 'validation' not in dataset:
        validation_test = dataset['test'].train_test_split(0.5, seed=0)
        dataset['validation'] = validation_test['train']
        dataset['test']=validation_test['test']

    if 'validation' not in dataset and 'test' not in dataset:
        train_val_test = dataset["train"].train_test_split(train_size=0.85, seed=0)
        val_test = train_val_test["test"].train_test_split(0.5, seed=0)
        dataset["train"] = train_val_test["train"]
        dataset["validation"] = val_test["train"]
        dataset["test"] = val_test["test"]
        
    return dataset

def fix_labels(dataset, label_key='labels'):
    if type(dataset['train'][label_key][0]) in [int,list,float]:
        return dataset
    labels=set(fc.flatten(dataset[k][label_key] for k in {"train"}))
    if set(labels)=={'entailment','neutral','contradiction'}:
        order=lambda x:dict(fc.flip(enumerate(['entailment','neutral','contradiction']))).get(x,x)
    else:
        order=str
    labels=sorted(labels, key=order)
    dataset=dataset.cast_column(label_key, datasets.ClassLabel(names=labels))
    return dataset