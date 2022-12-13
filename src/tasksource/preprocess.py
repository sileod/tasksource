from collections.abc import Iterable
from dotwiz import DotWiz
from dataclasses import dataclass
from typing import Union
import itertools
import funcy as fc
import exrex 
import pandas as pd
import magicattr 
import numpy as np
import re

def parse_var_name(var_name):
    regex = r"([^_]+)__([^_]+)"
    task_config = re.search(regex, var_name)
    task_config = task_config.groups()[-1] if task_config else None
    regex = r"___([^_]+)"
    config = re.search(regex, var_name)
    config = config.groups()[-1] if config else None
    dataset_name = var_name.split('__')[0]
    return dataset_name, config, task_config

class Preprocessing(DotWiz):
    default_splits = ('train','validation','test')
    
    @staticmethod
    def __map_to_target(x,fn : lambda x:None, target):
        x[target]=fn(x)
        return x
    
    def __call__(self,dataset):
        for k,v in zip(self.default_splits, self.splits):
            if v and k!=v:
                dataset[k]=dataset[v]
                del dataset[v]
        for k in list(dataset.keys()):
            if k not in self.default_splits:
                del dataset[k]
        dataset=dataset.rename_columns({v:k for k,v in self.to_dict().items()
                                        if (k and k!='splits' and type(v)==str)})
        for k in self.to_dict().keys():
            v=getattr(self, k)
            print(type(v))
            if callable(v):
                dataset=dataset.map(self.__map_to_target,
                                    fn_kwargs={'fn':v,'target':k})
                #dataset=v(dataset,k)
        return dataset
    
@dataclass
class cat(Preprocessing):
    fields:Union[str,list]=None
    separator:str=' '
        
    def __call__(self, dataset, target):
        def f(example):
            y=[np.char.array(example[f])+sep 
                   for f,sep in zip(self.fields[::-1],itertools.repeat(self.separator))]
            y=list(sum(*y))
            if len(y)==1:
                y=y[0]
            example[target]=y
            return example
        return dataset.map(f)

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

class dotgetter(dict):
    def __init__(self, path=''):
        self.path=path

    def __getattr__(self, k):
        return self.__class__(f'{self.path}.{k}'.lstrip('.'))
    
    def __getitem__(self, i):
        return self.__class__(f'{self.path}[{i}]')

    def __call__(self, example=None):
        print(self.path)
        return magicattr.get(DotWiz(example), self.path)


@dataclass
class Classification(Preprocessing):
    sentence1:str='sentence1'
    sentence2:str='sentence2'
    labels:Iterable='labels'
    splits:list=Preprocessing.default_splits
    dataset_name:str = None
    config_name:str = None
@dataclass
class TokenClassification(Preprocessing):
    tokens:str='tokens'
    labels:str='labels'
    splits:list=Preprocessing.default_splits
    dataset_name:str = None
    config_name:str = None
        
@dataclass
class MultipleChoice(Preprocessing):
    inputs:str='input'
    choices:Iterable=tuple()
    labels:str='labels'
    splits:list=Preprocessing.default_splits
    choices_list:str=None
    dataset_name:str = None
    config_name:str = None

    def __post_init__(self):
        for i, c in enumerate(self.choices):
            setattr(self,f'choice{i}',c)
        delattr(self,'choices')
        if not self.choices_list:
            delattr(self,'choices_list')

        
get=dotgetter()
constant = pretty(fc.constantly)
regen = lambda x: list(exrex.generate(x))