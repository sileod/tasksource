from .preprocess import Preprocessing
import re
import pandas as pd
from . import tasks
from .metadata import dataset_rank
from datasets import load_dataset
import funcy as fc
import os


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

def list_tasks(tasks_path=f'{os.path.dirname(__file__)}/tasks.py'):
    task_order = open(tasks_path).readlines()
    task_order = [x.split('=')[0].rstrip() for x in task_order if '=' in x]
    task_order = [x for x in task_order if x.isidentifier()]
    task_order = fc.flip(dict(enumerate(task_order)))

    l = []
    for key in dir(tasks):
        value=getattr(tasks, key)
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
    df = df.sort_values('rank')
    del df['rank']
    return df

task_df = list_tasks()

def load_preprocessing(dataset_name, config_name=None, task_name=None):
    y = task_df
    y = y[y.dataset_name.map(lambda x:x==dataset_name)]
    y = y[y.config_name.map(lambda x:x==config_name)]
    y = y[y.task_name.map(lambda x:x==task_name)]
    return getattr(tasks,y.preprocessing_name.iloc[0])

def load_task(*args,**kwargs):
    return load_preprocessing(*args,**kwargs)(load_dataset(*args,**kwargs))