from .preprocess import Preprocessing
import re
import pandas as pd
from . import tasks

def parse_var_name(var_name):
    regex = r"([^_]+)__([^_]+)"
    task_config = re.search(regex, var_name)
    task_config = task_config.groups()[-1] if task_config else None
    regex = r"___([^_]+)"
    config = re.search(regex, var_name)
    config = config.groups()[-1] if config else None
    dataset_name = var_name.split('__')[0]
    return dataset_name, config, task_config

def list_tasks():
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
                'task_type': value.__class__.__name__,'parsing': value,}]
    return pd.DataFrame(l)

task_df = list_tasks()

def load_preprocessing(dataset_name, config_name=None, task_name=None):
    y = task_df
    print(len(y))
    y = y[y.dataset_name.map(lambda x:x==dataset_name)]
    y = y[y.config_name.map(lambda x:x==config_name)]
    y = y[y.task_name.map(lambda x:x==task_name)]
    return getattr(tasks,y.preprocessing_name.iloc[0])