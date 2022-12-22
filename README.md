# tasksource

Huggingface Datasets is a great library, but it lacks standardization, and datasets require preprocessings to be used interchangeably.
Meet `tasksource` : a collection of task preprocessings to facilitate multi-task learning and reproducibility.

```python
import tasksource
from datasets import load_dataset

tasksource.bigbench(load_dataset('bigbench', 'movie_recommendation'))
```

Each dataset is mapped to a `MultipleChoice`, `Classification`, or `TokenClassification` task with standardized fields.
We do not support generation tasks as they are addressed by [promptsource](https://github.com/bigscience-workshop/promptsource).

All implemented preprocessings can be found in [tasks.py](https://github.com/sileod/tasksource/blob/main/src/tasksource/tasks.py). Each preprocessing is a function that takes a dataset as input and returns a standardized dataset. The preprocessing is designed to be human-readable. Adding a new preprocessing only takes a few lines, e.g:

```python
cos_e = tasksource.MultipleChoice('question',
    choices_list='choices',
    labels= lambda x: x['choices_list'].index(x['answer']),
    config_name='v1.0')
```

### Installation and usage:
`pip install tasksource`

List tasks:
```python
from tasksource import list_tasks, load_task
df=list_tasks()
```
Iterate over tasks:
```python
for _,x in df[df.task_type=="MultipleChoice"].iterrows():
    dataset = load_task(x.dataset_name,x.config_name, x.task_name)
```

See supported tasks in [tasks.md](https://github.com/sileod/tasksource/blob/main/tasks.md)
+200 MultipleChoic tasks, +100 Classification tasks, and 4 TokenClassification tasks.

 ### contact
 `damien.sileo@inria.fr`
```bib
@misc{sileod23-tasksource,
  author = {Sileo, Damien},
  doi = {10.5281/zenodo.7473446},
  month = {01},
  title = {{tasksource: preprocessings for reproducibility and multitask-learning}},
  url = {https://github.com/sileod/tasksource},
  version = {1.5.0},
  year = {2023}}
```
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
