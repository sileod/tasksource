# tasksource: datasets preprocessings for extreme multitask learning

Huggingface Datasets is a great library, but it lacks standardization, and datasets require some preprocessings to be used interchangeably.
Meet taskrouce: a collection of task preprocessing to facilitate multi-task learning and reproducibility.

```python
import tasksource
from datasets import load_dataset

tasksource.bigbench(load_dataset('bigbench', 'movie_recommendation'))
```

Each dataset is mapped to a `MultipleChoice`, `Classification`, or `TokenClassification` task with standardized fields.
We don't support generation tasks as they are addressed by [promptsource](https://github.com/bigscience-workshop/promptsource).

All implemented preprocessings can be found in [tasks.py](https://github.com/sileod/tasksource/blob/main/src/tasksource/tasks.py). Each preprocessing is a function that takes a dataset as input and returns a standardized dataset.

The annotation format is designed to be human readable. Adding a new preprocessing only takes a few lines, e.g:

```python
cos_e = MultipleChoice('question',
    choices_list='choices',
    labels= lambda x: x['choices_list'].index(x['answer']),
    config_name='v1.0')
```

 ### contact
 `damien.sileo@inria.fr`

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
