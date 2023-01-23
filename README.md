## tasksource: 480+ dataset harmonization preprocessings with structured annotations for frictionless extreme multi-task learning and evaluation

Huggingface Datasets is a great library, but it lacks standardization, and datasets require preprocessing work to be used interchangeably.
`tasksource` automates this and facilitates multi-task learning scaling and reproducibility.

```python
import tasksource
from datasets import load_dataset

tasksource.bigbench(load_dataset('bigbench', 'movie_recommendation')) # returns standardized MultipleChoice dataset
```

Each dataset is mapped to either `MultipleChoice`, `Classification`, or `TokenClassification` dataset with standardized fields.
We do not support generation tasks as they are addressed by [promptsource](https://github.com/bigscience-workshop/promptsource).

All implemented preprocessings can be found in [tasks.py](https://github.com/sileod/tasksource/blob/main/src/tasksource/tasks.py) or [tasks.md](https://github.com/sileod/tasksource/blob/main/tasks.md). Each preprocessing is a function that takes a dataset as input and returns a standardized dataset. The preprocessing annotation is designed to be human-readable: adding a new preprocessing only takes a few lines, e.g:

```python
cos_e = tasksource.MultipleChoice(
    'question',
    choices_list='choices',
    labels= lambda x: x['choices_list'].index(x['answer']),
    config_name='v1.0')
```

### Installation and usage:
`pip install tasksource`

Get the task index and iterate over harmonized tasks:
```python
from tasksource import list_tasks, load_task
df = list_tasks()

for id in df[df.task_type=="MultipleChoice"].id:
    dataset = load_task(id)
    # all yielded datasets can be used interchangeably
```

See supported 480+ tasks in [tasks.md](https://github.com/sileod/tasksource/blob/main/tasks.md) (+200 MultipleChoice tasks, +200 Classification tasks). Feel free to request or propose a new task.

### Pretrained model:

I pretrained models on tasksource and obtained state-of-the-art results:
<https://huggingface.co/sileod/deberta-v3-base-tasksource-nli>

 ### Contact
I can help you integrate tasksource in your experiments. `damien.sileo@inria.fr`

More details on this [article:](https://arxiv.org/abs/2301.05948) 
```bib
@article{sileo2023tasksource,
  title={tasksource: Structured Dataset Preprocessing Annotations for Frictionless Extreme Multi-Task Learning and Evaluation},
  author={Sileo, Damien},
  url= {https://arxiv.org/abs/2301.05948},
  journal={arXiv preprint arXiv:2301.05948},
  year={2023}
}
```
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
