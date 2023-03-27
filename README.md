## tasksource: 500+ dataset harmonization preprocessings for frictionless extreme multi-task learning and evaluation

Huggingface Datasets is a great library, but it lacks standardization, and datasets require preprocessing work to be used interchangeably.
`tasksource` automates this and facilitates reproducible multi-task learning scaling and meta-learning.

Each dataset is standardized to either `MultipleChoice`, `Classification`, or `TokenClassification` dataset with identical fields. We focus on discriminative tasks (= with negative examples or classes) and do not yet support generation tasks as they are addressed by [promptsource](https://github.com/bigscience-workshop/promptsource). All implemented preprocessings are in [tasks.py](https://github.com/sileod/tasksource/blob/main/src/tasksource/tasks.py) or [tasks.md](https://github.com/sileod/tasksource/blob/main/tasks.md). A preprocessing is a function that accepts a dataset and returns the standardized dataset. Preprocessing code is concise and human-readable.

### Installation and usage:
`pip install tasksource`
```python
from tasksource import list_tasks, load_task
df = list_tasks()

for id in df[df.task_type=="MultipleChoice"].id:
    dataset = load_task(id) # all yielded datasets can be used interchangeably
```

See supported 500+ tasks in [tasks.md](https://github.com/sileod/tasksource/blob/main/tasks.md) (+200 MultipleChoice tasks, +200 Classification tasks) and feel free to request a new task. Datasets are downloaded to `$HF_DATASETS_CACHE` (as any huggingface dataset), so be sure to have >100GB of space there.

### Pretrained model:

Text encoder pretrained on tasksource reached state-of-the-art results: [🤗/deberta-v3-base-tasksource-nli](https://hf.co/sileod/deberta-v3-base-tasksource-nli)
Tasksource pretraining should be quite helpful for RLHF reward models pretraining.

 ### Contact and citation
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
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
