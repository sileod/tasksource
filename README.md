## tasksource: 500+ dataset harmonization preprocessings for effortless extreme multi-task learning and evaluation

Huggingface Datasets is an excellent library, but it lacks standardization, and datasets often require preprocessing work to be used interchangeably.
`tasksource` streamlines interchangeable datasets usage to scale evaluation or multi-task learning.

Each dataset is standardized to a  `MultipleChoice`, `Classification`, or `TokenClassification` template with canonical fields. We focus on discriminative tasks (= with negative examples or classes) and do not yet support generation tasks as they are addressed by [promptsource](https://github.com/bigscience-workshop/promptsource). All implemented preprocessings are in [tasks.py](https://github.com/sileod/tasksource/blob/main/src/tasksource/tasks.py) or [tasks.md](https://github.com/sileod/tasksource/blob/main/tasks.md). A preprocessing is a function that accepts a dataset and returns the standardized dataset. Preprocessing code is concise and human-readable.

### Installation and usage:
`pip install tasksource`
```python
from tasksource import list_tasks, load_task
df = list_tasks() # takes some time

for id in df[df.task_type=="MultipleChoice"].id:
    dataset = load_task(id) # all yielded datasets can be used interchangeably
```

Browse the 500+ curated tasks in tasks.md (200+ MultipleChoice tasks, 200+ Classification tasks), and feel free to request a new task. Datasets are downloaded to $HF_DATASETS_CACHE (like any Hugging Face dataset), so ensure you have more than 100GB of space available.

Pretrained Model
### Pretrained model:

Text encoder pretrained on tasksource reached state-of-the-art results: [🤗/deberta-v3-base-tasksource-nli](https://hf.co/sileod/deberta-v3-base-tasksource-nli)

Tasksource pretraining is notably helpful for RLHF reward modeling.

 ### Contact and citation
For help integrating tasksource into your experiments, please contact [damien.sileo@inria.fr](mailto:damien.sileo@inria.fr).

For more details, refer to this [article:](https://arxiv.org/abs/2301.05948) 
```bib
@article{sileo2023tasksource,
  title={tasksource: Structured Dataset Preprocessing Annotations for Frictionless Extreme Multi-Task Learning and Evaluation},
  author={Sileo, Damien},
  url= {https://arxiv.org/abs/2301.05948},
  journal={arXiv preprint arXiv:2301.05948},
  year={2023}
}
```
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
