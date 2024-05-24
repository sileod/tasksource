## tasksource ![](https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/5fc0bcb41160c47d1d43856b/j06-U5e2Tifi2xOnTudqS.jpeg?w=20&h=20&f=face) 600+ curated datasets and preprocessings for instant and interchangeable use

Huggingface Datasets is an excellent library, but it lacks standardization, and datasets often require preprocessing work to be used interchangeably.
`tasksource` streamlines interchangeable datasets usage to scale evaluation or multi-task learning.

Each dataset is standardized to a `MultipleChoice`, `Classification`, or `TokenClassification` template with canonical fields. We focus on discriminative tasks (= with negative examples or classes) for our annotations but also provide a `SequenceToSequence` template. All implemented preprocessings are in [tasks.py](https://github.com/sileod/tasksource/blob/main/src/tasksource/tasks.py) or [tasks.md](https://github.com/sileod/tasksource/blob/main/tasks.md). A preprocessing is a function that accepts a dataset and returns the standardized dataset. Preprocessing code is concise and human-readable.

### Installation and usage:
`pip install tasksource`
```python
from tasksource import list_tasks, load_task
df = list_tasks(multilingual=False) # takes some time

for id in df[df.task_type=="MultipleChoice"].id:
    dataset = load_task(id) # all yielded datasets can be used interchangeably
```

Browse the 500+ curated tasks in tasks.md (200+ MultipleChoice tasks, 200+ Classification tasks), and feel free to request a new task. Datasets are downloaded to `$HF_DATASETS_CACHE` (like any Hugging Face dataset), so ensure you have more than 100GB of space available.

You can now also use:
```python
load_dataset("tasksource/data", "glue/rte",max_rows=30_000)
```

### Pretrained models:

Text encoder pretrained on tasksource reached state-of-the-art results: [🤗/deberta-v3-base-tasksource-nli](https://hf.co/sileod/deberta-v3-base-tasksource-nli)

Tasksource pretraining is notably helpful for RLHF reward modeling or any kind of classification, including zero-shot. You can also find a large and a multilingual version.

### tasksource-instruct

The repo also contains some recasting code to convert tasksource datasets to instructions, providing one of the richest instruction-tuning datasets:
[🤗/tasksource-instruct-v0](https://hf.co/datasets/tasksource/tasksource-instruct-v0)


### tasksource-label-nli

We also recast all classification tasks as natural language inference, to improve entailment-based zero-shot classification detection:
[🤗/zero-shot-label-nli](https://huggingface.co/datasets/tasksource/zero-shot-label-nli)

### Write and use custom preprocessings

```python
from tasksource import MultipleChoice

codah = MultipleChoice('question_propmt',choices_list='candidate_answers',
    labels='correct_answer_idx',
    dataset_name='codah', config_name='codah')
    
winogrande = MultipleChoice('sentence',['option1','option2'],'answer',
    dataset_name='winogrande',config_name='winogrande_xl',
    splits=['train','validation',None]) # test labels are not usable
    
tasks = [winogrande.load(), codah.load()]) #  Aligned datasets (same columns) can be used interchangably  
```

 ### Citation and contact

For more details, refer to this [article:](https://arxiv.org/abs/2301.05948) 
```bib
@inproceedings{sileo-2024-tasksource,
    title = "tasksource: A Large Collection of {NLP} tasks with a Structured Dataset Preprocessing Framework",
    author = "Sileo, Damien",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.1361",
    pages = "15655--15684",
}
```
For help integrating tasksource into your experiments, please contact [damien.sileo@inria.fr](mailto:damien.sileo@inria.fr).

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
