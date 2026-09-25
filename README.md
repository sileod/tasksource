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

Inputs are kept raw by default. When the inputs alone do not say what to predict, an annotation carries a `question` ("Is this search query a well-formed question?"), exposed as `dataset.question`; `load_task(id, prompted=True)` appends it to the inputs, and the instruct and typed-decision recasts use it as their instruction.

Browse the 500+ curated tasks in tasks.md (tasks kept out on purpose, such as evaluation benchmarks, are in [parked.py](https://github.com/sileod/tasksource/blob/main/src/tasksource/parked.py) with the reason) (200+ MultipleChoice tasks, 200+ Classification tasks), and feel free to request a new task. Datasets are downloaded to `$HF_DATASETS_CACHE` (like any Hugging Face dataset), so ensure you have more than 100GB of space available.

Some annotations are distributions rather than single labels: annotator votes, rater shares, survey counts. These `SoftLabeling` annotations load as probabilities with `load_task(id, soft=True)` (`labels` over `options`); most also have a hard view, their majority label on rows with clear agreement, which is what `load_task(id)` and the default `list_tasks()` give. `list_tasks(soft=True)` lists the soft views, including annotations that only make sense as distributions (e.g. ProtoQA survey answers). Each records whether it summarizes annotator votes or mean ratings, and how many annotators judged an item: vote shares from three annotators are coarse (0, 1/3, 2/3, 1), and `min_annotators=5` leaves them out. They are listed at the end of tasks.md.

Licenses are available on demand. `list_tasks(license_use="commercial")` keeps tasks whose sources allow commercial use (also `non-commercial`, `unspecified`, or a list), and `task_licenses()` gives each task's licenses and where they come from. They are read from the Hub cards of the datasets a task loads and of their originals, plus [Data Provenance Initiative](https://www.dataprovenance.org/) annotations, both snapshotted in the package (`task_licenses(fresh=True)` reads the current cards). `license_use` takes the most restrictive license found; `other`, bare `cc` and missing licenses are `unspecified`. This is a best-effort filter, not legal advice.

### Pretrained models:

Text encoder pretrained on tasksource reached state-of-the-art results: [🤗/deberta-v3-base-tasksource-nli](https://hf.co/sileod/deberta-v3-base-tasksource-nli)

Tasksource pretraining is notably helpful for RLHF reward modeling or any kind of classification, including zero-shot. You can also find a large and a multilingual version.

### tasksource-instruct

The repo also contains some recasting code to convert tasksource datasets to instructions, providing one of the richest instruction-tuning datasets:
[🤗/tasksource-instruct-v0](https://hf.co/datasets/tasksource/tasksource-instruct-v0)


### tasksource-label-nli

We also recast all classification tasks as natural language inference, to improve entailment-based zero-shot classification detection:
[🤗/zero-shot-label-nli](https://huggingface.co/datasets/tasksource/zero-shot-label-nli)

### tasksource-jev-typed-decisions

Tasksource classification, multiple-choice, and vetted token tasks can be recast as
runtime-defined typed decisions (the Jev / System One request format: `choice`,
`score`, and `noul` questions over a state). The canonical representation keeps
the state, instructions, criteria, integer label, and textual answer separate:

[🤗 tasksource/tasksource-jev-typed-decisions](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions)

The [Jev build runbook](docs/jev/README.md) covers smoke tests, resumable builds,
validation, and publication.

```python
from tasksource import load_task, render_typed_decision

dataset = load_task("glue/rte", recast="jev")
request = render_typed_decision(dataset["train"][0], model="openjev")

```

The canonical conversion is deterministic and does not paraphrase criteria,
except that a final "all/none of the above" becomes "all/none of the other
options". Multiple-choice criteria keep every source option and are permuted per
row, seeded by task, split, and row index, so the gold slot carries no signal;
rows whose options refer to other options by position or letter keep their
order. The published 1M corpus adds explicit, deterministic, low-frequency
subrecasts for label verification (`noul`), criterion-order invariance, and
manually vetted instruction variation. Every row records its source, normalized
`train`/`dev`/`test` split, and variant. BIG-bench, MMLU, and BLiMP are excluded.
The flat rows carry `group_id` and `question_id`; `render_typed_decision_group`
combines related canonical decisions into one multi-question request.
Publication keeps each source-row group together under the 500k cap and uses
reviewed question and paired-field wording to reduce repeated boilerplate.

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

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
