## <img src="https://cdn-avatars.huggingface.co/v1/production/uploads/5fc0bcb41160c47d1d43856b/j06-U5e2Tifi2xOnTudqS.jpeg" alt="" height="32" align="absmiddle"> tasksource: 1,100+ decision tasks, standardized for interchangeable use

tasksource is a collection of **decision tasks**: classification, multiple choice and token labeling,
with negative examples or classes, standardized so that they can be used interchangeably for
multi-task training, evaluation and reward modeling. Labels come from human annotation or from
procedures that verify them, and the focus is on training data: evaluation benchmarks are kept separate.

Hugging Face Datasets is an excellent library, but datasets lack a common format and often need
preprocessing before they can be used together. Each tasksource annotation maps a dataset to one of three
templates with canonical fields: `Classification`, `MultipleChoice` or `TokenClassification`. An annotation
is a short, readable function from a dataset to its standardized form.

- 570 English, 550 multilingual and 45 vision tasks: browse the [English](catalog_english.md),
  [multilingual](catalog_multilingual.md) and [vision](catalog_vision.md) catalogs.
- Soft labels (annotator votes and rating distributions), license filters, and per-task questions that say what to decide.
- Recasts to typed decisions ([tasksource-jev-typed-decisions](#tasksource-jev-typed-decisions)),
  [instructions](#tasksource-instruct) and [label NLI](#tasksource-label-nli).

### Usage

```bash
pip install tasksource
```

If you use tasksource, its standardized task annotations, or datasets derived from them in published research, please cite the tasksource paper (Sileo, LREC-COLING 2024). GitHub's "Cite this repository" button gives the same reference ([`CITATION.cff`](CITATION.cff)).

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
    doi = "10.63317/2h4pkr685mkq",
    pages = "15655--15684",
}
```

```python
from tasksource import list_tasks, load_task

df = list_tasks(multilingual=False) # takes some time
for id in df[df.task_type=="MultipleChoice"].id:
    dataset = load_task(id) # all yielded datasets can be used interchangeably
```

The [example catalog](docs/examples.md) shows one real example per task: input, question, options and
gold answer. Evaluation benchmarks are kept out of the training catalogs; load them with
`from tasksource import eval_only`. Feel free to request a new task.

Datasets are downloaded to `$HF_DATASETS_CACHE`, like any Hugging Face dataset; large multi-task runs
need more than 100GB.

#### Vision

`list_tasks(vision=True)` lists the visual tasks and `load_task(id, vision=True)` loads them, with images.

#### Questions

Inputs are kept raw. When they do not say what to predict, a task carries a `question`
("Is this search query a well-formed question?"), exposed as `dataset.question`;
`load_task(id, prompted=True)` appends it to the inputs.

#### Soft labels

Some tasks are distributions rather than single labels: annotator votes, rater shares, survey counts.
`load_task(id, soft=True)` loads them as probabilities over the options, and `list_tasks(soft=True)`
lists them. By default, `load_task(id)` gives the majority label on rows with clear agreement.

#### Licenses

`list_tasks(license_use="commercial")` keeps tasks whose sources allow commercial use (also
`non-commercial`, `unspecified`, or a list), and `task_licenses()` gives each task's licenses and their
sources. This is a best-effort filter, not legal advice.

The tasksource code is under the [BSD 3-Clause license](LICENSE). Original documentation and the task
annotations we write (mappings, questions, label names, recasts) are under [CC BY 4.0](LICENSE-CC-BY-4.0).
Third-party datasets, including the sources that tasksource loads and our mirrors or repackages of
them, keep their own licenses, which `task_licenses()` reports.

### tasksource-jev-typed-decisions

Tasks can be recast as typed decisions: a state and one or more questions of three kinds, `choice`
(pick one option), `score` (an ordered scale) and `noul` (yes/no or a probability). This is the Jev / System One request format.

[🤗 tasksource/tasksource-jev-typed-decisions](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions)
is the published corpus:

| config | content |
|---|---|
| `default` | 1M train rows, a mix steered by source quality, difficulty and option count |
| `full` | 2.5M train rows from every source |
| `vision` | decisions over images |

It includes
[🤗 tasksource/procedural-typed-decisions](https://huggingface.co/datasets/tasksource/procedural-typed-decisions):
generated states (ledgers, schedules, graphs, policies, records) with exact labels at five difficulty levels.

```python
from tasksource import load_task, render_typed_decision

dataset = load_task("glue/rte", recast="jev")
request = render_typed_decision(dataset["train"][0], model="openjev")
```

Multiple-choice options are shuffled per row and over-represented labels are capped, so neither the gold
position nor the most common label pays. Benchmarks such as MMLU, BIG-bench and BLiMP are excluded.
See the [dataset card](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions) for the row format and the mixes.

### Pretrained models

Text encoder pretrained on tasksource reached state-of-the-art results: [🤗/deberta-v3-base-tasksource-nli](https://hf.co/sileod/deberta-v3-base-tasksource-nli)

Tasksource pretraining is notably helpful for RLHF reward modeling or any kind of classification, including zero-shot. You can also find a large and a multilingual version.

### tasksource-instruct

Tasks recast as instructions (`recast="instruct"`), one of the richest instruction-tuning datasets:
[🤗/tasksource-instruct](https://hf.co/datasets/tasksource/tasksource-instruct)

### tasksource-label-nli

Classification tasks recast as natural language inference, for entailment-based zero-shot classification:
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

tasks = [winogrande.load(), codah.load()]  # aligned datasets (same columns) can be used interchangeably
```

### Contact

For help integrating tasksource into your experiments, please contact [damien.sileo@inria.fr](mailto:damien.sileo@inria.fr).
