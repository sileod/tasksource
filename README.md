## tasksource ![](https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/5fc0bcb41160c47d1d43856b/j06-U5e2Tifi2xOnTudqS.jpeg?w=20&h=20&f=face) 1,100+ decision tasks, standardized for interchangeable use

tasksource is a collection of **decision tasks**: classification, multiple choice and token labeling,
with negative examples or classes, standardized so that they can be used interchangeably for
multi-task training, evaluation and reward modeling.

Hugging Face Datasets is an excellent library, but datasets lack a common format and often need
preprocessing before they can be used together. Each tasksource annotation maps a dataset to one of three
templates with canonical fields: `Classification`, `MultipleChoice` or `TokenClassification`. An annotation
is a short, readable function from a dataset to its standardized form.

- 570 English, 550 multilingual and 45 vision tasks: browse the [English](catalog_english.md),
  [multilingual](catalog_multilingual.md) and [vision](catalog_vision.md) catalogs.
- Soft labels (annotator votes and rating distributions), license filters, and per-task questions that say what to decide.
- Recasts to typed decisions ([tasksource-jev-typed-decisions](#tasksource-jev-typed-decisions)),
  instructions and label NLI.

### Usage

```bash
pip install tasksource
```

```python
from tasksource import list_tasks, load_task

df = list_tasks(multilingual=False) # takes some time
for id in df[df.task_type=="MultipleChoice"].id:
    dataset = load_task(id) # all yielded datasets can be used interchangeably
```

#### Task catalogs

Feel free to request a new task. The [English example gallery](docs/examples/english/README.md) shows formatted
inputs, questions and gold answers, with a searchable HTML version.
The English catalog has about 460 Classification, 100 MultipleChoice and 15
TokenClassification tasks. Evaluation benchmarks are kept separate in
[eval_only.py](src/tasksource/eval_only.py). Annotations excluded for duplicates,
source problems, or unsound supervision remain in
[parked.py](src/tasksource/parked.py). Both record the reason for exclusion.
Use `from tasksource import eval_only` to access a benchmark annotation directly;
these annotations are excluded from the training catalogs.

The [discovery tools](src/tasksource/task_discovery/README.md) retrieve and deduplicate
dataset candidates, link source and paper identities, and retain structured review
history. Browse the [dataset prospects](docs/prospects.md) for screened sources
awaiting deeper checks; these are separate from the implemented task catalogs.

Visual annotations are in [vision_tasks.py](src/tasksource/vision_tasks.py).
Discover them with `list_tasks(vision=True)` and load them with
`load_task(id, vision=True)`. Use `recast="jev"` or `recast="instruct"` to keep
images alongside the rendered decisions or prompts.

Datasets are downloaded to `$HF_DATASETS_CACHE`, like any Hugging Face dataset.
Ensure you have more than 100GB of space available for large multi-task runs.

#### Prompts

Inputs are kept raw by default. When the inputs alone do not say what to predict,
an annotation carries a `question` ("Is this search query a well-formed question?"),
exposed as `dataset.question`.

`load_task(id, prompted=True)` appends the question to the inputs. The instruct
and typed-decision recasts use it as their instruction. For a per-example question
or query context, annotations can use `{sentence2}` in the question string:
`question="Which role does the marked argument play?\n{sentence2}"`. Recasts put
`sentence1` in the state and substitute `sentence2` into the question; raw fields
stay unchanged. Ordinary paired-text tasks retain both texts in the state.

#### Soft labels

Some annotations are distributions rather than single labels: annotator votes,
rater shares, survey counts. These `SoftLabeling` annotations load as
probabilities with `load_task(id, soft=True)` (`labels` over `options`). Most also
have a hard view: their majority label on rows with clear agreement, which is
what `load_task(id)` and the default `list_tasks()` give.

`list_tasks(soft=True)` lists the soft views, including annotations that only
make sense as distributions, such as ProtoQA survey answers. They are listed at
the end of [catalog_english.md](catalog_english.md).

Each records whether it summarizes annotator votes or mean ratings, and how many
annotators judged an item. Vote shares from three annotators are coarse
(0, 1/3, 2/3, 1); `min_annotators=5` leaves them out.

#### Licenses

`list_tasks(license_use="commercial")` keeps tasks whose sources allow commercial
use. The filter also accepts `non-commercial`, `unspecified`, or a list.
`task_licenses()` gives each task's licenses and where they come from.

Licenses are read from the Hub cards of the datasets a task loads and of their
originals, plus [Data Provenance Initiative](https://www.dataprovenance.org/)
annotations. Both are snapshotted in the package;
`task_licenses(fresh=True)` reads the current cards.

`license_use` takes the most restrictive license found. `other`, bare `cc`, and
missing licenses are `unspecified`. This is a best-effort filter, not legal advice.

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

Every classification, multiple-choice and vetted token task can be recast as a typed decision: a
state and one or more questions of three kinds, `choice` (pick one option), `score` (an ordered scale)
and `noul` (yes/no or a probability). This is the Jev / System One request format.

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

The conversion is deterministic. Multiple-choice options are permuted per row, so the gold position carries
no signal, and over-represented labels are capped so that always answering the common label does not pay.
BIG-bench, MMLU and BLiMP are excluded. The [dataset card](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions)
documents the row format, variants and mixing; the [build runbook](docs/jev/README.md) covers building and publishing.

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
