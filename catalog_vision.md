57 visual tasks. Load one with `load_task(id, vision=True)`; the annotations are in [vision_tasks.py](src/tasksource/vision_tasks.py). Evaluation benchmarks are in [eval_only.py](src/tasksource/eval_only.py); other excluded annotations are in [parked.py](src/tasksource/parked.py).

| # | id | type | dataset | question |
|--:|---|---|---|:-:|
| 1 | [nlvr2](src/tasksource/vision_tasks.py#L162) | VisualClassification | [pingzhili/nlvr2](https://hf.co/datasets/pingzhili/nlvr2) | ✓ |
| 2 | [aokvqa](src/tasksource/vision_tasks.py#L169) | VisualMultipleChoice | [HuggingFaceM4/A-OKVQA](https://hf.co/datasets/HuggingFaceM4/A-OKVQA) |  |
| 3 | [scienceqa-img](src/tasksource/vision_tasks.py#L175) | VisualMultipleChoice | [derek-thomas/ScienceQA](https://hf.co/datasets/derek-thomas/ScienceQA) |  |
| 4 | [ai2d](src/tasksource/vision_tasks.py#L182) | VisualMultipleChoice | [tasksource/ai2d](https://hf.co/datasets/tasksource/ai2d) |  |
| 5 | [figureqa](src/tasksource/vision_tasks.py#L188) | VisualClassification | [vikhyatk/figureqa](https://hf.co/datasets/vikhyatk/figureqa) | ✓ |
| 6 | [mind2web/action](src/tasksource/vision_tasks.py#L210) | VisualClassification | [tasksource/multimodal-mind2web](https://hf.co/datasets/tasksource/multimodal-mind2web) | ✓ |
| 7 | [mind2web/element](src/tasksource/vision_tasks.py#L217) | VisualMultipleChoice | [tasksource/multimodal-mind2web](https://hf.co/datasets/tasksource/multimodal-mind2web) | ✓ |
| 8 | [mind2web/x10](src/tasksource/vision_tasks.py#L223) | VisualClassification | [tasksource/multimodal-mind2web](https://hf.co/datasets/tasksource/multimodal-mind2web) | ✓ |
| 9 | [mind2web/y10](src/tasksource/vision_tasks.py#L230) | VisualClassification | [tasksource/multimodal-mind2web](https://hf.co/datasets/tasksource/multimodal-mind2web) | ✓ |
| 10 | [mind2web/grid5](src/tasksource/vision_tasks.py#L237) | VisualClassification | [tasksource/multimodal-mind2web](https://hf.co/datasets/tasksource/multimodal-mind2web) | ✓ |
| 11 | [mind2web/grid7](src/tasksource/vision_tasks.py#L244) | VisualClassification | [tasksource/multimodal-mind2web](https://hf.co/datasets/tasksource/multimodal-mind2web) | ✓ |
| 12 | [m3cot](src/tasksource/vision_tasks.py#L252) | VisualMultipleChoice | [LightChen2333/M3CoT](https://hf.co/datasets/LightChen2333/M3CoT) |  |
| 13 | [exams-v](src/tasksource/vision_tasks.py#L272) | VisualClassification | [MBZUAI/EXAMS-V](https://hf.co/datasets/MBZUAI/EXAMS-V) |  |
| 14 | [visualsphinx](src/tasksource/vision_tasks.py#L283) | VisualMultipleChoice | [VisualSphinx/VisualSphinx-V1-RL-20K](https://hf.co/datasets/VisualSphinx/VisualSphinx-V1-RL-20K) |  |
| 15 | [muslr/tfu](src/tasksource/vision_tasks.py#L293) | VisualClassification | [Aiden0526/MuSLR](https://hf.co/datasets/Aiden0526/MuSLR) |  |
| 16 | [muslr/mc](src/tasksource/vision_tasks.py#L302) | VisualMultipleChoice | [Aiden0526/MuSLR](https://hf.co/datasets/Aiden0526/MuSLR) |  |
| 17 | [iconqa/text](src/tasksource/vision_tasks.py#L312) | VisualMultipleChoice | [tasksource/iconqa-text](https://hf.co/datasets/tasksource/iconqa-text) |  |
| 18 | [view2space/mcq](src/tasksource/vision_tasks.py#L319) | VisualMultipleChoice | [tasksource/view2space](https://hf.co/datasets/tasksource/view2space) |  |
| 19 | [visual7w](src/tasksource/vision_tasks.py#L335) | VisualMultipleChoice | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) |  |
| 20 | [clevr/yesno](src/tasksource/vision_tasks.py#L340) | VisualClassification | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) |  |
| 21 | [mapqa/yesno](src/tasksource/vision_tasks.py#L345) | VisualClassification | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) |  |
| 22 | [tqa](src/tasksource/vision_tasks.py#L350) | VisualMultipleChoice | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) |  |
| 23 | [hateful-memes](src/tasksource/vision_tasks.py#L356) | VisualClassification | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) |  |
| 24 | [clevr/color](src/tasksource/vision_tasks.py#L364) | VisualClassification | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) |  |
| 25 | [clevr/shape](src/tasksource/vision_tasks.py#L369) | VisualClassification | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) |  |
| 26 | [clevr/size](src/tasksource/vision_tasks.py#L374) | VisualClassification | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) |  |
| 27 | [clevr/material](src/tasksource/vision_tasks.py#L379) | VisualClassification | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) |  |
| 28 | [intergps](src/tasksource/vision_tasks.py#L384) | VisualMultipleChoice | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) |  |
| 29 | [clevr/count](src/tasksource/vision_tasks.py#L389) | VisualClassification | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) | ✓ |
| 30 | [tallyqa/count](src/tasksource/vision_tasks.py#L396) | VisualClassification | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) | ✓ |
| 31 | [vsr/yesno](src/tasksource/vision_tasks.py#L403) | VisualClassification | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) | ✓ |
| 32 | [rico-widget/grid7](src/tasksource/vision_tasks.py#L481) | VisualClassification | [bevaya/RICO-WidgetCaptioning](https://hf.co/datasets/bevaya/RICO-WidgetCaptioning) | ✓ |
| 33 | [rico-widget/element](src/tasksource/vision_tasks.py#L491) | VisualMultipleChoice | [bevaya/RICO-WidgetCaptioning](https://hf.co/datasets/bevaya/RICO-WidgetCaptioning) | ✓ |
| 34 | [lvis/region](src/tasksource/vision_tasks.py#L500) | VisualClassification | [tasksource/coco-regions](https://hf.co/datasets/tasksource/coco-regions) | ✓ |
| 35 | [coco/panoptic-region](src/tasksource/vision_tasks.py#L506) | VisualClassification | [tasksource/coco-regions](https://hf.co/datasets/tasksource/coco-regions) | ✓ |
| 36 | [doclaynet/region](src/tasksource/vision_tasks.py#L512) | VisualClassification | [tasksource/doclaynet-region](https://hf.co/datasets/tasksource/doclaynet-region) | ✓ |
| 37 | [bapps/preference](src/tasksource/vision_tasks.py#L518) | VisualMultipleChoice | [tasksource/bapps](https://hf.co/datasets/tasksource/bapps) | ✓ |
| 38 | [spair71k/grid7](src/tasksource/vision_tasks.py#L524) | VisualClassification | [tasksource/spair71k-grid](https://hf.co/datasets/tasksource/spair71k-grid) | ✓ |
| 39 | [superclevr/yesno](src/tasksource/vision_tasks.py#L540) | VisualClassification | [tasksource/superclevr](https://hf.co/datasets/tasksource/superclevr) |  |
| 40 | [superclevr/count](src/tasksource/vision_tasks.py#L542) | VisualClassification | [tasksource/superclevr](https://hf.co/datasets/tasksource/superclevr) |  |
| 41 | [superclevr/color](src/tasksource/vision_tasks.py#L544) | VisualClassification | [tasksource/superclevr](https://hf.co/datasets/tasksource/superclevr) |  |
| 42 | [superclevr/shape](src/tasksource/vision_tasks.py#L546) | VisualClassification | [tasksource/superclevr](https://hf.co/datasets/tasksource/superclevr) |  |
| 43 | [superclevr/size](src/tasksource/vision_tasks.py#L548) | VisualClassification | [tasksource/superclevr](https://hf.co/datasets/tasksource/superclevr) |  |
| 44 | [superclevr/material](src/tasksource/vision_tasks.py#L550) | VisualClassification | [tasksource/superclevr](https://hf.co/datasets/tasksource/superclevr) |  |
| 45 | [gameqa/mcq](src/tasksource/vision_tasks.py#L568) | VisualMultipleChoice | [tasksource/gameqa](https://hf.co/datasets/tasksource/gameqa) |  |
| 46 | [valen-game/optimal-action](src/tasksource/vision_tasks.py#L572) | VisualClassification | [tasksource/valen-game](https://hf.co/datasets/tasksource/valen-game) | ✓ |
| 47 | [t2isafety/toxicity](src/tasksource/vision_tasks.py#L578) | VisualClassification | [tasksource/t2i-safety](https://hf.co/datasets/tasksource/t2i-safety) | ✓ |
| 48 | [met-meme/sentiment](src/tasksource/vision_tasks.py#L608) | VisualClassification | [Anthony3456347095/MET-Meme](https://hf.co/datasets/Anthony3456347095/MET-Meme) | ✓ |
| 49 | [met-meme/intention](src/tasksource/vision_tasks.py#L611) | VisualClassification | [Anthony3456347095/MET-Meme](https://hf.co/datasets/Anthony3456347095/MET-Meme) | ✓ |
| 50 | [met-meme/offensiveness](src/tasksource/vision_tasks.py#L614) | VisualClassification | [Anthony3456347095/MET-Meme](https://hf.co/datasets/Anthony3456347095/MET-Meme) | ✓ |
| 51 | [met-meme/metaphor](src/tasksource/vision_tasks.py#L618) | VisualClassification | [Anthony3456347095/MET-Meme](https://hf.co/datasets/Anthony3456347095/MET-Meme) | ✓ |
| 52 | [memeqa/mcq](src/tasksource/vision_tasks.py#L622) | VisualMultipleChoice | [tasksource/memeqa](https://hf.co/datasets/tasksource/memeqa) |  |
| 53 | [tron/mcq](src/tasksource/vision_tasks.py#L627) | VisualMultipleChoice | [tasksource/tron-mcq](https://hf.co/datasets/tasksource/tron-mcq) |  |
| 54 | [trace/mcq](src/tasksource/vision_tasks.py#L633) | VisualMultipleChoice | [tasksource/trace-mcq](https://hf.co/datasets/tasksource/trace-mcq) |  |
| 55 | [mentis-oculi/paper-fold](src/tasksource/vision_tasks.py#L639) | VisualMultipleChoice | [tasksource/mentis-oculi-paper-fold](https://hf.co/datasets/tasksource/mentis-oculi-paper-fold) |  |
| 56 | [sphinx/mcq](src/tasksource/vision_tasks.py#L646) | VisualMultipleChoice | [tasksource/sphinx-mcq](https://hf.co/datasets/tasksource/sphinx-mcq) |  |
| 57 | [visgym/mcq](src/tasksource/vision_tasks.py#L652) | VisualMultipleChoice | [tasksource/visgym-mcq](https://hf.co/datasets/tasksource/visgym-mcq) |  |

### Candidate grounding

`rico-widget/element` selects a widget using human instructions and native RICO
semantic candidate boxes. It preserves native train/validation/test splits and
uses `VisualMultipleChoice`. Missing or ambiguous targets, unsupported annotation
canvases, and rows exceeding the existing 26-choice limit are excluded. It never
inserts the target or creates distractors. The source revision is the same as
`rico-widget/grid7`.

Set-of-Mark is an optional deterministic loader augmentation, not a duplicate
catalog task. See [grounding usage](docs/jev/README.md#grounding-and-set-of-mark)
and the [pinned candidate pilot](docs/jev/rico-grounding-audit.json). Mind2Web can
be excluded by original or mirror Hub identity. Grid refinement remains a separate
repackaged crop transformation using the same displayed-image box convention.

### Game decisions

`gameqa/mcq` uses native GameQA options; `valen-game/optimal-action` asks whether each proposed direction is optimal, retaining every tied optimal action. Its Tasksource view selects English source rules before sampling (60,000 train / 896 test decisions); the mirror retains both source languages. Both use [reproducibly joined image mirrors](scripts/repackage_dataset/games.py), preserve native train/test partitions and encoded images, and exclude shared evaluation screenshots/layouts from training. See the [source and recast audit](docs/jev/game-expansion-audit.md).

### Image safety

`t2isafety/toxicity` classifies generated images as safe/unsafe under an explicit six-category toxicity policy. The [reproducible mirror](scripts/repackage_dataset/visual_safety.py) retains 30,781 train and 366 test judgments, native image bytes and splits. Generation prompts are excluded from inputs; demographic prediction, privacy labels and unreviewed real images are outside this view. See the [source and integrity audit](docs/jev/audits/visual-safety-audit.json).

### Memes

`memeqa/mcq` uses the native single-answer, human-verified None-minus questions; multiselect QAs are excluded. The [reproducible archive join](scripts/repackage_dataset/memes.py) preserves the original meme-level training/development/test partitions and encoded images. Automatically constructed source distractors are explicitly recorded in annotation provenance; image research-use terms are separate from the annotation repository's MIT license.

`met-meme/sentiment`, `/intention`, `/offensiveness`, and `/metaphor` load the existing image-embedded Hub upload directly. Only English source records are used. The Hub's partitions are retained, with exact held-out image copies removed from earlier splits; these are not claimed to be the author's original splits. The original Kaggle license is Unknown and is marked unspecified. See the [coarse source and recast audit](docs/jev/audits/meme-safety-batch-audit.json).

### Procedural visual decisions

`tron/mcq`, `trace/mcq` and `mentis-oculi/paper-fold` use the existing visual MC template. Their [reproducible static builders](scripts/repackage_dataset/procedural/README.md) retain native image bytes, record seeds/source pins and verify exactly one accepted option. TRACE retains its native partitions; TRON and Mentis use fresh split-specific seeds with held-out image-copy exclusions.

Initial scope: five TRON environments (100,000/83 train/validation QAs), TRACE (100,000/2,000), and Mentis paper folding (100,000/100). The Mentis pilot does not ingest published benchmark puzzles. Licenses differ: TRON research-use, TRACE CC BY 4.0 with asset credits, Mentis procedural paper-fold Apache-2.0. See the [coarse visual and recast audit](docs/jev/audits/procedural-vision-audit.json).

SPHINX adds a fresh four-family MC view (`sphinx/mcq`, 32/8 pilot; 100k build running). VisGym adds uniquely solved 2D maze next moves (`visgym/mcq`, 100,000/100). See the [suite builders](scripts/repackage_dataset/procedural/suites.py) and [coarse image/recast audit](docs/jev/audits/procedural-suites-audit.json). SPHINX retains native same-scene choices and close count alternatives; VisGym retains its four cardinal actions. Neither imports generator dependencies during normal Tasksource loading. VisualSphinx remains the existing curated RL-20K task; Gym-V is excluded from expansion.
