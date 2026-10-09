44 visual tasks. Load one with `load_task(id, vision=True)`; the annotations are in [vision_tasks.py](src/tasksource/vision_tasks.py). Evaluation benchmarks are in [eval_only.py](src/tasksource/eval_only.py); other excluded annotations are in [parked.py](src/tasksource/parked.py).

| # | id | type | dataset | question |
|--:|---|---|---|:-:|
| 1 | [nlvr2](src/tasksource/vision_tasks.py#L160) | VisualClassification | [pingzhili/nlvr2](https://hf.co/datasets/pingzhili/nlvr2) | ✓ |
| 2 | [aokvqa](src/tasksource/vision_tasks.py#L167) | VisualMultipleChoice | [HuggingFaceM4/A-OKVQA](https://hf.co/datasets/HuggingFaceM4/A-OKVQA) |  |
| 3 | [scienceqa-img](src/tasksource/vision_tasks.py#L173) | VisualMultipleChoice | [derek-thomas/ScienceQA](https://hf.co/datasets/derek-thomas/ScienceQA) |  |
| 4 | [ai2d](src/tasksource/vision_tasks.py#L180) | VisualMultipleChoice | [tasksource/ai2d](https://hf.co/datasets/tasksource/ai2d) |  |
| 5 | [figureqa](src/tasksource/vision_tasks.py#L186) | VisualClassification | [vikhyatk/figureqa](https://hf.co/datasets/vikhyatk/figureqa) | ✓ |
| 6 | [mind2web/action](src/tasksource/vision_tasks.py#L208) | VisualClassification | [tasksource/multimodal-mind2web](https://hf.co/datasets/tasksource/multimodal-mind2web) | ✓ |
| 7 | [mind2web/element](src/tasksource/vision_tasks.py#L215) | VisualMultipleChoice | [tasksource/multimodal-mind2web](https://hf.co/datasets/tasksource/multimodal-mind2web) | ✓ |
| 8 | [mind2web/x10](src/tasksource/vision_tasks.py#L221) | VisualClassification | [tasksource/multimodal-mind2web](https://hf.co/datasets/tasksource/multimodal-mind2web) | ✓ |
| 9 | [mind2web/y10](src/tasksource/vision_tasks.py#L228) | VisualClassification | [tasksource/multimodal-mind2web](https://hf.co/datasets/tasksource/multimodal-mind2web) | ✓ |
| 10 | [mind2web/grid5](src/tasksource/vision_tasks.py#L235) | VisualClassification | [tasksource/multimodal-mind2web](https://hf.co/datasets/tasksource/multimodal-mind2web) | ✓ |
| 11 | [mind2web/grid7](src/tasksource/vision_tasks.py#L242) | VisualClassification | [tasksource/multimodal-mind2web](https://hf.co/datasets/tasksource/multimodal-mind2web) | ✓ |
| 12 | [m3cot](src/tasksource/vision_tasks.py#L250) | VisualMultipleChoice | [LightChen2333/M3CoT](https://hf.co/datasets/LightChen2333/M3CoT) |  |
| 13 | [exams-v](src/tasksource/vision_tasks.py#L270) | VisualClassification | [MBZUAI/EXAMS-V](https://hf.co/datasets/MBZUAI/EXAMS-V) |  |
| 14 | [visualsphinx](src/tasksource/vision_tasks.py#L281) | VisualMultipleChoice | [VisualSphinx/VisualSphinx-V1-RL-20K](https://hf.co/datasets/VisualSphinx/VisualSphinx-V1-RL-20K) |  |
| 15 | [muslr/tfu](src/tasksource/vision_tasks.py#L291) | VisualClassification | [Aiden0526/MuSLR](https://hf.co/datasets/Aiden0526/MuSLR) |  |
| 16 | [muslr/mc](src/tasksource/vision_tasks.py#L300) | VisualMultipleChoice | [Aiden0526/MuSLR](https://hf.co/datasets/Aiden0526/MuSLR) |  |
| 17 | [iconqa/text](src/tasksource/vision_tasks.py#L310) | VisualMultipleChoice | [tasksource/iconqa-text](https://hf.co/datasets/tasksource/iconqa-text) |  |
| 18 | [view2space/mcq](src/tasksource/vision_tasks.py#L317) | VisualMultipleChoice | [tasksource/view2space](https://hf.co/datasets/tasksource/view2space) |  |
| 19 | [visual7w](src/tasksource/vision_tasks.py#L333) | VisualMultipleChoice | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) |  |
| 20 | [clevr/yesno](src/tasksource/vision_tasks.py#L338) | VisualClassification | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) |  |
| 21 | [mapqa/yesno](src/tasksource/vision_tasks.py#L343) | VisualClassification | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) |  |
| 22 | [tqa](src/tasksource/vision_tasks.py#L348) | VisualMultipleChoice | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) |  |
| 23 | [hateful-memes](src/tasksource/vision_tasks.py#L354) | VisualClassification | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) |  |
| 24 | [clevr/color](src/tasksource/vision_tasks.py#L362) | VisualClassification | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) |  |
| 25 | [clevr/shape](src/tasksource/vision_tasks.py#L367) | VisualClassification | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) |  |
| 26 | [clevr/size](src/tasksource/vision_tasks.py#L372) | VisualClassification | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) |  |
| 27 | [clevr/material](src/tasksource/vision_tasks.py#L377) | VisualClassification | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) |  |
| 28 | [intergps](src/tasksource/vision_tasks.py#L382) | VisualMultipleChoice | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) |  |
| 29 | [clevr/count](src/tasksource/vision_tasks.py#L387) | VisualClassification | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) | ✓ |
| 30 | [tallyqa/count](src/tasksource/vision_tasks.py#L394) | VisualClassification | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) | ✓ |
| 31 | [vsr/yesno](src/tasksource/vision_tasks.py#L401) | VisualClassification | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) | ✓ |
| 32 | [rico-widget/grid7](src/tasksource/vision_tasks.py#L479) | VisualClassification | [bevaya/RICO-WidgetCaptioning](https://hf.co/datasets/bevaya/RICO-WidgetCaptioning) | ✓ |
| 33 | [rico-widget/element](src/tasksource/vision_tasks.py#L489) | VisualMultipleChoice | [bevaya/RICO-WidgetCaptioning](https://hf.co/datasets/bevaya/RICO-WidgetCaptioning) | ✓ |
| 34 | [lvis/region](src/tasksource/vision_tasks.py#L498) | VisualClassification | [tasksource/coco-regions](https://hf.co/datasets/tasksource/coco-regions) | ✓ |
| 35 | [coco/panoptic-region](src/tasksource/vision_tasks.py#L504) | VisualClassification | [tasksource/coco-regions](https://hf.co/datasets/tasksource/coco-regions) | ✓ |
| 36 | [doclaynet/region](src/tasksource/vision_tasks.py#L510) | VisualClassification | [tasksource/doclaynet-region](https://hf.co/datasets/tasksource/doclaynet-region) | ✓ |
| 37 | [bapps/preference](src/tasksource/vision_tasks.py#L516) | VisualMultipleChoice | [tasksource/bapps](https://hf.co/datasets/tasksource/bapps) | ✓ |
| 38 | [spair71k/grid7](src/tasksource/vision_tasks.py#L522) | VisualClassification | [tasksource/spair71k-grid](https://hf.co/datasets/tasksource/spair71k-grid) | ✓ |
| 39 | [superclevr/yesno](src/tasksource/vision_tasks.py#L538) | VisualClassification | [tasksource/superclevr](https://hf.co/datasets/tasksource/superclevr) |  |
| 40 | [superclevr/count](src/tasksource/vision_tasks.py#L540) | VisualClassification | [tasksource/superclevr](https://hf.co/datasets/tasksource/superclevr) |  |
| 41 | [superclevr/color](src/tasksource/vision_tasks.py#L542) | VisualClassification | [tasksource/superclevr](https://hf.co/datasets/tasksource/superclevr) |  |
| 42 | [superclevr/shape](src/tasksource/vision_tasks.py#L544) | VisualClassification | [tasksource/superclevr](https://hf.co/datasets/tasksource/superclevr) |  |
| 43 | [superclevr/size](src/tasksource/vision_tasks.py#L546) | VisualClassification | [tasksource/superclevr](https://hf.co/datasets/tasksource/superclevr) |  |
| 44 | [superclevr/material](src/tasksource/vision_tasks.py#L548) | VisualClassification | [tasksource/superclevr](https://hf.co/datasets/tasksource/superclevr) |  |

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
