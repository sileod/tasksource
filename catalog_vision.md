33 visual tasks. Load one with `load_task(id, vision=True)`; the annotations are in [vision_tasks.py](src/tasksource/vision_tasks.py). Evaluation benchmarks are in [eval_only.py](src/tasksource/eval_only.py); other excluded annotations are in [parked.py](src/tasksource/parked.py).

| # | id | type | dataset | question |
|--:|---|---|---|:-:|
| 1 | [nlvr2](src/tasksource/vision_tasks.py#L134) | VisualClassification | [pingzhili/nlvr2](https://hf.co/datasets/pingzhili/nlvr2) | ✓ |
| 2 | [aokvqa](src/tasksource/vision_tasks.py#L141) | VisualMultipleChoice | [HuggingFaceM4/A-OKVQA](https://hf.co/datasets/HuggingFaceM4/A-OKVQA) |  |
| 3 | [scienceqa-img](src/tasksource/vision_tasks.py#L147) | VisualMultipleChoice | [derek-thomas/ScienceQA](https://hf.co/datasets/derek-thomas/ScienceQA) |  |
| 4 | [ai2d](src/tasksource/vision_tasks.py#L154) | VisualMultipleChoice | [tasksource/ai2d](https://hf.co/datasets/tasksource/ai2d) |  |
| 5 | [figureqa](src/tasksource/vision_tasks.py#L160) | VisualClassification | [vikhyatk/figureqa](https://hf.co/datasets/vikhyatk/figureqa) | ✓ |
| 6 | [mind2web/action](src/tasksource/vision_tasks.py#L182) | VisualClassification | [tasksource/multimodal-mind2web](https://hf.co/datasets/tasksource/multimodal-mind2web) | ✓ |
| 7 | [mind2web/element](src/tasksource/vision_tasks.py#L189) | VisualMultipleChoice | [tasksource/multimodal-mind2web](https://hf.co/datasets/tasksource/multimodal-mind2web) | ✓ |
| 8 | [mind2web/x10](src/tasksource/vision_tasks.py#L195) | VisualClassification | [tasksource/multimodal-mind2web](https://hf.co/datasets/tasksource/multimodal-mind2web) | ✓ |
| 9 | [mind2web/y10](src/tasksource/vision_tasks.py#L202) | VisualClassification | [tasksource/multimodal-mind2web](https://hf.co/datasets/tasksource/multimodal-mind2web) | ✓ |
| 10 | [mind2web/grid5](src/tasksource/vision_tasks.py#L209) | VisualClassification | [tasksource/multimodal-mind2web](https://hf.co/datasets/tasksource/multimodal-mind2web) | ✓ |
| 11 | [mind2web/grid7](src/tasksource/vision_tasks.py#L216) | VisualClassification | [tasksource/multimodal-mind2web](https://hf.co/datasets/tasksource/multimodal-mind2web) | ✓ |
| 12 | [m3cot](src/tasksource/vision_tasks.py#L224) | VisualMultipleChoice | [LightChen2333/M3CoT](https://hf.co/datasets/LightChen2333/M3CoT) |  |
| 13 | [exams-v](src/tasksource/vision_tasks.py#L244) | VisualClassification | [MBZUAI/EXAMS-V](https://hf.co/datasets/MBZUAI/EXAMS-V) |  |
| 14 | [visualsphinx](src/tasksource/vision_tasks.py#L255) | VisualMultipleChoice | [VisualSphinx/VisualSphinx-V1-RL-20K](https://hf.co/datasets/VisualSphinx/VisualSphinx-V1-RL-20K) |  |
| 15 | [muslr/tfu](src/tasksource/vision_tasks.py#L265) | VisualClassification | [Aiden0526/MuSLR](https://hf.co/datasets/Aiden0526/MuSLR) |  |
| 16 | [muslr/mc](src/tasksource/vision_tasks.py#L274) | VisualMultipleChoice | [Aiden0526/MuSLR](https://hf.co/datasets/Aiden0526/MuSLR) |  |
| 17 | [iconqa/text](src/tasksource/vision_tasks.py#L284) | VisualMultipleChoice | [tasksource/iconqa-text](https://hf.co/datasets/tasksource/iconqa-text) |  |
| 18 | [view2space/mcq](src/tasksource/vision_tasks.py#L291) | VisualMultipleChoice | [tasksource/view2space](https://hf.co/datasets/tasksource/view2space) |  |
| 19 | [visual7w](src/tasksource/vision_tasks.py#L307) | VisualMultipleChoice | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) |  |
| 20 | [clevr/yesno](src/tasksource/vision_tasks.py#L312) | VisualClassification | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) |  |
| 21 | [mapqa/yesno](src/tasksource/vision_tasks.py#L317) | VisualClassification | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) |  |
| 22 | [tqa](src/tasksource/vision_tasks.py#L322) | VisualMultipleChoice | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) |  |
| 23 | [hateful-memes](src/tasksource/vision_tasks.py#L328) | VisualClassification | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) |  |
| 24 | [clevr/color](src/tasksource/vision_tasks.py#L336) | VisualClassification | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) |  |
| 25 | [clevr/shape](src/tasksource/vision_tasks.py#L341) | VisualClassification | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) |  |
| 26 | [clevr/size](src/tasksource/vision_tasks.py#L346) | VisualClassification | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) |  |
| 27 | [clevr/material](src/tasksource/vision_tasks.py#L351) | VisualClassification | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) |  |
| 28 | [intergps](src/tasksource/vision_tasks.py#L356) | VisualMultipleChoice | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) |  |
| 29 | [clevr/count](src/tasksource/vision_tasks.py#L361) | VisualClassification | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) | ✓ |
| 30 | [tallyqa/count](src/tasksource/vision_tasks.py#L368) | VisualClassification | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) | ✓ |
| 31 | [vsr/yesno](src/tasksource/vision_tasks.py#L375) | VisualClassification | [HuggingFaceM4/the_cauldron](https://hf.co/datasets/HuggingFaceM4/the_cauldron) | ✓ |
| 32 | [rico-widget/grid7](src/tasksource/vision_tasks.py#L453) | VisualClassification | [bevaya/RICO-WidgetCaptioning](https://hf.co/datasets/bevaya/RICO-WidgetCaptioning) | ✓ |
| 33 | [rico-widget/element](src/tasksource/vision_tasks.py#L463) | VisualMultipleChoice | [bevaya/RICO-WidgetCaptioning](https://hf.co/datasets/bevaya/RICO-WidgetCaptioning) | ✓ |

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
