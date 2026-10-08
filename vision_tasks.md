18 visual tasks. Load one with `load_task(id, vision=True)`; the annotations are in [vision_tasks.py](src/tasksource/vision_tasks.py), and tasks kept out on purpose are in [parked.py](src/tasksource/parked.py).

| # | id | type | dataset | question |
|--:|---|---|---|:-:|
| 1 | [nlvr2](src/tasksource/vision_tasks.py#L38) | VisualClassification | [pingzhili/nlvr2](https://hf.co/datasets/pingzhili/nlvr2) | ✓ |
| 2 | [aokvqa](src/tasksource/vision_tasks.py#L45) | VisualMultipleChoice | [HuggingFaceM4/A-OKVQA](https://hf.co/datasets/HuggingFaceM4/A-OKVQA) |  |
| 3 | [scienceqa-img](src/tasksource/vision_tasks.py#L51) | VisualMultipleChoice | [derek-thomas/ScienceQA](https://hf.co/datasets/derek-thomas/ScienceQA) |  |
| 4 | [ai2d](src/tasksource/vision_tasks.py#L58) | VisualMultipleChoice | [tasksource/ai2d](https://hf.co/datasets/tasksource/ai2d) |  |
| 5 | [figureqa](src/tasksource/vision_tasks.py#L64) | VisualClassification | [vikhyatk/figureqa](https://hf.co/datasets/vikhyatk/figureqa) | ✓ |
| 6 | [mind2web/action](src/tasksource/vision_tasks.py#L103) | VisualClassification | [tasksource/multimodal-mind2web](https://hf.co/datasets/tasksource/multimodal-mind2web) | ✓ |
| 7 | [mind2web/element](src/tasksource/vision_tasks.py#L110) | VisualMultipleChoice | [tasksource/multimodal-mind2web](https://hf.co/datasets/tasksource/multimodal-mind2web) | ✓ |
| 8 | [mind2web/x10](src/tasksource/vision_tasks.py#L116) | VisualClassification | [tasksource/multimodal-mind2web](https://hf.co/datasets/tasksource/multimodal-mind2web) | ✓ |
| 9 | [mind2web/y10](src/tasksource/vision_tasks.py#L123) | VisualClassification | [tasksource/multimodal-mind2web](https://hf.co/datasets/tasksource/multimodal-mind2web) | ✓ |
| 10 | [mind2web/grid5](src/tasksource/vision_tasks.py#L130) | VisualClassification | [tasksource/multimodal-mind2web](https://hf.co/datasets/tasksource/multimodal-mind2web) | ✓ |
| 11 | [mind2web/grid7](src/tasksource/vision_tasks.py#L137) | VisualClassification | [tasksource/multimodal-mind2web](https://hf.co/datasets/tasksource/multimodal-mind2web) | ✓ |
| 12 | [m3cot](src/tasksource/vision_tasks.py#L145) | VisualMultipleChoice | [LightChen2333/M3CoT](https://hf.co/datasets/LightChen2333/M3CoT) |  |
| 13 | [exams-v](src/tasksource/vision_tasks.py#L165) | VisualClassification | [MBZUAI/EXAMS-V](https://hf.co/datasets/MBZUAI/EXAMS-V) |  |
| 14 | [visualsphinx](src/tasksource/vision_tasks.py#L176) | VisualMultipleChoice | [VisualSphinx/VisualSphinx-V1-RL-20K](https://hf.co/datasets/VisualSphinx/VisualSphinx-V1-RL-20K) |  |
| 15 | [muslr/tfu](src/tasksource/vision_tasks.py#L186) | VisualClassification | [Aiden0526/MuSLR](https://hf.co/datasets/Aiden0526/MuSLR) |  |
| 16 | [muslr/mc](src/tasksource/vision_tasks.py#L195) | VisualMultipleChoice | [Aiden0526/MuSLR](https://hf.co/datasets/Aiden0526/MuSLR) |  |
| 17 | [iconqa/text](src/tasksource/vision_tasks.py#L205) | VisualMultipleChoice | [tasksource/iconqa-text](https://hf.co/datasets/tasksource/iconqa-text) |  |
| 18 | [view2space/mcq](src/tasksource/vision_tasks.py#L212) | VisualMultipleChoice | [tasksource/view2space](https://hf.co/datasets/tasksource/view2space) |  |
