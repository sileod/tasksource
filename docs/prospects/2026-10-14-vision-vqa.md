# English image VQA prospects — 2026-10-14

Provisional machine-screened leads only. No tasks were implemented from this list. HF tags, declared splits and model reviews do not establish answer quality, a usable closed-set projection, or redistribution rights. Sources with audio also need an independently valid image-only view.

Inventory: 2,044 repositories; 1,881 screened after catalog, queue and prior-stage matching; 12 provisional handoffs. The image-classification and PWC stages continue separately.

| Dataset | Model-assessed annotation origin | Model-assessed checks |
|---|---|---|
| [BrundageLab/synthetic_wildlife_health](https://huggingface.co/datasets/BrundageLab/synthetic_wildlife_health) | procedural | unknown |
| [choucsan/OpenUAV-QA](https://huggingface.co/datasets/choucsan/OpenUAV-QA) | procedural | executable |
| [dunktra/dermacheck-temporal-pairs](https://huggingface.co/datasets/dunktra/dermacheck-temporal-pairs) | procedural | unknown |
| [ebrukilic/vizwiz_vqa_dataset](https://huggingface.co/datasets/ebrukilic/vizwiz_vqa_dataset) | human | unknown |
| [kagakouko/LSI-108K](https://huggingface.co/datasets/kagakouko/LSI-108K) | procedural | executable |
| [orrzohar/EMID-Emotion-Matching](https://huggingface.co/datasets/orrzohar/EMID-Emotion-Matching) | procedural | unknown |
| [RyanWW/DynSuperCLEVR](https://huggingface.co/datasets/RyanWW/DynSuperCLEVR) | procedural | executable |
| [tumor-vqa/DeepTumorVQA_2.0](https://huggingface.co/datasets/tumor-vqa/DeepTumorVQA_2.0) | procedural | independent_review |
| [TYTSTQ/ordinary-bench-multiview](https://huggingface.co/datasets/TYTSTQ/ordinary-bench-multiview) | procedural | unknown |
| [TYTSTQ/ordinary-bench](https://huggingface.co/datasets/TYTSTQ/ordinary-bench) | procedural | unknown |
| [Vasanthaleela/engineering-drawings-as1100](https://huggingface.co/datasets/Vasanthaleela/engineering-drawings-as1100) | human | unknown |
| [ZoeyZou/grm-reproduce-align30k-beavertails-v](https://huggingface.co/datasets/ZoeyZou/grm-reproduce-align30k-beavertails-v) | procedural | unknown |

The [JSONL judgments](2026-10-14-vision-vqa.jsonl) retain source split evidence, uncertainty distributions, reviewer IDs and state hashes. Human annotation and verification claims remain source-specific leads for later inspection.

Replay using the image queue in [`task_discovery/README.md`](../../src/tasksource/task_discovery/README.md), with stages `multiple-choice`, `visual-question-answering`, `image-classification` in that order. Known source identities include the visual roster and original repositories behind mirrors.
