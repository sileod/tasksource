# Game decisions — 2026-10-13

Two short visual annotations reuse the existing templates and grouped-QA loader. Separate image archives make one-time repackaging necessary; the [reproducible script](../../scripts/repackage_dataset/games.py) is linked from each mirror's dataset card.

| Task | Original source | Mirror | Train / test decisions | License |
|---|---|---|---:|---|
| `gameqa/mcq` | [GameQA-140K](https://huggingface.co/datasets/OpenMOSS-Team/GameQA-140K) | [tasksource/gameqa](https://huggingface.co/datasets/tasksource/gameqa) | 80,379 / 10,493 | MIT |
| `valen-game/optimal-action` | [Valen-Eval-Game](https://huggingface.co/datasets/Valen-Team/Valen-Eval-Game) | [tasksource/valen-game](https://huggingface.co/datasets/tasksource/valen-game) | 120,000 / 2,000 | Apache 2.0 |

GameQA retains native MCQs only. Numeric answers refer to source option positions, not necessarily the option's numeric value. Source prefixes and embedded option blocks are removed before Jev permutation; equivalent duplicate options collapse with the gold answer remapped. Malformed/unresolved choices are excluded. Training copies of held-out screenshots account for 4,818 additional exclusions. No distractors or validation split are invented.

Valen has 30,000 training screenshots and 500 test screenshots despite “Eval” in its name. Every screenshot supplies four action-conditioned yes/no judgments. Positive native probability means optimal, so tied actions remain positive rather than being arbitrarily reduced to one answer. The mirror retains English and Chinese source rules; the Tasksource annotation now selects English before sampling (60,000 train / 896 test decisions). Probabilities, hidden symbolic states, solution traces and solver provenance remain in metadata. Its selected test layouts are not an unbiased full-game benchmark.

Image bytes remain unchanged, with one image stored per QA group. Retained train/test image hashes are disjoint; canonical Valen layout groups are also held out. No conflicting gold labels were found for identical image/question/options. GameQA has 539 repeated native prompts with agreeing labels; export deduplication handles repeated requests.

Two images per task were read and checked through the shared loader/Jev recast. The word-search letter and enemy count are visibly correct; the one-step Sokoban push is correct. An independent forward BFS confirms both optimal actions in the six-step Sokoban test example. These are coarse checks, not certification of all game labels.

The [machine-readable audit](game-expansion-audit.json) records source revisions, file hashes, exclusions, mirror revisions and the reviewed examples.

Published a pilot in [vision revision 0100b447](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions/commit/0100b447ed4ee8b55bb9f89e725b62420ac71d2f): **1,000 training rows per task**, 2,000 additions total; the config now has **68,085 training rows**. Native gold and image hashes were checked for all additions; one actual pilot image per task was also read. All prior Parquet blobs, evaluation shards and the root card body were preserved.

Expanded in [v2026.10.13.3](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions/commit/bfb636a73eb3250e073fafbbb79b989141ca5517): **20,000 training rows per game family**, adding 38,000 rows for **106,085 vision training rows** total. New rows are shuffled together with seed 0; every native gold answer and encoded image hash was verified. New Valen rows are English-only, giving 19,493 English and 507 Chinese rows after preserving the bilingual pilot. GameQA IDs include the source record because native question IDs repeat. Existing shards and evaluation remain unchanged.
