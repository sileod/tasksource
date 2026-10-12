# Example galleries

[English tasks](english/README.md) · [Searchable HTML](english/index.html)

The English gallery lists every task in `list_tasks()`. It shows up to three real training examples per task, retaining the published Jev state, question, criterion order and gold distribution. Boolean decisions show the probability of true. The HTML shows the first example immediately, with input and decision side by side. Long fields show the beginning and end; expand “Full text” for the complete content. Two more examples are folded below each task. The Markdown and JSON retain full texts.

For tasks absent from the published `full` config, optional source loads show canonical preprocessed fields with decoded class labels instead. These are labeled separately, with provenance in `examples.json`. Failed or timed-out loads are reported, not replaced by fabricated examples. These samples illustrate the transformations; they are not a comprehensive quality audit.

Regenerate with [write_example_gallery.py](../../scripts/write_example_gallery.py):

```bash
PYTHONPATH=src python scripts/write_example_gallery.py --examples 3 --load-missing
# One to five examples; pin the published snapshot to reproduce its rows:
PYTHONPATH=src python scripts/write_example_gallery.py --examples 5 --revision COMMIT
# Refresh selected tasks from current preprocessing (explicitly labeled local previews):
PYTHONPATH=src python scripts/write_example_gallery.py --refresh-task maven-arg/role --refresh-task lex_glue/eurlex
# Serve the searchable gallery locally:
python -m http.server 8000 --directory docs/examples/english
```

Open `http://localhost:8000`. The HTML also opens directly as a local file and requires no external scripts. GitHub renders the Markdown task pages; it displays HTML as source.

The generator scans only published training shards in filename/row order. Canonical fallbacks use Tasksource's default seed 0 and a per-task timeout (90 seconds by default). The JSON records the export revision and generator hash, plus source provenance for canonical fallbacks. Reproducing a fallback also requires the same Tasksource code and available source revisions.
