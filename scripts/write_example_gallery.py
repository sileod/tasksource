"""Render real published Jev examples, indexed by the complete English catalog.

PYTHONPATH=src python scripts/write_example_gallery.py --examples 3
The snapshot is pinned in examples.json; rerun with --revision to reproduce it.
Use --load-missing for bounded canonical source loads; failures stay visible.
"""
import argparse
import hashlib
import html
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from tasksource import list_tasks

REPO = 'tasksource/tasksource-jev-typed-decisions'
ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['source', 'id', 'state', 'question', 'kind', 'options', 'target', 'split']


def collect(revision, count):
    info = HfApi().dataset_info(REPO, revision=revision)
    examples = {}
    files = sorted(s.rfilename for s in info.siblings
                   if s.rfilename.startswith('full/train') and s.rfilename.endswith('.parquet'))
    for name in files:
        path = hf_hub_download(REPO, name, repo_type='dataset', revision=info.sha)
        for batch in pq.ParquetFile(path).iter_batches(columns=FIELDS):
            # Avoid materializing large text columns once a batch's tasks are filled.
            sources = batch.column('source').to_pylist()
            for index, source in enumerate(sources):
                selected = examples.setdefault(source, [])
                if len(selected) < count:
                    row = batch.slice(index, 1).to_pylist()[0]
                    if all(row['id'] != previous['id'] for previous in selected):
                        selected.append(row)
        print(name, len(examples), 'sources', flush=True)
    return info.sha, examples


def source_examples(task, count, current_recast=False):
    """Canonical fallback for tasks absent from the exported snapshot."""
    from datasets import ClassLabel
    from tasksource import load_task, task_provenance
    data = load_task(task, max_rows=count, max_rows_eval=count)
    if 'train' not in data or not len(data['train']):
        raise ValueError('No labeled training rows')
    split = data['train']
    def decode(value, feature):
        if isinstance(feature, ClassLabel):
            return feature.int2str(value) if value >= 0 else value
        if hasattr(feature, 'feature') and isinstance(value, list):
            return [decode(v, feature.feature) for v in value]
        return value
    question = getattr(data, 'question', None)
    label_feature = split.features.get('labels')
    if hasattr(label_feature, 'feature'):
        label_feature = label_feature.feature
    criteria = list(label_feature.names) if isinstance(label_feature, ClassLabel) else []
    if not question and 'tokens' not in split.features:
        from tasksource.jev.questions import default_question
        question = default_question(criteria, 'sentence2' in split.features)
    if current_recast and (isinstance(split.features.get('labels'), ClassLabel) or 'inputs' in split.features):
        from tasksource.recast import recast_jev
        from scripts.build_jev_dataset import to_training_row
        recast = recast_jev(data, task=task, question=question)['train']
        return [dict(to_training_row(row, i, task, 'train'),
                     provenance=task_provenance(task), origin='current local Jev recast (not yet published)')
                for i, row in enumerate(recast.select(range(min(count, len(recast)))))]
    return [dict(source=task, id=f'{task}:canonical:sample:{i}', split='train',
                 canonical={k: decode(v, split.features[k]) for k, v in row.items() if k != 'metadata'},
                 question=question or '', criteria=criteria, provenance=task_provenance(task), row_metadata=row.get('metadata'), origin='canonical preprocessing (not a published Jev row)')
            for i, row in enumerate(split.select(range(min(count, len(split)))))]


def load_missing(tasks, examples, count, timeout, current_recast=False):
    """Bound each source load in its own process; retain errors rather than guessing."""
    def load(task):
        try:
            result = subprocess.run([sys.executable, '-m', 'scripts.write_example_gallery', '--source-task', task,
                                     '--examples', str(count)] + (['--current-recast'] if current_recast else []), capture_output=True, text=True, timeout=timeout, cwd=ROOT)
            if result.returncode:
                raise ValueError(result.stderr.strip().splitlines()[-1])
            return task, json.loads(result.stdout.split('GALLERY_RESULT=')[-1]), None
        except (subprocess.TimeoutExpired, ValueError) as error:
            return task, [], str(error) if not isinstance(error, subprocess.TimeoutExpired) else f'Source load exceeded {timeout}s'
    errors = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for task, rows, error in pool.map(load, [t for t in tasks if not examples.get(t)]):
            examples[task] = rows
            if error:
                errors[task] = error
            print(task, error or f'{len(rows)} examples', flush=True)
    return errors


def format_example(row):
    """Keep criteria order and the complete gold distribution, including soft labels."""
    if 'canonical' in row:
        from tasksource.preprocess import classification_prompt
        values = row['canonical']
        state, question = classification_prompt(values, row['question']) if 'sentence1' in values else (None, row['question'])
        fields = ([('State', state)] if state is not None else []) + [('Question', question)]
        if row.get('criteria'):
            fields.append(('Criteria (label vocabulary)', '\n'.join(row['criteria'])))
        if 'tokens' in values and isinstance(values.get('labels'), list) and len(values['tokens']) == len(values['labels']):
            return fields + [('Tokens / gold tags', '\n'.join(f'{token} → {label}' for token, label in zip(values['tokens'], values['labels'])))]
        return fields + [('Gold labels' if key == 'labels' else key,
                          value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2))
                         for key, value in values.items() if key not in ('sentence1', 'sentence2')]
    if row['kind'] == 'noul' and not row['options'] and len(row['target']) == 1:
        return [('State', row['state']), ('Question', row['question']),
                ('Decision', 'Boolean (noul)'), ('Gold target', f"Probability of true: {row['target'][0]:g}")]
    if len(row['options']) != len(row['target']):
        raise ValueError(f"Misaligned target: {row['id']}")
    options = '\n'.join(f'{i + 1}. {option}' for i, option in enumerate(row['options']))
    gold = '\n'.join(f'{i + 1}. {option} — {weight:g}'
                     for i, (option, weight) in enumerate(zip(row['options'], row['target'])) if weight)
    return [('State', row['state']), ('Question', row['question']),
            (f"Criteria ({row['kind']})", options), ('Gold target (weight)', gold)]


def html_text(value, limit=700):
    """Show a middle-cropped preview; retain the complete text in a native foldout."""
    full = '<pre>' + html.escape(value) + '</pre>'
    if len(value) <= limit:
        return full
    preview = value[:limit * 3 // 4] + '\n… [middle omitted] …\n' + value[-limit // 4:]
    return '<pre>' + html.escape(preview) + '</pre><details><summary>Full text (' + f'{len(value):,}' + ' characters)</summary>' + full + '</details>'


def write(output, revision, examples, count, errors=None):
    output.mkdir(parents=True, exist_ok=True)
    catalog = list_tasks()
    tasks = [dict(id=r.id, type=r.task_type, dataset=r.dataset_name,
                  examples=examples.get(r.id, [])[:count], error=(errors or {}).get(r.id)) for r in catalog.itertuples()]
    covered = sum(bool(t['examples']) for t in tasks)
    summary = f'{covered} of {len(tasks)} English catalog tasks have examples.'
    note = ('Real examples from the published full Jev training config. Criteria stay in exported order; '
            'gold weights are shown exactly. Source-loaded canonical fallbacks and current local recast previews are explicitly labeled. These examples illustrate preprocessing, not a quality guarantee. '
            'Tasks without an exported example remain listed; this does not imply their source is unavailable.')
    payload = dict(repo=REPO, revision=revision, examples_per_task=count, seed=0,
                   generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), tasks=tasks)
    (output / 'examples.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    md = ['# English example gallery', '', summary, '', note, '',
          '[Searchable HTML](index.html) · [Machine-readable examples](examples.json)', '',
          f'[Published snapshot](https://huggingface.co/datasets/{REPO}/tree/{revision})', '']
    index = md + ['| Task | Type | Examples |', '|---|---|--:|']
    (output / 'tasks').mkdir(exist_ok=True)
    articles = []
    esc = html.escape
    for task in tasks:
        task_id = task['id']
        anchor = 'task-' + hashlib.sha256(task_id.encode()).hexdigest()[:12]
        header = f"{task_id} · {task['type']} · {task['dataset']}"
        md = [f'# {task_id}', '', f"{task['type']} · `{task['dataset']}`", '']
        body = []
        for i, row in enumerate(task['examples'], 1):
            provenance = f"{row.get('origin', 'published Jev')} · {row['split']} · {row['id']}"
            md += [f'<details><summary>Example {i}</summary>', '', provenance, '']
            inputs, decisions = [], []
            for label, value in format_example(row):
                # HTML pre handles arbitrary source text, including Markdown fences/HTML.
                md += [f'**{label}**', '', f'<pre>{esc(value)}</pre>', '']
                target = decisions if label in ('Question', 'Decision', 'Gold target', 'Gold labels') or label.startswith(('Criteria', 'Gold target')) else inputs
                style = ' gold' if label.startswith('Gold') else ''
                target.append(f'<div class="field{style}"><h4>{esc(label)}</h4>' + html_text(value) + '</div>')
            md += ['</details>', '']
            rendered = '<div class="example-grid"><div>' + ''.join(inputs) + '</div><div>' + ''.join(decisions) + '</div></div>'
            rendered += f'<details class="provenance"><summary>Source and row ID</summary>{esc(provenance)}</details>'
            body.append('<section class="primary-example">' + rendered + '</section>' if i == 1 else
                        f'<details class="extra-example"><summary>Example {i}</summary>' + rendered + '</details>')
        if not body:
            message = 'No training example in this published Jev snapshot.' + (' Source load: ' + task['error'] if task['error'] else '')
            md += [message, '']
            body.append(f'<p class="missing">{message}</p>')
        (output / 'tasks' / f'{anchor}.md').write_text('\n'.join(md))
        index.append(f"| [{task_id}](tasks/{anchor}.md) | {task['type']} | {len(task['examples'])} |")
        articles.append(f'<article id="{anchor}" data-search="{esc(header.lower(), quote=True)}" '
                        f'data-covered="{bool(task["examples"])}"><h2><a href="#{anchor}">{esc(task_id)}</a></h2>'
                        f'<p class="task-meta">{esc(task["type"])} · {esc(task["dataset"])}</p>' + ''.join(body) + '</article>')
    active = {'task-' + hashlib.sha256(t['id'].encode()).hexdigest()[:12] + '.md' for t in tasks}
    for old in (output / 'tasks').glob('task-*.md'):
        if old.name not in active:
            old.unlink()
    (output / 'README.md').write_text('\n'.join(index) + '\n')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tasksource English examples</title><style>
*{box-sizing:border-box}body{font:15px/1.45 system-ui;max-width:1220px;margin:1.5rem auto;padding:0 1rem;color:#17212b;background:#fafafa}
a{color:#175fa8}article{border:1px solid #dbe0e7;border-radius:8px;background:white;padding:1rem;margin:.8rem 0}
h1{font-size:1.7rem}h2{font-size:1.1rem;margin:0}h4{font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;color:#536071;margin:0 0 .25rem}
.task-meta{font-size:.8rem;color:#667085;margin:.15rem 0 .7rem}.example-grid{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);gap:1.2rem}
.field{margin-bottom:.6rem}pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:0;background:#f4f6f8;border-radius:4px;padding:.6rem;font:13px/1.45 ui-monospace,monospace}
.field > pre{max-height:16rem;overflow:auto}details pre{max-height:70vh;overflow:auto}
.gold pre{background:#edf7ef;color:#185a31}.gold h4{color:#185a31}
summary{cursor:pointer;font-size:.8rem;padding:.3rem 0;color:#175fa8}.provenance{font-size:.75rem;color:#667085;overflow-wrap:anywhere}
.extra-example{border-top:1px solid #edf0f4}.extra-example[open]{padding-top:.4rem}
input{font:inherit;padding:.45rem;border:1px solid #ccd3db;border-radius:5px;max-width:90%}
nav{position:sticky;top:0;background:#fafafaf5;padding:.7rem 0;z-index:1;border-bottom:1px solid #dbe0e7;display:flex;gap:.8rem;align-items:center;flex-wrap:wrap}
nav label{font-size:.8rem}#search{width:350px}#count,.missing{color:#667085;font-size:.8rem}
@media(max-width:700px){.example-grid{grid-template-columns:1fr;gap:0}body{padding:0 .6rem}article{padding:.75rem}}
</style><h1>English example gallery</h1>'''
    page += f'<p>{esc(summary)} First example visible; two more available below each task.</p><details><summary>About these examples</summary><p>{esc(note)}</p></details><p><a href="README.md">Markdown</a> · <a href="examples.json">JSON</a> · <a href="https://huggingface.co/datasets/{REPO}/tree/{revision}">Published snapshot</a></p>'
    page += '''<nav><input id="search" type="search" placeholder="Filter task name, type or source" aria-label="Filter tasks">
<label><input id="missing" type="checkbox"> Missing examples only</label> <span id="count"></span></nav>'''
    page += '\n'.join(articles)
    page += '''<script>
const search=document.querySelector('#search'), missing=document.querySelector('#missing');
function filter(){let count=0;for(const task of document.querySelectorAll('article')){
 task.hidden=!task.dataset.search.includes(search.value.toLowerCase())||(missing.checked&&task.dataset.covered==='True');
 if(!task.hidden)count++;}document.querySelector('#count').textContent=count+' tasks';}
search.addEventListener('input',filter);missing.addEventListener('change',filter);filter();
</script></html>'''
    (output / 'index.html').write_text(page.replace('\r', '&#13;').replace(' \n', '&#32;\n').replace('\t\n', '&#9;\n'))
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples', type=int, choices=range(1, 6), default=3)
    parser.add_argument('--source-task', help=argparse.SUPPRESS)
    parser.add_argument('--current-recast', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--refresh-task', action='append', default=[], help='Refresh a task from its source into the existing snapshot; label local previews')
    parser.add_argument('--load-missing', action='store_true', help='Try canonical source loads for missing exports')
    parser.add_argument('--timeout', type=int, default=90, help='Seconds per fallback source load')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/examples/english')
    args = parser.parse_args()
    if args.source_task:
        print('GALLERY_RESULT=' + json.dumps(source_examples(args.source_task, args.examples, args.current_recast)))
    elif args.refresh_task:
        snapshot = json.loads((args.output / 'examples.json').read_text())
        examples = {t['id']: t['examples'] for t in snapshot['tasks']}
        errors = {t['id']: t['error'] for t in snapshot['tasks'] if t.get('error')}
        for task in args.refresh_task:
            if task not in examples:
                parser.error(f'Unknown task: {task}')
            examples[task] = []
            errors.pop(task, None)
        errors.update(load_missing(args.refresh_task, examples, args.examples, args.timeout, current_recast=True))
        write(args.output, snapshot['revision'], examples, args.examples, errors)
    else:
        revision, examples = collect(args.revision, args.examples)
        errors = load_missing(list_tasks().id, examples, args.examples, args.timeout) if args.load_missing else {}
        write(args.output, revision, examples, args.examples, errors)
