"""Bounded English-focused paper discovery; emits prospects, never task annotations."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

from tasksource.task_discovery.discover_task_prospects import write_rows, save_prospects
from tasksource.task_discovery.jev_review import run as review
from tasksource.task_discovery.scheduled import control, read_rows, save_json

QUERIES = {
    'general': ('English text classification dataset human annotated',
                'English multiple choice dataset training',
                'English natural language inference corpus',
                'English relevance ranking dataset judgments'),
    'medical': ('biomedical text classification annotated corpus English',
                'clinical relation extraction corpus English',
                'medical multiple choice dataset English',
                'biomedical evidence inference annotated dataset'),
    'legal': ('legal text classification annotated dataset English',
              'legal entailment dataset English',
              'legal relevance dataset expert annotations',
              'legal multiple choice dataset English'),
}
MODEL = 'openrouter/~typesafe/jev-latest'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('build/discovery/openalex-en'))
    parser.add_argument('--pilot-only', action='store_true')
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    history = output/'previously-screened.jsonl'
    if not history.exists():
        paths = sorted(Path('build/discovery').glob('prospects-*/checked.jsonl'))
        paths += sorted(Path('build/discovery/exhaustive-queue').glob('*/batches/*/checked.jsonl'))
        paths += [Path('build/discovery/all-hf-multiple-choice-en-screen/checked.jsonl')]
        write_rows(history, [dict(name=r['name'], huggingface=r.get('huggingface', []))
                            for p in paths if p.exists() for r in read_rows(p)])
    def run(command, directory):
        directory.mkdir(parents=True, exist_ok=True)
        save_json(directory/'command.json', command)
        with (directory/'run.log').open('a') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    try:
        for domain, queries in QUERIES.items():
            directory = output/domain
            inventory = directory/'inventory.jsonl'
            if not inventory.exists():
                command = [sys.executable, '-u', '-m', 'tasksource.task_discovery', 'retrieve',
                           '--source', 'openalex', '--language', 'en', '--limit', '100',
                           '--cache', str(output/'cache'), '--output', str(inventory)]
                for query in queries:
                    command += ['--query', query]
                run(command, directory)
        for phase, limit in [('pilot', 12)] + ([] if args.pilot_only else [('expanded', 188)]):
            for domain in QUERIES:
                directory = output/domain/phase
                if (directory/'manifest.json').exists():
                    continue
                directory.mkdir(parents=True, exist_ok=True)
                save_json(output/'status.json', dict(status='running', domain=domain, phase=phase))
                inventory = output/domain/'inventory.jsonl'
                if phase == 'expanded':
                    # Abstract-only pilots often lack train/resource evidence. Prefer
                    # explicit distributions with cards/READMEs for deeper screening.
                    leads = [r for r in read_rows(inventory) if r.get('huggingface') or r.get('github')]
                    inventory = output/domain/'resource-leads.jsonl'
                    write_rows(inventory, leads)
                if not control([MODEL], directory/'control.json'):
                    raise RuntimeError('Reviewer control failed; stopping before the batch')
                command = [sys.executable, '-u', '-m', 'tasksource.task_discovery.discover_task_prospects',
                           str(inventory), '--output', str(directory),
                           '--limit', str(limit), '--language', 'en', '--model', MODEL,
                           '--allow-verified-synthetic', '--exclude', str(history)]
                for checked in sorted(output.glob('*/*/checked.jsonl')):
                    if checked.parent != directory:
                        command += ['--exclude', str(checked)]
                run(command, directory)
                print(f'Completed {domain}/{phase}', flush=True)
        if not args.pilot_only:
            final = output/'english-confirmed'
            if not (final/'manifest.json').exists():
                rows = [r for p in sorted(output.glob('*/expanded/handoff.jsonl')) for r in read_rows(p)]
                final.mkdir(parents=True, exist_ok=True)
                if rows:
                    if not control([MODEL], final/'control.json'):
                        raise RuntimeError('Language review control failed')
                    review(rows, final/'review', output/'cache', [MODEL], languages=['en'])
                    rows = read_rows(final/'review/reviews.jsonl')
                manifest = save_prospects(rows, final, ['human', 'procedural'], True)
                save_json(final/'manifest.json', manifest)
        save_json(output/'status.json', dict(status='pilot_complete' if args.pilot_only else 'complete'))
    except Exception as error:
        save_json(output/'status.json', dict(status='stopped', error=str(error)))
        raise


if __name__ == '__main__':
    main()
