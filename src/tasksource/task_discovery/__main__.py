"""python -m tasksource.task_discovery {retrieve,match,link} --help"""
import argparse
import hashlib
import json
from pathlib import Path

from .match import catalog_hash, catalog_index, match, previous_decisions
from .paperswithcode import retrieve
from .link import link


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fetch = commands.add_parser('retrieve', help='Retrieve Papers with Code candidates')
    fetch.add_argument('--snapshot', type=Path, help='Local copy of the pinned .json.gz export')
    fetch.add_argument('--all', action='store_true', help='Include datasets without decision-oriented task tags')
    fetch.add_argument('--task', action='append', default=[], help='Case-insensitive task-tag substring; repeat for OR')
    compare = commands.add_parser('match', help='Deduplicate and compare any canonical candidate JSONL')
    compare.add_argument('input', type=Path)
    compare.add_argument('--unmatched-only', action='store_true')
    compare.add_argument('--unreviewed-only', action='store_true', help='Exclude catalog matches and previously assessed sources')
    enrich = commands.add_parser('link', help='Link papers and suggest Hub mirrors using cached metadata')
    enrich.add_argument('input', type=Path)
    enrich.add_argument('--cache', type=Path, required=True)
    enrich.add_argument('--offline', action='store_true', help='Replay saved API responses without network access')
    enrich.add_argument('--limit', type=int, default=20, help='Maximum candidate records to enrich (default 20)')
    enrich.add_argument('--no-hub-search', action='store_true')
    for command in (fetch, compare, enrich):
        command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'retrieve':
        rows, manifest = retrieve(decision_only=not args.all, snapshot=args.snapshot)
        if args.task:
            rows = [r for r in rows if any(q.casefold() in t.casefold() for q in args.task for t in r['tasks'])]
        manifest.update(task_queries=args.task, output_rows=len(rows))
    elif args.command == 'match':
        raw = args.input.read_bytes()
        names, repos = catalog_index()
        decisions = previous_decisions()
        rows = list(match([json.loads(line) for line in raw.splitlines() if line.strip()], names, repos, decisions))
        manifest = dict(input_sha256=hashlib.sha256(raw).hexdigest(), catalog_sha256=catalog_hash(names, repos),
                        unique_candidates=len(rows), catalog_matches=sum(bool(r['matches']) for r in rows),
                        previous_decisions_sha256=hashlib.sha256(json.dumps(decisions,sort_keys=True).encode()).hexdigest(),
                        previously_reviewed=sum(r['status']=='previously_reviewed' for r in rows))
        if args.unmatched_only:
            rows = [r for r in rows if not r['matches']]
        if args.unreviewed_only:
            rows = [r for r in rows if r['status']=='candidate']
        manifest.update(unmatched_only=args.unmatched_only, unreviewed_only=args.unreviewed_only, output_rows=len(rows))
    else:
        if args.limit < 1:
            parser.error('--limit must be positive')
        raw = args.input.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        rows[:args.limit] = [link(r, args.cache, args.offline, not args.no_hub_search) for r in rows[:args.limit]]
        manifest = dict(input_sha256=hashlib.sha256(raw).hexdigest(), linked_candidates=min(args.limit,len(rows)),
                        offline=args.offline, search_hub=not args.no_hub_search, output_rows=len(rows))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(row, sort_keys=True, ensure_ascii=False)+'\n' for row in rows))
    manifest['output_sha256'] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    print(json.dumps(manifest, sort_keys=True))


if __name__ == '__main__':
    main()
