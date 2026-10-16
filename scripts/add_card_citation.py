"""Add the tasksource citation to every ``tasksource/`` dataset card on the Hub.

    PYTHONPATH=.:src python scripts/add_card_citation.py            # report what would change
    PYTHONPATH=.:src python scripts/add_card_citation.py --push     # one card-only commit per dataset

Only the card body changes (``tasksource.citation.with_citation``): a section that already cites
tasksource is refreshed, a citation of the original data is kept, and YAML metadata is untouched.
Repositories without a card are listed, not given one.
"""
import argparse

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from tasksource.citation import with_citation


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--author', default='tasksource')
    parser.add_argument('--push', action='store_true')
    args = parser.parse_args()
    api, changed, missing = HfApi(), [], []
    for info in api.list_datasets(author=args.author):
        try:
            text = open(hf_hub_download(info.id, 'README.md', repo_type='dataset', force_download=True)).read()
        except EntryNotFoundError:
            missing.append(info.id); continue
        head, body = (text.split('---', 2)[:2], text.split('---', 2)[2]) if text.startswith('---') else (None, text)
        updated = with_citation(body)
        if updated == body:
            continue
        changed.append(info.id)
        if args.push:
            content = f'---{head[1]}---{updated}' if head else updated
            api.upload_file(path_or_fileobj=content.encode(), path_in_repo='README.md', repo_id=info.id,
                            repo_type='dataset', commit_message='Card: tasksource citation')
            print('updated', info.id, flush=True)
    print(f'{len(changed)} cards {"updated" if args.push else "to update"}; {len(missing)} without a card: {", ".join(missing)}')


if __name__ == '__main__':
    main()
