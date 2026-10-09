"""Upload reproducible mirrors; also available as python -m scripts.repackage_dataset."""
import argparse
import inspect
import json
import sys
from pathlib import Path

# Support direct execution as well as module imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import DatasetDict
from huggingface_hub import DatasetCard
from tasksource.metadata.originals import ORIGINALS
from scripts.repackage_dataset.text import (
    sharc, numer_sense, clutrr, wellformed, humicroedit, ethos,
    multilingual_sentiments, mms, chaos_mnli_ambiguity, measuring_hate_speech_votes,
    lewidi, wikipedia_detox_votes, webinstruct, _wikipedia_detox_votes,
)
from scripts.repackage_dataset.vision import view2space, ai2d, iconqa_text, mind2web
from scripts.repackage_dataset.browser import websrc
from scripts.repackage_dataset.mind2web import mind2web_dom
from scripts.repackage_dataset.superclevr import superclevr
from scripts.repackage_dataset.regions import coco_regions, doclaynet_region, bapps, spair71k_grid


BUILDERS = {"mind2web_dom": ("tasksource/mind2web-dom", mind2web_dom, "per-config"),
            "websrc": ("tasksource/websrc", websrc, "per-config"),
            "superclevr": ("tasksource/superclevr", superclevr),
            "coco_regions": ("tasksource/coco-regions", coco_regions, "per-config"),
            "doclaynet_region": ("tasksource/doclaynet-region", doclaynet_region),
            "bapps": ("tasksource/bapps", bapps),
            "spair71k_grid": ("tasksource/spair71k-grid", spair71k_grid),
            "ai2d": ("tasksource/ai2d", ai2d),
            "iconqa_text": ("tasksource/iconqa-text", iconqa_text),
            "mind2web": ("tasksource/multimodal-mind2web", mind2web),
            "view2space": ("tasksource/view2space", view2space),
            "webinstruct": ("tasksource/webinstruct", webinstruct, "per-config"),
            "sharc": ("tasksource/sharc", sharc), "numer_sense": ("tasksource/numer_sense", numer_sense),
            "clutrr": ("tasksource/clutrr", clutrr), "wellformed": ("tasksource/google_wellformed_query", wellformed),
            "humicroedit": ("tasksource/humicroedit", humicroedit, "subtask-1"), "ethos": ("tasksource/ethos", ethos, "multilabel"),
            "multilingual_sentiments": ("tasksource/multilingual-sentiments", multilingual_sentiments),
            "mms": ("tasksource/mms", mms, "per-language"),
            "chaos_mnli_ambiguity": ("tasksource/chaos-mnli-ambiguity", chaos_mnli_ambiguity),
            "measuring_hate_speech_votes": ("tasksource/measuring-hate-speech-votes", measuring_hate_speech_votes),
            "lewidi": ("tasksource/lewidi", lewidi, "per-config"),
            "wikipedia_detox_votes": ("tasksource/wikipedia-detox-votes", wikipedia_detox_votes, "per-config")}


# license metadata for repackaged sets, as the originals state it
LICENSES = {"tasksource/mind2web-dom": "cc-by-4.0",
            "tasksource/websrc": "cc-by-4.0",
            "tasksource/superclevr": "mit",
            "tasksource/coco-regions": ["other", "cc-by-4.0", "cc-by-2.0",
                                      "cc-by-nc-2.0", "cc-by-nc-sa-2.0"],
            "tasksource/doclaynet-region": "cdla-permissive-1.0",
            "tasksource/bapps": "other", "tasksource/spair71k-grid": "other",
            "tasksource/ai2d": "cc-by-sa-4.0", "tasksource/iconqa-text": "cc-by-nc-sa-4.0",
            "tasksource/multimodal-mind2web": "openrail", "tasksource/view2space": "cc-by-4.0", "tasksource/measuring-hate-speech-votes": "cc-by-4.0", "tasksource/lewidi": "other",
            "tasksource/wikipedia-detox-votes": "cc0-1.0",
            "tasksource/webinstruct": "apache-2.0"}


def push_card(repo, build):
    card = DatasetCard.load(repo)
    card.data.source_datasets = ORIGINALS[repo]
    if repo in LICENSES:
        card.data.license = LICENSES[repo]
    sources = ", ".join(f"[{name}](https://huggingface.co/datasets/{name})" for name in ORIGINALS[repo])
    sources = f"Original data: {sources}. " if sources else ""  # an empty entry: the original is not on the Hub
    storage = 'native ImageFolder' if repo == 'tasksource/view2space' else 'parquet'
    card.text = (f"\n# {repo.split('/')[1]}\n\n{inspect.cleandoc(build.__doc__)}\n\n{sources}"
                 f"Repackaged as {storage} for [tasksource](https://github.com/sileod/tasksource) by "
                 "[scripts/repackage_dataset/](https://github.com/sileod/tasksource/tree/main/scripts/repackage_dataset).\n")
    if repo in ('tasksource/coco-regions', 'tasksource/doclaynet-region', 'tasksource/bapps', 'tasksource/spair71k-grid', 'tasksource/superclevr', 'tasksource/websrc'):
        report = json.loads((Path('build') / (repo.split('/')[1] + '-release') / 'provenance.json').read_text())
        reports = {'default': report} if 'splits' in report else report
        card.text += '\n## Release scope\n\nBounded samples of the eligible native annotations; full classification ontologies are retained.\n\n'
        for config, info in reports.items():
            counts = ', '.join(f'{split}: {count["rows"]:,}' for split, count in info['splits'].items())
            card.text += f'- {config}: {counts}.\n'
        card.text += '\nSee [provenance.json](provenance.json) for source pins, selection seeds, rendering configuration and exclusions.\n'
    if repo == 'tasksource/webinstruct':
        prepared_card = Path('build/webinstruct-release/README.md')
        source_card = prepared_card if prepared_card.exists() else Path(__file__).resolve().parents[1] / 'dataset_cards/webinstruct.md'
        card.text = source_card.read_text().split('---', 2)[2]
    card.push_to_hub(repo)
    if repo in ('tasksource/view2space', 'tasksource/ai2d', 'tasksource/iconqa-text', 'tasksource/multimodal-mind2web',
                'tasksource/coco-regions', 'tasksource/doclaynet-region', 'tasksource/bapps', 'tasksource/spair71k-grid', 'tasksource/superclevr', 'tasksource/websrc', 'tasksource/mind2web-dom'):
        from huggingface_hub import HfApi
        directory = Path('build') / ('view2space-release-imagefolder' if repo == 'tasksource/view2space' else repo.split('/')[1] + '-release')
        for filename in ('provenance.json', 'excluded-questions.jsonl'):
            HfApi().upload_file(path_or_fileobj=directory / filename, path_in_repo=filename,
                repo_id=repo, repo_type='dataset', commit_message='Record visual conversion provenance and exclusions')
    if repo == 'tasksource/webinstruct':
        from huggingface_hub import HfApi
        HfApi().upload_file(path_or_fileobj='build/webinstruct-release/provenance.json', path_in_repo='provenance.json',
                           repo_id=repo, repo_type='dataset', commit_message='Record WebInstruct preprocessing provenance')
        provenance = json.loads(Path('build/webinstruct-release/provenance.json').read_text())
        for filename, field in [('bad-examples.jsonl', 'removal_manifest_sha256'), ('repairs.jsonl', 'repairs_sha256')]:
            if field in provenance:
                HfApi().upload_file(path_or_fileobj=Path('build/webinstruct-release') / filename, path_in_repo=filename,
                                   repo_id=repo, repo_type='dataset', commit_message='Record WebInstruct audit decisions')
        if 'audit' in provenance:
            for filename in ['audit-prompt.txt', 'audit-settings.json', 'screen-verdicts.jsonl', 'confirm-verdicts.jsonl']:
                path = Path('build/webinstruct-release') / filename
                if path.exists():
                    HfApi().upload_file(path_or_fileobj=path, path_in_repo=filename, repo_id=repo,
                                       repo_type='dataset', commit_message='Document WebInstruct presentation audit')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+", choices=BUILDERS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--card-only", action="store_true")
    parser.add_argument("--bad-examples", type=Path, help="WebInstruct only: confirmed-removal manifest")
    parser.add_argument("--repairs", type=Path, help="WebInstruct only: verified presentation repairs")
    parser.add_argument("--max-rows", type=int, help="Rendered mirrors: maximum training examples (default 5000)")
    parser.add_argument("--max-rows-eval", type=int, help="Rendered mirrors: maximum examples per native evaluation split (default 100)")
    parser.add_argument("--seed", type=int, help="Rendered mirrors: deterministic source annotation sampling seed")
    args = parser.parse_args()
    options = {key: getattr(args, key) for key in ("max_rows", "max_rows_eval", "seed")
               if getattr(args, key) is not None}
    if any(options.get(key, 1) < 1 for key in ("max_rows", "max_rows_eval")):
        parser.error("Row limits must be positive")
    if (args.bad_examples or args.repairs) and args.names != ['webinstruct']:
        parser.error('--bad-examples and --repairs apply only to webinstruct')
    for name in args.names:
        repo, build, *config = BUILDERS[name]
        assert repo in ORIGINALS, f"add {repo} to tasksource/metadata/originals.py"
        if options.keys() - inspect.signature(build).parameters.keys():
            parser.error(f"Sampling arguments are not supported by {name}")
        if not args.card_only:
            dataset = build(args.bad_examples, args.repairs) if name == 'webinstruct' else build(**options)
            if isinstance(dataset, Path):  # native imagefolder: retain one copy of each image
                print(repo, json.loads((dataset / 'provenance.json').read_text()))
                if args.dry_run:
                    continue
                from huggingface_hub import HfApi
                api = HfApi()
                api.create_repo(repo, repo_type='dataset', exist_ok=True)
                api.upload_large_folder(repo_id=repo, repo_type='dataset', folder_path=dataset,
                    allow_patterns=['data.zip', 'README.md', 'provenance.json', 'excluded-questions.jsonl'],
                    num_workers=8)
                push_card(repo, build)
                continue
            # a builder returns one DatasetDict, or {config: DatasetDict} for per-config repos
            configs = dataset if config in (["per-language"], ["per-config"]) else {config[0] if config else "default": dataset}
            for config_name, splits in configs.items():
                print(repo, config_name, splits, sep="\n")
                if not args.dry_run:
                    splits.push_to_hub(repo, config_name=config_name)
            if args.dry_run:
                continue
        push_card(repo, build)


if __name__ == "__main__":
    main()
