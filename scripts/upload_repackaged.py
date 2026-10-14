"""Upload reproducible mirrors; also available as python -m scripts.repackage_dataset."""
import argparse
import inspect
import json
import sys
from pathlib import Path

# Support direct execution as well as module imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import Dataset, DatasetDict
from huggingface_hub import DatasetCard
from tasksource.metadata.originals import ORIGINALS
from scripts.repackage_dataset.text import (
    sharc, numer_sense, clutrr, wellformed, humicroedit, ethos,
    multilingual_sentiments, mms, chaos_mnli_ambiguity, measuring_hate_speech_votes,
    lewidi, wikipedia_detox_votes, webinstruct, _wikipedia_detox_votes, discosense,
)
from scripts.repackage_dataset.vision import view2space, ai2d, iconqa_text, mind2web
from scripts.repackage_dataset.browser import websrc
from scripts.repackage_dataset.mind2web import mind2web_dom
from scripts.repackage_dataset.relevance import antique, sentfin, wands, scirepeval_search
from scripts.repackage_dataset.specialized_text import acord, edos, evidence_inference
from scripts.repackage_dataset.dialogue import slurp, mcscript2
from scripts.repackage_dataset.stance import vast, fnc1, perspectrum, ukpconvarg, discogem
from scripts.repackage_dataset.specialized_text import pubmed_rct
from scripts.repackage_dataset.linguistic import semcor, streusle, spider
from scripts.repackage_dataset.relations import matres
from scripts.repackage_dataset.superclevr import superclevr
from scripts.repackage_dataset.games import gameqa, valen_game
from scripts.repackage_dataset.target_sentiment import uk_election
from scripts.repackage_dataset.visual_safety import t2i_safety
from scripts.repackage_dataset.memes import memeqa
from scripts.repackage_dataset.procedural.tron import tron_mc,repair_tron_mc
from scripts.repackage_dataset.procedural.trace import trace_mc
from scripts.repackage_dataset.procedural.trace_generate import trace_expanded_mc
from scripts.repackage_dataset.procedural.mentis import mentis_paper_fold
from scripts.repackage_dataset.procedural.common import file_sha256
from scripts.repackage_dataset.regions import coco_regions, doclaynet_region, bapps, spair71k_grid


BUILDERS = {"tron-mcq": ("tasksource/tron-mcq", tron_mc),
            "tron-mcq-repair": ("tasksource/tron-mcq", repair_tron_mc),
            "trace-expanded-mcq": ("tasksource/trace-mcq", trace_expanded_mc),
            "trace-mcq": ("tasksource/trace-mcq", trace_mc),
            "mentis-oculi-paper-fold": ("tasksource/mentis-oculi-paper-fold", mentis_paper_fold),
            "memeqa": ("tasksource/memeqa", memeqa),
            "gameqa": ("tasksource/gameqa", gameqa),
            "uk-election": ("tasksource/uk-election", uk_election),
            "t2i-safety": ("tasksource/t2i-safety", t2i_safety),
            "valen_game": ("tasksource/valen-game", valen_game),
            "discosense": ("tasksource/discosense", discosense),
            "matres": ("tasksource/matres", matres),
            "semcor": ("tasksource/semcor", semcor),
            "streusle": ("tasksource/streusle", streusle),
            "spider": ("tasksource/spider", spider, "per-config"),
            "slurp": ("tasksource/slurp", slurp),
            "antique": ("tasksource/antique", antique),
            "pubmed-rct": ("tasksource/pubmed-rct", pubmed_rct),
            "vast": ("tasksource/vast", vast), "fnc1": ("tasksource/fnc1", fnc1),
            "perspectrum": ("tasksource/perspectrum", perspectrum),
            "ukpconvarg": ("tasksource/ukpconvarg", ukpconvarg), "discogem": ("tasksource/discogem", discogem),
            "mcscript2": ("tasksource/mcscript2", mcscript2),
            "sentfin": ("tasksource/sentfin", sentfin),
            "acord": ("tasksource/acord", acord),
            "edos": ("tasksource/edos", edos),
            "evidence_inference": ("tasksource/evidence-inference", evidence_inference),
            "wands": ("tasksource/wands", wands),
            "scirepeval_search": ("tasksource/scirepeval-search", scirepeval_search),
            "mind2web_dom": ("tasksource/mind2web-dom", mind2web_dom, "per-config"),
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
LICENSES = {"tasksource/tron-mcq": "other", "tasksource/trace-mcq": "cc-by-4.0",
            "tasksource/mentis-oculi-paper-fold": "apache-2.0",
            "tasksource/memeqa": "other", "tasksource/gameqa": "mit", "tasksource/valen-game": "apache-2.0",
            "tasksource/uk-election": "cc-by-4.0",
            "tasksource/t2i-safety": "apache-2.0",
            "tasksource/discosense": "apache-2.0",
            "tasksource/antique": "other", "tasksource/semcor": "other", "tasksource/streusle": "cc-by-sa-4.0",
            "tasksource/spider": "cc-by-sa-4.0", "tasksource/slurp": "cc-by-4.0", "tasksource/mcscript2": "other",
            "tasksource/pubmed-rct": "other",
            "tasksource/vast": "other", "tasksource/fnc1": "other", "tasksource/perspectrum": "cc-by-sa-4.0",
            "tasksource/ukpconvarg": "cc-by-4.0", "tasksource/discogem": "other",
            "tasksource/sentfin": "mit", "tasksource/wands": "mit",
            "tasksource/acord": "cc-by-4.0", "tasksource/edos": "cc0-1.0",
            "tasksource/evidence-inference": "other",
            "tasksource/scirepeval-search": "other",
            "tasksource/mind2web-dom": "cc-by-4.0",
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


def push_card(repo, build, directory=None):
    card = DatasetCard.load(repo)
    card.data.source_datasets = ORIGINALS[repo]
    if repo in LICENSES:
        card.data.license = LICENSES[repo]
    sources = ", ".join(f"[{name}](https://huggingface.co/datasets/{name})" for name in ORIGINALS[repo])
    directory = directory or Path('build') / ('view2space-release-imagefolder' if repo == 'tasksource/view2space' else repo.split('/')[1] + '-release')
    provenance = directory / 'provenance.json'
    if provenance.exists():
        report = json.loads(provenance.read_text())
        if report.get('annotation_origin') == 'procedural':
            card.data.language = ['en']
            card.data.task_categories = ['multiple-choice', 'visual-question-answering']
            card.data.tags = sorted(set(getattr(card.data,'tags',None) or []) | {'decision-model', 'procedural'})
            if repo == 'tasksource/tron-mcq':
                card.data.license_name = 'tron-research-use'
                card.data.license_link = report['license_url']
        upstream = report.get('source_url') or report.get('source', '')
        if isinstance(upstream, str) and upstream.startswith(('https://', 'http://')):
            sources = ', '.join(filter(None, (sources, f'[original distribution]({upstream})')))
        if report.get('verifier_source'):
            sources += f", [upstream generator and verifier]({report['verifier_source']})"
    sources = f"Original data: {sources}. " if sources else ""  # an empty entry: the original is not on the Hub
    script = Path(inspect.getsourcefile(build)).resolve().relative_to(Path(__file__).resolve().parents[1]).as_posix()
    storage = 'native ImageFolder' if repo == 'tasksource/view2space' else 'parquet'
    card.text = (f"\n# {repo.split('/')[1]}\n\n{inspect.cleandoc(build.__doc__)}\n\n{sources}"
                 f"Repackaged as {storage} for [tasksource](https://github.com/sileod/tasksource) by "
                 f"[{script}](https://github.com/sileod/tasksource/blob/main/{script}#L{inspect.getsourcelines(build)[1]}).\n")
    if provenance.exists():
        card.text += '\n[Conversion provenance](provenance.json): source revisions, hashes, configuration and exclusions.\n'
    if (directory / 'asset-notices.json').exists():
        card.text += '\n[Upstream asset credits and licenses](asset-notices.json) accompany the original rendered images.\n'
    if (directory / 'generation-checkpoints.json').exists():
        card.text += '\n[Generation checkpoints](generation-checkpoints.json) record each chunk’s seeds and exclusions. Reproduce with the [resumable scale script](https://github.com/sileod/tasksource/blob/main/scripts/repackage_dataset/procedural/scale.py).\n'
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
    if provenance.exists() and repo != 'tasksource/webinstruct':
        from huggingface_hub import HfApi
        for filename in ('provenance.json', 'excluded-questions.jsonl', 'asset-notices.json', 'generation-checkpoints.json'):
            if not (directory / filename).exists():continue
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

def load_prepared(directory, repo):
    """Publish reviewed split data without repeating expensive preparation."""
    report = json.loads((directory / 'provenance.json').read_text())
    if report.get('mirror') != repo:raise ValueError('Prepared build belongs to another mirror')
    hashes = report.get('parquet_sha256', {})
    if not hashes or {Path(n).stem for n in hashes} != set(report['splits']):
        raise ValueError('Prepared build requires hashes for every split')
    for name,digest in hashes.items():
        if Path(name).name != name or file_sha256(directory / name) != digest:
            raise ValueError(f'Prepared split checksum mismatch: {name}')
    dataset = DatasetDict({Path(name).stem:Dataset.from_parquet(str(directory / name)) for name in hashes})
    if any(len(rows) != report['splits'][split]['rows'] for split,rows in dataset.items()):
        raise ValueError('Prepared split row count mismatch')
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+", choices=BUILDERS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--card-only", action="store_true")
    parser.add_argument("--prepared", type=Path, help="Publish a reviewed, checksummed default-config build without regenerating it")
    parser.add_argument("--bad-examples", type=Path, help="WebInstruct only: confirmed-removal manifest")
    parser.add_argument("--repairs", type=Path, help="WebInstruct only: verified presentation repairs")
    parser.add_argument("--max-rows", type=int, help="Rendered mirrors: maximum training examples (default 5000)")
    parser.add_argument("--max-rows-eval", type=int, help="Rendered mirrors: maximum examples per native evaluation split (default 100)")
    parser.add_argument("--seed", type=int, help="Rendered mirrors: deterministic source annotation sampling seed")
    parser.add_argument("--generator-task", dest="task_ids", action="append", help="Procedural builders: restrict native task/environment IDs; repeat to select several")
    parser.add_argument("--workers", type=int, help="Fresh TRACE generation: parallel workers")
    args = parser.parse_args()
    options = {key: getattr(args, key) for key in ("max_rows", "max_rows_eval", "seed", "task_ids", "workers")
               if getattr(args, key) is not None}
    if args.prepared and (len(args.names)!=1 or options or args.card_only or args.bad_examples or args.repairs):
        parser.error('--prepared requires one builder and no generation/card-only options')
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
            if args.prepared and config:parser.error('--prepared supports default-config mirrors only')
            dataset = (load_prepared(args.prepared,repo) if args.prepared else
                       build(args.bad_examples,args.repairs) if name=='webinstruct' else build(**options))
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
        push_card(repo, build, args.prepared)


if __name__ == "__main__":
    main()
