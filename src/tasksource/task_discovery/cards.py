"""Mine implemented datasets' cached Hub cards; no corpus downloads or model calls."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import re

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from .identifiers import repository_links


def card_links(text):
    github=repository_links(text)['github']
    # Infrastructure links do not identify the dataset.
    github=[u for u in github if u.casefold().rstrip('/') not in {
        'https://github.com/huggingface/datasets','https://github.com/pytorch/text',
        'https://github.com/tensorflow/datasets','https://github.com/sileod/tasksource',
        'https://github.com/google-research/google-research'}]
    papers=set(re.findall(r"""https?://(?:www\.)?(?:arxiv\.org/(?:abs|pdf)/|(?:dx\.)?doi\.org/|semanticscholar\.org/paper/|aclanthology\.org/)[^\s<>"'{}\[\]()`\\]+""",text))
    papers.update('arxiv:'+v for v in re.findall(r'(?:arxiv\s*:\s*|eprint\s*=\s*[{\"])(\d{4}\.\d{4,5}(?:v\d+)?)',text,re.I))
    return dict(github=github,papers=[dict(url=u.rstrip('.,);]')) for u in sorted(papers)])


def mine_cards(repos, limit=None, offline=False):
    def read(repo):
        try:
            path=Path(hf_hub_download(repo,'README.md',repo_type='dataset',local_files_only=offline))
            raw=path.read_bytes();revision=path.parent.name
            return dict(name=repo,tasks=sorted(repos[repo]),**card_links(raw.decode('utf-8')),
                sources=[dict(source='dataset_card',url=f'https://huggingface.co/datasets/{repo}/blob/{revision}/README.md')],
                revision=revision,sha256=hashlib.sha256(raw).hexdigest())
        except (HfHubHTTPError,OSError,ValueError) as error:
            return dict(name=repo,tasks=sorted(repos[repo]),error=str(error))
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(read,sorted(repos)[:limit]))
