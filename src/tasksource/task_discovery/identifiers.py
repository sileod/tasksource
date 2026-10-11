"""Canonical source identities shared by discovery, linking and review history."""
import re


def normalized(name):
    return re.sub(r'[^\w]', '', name.casefold()).replace('_', '')


def source_id(value, kind):
    value = value.casefold().removeprefix('https://').removeprefix('http://').rstrip('/')
    if kind == 'huggingface':
        return value.removeprefix('huggingface.co/datasets/').split('/tree/')[0]
    if kind == 'github':
        return '/'.join(value.removeprefix('github.com/').split('/')[:2]).removesuffix('.git')
    if 'semanticscholar.org/paper/' in value:
        return 's2:'+value.rsplit('/', 1)[-1]
    value = value.removeprefix('doi.org/').removeprefix('dx.doi.org/') if value.startswith(('doi.org/', 'dx.doi.org/')) else value
    if re.match(r'^10\.\d+/', value):
        value = 'doi:'+value
    value = re.sub(r'^(?:www\.)?arxiv\.org/(?:abs|pdf)/|^arxiv:', 'arxiv:', value)
    return re.sub(r'(arxiv:[\d.]+)v\d+(?:\.pdf)?$', r'\1', value).removesuffix('.pdf')


def paper_ids(papers):
    values = []
    for paper in papers:
        if isinstance(paper, str):
            values.append(paper)
        else:
            values.append(paper.get('url', ''))
            if paper.get('semantic_scholar_id'):
                values.append('s2:'+paper['semantic_scholar_id'])
            for kind, value in paper.get('external_ids', {}).items():
                if kind in ('ArXiv', 'DOI'):
                    values.append(kind+':'+value)
    return {source_id(v, 'papers') for v in values if v}
