"""Native MemeQA choices joined once to the original meme image archives."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
import zipfile

from .games import _package

MEMEQA = 'https://github.com/npnkhoi/memeqa'
MEMEQA_REVISION = 'e48b332a02f7b25e08459dd7dcf69f19b430c621'
IMAGE_SOURCE = 'https://github.com/di-dimitrov/SEMEVAL-2021-task6-corpus'
IMAGE_REVISION = '796b32fa76b7a5814a48d1870d163058e81e4ce1'


def memeqa_record(row, split):
    if row['general_type'] != 'single':return None, 'native_multiselect_outside_mc_view'
    choices, label = row['options'], row['answer_key']
    if not 2 <= len(choices) <= 26 or any(not c.strip() for c in choices):return None, 'invalid_choices'
    if len({c.strip().casefold() for c in choices}) != len(choices):return None, 'duplicate_choices'
    if type(label) != int or not 0 <= label < len(choices):raise ValueError('Invalid native MemeQA gold index')
    return dict(inputs=row['question'],choices_list=choices,labels=label,
        metadata=json.dumps(dict(id=row['id'],source_split=split,source_image=row['img'],
            question_type=row['specific_type'],source_dataset=MEMEQA,source_revision=MEMEQA_REVISION))), None


def memeqa():
    """MemeQA's native single-answer MCQs, preserving meme-level train/dev/test.

    Original annotations: https://github.com/npnkhoi/memeqa. Human interpretations,
    automatically constructed distractors and human answer verification (ACL 2025).
    Uses none_minus only; native multiselect QAs are omitted, never forced to one
    answer. Joins the original SemEval images without decoding/re-encoding them.
    Annotation repository: MIT; meme images retain their original rights, which
    are not covered by the annotation code license. No synthetic splits/options.
    """
    directory = Path('build/source-archives/memeqa');directory.mkdir(parents=True,exist_ok=True)
    files, groups, excluded = {}, {}, []
    def fetch(repo, revision, name):
        path = directory/revision/Path(name).name;path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            temporary=path.with_suffix('.tmp')
            with urlopen(f'{repo.replace("github.com", "raw.githubusercontent.com")}/{revision}/{name}',timeout=60) as response:
                temporary.write_bytes(response.read())
            temporary.replace(path)
        files[name]=str(path)
        return path
    images = [fetch(IMAGE_SOURCE,IMAGE_REVISION,'data/'+name+'_set_task3.zip') for name in ('training','dev','test')]
    archive = directory/IMAGE_REVISION/'images.zip'
    if not archive.exists():
        with zipfile.ZipFile(archive.with_suffix('.tmp'),'w') as output:
            for path in images:
                with zipfile.ZipFile(path) as source:
                    for name in source.namelist():
                        if name.endswith('.png'):output.writestr(Path(name).name,source.read(name))
        archive.with_suffix('.tmp').replace(archive)
    for split, source_split in [('train','train'),('validation','dev'),('test','test')]:
        path=fetch(MEMEQA,MEMEQA_REVISION,f'data/none_minus/{source_split}.json')
        groups[split]=defaultdict(list)
        for row in json.loads(path.read_text()):
            qa,reason=memeqa_record(row,source_split)
            if reason:excluded.append(dict(id=row['id'],split=split,reason=reason))
            else:groups[split][row['img']].append(qa)
    result=_package('memeqa',MEMEQA,MEMEQA_REVISION,groups,{s:str(archive) for s in groups},
        files,excluded,True,memeqa,'other',shuffle_seed=0,
        annotation_provenance='human interpretations and answer verification; automatically constructed distractors')
    path=Path('build/memeqa-release/provenance.json');report=json.loads(path.read_text())
    report.update(source_url=MEMEQA,license_url=MEMEQA+'/blob/'+MEMEQA_REVISION+'/LICENSE',
        image_source=IMAGE_SOURCE,image_revision=IMAGE_REVISION,
        license_scope='MIT annotation repository; original meme image rights remain separate',
        conversion_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path.write_text(json.dumps(report,indent=2)+'\n')
    return result
