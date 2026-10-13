"""Join native game annotations to archived screenshots; retain encoded images."""
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

from datasets import Features, Image, IterableDataset, IterableDatasetDict, Sequence, Value
from huggingface_hub import hf_hub_download
from PIL import Image as PILImage

from .vision import _materialize

GAMEQA = 'OpenMOSS-Team/GameQA-140K'
GAMEQA_REVISION = '47fb49645e94a370d5b68a45d986b0a951bbd783'
VALEN = 'Valen-Team/Valen-Eval-Game'
VALEN_REVISION = '4f25038cd30d9a96aa377eea176a4d2e1680a35c'
_OPTION_HEADER = re.compile(r'\*{0,2}(?:Options|Optoins)\s*:\*{0,2}|The options are as follows:|Answer using one of the following options with its corresponding letter:', re.I)


def gameqa_record(row, index):
    options = row.get('options') or []
    if not 2 <= len(options) <= 26:return None, 'no_native_multiple_choice'
    prefixes = [re.match(r'^([A-Z])[.:]\s*', str(o)) for o in options]
    if any(prefixes) and [m[1] if m else None for m in prefixes]!=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(options)]):
        return None, 'misaligned_source_options'
    choices = [re.sub(r'^[A-Z][.:]\s*', '', str(o)).strip() for o in options]
    answer = str(row['answer']).strip()
    marker = _OPTION_HEADER.search(row['question'])
    if re.fullmatch('[A-Z]', answer):label = ord(answer)-ord('A')
    elif answer.isdigit() and marker:label = int(answer)-1
    elif choices.count(answer)==1:label = choices.index(answer)
    else:return None, 'unresolved_answer'
    if not 0 <= label < len(choices):return None, 'invalid_answer_index'
    if not all(choices):return None, 'empty_option'
    # Equivalent duplicate options collapse without changing the gold answer's meaning.
    unique = list(dict.fromkeys(choices))
    if len(unique)<2:return None, 'fewer_than_two_unique_options'
    question = row['question'][:marker.start()].strip() if marker else row['question'].strip()
    for suffix in ('Give the number of the correct option.', 'Choose the correct option and give the analysis.'):
        question = question.removesuffix(suffix).strip()
    if not marker and re.search(r'(?m)^\s*[A-Z][.:]\s', question):return None, 'unparsed_option_block'
    return dict(inputs=question,choices_list=unique,labels=unique.index(choices[label]),
        metadata=json.dumps(dict(id=str(index),source_dataset=GAMEQA,source_revision=GAMEQA_REVISION,
            source_id=row['data_id'],game=row['game_name'],question_id=row['question_id'],
            source_question=row['question'],source_options=options,source_answer=row['answer']),ensure_ascii=False)), None


def valen_records(row):
    content = row['request']['state']['messages']
    assert len(content)==1 and content[0]['role']=='user'
    parts = content[0]['content']
    assert [p['type'] for p in parts]==['text','image_url']
    assert parts[1]['image_url']['url']==row['assets'][0]['path']
    question = row['request']['questions']['next_action']
    probabilities = row['targets']['next_action']['probabilities']
    assert question['type']=='choice' and set(question['criteria'])==set(probabilities)=={'up','down','left','right'}
    assert all(0<=p<=1 for p in probabilities.values()) and abs(sum(probabilities.values())-1)<1e-8
    for action in question['criteria']:
        yield dict(inputs=parts[0]['text']+f'\n\nIs moving {action} an optimal next action, minimizing total moves to finish?',
            labels='yes' if probabilities[action]>0 else 'no',
            metadata=json.dumps(dict(id=row['meta']['record_id']+':'+action,source_dataset=VALEN,
                source_revision=VALEN_REVISION,split_group_id=row['group_id'],source_row=row['meta']['record_id'],
                action=action,source_probabilities=probabilities,source_metadata=row['meta']),ensure_ascii=False))


def _package(name, source, revision, groups, archives, files, excluded, mc, conversion, license, shuffle_seed=None,
             annotation_provenance='procedural game/solver labels; source rationales and hidden states are never model inputs'):
    features = Features(dict(images=Sequence(Image(decode=False)),image_group_id=Value('string'),
        qa=[dict(inputs=Value('string'),labels=Value('int64' if mc else 'string'),metadata=Value('string'),
                 **({'choices_list':Sequence(Value('string'))} if mc else {}))]))
    heldout_images = set()
    heldout_groups = {json.loads(qa['metadata']).get('split_group_id') for qas in groups['test'].values() for qa in qas}-{None}
    def generate(split):
        with zipfile.ZipFile(archives[split]) as archive:
            for path,qas in groups[split].items():
                image = archive.read(path)
                digest = hashlib.sha256(image).hexdigest()
                family = json.loads(qas[0]['metadata']).get('split_group_id')
                if split=='train' and (digest in heldout_images or family in heldout_groups):
                    excluded.extend(dict(id=json.loads(q['metadata'])['id'],split=split,reason='heldout_image_or_layout') for q in qas)
                    continue
                with PILImage.open(io.BytesIO(image)) as decoded:decoded.verify()
                if split=='test':heldout_images.add(digest)
                for qa in qas:
                    metadata = json.loads(qa['metadata']);metadata['image_sha256']=digest
                    qa['metadata']=json.dumps(metadata,ensure_ascii=False)
                yield dict(images=[{'bytes':image,'path':None}],image_group_id=digest,qa=qas)
    # Read holdouts first so shared screenshots and canonical layouts cannot enter training.
    data = IterableDatasetDict({s:IterableDataset.from_generator(generate,gen_kwargs={'split':s},features=features)
                                for s in ('test','train')})
    result = _materialize(data,name,source,revision,excluded,conversion,license,
                          f'https://huggingface.co/datasets/{source}')
    directory = Path('build')/(name+'-release')
    if shuffle_seed is not None:
        result = result.shuffle(seed=shuffle_seed)
        for split, rows in result.items():rows.to_parquet(str(directory/(split+'.parquet')))
    report = json.loads((directory/'provenance.json').read_text())
    if shuffle_seed is not None:
        report['ordering'] = dict(method='datasets.shuffle', seed=shuffle_seed, unit='image_group', within='native_split')
    report.update(conversion_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_file_sha256={n:hashlib.file_digest(open(p,'rb'),'sha256').hexdigest() for n,p in files.items()},
        exclusions_by_reason=dict(Counter(r['reason'] for r in excluded)),
        annotation_provenance=annotation_provenance)
    (directory/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    return result


def gameqa():
    """GameQA native multiple-choice questions, with original screenshot bytes.

    Full native train/test JSON and image archives are joined once. Open-ended
    questions are omitted; no distractors are generated. Options retain source
    order, equivalent duplicates collapse, and gold indices are remapped.
    Embedded option blocks are removed before Jev permutation. Screenshots are
    grouped across QAs and test images take precedence over training copies.
    Image groups are shuffled within each native split with seed 0; QA order is retained.
    Procedural game labels are used; generated rationales are excluded from inputs.
    Original source declares MIT. No validation split is fabricated.
    """
    files = {n:hf_hub_download(GAMEQA,n,repo_type='dataset',revision=GAMEQA_REVISION)
             for n in ('games_data.json','games_data_test.json','games_images.zip','games_images_test.zip')}
    groups = {};excluded = []
    for split,suffix in [('test','_test'),('train','')]:
        groups[split] = defaultdict(list)
        for index,row in enumerate(json.loads(Path(files[f'games_data{suffix}.json']).read_text())):
            qa,reason = gameqa_record(row,index)
            if reason:excluded.append(dict(id=str(index),split=split,reason=reason))
            else:groups[split][row['image']].append(qa)
    return _package('gameqa',GAMEQA,GAMEQA_REVISION,groups,
        {s:files[f'games_images{suffix}.zip'] for s,suffix in [('train',''),('test','_test')]},files,excluded,True,gameqa,'mit',shuffle_seed=0)


def valen_game():
    """Valen Sokoban: native optimal-action sets as action-conditioned yes/no decisions.

    All four native directions are supervised, including every equally optimal
    action. Original English/Chinese rules, train/test layout partitions, image
    bytes, native probabilities and solver provenance are retained. Hidden states
    and solutions stay in metadata. Four QAs share one screenshot. Evaluation is
    the source's selected single-step subset, not an unbiased full-game benchmark.
    Original source declares Apache 2.0. No validation split is fabricated.
    """
    names = ('train_sokoban/train.jsonl','train_sokoban/assets.zip',
             'eval_sokoban/single_step.jsonl','eval_sokoban/single_step_assets.zip')
    files = {n:hf_hub_download(VALEN,n,repo_type='dataset',revision=VALEN_REVISION) for n in names}
    groups = {};excluded = []
    for split,name in [('train',names[0]),('test',names[2])]:
        groups[split] = {}
        for line in Path(files[name]).read_text().splitlines():
            row = json.loads(line);path = row['assets'][0]['path']
            assert path not in groups[split], 'Duplicate native Valen screenshot'
            groups[split][path] = list(valen_records(row))
    # Verify the publisher's per-image hashes without decoding/re-encoding pixels.
    for name,archive_name in [(names[0],names[1]),(names[2],names[3])]:
        with zipfile.ZipFile(files[archive_name]) as archive:
            for line in Path(files[name]).read_text().splitlines():
                asset = json.loads(line)['assets'][0]
                assert hashlib.sha256(archive.read(asset['path'])).hexdigest()==asset['sha256']
    return _package('valen-game',VALEN,VALEN_REVISION,groups,
        {'train':files[names[1]],'test':files[names[3]]},files,excluded,False,valen_game,'apache-2.0')
