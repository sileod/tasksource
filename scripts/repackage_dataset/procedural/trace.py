"""Make verifier-checked MC views of TRACE's existing native static release."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

from huggingface_hub import snapshot_download
import pyarrow.parquet as pq

from .common import checkout, seed_for, verified_choices, write_static

SOURCE = 'maveryn/trace'
REVISION = '4f01a535f489cb1db71ca43d10cdbc38a1746b5c'
GENERATOR_REVISION = 'd3440e95d8e9f39b0fe01731065bd166ff9bcd65'


def trace_question(prompt, scene_id=None):
    """Keep the question and answer semantics, remove JSON demonstrations/contracts."""
    lines = prompt.removeprefix('<image>').strip().splitlines()
    contract = re.compile(r'^(?:Answer (?:field|format)|(?:Required|Final|Use this) answer format|Format for the "answer" field):\s*(?:set\s+"?answer"?\s+to\s+)?(.+)$',re.I)
    start = next((i for i,line in enumerate(lines) if line.startswith(('Return a JSON object','Return JSON')) or contract.match(line)),None)
    if start is None:raise ValueError('Unknown TRACE output contract')
    question = '\n'.join(lines[:start])
    semantics = next((contract.match(line) for line in lines[start:] if contract.match(line)),None)
    if semantics:
        question += '\nChoose '+semantics[1]
    if scene_id=='toggle_grid':
        question = 'Pressing a switch flips that cell and its orthogonal neighbors (up, down, left, right).\n'+question
    if not question.strip():raise ValueError('Empty TRACE question')
    return question.strip()


def answer_text(answer):
    value = answer['value']
    return value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)


def asset_notices(code):
    """Retain upstream credits and licenses, including per-icon attribution."""
    paths = [code/'THIRD_PARTY_NOTICES.md',code/'LICENSE',
             code/'src/trace_tasks/resources/assets/icons/icon-sources.json']
    paths += [p for p in (code/'src/trace_tasks/resources/assets').rglob('*') if p.is_file()
              and ('licenses' in p.parts or p.name.upper().startswith(('LICENSE','OFL','NOTICE')))]
    return {str(p.relative_to(code)):p.read_text() for p in paths}


def trace_mc(max_rows=5000, max_rows_eval=100, seed=42, task_ids=()):
    """TRACE static MC reuses original rendered images and native train/validation rows.

    Source: https://huggingface.co/datasets/maveryn/trace. Distractors are answers
    from other native training instances of the same task and answer type, rejected
    by TRACE's exact-JSON answer scorer for the current example. No annotation targets,
    JSON demonstrations or execution traces enter model inputs. Native labels and image
    bytes are preserved; no new TRACE generation or model-derived labels are needed.
    Dataset: CC BY 4.0; verifier code: Apache-2.0; original asset notices remain applicable.
    """
    code, code_sha = checkout('maveryn/trace',GENERATOR_REVISION)
    sys.path.insert(0,str(code/'src'))
    from trace_tasks import score_trace_response
    import trace_tasks
    if not str(trace_tasks.__file__).startswith(str(code)):raise ValueError('A different TRACE verifier is loaded')
    root = Path(snapshot_download(SOURCE,repo_type='dataset',revision=REVISION,cache_dir='/tmp/tasksource-hf',
        allow_patterns=['data/train/*.parquet','data/validation/*.parquet']))
    files = {split:sorted((root/'data'/split).glob('*.parquet')) for split in ('train','validation')}
    pools, selected, eligible = defaultdict(list), {},{}
    for split,paths in files.items():
        index = []
        for path in paths:
            columns = ['instance_id','task','answer_gt']
            offset = 0
            for batch in pq.ParquetFile(path).iter_batches(columns=columns,batch_size=512):
                for i,row in enumerate(batch.to_pylist()):
                    if task_ids and row['task'] not in task_ids:continue
                    answer = json.loads(row['answer_gt'])
                    if split=='train':pools[(row['task'],answer['type'])].append((answer_text(answer),row['instance_id']))
                    index.append((seed_for(seed,row['instance_id']),path.name,offset+i))
                offset += batch.num_rows
        eligible[split] = len(index)
        limit = max_rows if split=='train' else max_rows_eval
        selected[split] = {(name,i) for _,name,i in sorted(index)[:limit]}
    splits,excluded,heldout_images = {},[],set()
    for split in ('validation','train'):
        rows = [];splits[split] = rows
        for path in files[split]:
            offset = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
                for i,row in enumerate(batch.to_pylist()):
                    images = row['images']
                    if len(images)!=1:raise ValueError('TRACE expects one original image')
                    image = images[0]['bytes'];digest = hashlib.sha256(image).hexdigest()
                    if split=='validation':heldout_images.add(digest)
                    if (path.name,offset+i) not in selected[split]:continue
                    try:
                        if split=='train' and digest in heldout_images:raise ValueError('Image belongs to native validation')
                        answer = json.loads(row['answer_gt'])
                        def verify(candidate):
                            return score_trace_response(response=json.dumps(dict(answer=candidate)),answer_gt=answer,
                                annotation_gt=json.loads(row['annotation_gt']),reward_contract=json.loads(row['reward_contract']),
                                trace_reward_mode='answer',trace_answer_scoring='exact_json')['accuracy']
                        choices,label,alignment = verified_choices(answer_text(answer),pools[(row['task'],answer['type'])],
                            verify,seed_for(seed,row['instance_id'],'choices'))
                        raw_choices = choices
                        choices = [f'Label {c}' if len(c)==1 and c.isalpha() else c for c in choices]
                        if len(set(choices))!=len(choices):raise ValueError('Duplicate displayed choices')
                        rows.append(dict(id=row['instance_id'],image=image,inputs=trace_question(row['prompt_answer'],row['scene_id']),
                            choices_list=choices,labels=label,metadata=dict(task=row['task'],domain=row['domain'],
                                scene_id=row['scene_id'],query_id=row['query_id'],source_split=split,source_revision=REVISION,
                                verifier_revision=GENERATOR_REVISION,answer_gt=answer,original_question=row['prompt_answer'],
                                trace_ref=row['trace_ref'],verifier_answers=raw_choices,
                                choice_method='same_task_training_answers',**alignment)))
                    except (ValueError,KeyError,TypeError) as error:
                        excluded.append(dict(id=row['instance_id'],split=split,reason=str(error)))
                offset += batch.num_rows
    report = dict(source='https://huggingface.co/datasets/'+SOURCE,revision=REVISION,
        verifier_source='https://github.com/maveryn/trace',verifier_revision=GENERATOR_REVISION,verifier_archive_sha256=code_sha,
        license='cc-by-4.0',license_url='https://huggingface.co/datasets/'+SOURCE+'/blob/'+REVISION+'/LICENSE',
        configuration=dict(seed=seed,max_rows=max_rows,max_rows_eval=max_rows_eval,task_ids=list(task_ids)),
        split_policy='Native train/validation; all native validation image hashes excluded from training.',
        annotation_origin='procedural',eligible=eligible,conversion_exclusions=excluded,
        context_repairs={'toggle_grid':f'https://github.com/maveryn/trace/blob/{GENERATOR_REVISION}/src/trace_tasks/tasks/puzzles/toggle_grid/shared/rules.py#L20'},
        exclusion_counts=dict(Counter(r['reason'] for r in excluded)),native_validation_images=len(heldout_images))
    result = write_static('trace-mcq',splits,report,trace_mc)
    path = Path('build/trace-mcq-release/asset-notices.json')
    path.write_text(json.dumps(asset_notices(code),sort_keys=True,ensure_ascii=False)+'\n')
    return result
