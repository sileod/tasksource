"""Retain TRACE's native release and add fresh verifier-checked training instances."""
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import io
import json
import multiprocessing
import os
from pathlib import Path
import sys

from huggingface_hub import hf_hub_download,snapshot_download
import pyarrow.parquet as pq
from PIL import Image

from .common import checkout,file_sha256,seed_for,verified_choices,write_static
from .trace import SOURCE,REVISION,GENERATOR_REVISION,answer_text,asset_notices,trace_question

BASE_REVISION = 'd3a6da49e1ec3e1013f83e372124e61745564198'


def pixel_hash(encoded):
    with Image.open(io.BytesIO(encoded)) as image:
        return hashlib.sha256(str(image.size).encode()+image.convert('RGBA').tobytes()).hexdigest()


def initialize_worker(code,pools,cache):
    global _context
    sys.path.insert(0,str(Path(code)/'src'))
    from trace_tasks import generate_task,score_trace_response
    _context = pools,Path(cache),generate_task,score_trace_response


def generate_one(job):
    try:return generated_row(*job,*_context),None
    except Exception as error:return None,f'{type(error).__name__}: {error}'


def generated_row(task,index,seed,pools,cache,generate,score):
    from trace_tasks.core.reward_contracts import resolve_reward_contract
    from trace_tasks.core.rlvr_export import _build_prompt_columns
    instance_seed = seed_for('tasksource-trace-extra-train',seed,task,index)
    identity = f'trace-generated:{task}:{instance_seed}'
    key = hashlib.sha256(identity.encode()).hexdigest()
    path = cache/(key+'.json');image_path = cache/(key+'.png')
    if path.exists():
        row = json.loads(path.read_text());image = image_path.read_bytes()
        if hashlib.sha256(image).hexdigest()!=row.pop('image_sha256'):raise ValueError('Checkpoint image checksum mismatch')
        return dict(image=image,**row)
    output = generate(task,seed=instance_seed)
    answer,annotation = output.answer_gt.to_dict(),output.annotation_gt.to_dict()
    contract = resolve_reward_contract(answer_type=answer['type'],annotation_type=annotation['type']).to_dict()
    def verify(candidate):
        return score(response=json.dumps(dict(answer=candidate)),answer_gt=answer,annotation_gt=annotation,
            reward_contract=contract,trace_reward_mode='answer',trace_answer_scoring='exact_json')['accuracy']
    raw,label,alignment = verified_choices(answer_text(answer),pools[(task,answer['type'])],verify,instance_seed)
    choices = [f'Label {c}' if len(c)==1 and c.isalpha() else c for c in raw]
    if len(set(choices))!=len(choices):raise ValueError('Duplicate displayed choices')
    prompt = _build_prompt_columns(dict(prompt=output.prompt,prompt_variants=output.prompt_variants),image_count=1)['prompt_answer']
    image = io.BytesIO();output.image.save(image,format='PNG');image = image.getvalue()
    row = dict(id=identity,inputs=trace_question(prompt,output.scene_id),
        choices_list=choices,labels=label,
        metadata=dict(task=task,seed=instance_seed,generated=True,source_split='generated_train',
            scene_id=output.scene_id,query_id=output.query_id,task_versions=output.task_versions,
            image_size=list(output.image.size),
            answer_gt=answer,original_question=prompt,source_revision=GENERATOR_REVISION,
            verifier_answers=raw,choice_method='same_task_native_training_answers',**alignment))
    image_path.write_bytes(image)
    temporary = path.with_suffix('.tmp');temporary.write_text(json.dumps(dict(image_sha256=hashlib.sha256(image).hexdigest(),**row)))
    temporary.replace(path)
    return dict(image=image,**row)


def trace_expanded_mc(max_rows=100000,max_rows_eval=2000,seed=42,workers=8):
    """TRACE MC retains native training/validation QAs and adds fresh training generations.

    Original data: https://huggingface.co/datasets/maveryn/trace. New instances use
    the pinned TRACE generator and independent seeds; native training answers supply
    same-task distractors checked with the original scorer. The native validation
    split is preserved, and image/question/gold duplicates and held-out image copies
    are excluded from the fresh training additions. Seeds and rendering versions are
    recorded; execution traces and gold targets stay out of model inputs. Dataset
    terms: CC BY 4.0, with the original asset credits and licenses retained.
    """
    code,code_sha = checkout('maveryn/trace',GENERATOR_REVISION)
    sys.path.insert(0,str(code/'src'))
    from trace_tasks import generate_task,score_trace_response
    import trace_tasks
    if not str(trace_tasks.__file__).startswith(str(code)):raise ValueError('A different TRACE package is loaded')
    local = os.environ.get('TASKSOURCE_TRACE_BASE')
    native_provenance = json.loads(Path(hf_hub_download('tasksource/trace-mcq','provenance.json',
        repo_type='dataset',revision=BASE_REVISION)).read_text())
    if local:
        base = Path(local)
        from scripts.upload_repackaged import load_prepared
        load_prepared(base,'tasksource/trace-mcq')
        if json.loads((base/'provenance.json').read_text())['parquet_sha256']!=native_provenance['parquet_sha256']:
            raise ValueError('Local TRACE base differs from the pinned native release')
        paths = {s:[base/(s+'.parquet')] for s in ('validation','train')}
    else:
        base = Path(snapshot_download('tasksource/trace-mcq',repo_type='dataset',revision=BASE_REVISION,
                                     cache_dir=os.environ.get('TASKSOURCE_DATA_CACHE'),allow_patterns=['data/*.parquet']))
        paths = {s:sorted((base/'data').glob(s+'-*.parquet')) for s in ('validation','train')}
    rows = {'validation':[],'train':[]};pools = defaultdict(list);heldout = set();known = set();excluded = Counter()
    def fingerprint(row):
        digest = pixel_hash(row['image'])
        return digest,hashlib.sha256(json.dumps([digest,row['inputs'],row['choices_list'][row['labels']]]).encode()).hexdigest()
    for split,files in paths.items():
        limit = max_rows if split=='train' else max_rows_eval
        for file in files:
            for batch in pq.ParquetFile(file).iter_batches(batch_size=32):
                for group in batch.to_pylist():
                    if split=='validation':heldout.add(pixel_hash(group['images'][0]['bytes']))
                    for qa in group['qa']:
                        metadata = json.loads(qa['metadata']);identity = metadata.pop('id')
                        if split=='train':
                            answer = metadata['answer_gt'];pools[(metadata['task'],answer['type'])].append((answer_text(answer),identity))
                        if len(rows[split])>=limit:continue
                        row = dict(image=group['images'][0]['bytes'],id=identity,metadata=metadata,
                                   **{k:qa[k] for k in ('inputs','choices_list','labels')})
                        row['inputs'] = trace_question(metadata['original_question'],metadata['scene_id'])
                        digest,key = fingerprint(row)
                        if split=='train' and digest in heldout:excluded['native_image_in_validation']+=1;continue
                        if key in known:excluded['duplicate_native_question']+=1;continue
                        rows[split].append(row);known.add(key)
    native_counts = {s:len(r) for s,r in rows.items()}
    tasks = sorted({task for task,kind in pools});needed = max_rows-len(rows['train'])
    versions = {p:importlib.metadata.version(p) for p in ('Pillow','CairoSVG','numpy','scipy','networkx')}
    versions['python'] = sys.version.split()[0]
    recipe = hashlib.sha256((file_sha256(__file__)+file_sha256(Path(__file__).with_name('trace.py'))+
        BASE_REVISION+str(seed)+json.dumps(versions,sort_keys=True)).encode()).hexdigest()
    cache = Path(os.environ.get('TASKSOURCE_GENERATOR_CACHE','/tmp'))/'tasksource-trace-generated'/recipe
    cache.mkdir(parents=True,exist_ok=True)
    if workers<1:raise ValueError('Workers must be positive')
    with ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize_worker,initargs=(code,pools,cache)) as executor:
        for start in range(0,2*needed,workers*8):
            if len(rows['train'])>=max_rows:break
            jobs = [(tasks[i%len(tasks)],i,seed) for i in range(start,min(start+workers*8,2*needed))]
            for row,error in executor.map(generate_one,jobs):
                if len(rows['train'])>=max_rows:break
                if error:excluded[error]+=1;continue
                digest,key = fingerprint(row)
                if digest in heldout:excluded['image_in_native_validation']+=1;continue
                if key in known:excluded['duplicate_question']+=1;continue
                rows['train'].append(row);known.add(key)
            print('TRACE fresh',len(rows['train'])-native_counts['train'],'target',needed,flush=True)
    if len(rows['train'])<max_rows:raise ValueError('Fresh TRACE candidate budget exhausted; checkpoints retained')
    report = dict(source='https://huggingface.co/datasets/'+SOURCE,revision=REVISION,
        verifier_source='https://github.com/maveryn/trace',verifier_revision=GENERATOR_REVISION,verifier_archive_sha256=code_sha,
        license='cc-by-4.0',license_url='https://huggingface.co/datasets/'+SOURCE+'/blob/'+REVISION+'/LICENSE',
        annotation_origin='procedural',native_counts=native_counts,generated_train=len(rows['train'])-native_counts['train'],
        native_base_parquet_sha256=native_provenance['parquet_sha256'],
        configuration=dict(seed=seed,max_rows=max_rows,max_rows_eval=max_rows_eval,base_revision=BASE_REVISION,
                           fresh_seed_namespace='tasksource-trace-extra-train',recipe_sha256=recipe),
        split_policy='Native partitions retained; fresh training excludes all native validation images and duplicate questions.',
        fresh_rendering_versions=versions,
        exclusion_counts=dict(excluded))
    result = write_static('trace-mcq',rows,report,trace_expanded_mc)
    Path('build/trace-mcq-release/asset-notices.json').write_text(json.dumps(asset_notices(code),sort_keys=True,ensure_ascii=False)+'\n')
    return result
