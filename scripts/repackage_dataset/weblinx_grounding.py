"""Join native WebLINX element decisions to their pre-action screenshots."""
import gzip
import hashlib
import io
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from huggingface_hub.errors import RemoteEntryNotFoundError
from tasksource.tasks import weblinx_record
from .grounding import grounding_qa, hub_file, materialize, ordered, xywh_box

CHAT = ('McGill-NLP/WebLINX', 'a30ff2cbecb75f2d04fac75a2b92721c6c9e3f13')
RAW = ('McGill-NLP/WebLINX-full', '36ef9f79b43df50e25b7f3b68e5c9f6ccf4160e8')
BOX = re.compile(r'\[\[bbox\]\] x=([-+.\deE]+) y=([-+.\deE]+) width=([-+.\deE]+) height=([-+.\deE]+)')
SPLITS = {'train':'train', 'validation':'valid', 'test':'test_iid',
          **{key:key for key in ('test_cat','test_geo','test_vis','test_web')}}


def native_query(row, excluded):
    """Reuse native UID supervision, retaining only pre-action context and CSS geometry."""
    record, reason = weblinx_record({**row, 'clean_html':''}, elements=True)
    key = f"weblinx:{row['demo']}:{row['turn']}"
    if reason:
        excluded.append(dict(source_row=key,reason=reason));return
    info = json.loads(record['metadata'])
    boxes = {}
    for uid, text in zip(info['candidate_ids'], record['choices_list']):
        match = BOX.search(text)
        if match:boxes[uid] = list(map(float, match.groups()))
    viewport = re.fullmatch(r'(\d+)h x (\d+)w', row['viewport'])
    if viewport is None or min(map(int,viewport.groups()))<=0 or info['target_id'] not in boxes:
        excluded.append(dict(source_row=key,reason='missing_native_geometry'));return
    fields = [('User dialogue',row['utterances']),('Previous actions',row['action_history'])]
    inputs = '\n'.join(label+':\n'+value.replace('</s><s>[INST]', '').strip() for label,value in fields if value and
                       value.strip() and value!='N o   i n s t r u c t o r   u t t e r a n c e ;')
    inputs += '\nOperation: ' + row['action'].partition('(')[0]
    return dict(demo=row['demo'],turn=int(row['turn']),inputs=inputs,boxes=boxes,
                target_id=info['target_id'],viewport=[int(viewport[2]),int(viewport[1])],source_row=key)


def visual_query(query, turn, image_size, screenshot, excluded):
    """Verify CSS/screenshot scaling against the recorded target's pixel geometry."""
    def reject(reason):excluded.append(dict(source_row=query['source_row'],reason=reason))
    width,height = image_size;vw,vh = query['viewport'];scale=width/vw
    if [turn.viewport_width,turn.viewport_height]!=[vw,vh] or abs(height-vh*scale)>2:
        reject('screenshot_viewport_mismatch');return
    boxes = {}
    for uid,box in query['boxes'].items():
        try:boxes[uid]=xywh_box([value*scale for value in box],width,height)
        except ValueError:continue
    target=boxes.get(query['target_id'])
    original=(turn.element or {}).get('bbox')
    if target is None or not original or not {'left','right','top','bottom'}<=original.keys():
        reject('invisible_or_unverified_target');return
    center=[(original['left']+original['right'])/2/width,(original['top']+original['bottom'])/2/height]
    if not all(target[axis]-2/image_size[axis]<=center[axis]<=target[axis+2]+2/image_size[axis] for axis in (0,1)):
        reject('target_coordinate_frame_mismatch');return
    metadata=dict(source_row=query['source_row'],trajectory_id=query['demo'],turn=query['turn'],
        image_group_id=f"weblinx:{query['demo']}:{screenshot}",image_size=image_size,viewport_css=[vw,vh],
        css_to_image_scale=scale,target_id=query['target_id'],native_target_bbox=original,
        source_dataset=RAW[0],source_revision=RAW[1],chat_revision=CHAT[1],annotation_origin='human browser interaction',
        screenshot=screenshot,input_view='visual')
    return grounding_qa(query['inputs'],target,boxes.values(),metadata,'weblinx',excluded)


def weblinx_grounding(max_rows=100000, seed=0):
    """Human WebLINX element grounding with native demonstration-level partitions.

    Requires weblinx==0.3.2 for preparation only. Native retrieved candidate boxes
    are joined to the replay's pre-action screenshot; mismatched viewports, invisible
    targets and inconsistent target geometry are excluded. Only prior dialogue/actions
    and the operation enter the input, never the current action arguments or future turns.
    Source screenshot bytes are unchanged. Native test_iid is the canonical test split;
    other held-out partitions remain separate. Original CC BY-NC-SA 4.0 data terms apply.
    """
    import weblinx
    if weblinx.__version__!='0.3.2':raise ValueError('Preparation requires weblinx==0.3.2')
    excluded,by_demo,ownership = [],defaultdict(list),{}
    for split,native in SPLITS.items():
        with gzip.open(hub_file(CHAT,f'data/chat/{native}.json.gz'),'rt') as stream:
            for line in stream:
                query=native_query(json.loads(line),excluded)
                if query is None:continue
                demo=query['demo']
                if demo in ownership and ownership[demo]!=split:raise ValueError('Native demonstration crosses splits')
                ownership[demo]=split;query['source_split']=native;by_demo[demo].append(query)
    def prepare(demo):
        try:path=hub_file(RAW,f'demonstrations/{demo}/replay.json')
        except RemoteEntryNotFoundError:
            excluded.extend(dict(source_row=q['source_row'],reason='missing_native_replay') for q in by_demo[demo]);return []
        replay=weblinx.Replay(json.loads(path.read_text()),demo_name=demo,base_dir=path.parent.parent)
        groups={}
        for query in by_demo[demo]:
            turn=replay[query['turn']]
            if not turn.has_screenshot():replay.assign_screenshot_to_turn(turn)
            state=turn.get('state') or {};screenshot=state.get('screenshot')
            if not screenshot or state.get('screenshot_status')!='good':
                excluded.append(dict(source_row=query['source_row'],reason='missing_or_broken_screenshot'));continue
            try:image=hub_file(RAW,f'demonstrations/{demo}/screenshots/{screenshot}')
            except RemoteEntryNotFoundError:
                excluded.append(dict(source_row=query['source_row'],reason='missing_native_screenshot'));continue
            with Image.open(image) as header:size=list(header.size)
            qa=visual_query(query,turn,size,screenshot,excluded)
            if qa is None:continue
            info=json.loads(qa['metadata']);info['source_split']=query['source_split'];qa['metadata']=json.dumps(info,sort_keys=True)
            group=groups.setdefault(screenshot,dict(path=image,image_group_id=info['image_group_id'],qa=[],split=ownership[demo]))
            group['qa'].append(qa)
        return list(groups.values())
    groups=[]
    with ThreadPoolExecutor(8) as pool:
        for index,prepared in enumerate(pool.map(prepare,ordered(by_demo,seed)),1):
            groups.extend(prepared)
            if index%50==0:print('WebLINX demonstrations',index,'/',len(by_demo),flush=True)
    hashes={str(group['path']):hashlib.sha256(group['path'].read_bytes()).hexdigest() for group in groups}
    priority={split:index for index,split in enumerate(['test','test_cat','test_geo','test_vis','test_web','validation','train'])}
    owners={}
    for group in groups:
        digest=hashes[str(group['path'])];split=group['split']
        if digest not in owners or priority[split]<priority[owners[digest]]:owners[digest]=split
    excluded.sort(key=lambda row:(row['source_row'],row['reason']))
    def rows(split):
        count=0
        for group in groups:
            if group['split']!=split:continue
            if owners[hashes[str(group['path'])]]!=split:
                excluded.append(dict(source_row=group['image_group_id'],reason='cross_split_screenshot_bytes'));continue
            qas=group['qa'][:max_rows-count] if split=='train' else group['qa']
            if qas:
                yield dict(images=[dict(bytes=group['path'].read_bytes(),path=None)],image_group_id=group['image_group_id'],qa=qas)
                count+=len(qas)
            if split=='train' and count>=max_rows:break
    return materialize({split:lambda split=split:rows(split) for split in SPLITS},'weblinx-grounding',[CHAT,RAW],
        excluded,weblinx_grounding,'https://huggingface.co/datasets/McGill-NLP/WebLINX-full',max_rows,seed,
        provenance=dict(license='cc-by-nc-sa-4.0',source_url='https://mcgill-nlp.github.io/weblinx/',annotation_origin='human',
            language=['en'],task_categories=['multiple-choice','visual-question-answering'],
            preparation_dependencies={'weblinx':'0.3.2'},box_convention='native CSS boxes scaled to the matching screenshot; raw event geometry corroborates gold'))
