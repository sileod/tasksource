"""Static, fresh procedural views of SPHINX and VisGym."""
from collections import Counter, deque
import io
import importlib.metadata
import json
import random
import re
import sys
from pathlib import Path

from .common import checkout, seed_for, verified_choices, write_static

SPHINX = ('maveryn/sphinx','28ef6c43629ab4491491a204167ef2f52dcd8182')
VISGYM = ('visgym/visgym','927271d107ad0196ad6aa597095ca57d01c6ddbb')
SPHINX_TASKS = ('charts_pie','transform_result_identify','transform_pair_infer','shape_count')


def nearby_choices(gold, verify, seed, lower=0, upper=None):
    """Close integer alternatives, checked against the current instance."""
    donors=[(str(gold+d),f'offset:{d}') for d in (-3,-2,-1,1,2,3)
            if gold+d>=lower and (upper is None or gold+d<=upper)]
    return verified_choices(str(gold),donors,verify,seed)


def sphinx_options(meta, seed):
    """Keep native same-scene candidates; image panel identities stay fixed."""
    gold=str(meta['answer']);question=meta['question']
    if 'options' in meta:
        labels=list(meta['options']);choices=[str(meta['options'][k]) for k in labels]
        question=re.sub(r'\s*\(a\)[–-]\(d\)','',question.split('\n\n')[0])
    elif meta.get('option_strategy')=='text_transform_synonyms':
        labels=meta['labels'];choices=meta['option_descs']
        question='Which transformation maps the left image to the right image?'
    elif 'labels' in meta:
        labels=meta['labels'];choices=['Image panel '+k.strip('()').upper() for k in labels]
        question+=' Panel letters refer to labels printed in the image.'
    else:
        lower,upper=meta['count_bounds']
        choices,label,alignment=nearby_choices(int(gold),lambda c:int(c==gold),seed,lower,upper)
        return question,choices,label,alignment
    label=labels.index(gold)
    if len(choices)<2 or len(set(choices))!=len(choices):raise ValueError('Invalid native choices')
    return question,choices,label,dict(verifier_scores=[int(i==label) for i in range(len(choices))],
                                     choice_method='native_same_scene_candidates')


def distances(grid, target):
    """Four-neighbor distances; used to check shortest-path decisions and ties."""
    target=tuple(target);distance={target:0};queue=deque([target])
    while queue:
        r,c=queue.popleft()
        for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
            p=(r+dr,c+dc)
            if 0<=p[0]<len(grid) and 0<=p[1]<len(grid[0]) and grid[p[0]][p[1]]==0 and p not in distance:
                distance[p]=distance[(r,c)]+1;queue.append(p)
    return distance


def unique_maze_move(grid, start, target):
    distance=distances(grid,target);start=tuple(start)
    if distance.get(start,0)<2:raise ValueError('Solved, unreachable or trivial maze state')
    scores=[int(distance.get((start[0]+dr,start[1]+dc))==distance[start]-1)
            for dr,dc in ((0,1),(-1,0),(0,-1),(1,0))]
    if sum(scores)!=1:raise ValueError('Multiple optimal next moves')
    return scores.index(1),scores


def _export(name, source, license, generate, max_rows, max_rows_eval, seed, config, builder=None):
    splits={};excluded=[]
    for split,limit in (('validation',max_rows_eval),('train',max_rows)):
        rows=[];splits[split]=rows
        for index in range(limit):
            instance_seed=seed_for(name,seed,split,index)
            try:
                image,question,choices,label,metadata=generate(instance_seed,index)
                if image.width<32 or image.height<32:raise ValueError('Image too small')
                data=io.BytesIO();image.save(data,format='PNG')
                rows.append(dict(id=f'{name}:{instance_seed}',image=data.getvalue(),inputs=question,
                    choices_list=choices,labels=label,metadata=dict(seed=instance_seed,source_revision=source[1],**metadata)))
            except (ValueError,RuntimeError,KeyError) as error:
                excluded.append(dict(split=split,seed=instance_seed,reason=str(error)))
    report=dict(source='https://github.com/'+source[0],revision=source[1],license=license,
        license_url='https://github.com/'+source[0]+'/blob/'+source[1]+'/LICENSE',annotation_origin='procedural',
        configuration=dict(seed=seed,max_rows=max_rows,max_rows_eval=max_rows_eval,**config),
        rendering_versions={p:importlib.metadata.version(p) for p in ('Pillow','numpy')},
        split_policy='Fresh split-specific seeds; image copies excluded across splits. No released benchmark examples ingested.',
        generation_exclusions=excluded,exclusion_counts=dict(Counter(r['reason'] for r in excluded)))
    from scripts.upload_repackaged import BUILDERS
    result=write_static(name,splits,report,builder or BUILDERS[name][1])
    root,archive_sha=checkout(*source)
    folder=Path('build')/(name+'-release')
    notices={p.name:p.read_text() for p in (root/'LICENSE',root/'NOTICE') if p.exists()}
    (folder/'asset-notices.json').write_text(json.dumps(notices,indent=2)+'\n')
    path=folder/'provenance.json';written=json.loads(path.read_text());written['source_archive_sha256']=archive_sha
    path.write_text(json.dumps(written,indent=2)+'\n')
    return result


def sphinx_mc(max_rows=1000,max_rows_eval=100,seed=42):
    """Fresh SPHINX visual MC: charts, transformations and geometric counting.

    Uses four reviewed task families, native same-scene choices and nearby count
    alternatives. Original panel labels stay fixed in images. Gold comes from the
    native constructive generator and its uniqueness guards, not a model judge.
    Code is MIT; shipped icon assets carry CC BY 4.0 attribution.
    """
    root,_=checkout(*SPHINX);sys.path.insert(0,str(root/'src'))
    from sphinx.registry import build_task_registry,build_motif_registry
    tasks=build_task_registry();motifs=build_motif_registry()
    def generate(s,index):
        task=SPHINX_TASKS[index%len(SPHINX_TASKS)]
        image,specs,meta=tasks[task].generate_instance(motifs,random.Random(s))
        question,choices,label,alignment=sphinx_options(meta,s)
        return image,question,choices,label,dict(task=task,native_metadata=meta,
            verification='native_constructive_gold_and_generator_uniqueness_guards',**alignment)
    result=_export('sphinx-mcq',SPHINX,'mit',generate,max_rows,max_rows_eval,seed,dict(tasks=SPHINX_TASKS))
    folder=Path('build/sphinx-mcq-release')
    (folder/'asset-notices.json').write_text(json.dumps({'code':(root/'LICENSE').read_text(),
        'icons':{'source':SPHINX[0],'license':(root/'src/sphinx/icons/LICENSE.txt').read_text()}},indent=2)+'\n')
    return result


def visgym_mc(max_rows=1000,max_rows_eval=100,seed=42):
    """Fresh VisGym maze decisions with a unique shortest-path next move.

    Uses the native Wilson maze generator and renderer, with intermediate
    reachable states and independently checked four-neighbor shortest paths.
    Tied optimal actions are excluded. The scope is procedural 2D mazes, without
    photographic, simulator or benchmark assets. Apache-2.0 with upstream notices.
    """
    root,_=checkout(*VISGYM);sys.path.insert(0,str(root))
    from gymnasium.envs.maze_2d.maze_2d import Maze2DEnv
    from gymnasium.envs.maze_2d.maze_solver import MazeSolver
    from PIL import Image
    def generate(s,index):
        size=(11,15,19)[index%3];env=Maze2DEnv(maze_width=size,maze_height=size,seed=s)
        try:
            env.reset(seed=s);state=env.get_init_state()
            path=MazeSolver().solve(env.maze_map,state['agent_location'],state['target_location'])
            if len(path)<4:raise ValueError('Trivial native maze')
            state['agent_location']=list(path[random.Random(s).randrange(len(path)-2)])
            env.reset(seed=s,init_state=state)
            label,scores=unique_maze_move(state['maze_map'],state['agent_location'],state['target_location'])
            return Image.fromarray(env.render()),'Move the blue dot toward the red square. Which first move is on the shortest path? Gray cells are walls; move one cell horizontally or vertically.',['Right','Up','Left','Down'],label,dict(task='maze2d_next_move',geometry=state,
                verifier_scores=scores,verification='native_maze_solver_and_independent_distance_check')
        finally:env.close()
    return _export('visgym-mcq',VISGYM,'apache-2.0',generate,max_rows,max_rows_eval,seed,dict(sizes=[11,15,19],scope='maze2d_next_move'),builder=visgym_mc)
