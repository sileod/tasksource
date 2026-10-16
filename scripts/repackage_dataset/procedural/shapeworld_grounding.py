"""Static grounding decisions from pinned ShapeWorld scenes and native object boxes."""
import hashlib
import io
import random
import sys
from collections import Counter

import numpy as np
from PIL import Image

from tasksource.grounding import normalized_box
from ..grounding import grounding_qa, materialize
from .common import checkout, seed_for

SOURCE = ('AlexKuhnle/ShapeWorld', 'e720bf46e57fc01326d04d639fa6133d9c12158f')
CONFIG = dict(world_size=192, entity_counts=(3,4,5,6), collision_tolerance=0.0,
              boundary_tolerance=0.0, shade_range=0.0,
              shapes=('square','triangle','pentagon','circle','ellipse'))


def scene_queries(entities):
    """Unique attribute references and unambiguous spatial relations, verified on the scene."""
    names = [entity['color']['name']+' '+entity['shape']['name'] for entity in entities]
    counts = Counter(names)
    boxes = [normalized_box([entity['bounding_box'][corner][axis]
                             for corner in ('topleft','bottomright') for axis in ('x','y')])
             for entity in entities]
    queries = [('Locate the '+name+'.', box, 'attribute')
               for name,box in zip(names,boxes) if counts[name]==1]
    for anchor,name in enumerate(names):
        if counts[name]!=1:continue
        for axis,relation in ((0,'left of'),(1,'above'),(0,'right of'),(1,'below')):
            def matches(box):
                return (box[axis+2]+.03<boxes[anchor][axis] if relation in ('left of','above')
                        else box[axis]-.03>boxes[anchor][axis+2])
            for target in sorted(set(names)):
                if counts[target]==1:continue  # spatial wording must disambiguate repeated attributes
                indices=[i for i,(other,box) in enumerate(zip(names,boxes)) if other==target and matches(box)]
                if len(indices)==1:
                    queries.append((f'Locate the {target} {relation} the {name}.',boxes[indices[0]],'spatial'))
    return boxes,queries


def shapeworld_grounding(max_rows=100000, max_rows_eval=1000, seed=0):
    """Procedurally verified color/shape and spatial grounding using ShapeWorld's renderer.

    Native scene objects provide every candidate box. Queries are retained only when
    exactly one object satisfies their attributes and, where used, spatial relation.
    Non-overlapping, fully visible scenes have three to six shapes. Independent
    split/scene seeds and image hashes prevent train/evaluation image overlap.
    Images are generated as PNG, stored once per scene with up to two grouped QAs.
    ShapeWorld's MIT license applies; this is a derived grounding task, not its
    original caption-agreement benchmark. No language model labels or distractors.
    """
    root,archive_hash=checkout(*SOURCE)
    sys.path.insert(0,str(root))
    from shapeworld.generators import RandomAttributesGenerator
    generator=RandomAttributesGenerator(**CONFIG)
    excluded=[];seen=set()
    def rows(split,limit):
        count=index=0
        while count<limit:
            identity=f'shapeworld:{split}:{index}';index+=1
            instance_seed=seed_for(seed,identity)
            random.seed(instance_seed);np.random.seed(instance_seed%(2**32))
            generator.initialize(split);world=generator()
            if world is None:
                excluded.append(dict(source_row=identity,reason='generation_failed'));continue
            model=world.model()
            try:boxes,queries=scene_queries(model['entities'])
            except ValueError:
                excluded.append(dict(source_row=identity,reason='native_box_outside_image'));continue
            picked=[]
            for kind in ('attribute','spatial'):
                options=[query for query in queries if query[2]==kind]
                if options:picked.append(random.choice(options))
            qas=[]
            for position,(instruction,target,kind) in enumerate(picked):
                qa=grounding_qa(instruction,target,boxes,dict(source_row=f'{identity}:{position}',
                    image_group_id=identity,image_size=[CONFIG['world_size']]*2,source_split=split,
                    source_dataset=SOURCE[0],source_revision=SOURCE[1],annotation_origin='procedural',
                    generation_seed=instance_seed,query_kind=kind,world_model=model),'shapeworld',excluded)
                if qa is not None:qas.append(qa)
            if not qas:
                excluded.append(dict(source_row=identity,reason='no_unique_grounding_query'));continue
            image=Image.fromarray(np.rint(np.clip(world.get_array(),0,1)*255).astype('uint8'))
            stream=io.BytesIO();image.save(stream,format='PNG');encoded=stream.getvalue()
            digest=hashlib.sha256(encoded).hexdigest()
            if digest in seen:
                excluded.append(dict(source_row=identity,reason='duplicate_generated_image'));continue
            seen.add(digest);qas=qas[:limit-count]
            yield dict(images=[dict(bytes=encoded,path=None)],image_group_id=identity,qa=qas)
            count+=len(qas)
    # Reserve generated held-out images first; later splits resample exact duplicates.
    limits={'test':max_rows_eval,'validation':max_rows_eval,'train':max_rows}
    return materialize({split:lambda split=split:rows(split,limits[split]) for split in limits},
        'shapeworld-grounding',[SOURCE],excluded,shapeworld_grounding,
        f'https://github.com/{SOURCE[0]}/blob/{SOURCE[1]}/LICENSE',max_rows,seed,
        provenance=dict(license='mit',source_url=f'https://github.com/{SOURCE[0]}',annotation_origin='procedural',
            language=['en'],task_categories=['multiple-choice','visual-question-answering'],
            generator_archive_sha256=archive_hash,generator_config=CONFIG,max_evaluation_questions=max_rows_eval,
            split_policy='independent split/scene seeds; exact image hashes disjoint',
            query_policy='unique color/shape references; strict box-separated spatial relations; up to one query per kind per scene'))
