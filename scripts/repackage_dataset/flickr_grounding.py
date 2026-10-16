"""Join official Flickr30k Entities phrase chains to unchanged Flickr30k photographs."""
import hashlib
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.request import urlretrieve

from tasksource.grounding import normalized_box
from .grounding import grounding_qa, hub_file, materialize, ordered
from .procedural.common import file_sha256

ANNOTATIONS = ('BryanPlummer/flickr30k_entities', '68b3d6f12d1d710f96233f6bd2b6de799d6f4e5b')
IMAGES = ('nlphuji/flickr30k', '2b239befc81b6e3f035ce6bd52f5f4d60f5625f7')
PHRASE = re.compile(r'\[/EN#(\d+)/[^\s]+ ([^\]]+)\]')


def annotation_file(name):
    path = Path('build/grounding-sources/flickr30k-entities') / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix('.part')
        urlretrieve(f'https://raw.githubusercontent.com/{ANNOTATIONS[0]}/{ANNOTATIONS[1]}/{name}', temporary)
        temporary.replace(path)
    return path


def phrase_boxes(xml):
    """Convert native 1-based inclusive pixels to normalized half-open boxes."""
    root = ET.fromstring(xml)
    width, height = (int(root.findtext('size/' + key)) for key in ('width', 'height'))
    boxes = defaultdict(list)
    for item in root.findall('object'):
        for box in item.findall('bndbox'):
            x1, y1, x2, y2 = (int(box.findtext(key)) for key in ('xmin', 'ymin', 'xmax', 'ymax'))
            try:geometry = normalized_box([(x1-1)/width, (y1-1)/height, x2/width, y2/height])
            except ValueError:continue
            for name in item.findall('name'):boxes[name.text].append(geometry)
    return [width, height], boxes


def marked_caption(sentence, target):
    """Keep the native caption and identify the queried occurrence, even for repeated text."""
    return PHRASE.sub(lambda match: ('[PHRASE]' + match[2] + '[/PHRASE]')
                      if match.start() == target.start() else match[2], sentence).strip()


def flickr30k_grounding(max_rows=100000, seed=0):
    """Human phrase grounding from Flickr30k Entities, with native image-level splits.

    Each question retains the complete caption and marks the queried phrase in text.
    Only chains with one native target box are included; other annotated image regions
    supply choices, never invented negatives. Known unrelated captions are excluded.
    Images are stored once with canonical grouped QAs, unchanged from the image source.
    The training cap counts questions, not image rows. Original images are provided
    for noncommercial research/education under Flickr terms; annotation/code access
    does not grant ownership of those images.
    """
    files = {name: annotation_file(name) for name in ('annotations.zip', 'train.txt', 'val.txt', 'test.txt', 'UNRELATED_CAPTIONS')}
    partitions = {split: set(files[name].read_text().split()) for split, name in
                  [('train', 'train.txt'), ('validation', 'val.txt'), ('test', 'test.txt')]}
    assert not partitions['train'] & partitions['validation'] and not partitions['train'] & partitions['test']
    assert not partitions['validation'] & partitions['test']
    unrelated = {tuple(line.split()) for line in files['UNRELATED_CAPTIONS'].read_text().splitlines() if line and not line.startswith('#')}
    archive_path = hub_file(IMAGES, 'flickr30k-images.zip')
    with zipfile.ZipFile(archive_path) as archive:
        members = {Path(name).stem: name for name in archive.namelist() if name.startswith('flickr30k-images/') and name.endswith('.jpg')}
        hashes = {key: hashlib.sha256(archive.read(members[key])).hexdigest() for ids in partitions.values() for key in ids}
    forbidden = {split: {hashes[key] for other, ids in partitions.items() for key in ids
                        if other == 'test' or split == 'train' and other == 'validation'} for split in ('train', 'validation')}
    excluded, generators = [], {}
    for split, identities in partitions.items():
        def examples(split=split, identities=identities):
            count = 0
            with zipfile.ZipFile(files['annotations.zip']) as annotations, zipfile.ZipFile(archive_path) as images:
                for identity in ordered(identities, seed):
                    if hashes[identity] in forbidden.get(split, set()):
                        excluded.append(dict(source_row=identity, reason='heldout_image_bytes'));continue
                    size, boxes = phrase_boxes(annotations.read(f'Annotations/{identity}.xml'))
                    pool = [box for values in boxes.values() for box in values]
                    qas = []
                    sentences = annotations.read(f'Sentences/{identity}.txt').decode().splitlines()
                    for number, sentence in enumerate(sentences, 1):
                        if (identity, str(number)) in unrelated:
                            excluded.append(dict(source_row=f'{identity}:{number}', reason='publisher_unrelated_caption'));continue
                        for target in PHRASE.finditer(sentence):
                            chain = boxes.get(target[1], [])
                            key = f'flickr30k:{identity}:{number}:{target.start()}'
                            if len(chain) != 1:
                                excluded.append(dict(source_row=key, reason='multiple_or_no_target_boxes'));continue
                            qa = grounding_qa('Caption: ' + marked_caption(sentence, target), chain[0], pool,
                                dict(source_row=key, image_group_id='flickr30k:' + identity, image_size=size,
                                     source_dataset=ANNOTATIONS[0], source_revision=ANNOTATIONS[1], source_split=split,
                                     phrase_id=target[1], caption_index=number, phrase=target[2], annotation_origin='human'),
                                'flickr30k', excluded)
                            if qa is not None:qas.append(qa)
                    if split == 'train':qas = qas[:max_rows-count]
                    if qas:
                        yield dict(images=[dict(bytes=images.read(members[identity]), path=None)],
                                   image_group_id='flickr30k:' + identity, qa=qas)
                        count += len(qas)
                    if split == 'train' and count >= max_rows:break
        generators[split] = examples
    return materialize(generators, 'flickr30k-grounding', [ANNOTATIONS, IMAGES], excluded,
        flickr30k_grounding, 'https://github.com/BryanPlummer/flickr30k_entities#flickr30k-entities-dataset', max_rows, seed,
        provenance=dict(source_url='https://github.com/BryanPlummer/flickr30k_entities', annotation_origin='human',
            language=['en'], task_categories=['multiple-choice', 'visual-question-answering'],
            input_sha256={name:file_sha256(path) for name,path in files.items()},
            box_convention='1-based inclusive source pixels to normalized zero-based half-open xyxy'))
