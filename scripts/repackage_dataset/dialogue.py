"""Small original-source conversions that avoid audio downloads or legacy loaders."""
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

from datasets import Dataset, DatasetDict, Features, List, Value

from .relevance import text_key, write_mirror

SLURP_REVISION = '8eb16545762be97ace75334109d73824217311f1'
MCSCRIPT_URL = 'https://fedora.clarin-d.uni-saarland.de/sfb1102/MCScript-2.0.zip'
MCSCRIPT_SHA256 = '50037b08123a4660e2e00ad07bcfa2a7ddc1d666b05b198d359cd2cda8f5e88c'
MOLWENI_REVISION = '2eb681d7e610ee3fda4ab3b4cc364554bd17ee67'
FRIENDSQA_REVISION = '9ce6c98a6acf13c87d96b6811e4409b5de43bdfe'


def _write_squad_mirror(name, splits, excluded, report):
    """Write a SQuAD-format mirror (``id``, ``title``, ``context``, ``question``, ``answers``,
    ``metadata``) as parquet, alongside its provenance and an exclusion manifest. Mirrors the layout
    ``write_mirror`` uses for classification/MC views, for the extractive-QA schema instead."""
    features = Features({'id': Value('string'), 'title': Value('string'), 'context': Value('string'),
        'question': Value('string'),
        'answers': {'text': List(Value('string')), 'answer_start': List(Value('int32'))},
        'metadata': Value('string')})
    directory = Path('build') / (name + '-release')
    directory.mkdir(parents=True, exist_ok=True)
    data = DatasetDict({split: Dataset.from_list(rows, features=features) for split, rows in splits.items()})
    for split, rows in data.items():
        rows.to_parquet(str(directory / (split + '.parquet')))
    report = dict(report, mirror=f'tasksource/{name}',
        splits={split: {'rows': len(rows)} for split, rows in data.items()},
        excluded_count=len(excluded),
        conversion_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (directory / 'provenance.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    (directory / 'excluded-questions.jsonl').write_text(
        ''.join(json.dumps(e, sort_keys=True) + '\n' for e in excluded))
    return data


def molweni():
    """Molweni multiparty Ubuntu-chat MRC, native MRC(withDiscourse) train/dev/test partitions.

    ``context`` and every answer's character offset are the official release's own SQuAD-style
    fields; both are kept verbatim rather than re-derived. Every one of the 30,066 questions across
    all three splits locates exactly (0 dropped). Speaker identity and the native dialogue index are
    kept in metadata, never in the model input. Original terms: Apache License 2.0 (repository LICENSE).
    """
    base = f'https://raw.githubusercontent.com/HIT-SCIR/Molweni/{MOLWENI_REVISION}/MRC(withDiscourse)'
    hashes, splits, excluded = {}, {}, []
    for split, native in [('train', 'train'), ('validation', 'dev'), ('test', 'test')]:
        raw = urllib.request.urlopen(f'{base}/{native}.json').read()
        hashes[f'MRC(withDiscourse)/{native}.json'] = hashlib.sha256(raw).hexdigest()
        dialogues = json.loads(raw)['data']['dialogues']
        rows = []
        for index, dialogue in enumerate(dialogues):
            context = dialogue['context']
            title = f'{native}-{index}'
            speakers = sorted(set(edu['speaker'] for edu in dialogue['edus']))
            for qa in dialogue['qas']:
                texts, starts = [], []
                for a in qa['answers']:
                    text, start = a['text'], a['answer_start']
                    if start is not None and start >= 0 and context[start:start + len(text)] == text:
                        texts.append(text); starts.append(start)
                if qa['answers'] and not texts:
                    excluded.append({'split': split, 'id': qa['id'], 'reason': 'unlocated_answer'})
                    continue
                meta = dict(id=qa['id'], title=title, dialogue_index=index, speakers=speakers,
                    num_utterances=len(dialogue['edus']), is_impossible=qa['is_impossible'])
                rows.append(dict(id=qa['id'], title=title, context=context, question=qa['question'],
                    answers={'text': texts, 'answer_start': starts}, metadata=json.dumps(meta, sort_keys=True, ensure_ascii=False)))
        splits[split] = rows
    report = dict(source='https://github.com/HIT-SCIR/Molweni', revision=MOLWENI_REVISION,
        source_hashes=hashes, license='apache-2.0',
        license_evidence='Repository LICENSE file: Apache License, Version 2.0.',
        split_policy='native MRC(withDiscourse) train/dev/test partitions; dev mapped to validation',
        packaging_script='scripts/repackage_dataset/dialogue.py:molweni')
    return _write_squad_mirror('molweni', splits, excluded, report)


def _friendsqa_locate(context, utterances, offsets, answer):
    uid = answer['utterance_id']
    if not 0 <= uid < len(offsets):
        return None
    info, text = offsets[uid], answer['answer_text']
    if answer['is_speaker']:
        start = context.find(text, info['prefix_start'], info['text_start'])
    else:
        tokens = utterances[uid]['utterance'].split(' ')
        start_i, end_i = answer['inner_start'], answer['inner_end']
        if not (0 <= start_i <= end_i < len(tokens)):
            return None
        start = info['text_start'] + sum(len(t) + 1 for t in tokens[:start_i])
    if start < 0 or context[start:start + len(text)] != text:
        return None
    return start


def friendsqa():
    """FriendsQA span QA over Friends transcript scenes, native trn/dev/tst partitions.

    The official release gives answers as ``utterance_id`` + an inner token span (or, for
    "who is speaking" questions, the speaker name with no span). Each dialogue is flattened to one
    ``Speaker, Speaker: utterance`` line per utterance, and every answer is relocated to a character
    offset in that flattened context, verified verbatim (simultaneous multi-speaker lines locate
    correctly because the gold speaker name is always a prefix of the joined speaker list). Only 2
    of 16,662 train answers carry a genuine source token-offset bug and cannot be relocated; those
    two gold spans are dropped and every other valid gold answer for their question is kept, so no
    question is dropped. Duplicate (text, offset) answers from different annotators are collapsed to
    one gold span. Original terms: Apache License 2.0 (repository LICENSE.txt).
    """
    base = f'https://raw.githubusercontent.com/emorynlp/FriendsQA/{FRIENDSQA_REVISION}/dat'
    hashes, splits, excluded = {}, {}, []
    dropped_answers = Counter()
    for split, native in [('train', 'trn'), ('validation', 'dev'), ('test', 'tst')]:
        raw = urllib.request.urlopen(f'{base}/friendsqa_{native}.json').read()
        hashes[f'friendsqa_{native}.json'] = hashlib.sha256(raw).hexdigest()
        scenes = json.loads(raw)['data']
        rows = []
        for scene in scenes:
            title = scene['title']
            for paragraph in scene['paragraphs']:
                utterances = paragraph['utterances:']
                lines, offsets, pos = [], [], 0
                for u in utterances:
                    prefix = ', '.join(u['speakers']) + ': '
                    line = prefix + u['utterance']
                    offsets.append({'prefix_start': pos, 'text_start': pos + len(prefix)})
                    lines.append(line)
                    pos += len(line) + 1  # one '\n' between utterances
                context = '\n'.join(lines)
                for qa in paragraph['qas']:
                    located = {}
                    for a in qa['answers']:
                        start = _friendsqa_locate(context, utterances, offsets, a)
                        if start is not None:
                            located[(a['answer_text'], start)] = True
                        else:
                            dropped_answers[split] += 1
                    if qa['answers'] and not located:
                        excluded.append({'split': split, 'id': qa['id'], 'reason': 'unlocated_answer'})
                        continue
                    texts = [t for t, s in located]
                    starts = [s for t, s in located]
                    meta = dict(id=qa['id'], title=title, utterance_ids=sorted(set(
                        a['utterance_id'] for a in qa['answers'])))
                    rows.append(dict(id=qa['id'], title=title, context=context, question=qa['question'],
                        answers={'text': texts, 'answer_start': starts}, metadata=json.dumps(meta, sort_keys=True, ensure_ascii=False)))
        splits[split] = rows
    report = dict(source='https://github.com/emorynlp/FriendsQA', revision=FRIENDSQA_REVISION,
        source_hashes=hashes, license='apache-2.0',
        license_evidence='Repository LICENSE.txt: Apache License, Version 2.0.',
        split_policy='native trn/dev/tst partitions mapped to train/validation/test',
        context_construction='one "Speaker, Speaker: utterance" line per native utterance, joined by newline; '
            'answers relocated from utterance_id + inner token span (or speaker name) to a character offset',
        dropped_answer_spans=dict(dropped_answers),
        packaging_script='scripts/repackage_dataset/dialogue.py:friendsqa')
    return _write_squad_mirror('friendsqa', splits, excluded, report)


def _slurp_intent(row):
    return row['scenario']+'_'+row['action']


def slurp():
    """SLURP original human transcripts and intent labels, without bundled audio.

    Native real train/devel/test partitions; synthetic speech is not included.
    Identical transcripts with inconsistent intents are excluded, and duplicate
    transcripts stay in their highest-priority native holdout (test, dev, train).
    One- and two-word transcripts are excluded as insufficient intent context.
    Intent is the native scenario/action pair; inconsistent raw intent aliases
    are retained in metadata. Original terms: CC BY 4.0.
    """
    splits={}; hashes={}; labels={};counts=Counter()
    for split,native in [('train','train'),('validation','devel'),('test','test')]:
        url=f'https://raw.githubusercontent.com/pswietojanski/slurp/{SLURP_REVISION}/dataset/slurp/{native}.jsonl'
        raw=urllib.request.urlopen(url).read();hashes[native]=hashlib.sha256(raw).hexdigest()
        splits[split]=[json.loads(line) for line in raw.splitlines() if line.strip()]
        for row in splits[split]:labels.setdefault(text_key(row['sentence']),set()).add(_slurp_intent(row))
    output={};seen=set()
    for split in ('test','validation','train'):
        output[split]=[]
        for row in splits[split]:
            key=text_key(row['sentence'])
            reason='ambiguous_intent' if len(labels[key])!=1 else 'short_transcript' if len(row['sentence'].split())<3 else 'duplicate_transcript' if key in seen else None
            if reason:counts[split+'/'+reason]+=1;continue
            seen.add(key)
            meta=dict(id=str(row['slurp_id']),split_group_id=key,scenario=row['scenario'],action=row['action'],
                native_intent=row['intent'],entities=row['entities'],tokens=[t['surface'] for t in row['tokens']],source_revision=SLURP_REVISION)
            output[split].append(dict(sentence1=row['sentence'],sentence2='',labels=_slurp_intent(row),metadata=json.dumps(meta,sort_keys=True)))
    return write_mirror('slurp',output,'string',dict(source='https://github.com/pswietojanski/slurp',revision=SLURP_REVISION,
        source_hashes=hashes,excluded=dict(counts),split_policy='native splits; exact transcript conflicts and overlap removed',
        intent_definition='native scenario + action',
        normalized_intents={s:sum(r['intent']!=_slurp_intent(r) for r in rows) for s,rows in splits.items()},
        conversion_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),license='cc-by-4.0'))


def mcscript2():
    """MCScript2.0 native two-answer reading comprehension, original XML partitions.

    Preserve stories, questions and native choices; exactly one gold answer and
    two distinct non-empty options are required. Story hashes prevent train/dev/
    test copies. Source URL/archive hash and per-split counts are recorded.
    Dataset terms: unspecified until original distribution terms are verified.
    """
    raw=urllib.request.urlopen(MCSCRIPT_URL).read()
    if hashlib.sha256(raw).hexdigest()!=MCSCRIPT_SHA256:raise ValueError('MCScript archive changed')
    output={};groups={};counts=Counter()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for split,native in [('train','train'),('validation','dev'),('test','test')]:
            output[split]=[];groups[split]=set()
            for story in ET.fromstring(archive.read(native+'-data.xml')):
                text=story.findtext('text');group=text_key(text);groups[split].add(group)
                for question in story.findall('./questions/question'):
                    choices=[a.attrib['text'] for a in question.findall('answer')]
                    gold=[i for i,a in enumerate(question.findall('answer')) if a.attrib['correct'].lower()=='true']
                    if len(choices)!=2 or len(set(choices))!=2 or not all(c.strip() for c in choices) or len(gold)!=1:
                        counts[split+'/invalid_choices']+=1;continue
                    meta=dict(id=story.attrib['id']+':'+question.attrib['id'],split_group_id=group,
                        scenario=story.attrib['scenario'],question_type=question.attrib['type'])
                    output[split].append(dict(inputs=f"Story: {text}\nQuestion: {question.attrib['text']}",
                        choices_list=choices,labels=gold[0],metadata=json.dumps(meta,sort_keys=True)))
    assert all(not groups[a]&groups[b] for a in groups for b in groups if a!=b),'MCScript story overlap'
    data=DatasetDict({s:Dataset.from_list(r) for s,r in output.items()})
    directory=Path('build/mcscript2-release');directory.mkdir(parents=True,exist_ok=True)
    for split,rows in data.items():rows.to_parquet(str(directory/(split+'.parquet')))
    provenance=dict(source_url=MCSCRIPT_URL,archive_sha256=MCSCRIPT_SHA256,excluded=dict(counts),
        splits={s:len(r) for s,r in data.items()},split_policy='native story partitions',
        conversion_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),license='unspecified')
    (directory/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (directory/'excluded-questions.jsonl').write_text('')
    return data


ESCONV_REVISION = '32c54c15e7651cd68dfab18eee39f63ff0cc1e82'


def esconv_turns(row, identity, source_row):
    history = []
    for index, turn in enumerate(row['dialog']):
        speaker = {'usr':'Seeker','sys':'Supporter'}[turn['speaker']]
        if turn['speaker']=='sys':
            yield dict(sentence1='Situation: '+row['situation']+'\n\nEarlier dialogue:\n'+'\n'.join(history),
                sentence2=turn['text'],labels=turn['strategy'],metadata=json.dumps(dict(
                    id=f'{source_row}:{index}',split_group_id=identity,turn=index,
                    source_revision=ESCONV_REVISION,annotation_origin='human'),sort_keys=True))
        history.append(speaker+': '+turn['text'])


def esconv():
    """ESConv supporter-strategy judgments from the authors' native dialogue splits.

    Read the official JSON-lines files despite their .txt suffix and preserve
    each annotated supporter turn separately, including consecutive turns with
    different strategies. Some existing conversions merge those turns and lose
    strategy alignment. Inputs contain only the situation, earlier dialogue and
    current reply; surveys, strategy names and future turns are excluded.
    Native conversations stay together; exact conversation duplicates favor
    test, validation, then train. Original terms: CC BY-NC 4.0.
    """
    from huggingface_hub import hf_hub_download
    repo='thu-coai/esconv';splits={};hashes={};excluded=Counter();seen=set()
    for split,name in [('test','test'),('validation','valid'),('train','train')]:
        path=Path(hf_hub_download(repo,name+'.txt',repo_type='dataset',revision=ESCONV_REVISION))
        raw=path.read_bytes();hashes[name+'.txt']=hashlib.sha256(raw).hexdigest();splits[split]=[]
        for index,line in enumerate(raw.splitlines()):
            row=json.loads(line)
            identity=hashlib.sha256(json.dumps([(t['speaker'],t['text']) for t in row['dialog']]).encode()).hexdigest()
            if identity in seen:excluded[split+'/duplicate_conversation']+=1;continue
            seen.add(identity);splits[split].extend(esconv_turns(row,identity,f'{split}:{index}'))
    return write_mirror('esconv',splits,'string',dict(source='https://huggingface.co/datasets/'+repo,
        revision=ESCONV_REVISION,source_hashes=hashes,excluded=dict(excluded),license='cc-by-nc-4.0',
        annotation_origin='human',split_policy='native conversation splits; duplicate conversations favor holdouts',
        conversion_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
