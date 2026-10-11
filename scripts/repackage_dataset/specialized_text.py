"""Pinned legal relevance, clinical inference and sexism sources; native labels only."""
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request
import zipfile

from huggingface_hub import hf_hub_download

from .relevance import text_key, write_mirror

ACORD_REVISION = '7ba1c6f2e0baaa32e78c78c2b02e9c57b102af4e'
EDOS_REVISION = 'b1f1fbcca9d42de0395432ada8fcfb5c35f68c77'
EVIDENCE_URL = 'https://evidence-inference.ebm-nlp.com/v2.0.tar.gz'
EVIDENCE_SHA256 = '6abe0d4ec0d331834981c0171c3c79d47515761867f82f1dc6066e43863a1586'


def acord():
    """ACORD attorney-annotated query/clause relevance, preserving native query splits.

    Join only explicit qrels against the official queries/corpus. Unjudged pairs
    are never converted to negatives. Grades 0–4 correspond to 1–5 attorney stars.
    Original dataset terms: CC BY 4.0. Archive, source revision and counts are
    recorded in provenance.json; metadata retains query and clause identities.
    """
    archive = hf_hub_download('theatticusproject/acord','ACORD Dataset & ReadMe.zip',
        repo_type='dataset',revision=ACORD_REVISION)
    splits, counts, groups = {}, Counter(), {}
    with zipfile.ZipFile(archive) as source:
        prefix = next(n.rsplit('/',1)[0]+'/' for n in source.namelist() if n.endswith('/queries.jsonl'))
        queries = {r['_id']:r['text'] for r in map(json.loads,source.read(prefix+'queries.jsonl').splitlines())}
        clauses = {r['_id']:r['text'] for r in map(json.loads,source.read(prefix+'corpus.jsonl').splitlines())}
        for split,native in [('train','train'),('validation','valid'),('test','test')]:
            records = {}
            for row in csv.DictReader(io.StringIO(source.read(prefix+f'qrels/{native}.tsv').decode()),delimiter='\t'):
                q,c,grade = row['query-id'],row['corpus-id'],int(row['score'])
                if q not in queries or c not in clauses or grade not in range(5):
                    raise ValueError('Invalid ACORD qrel')
                key=(q,c)
                if key in records:
                    if records[key]['labels'] != str(grade):
                        raise ValueError('Conflicting ACORD qrels')
                    counts['duplicate_qrels'] += 1
                    continue
                meta=dict(id=text_key(q)+':'+c,query_id=q,clause_id=c,split_group_id=text_key(queries[q]))
                records[key]=dict(sentence1=queries[q],sentence2=clauses[c],labels=str(grade),
                    metadata=json.dumps(meta,sort_keys=True))
                counts[f'{split}/grade/{grade}'] += 1
            splits[split]=list(records.values())
            groups[split]={text_key(queries[q]) for q,c in records}
    assert all(not groups[a]&groups[b] for a in groups for b in groups if a!=b), 'ACORD query overlap'
    return write_mirror('acord',splits,'string',dict(conversion_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source='theatticusproject/acord',revision=ACORD_REVISION,
        archive_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest(),counts=dict(counts),
        query_counts={s:len(g) for s,g in groups.items()},split_policy='native query partitions',license='cc-by-4.0'))


def edos():
    """EDOS native binary sexism judgments and original train/dev/test assignments.

    Original source: https://github.com/rewire-online/edos (CC0-1.0).
    Reads the aggregated annotation CSV, retaining all native labels, identities
    and split assignments. Category/vector labels remain in metadata; no generated
    explanations enter inputs. See provenance.json for the pinned source/hash.
    """
    url=f'https://raw.githubusercontent.com/rewire-online/edos/{EDOS_REVISION}/data/edos_labelled_aggregated.csv'
    raw=urllib.request.urlopen(url).read()
    splits={s:[] for s in ('train','validation','test')}
    for row in csv.DictReader(io.StringIO(raw.decode())):
        if row['label_sexist'] not in ('not sexist','sexist'):
            raise ValueError('Unknown EDOS label')
        split={'train':'train','dev':'validation','test':'test'}[row['split']]
        meta=dict(id=row['rewire_id'],split_group_id=text_key(row['text']),
            category=row['label_category'],vector=row['label_vector'],source_revision=EDOS_REVISION)
        splits[split].append(dict(sentence1=row['text'],sentence2='',labels=row['label_sexist'],
            metadata=json.dumps(meta,sort_keys=True)))
    # Preserve evaluation membership; discard training copies of evaluation texts.
    heldout={text_key(r['sentence1']) for s in ('validation','test') for r in splits[s]}
    before=len(splits['train'])
    splits['train']=[r for r in splits['train'] if text_key(r['sentence1']) not in heldout]
    return write_mirror('edos',splits,'string',dict(conversion_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_url=url,revision=EDOS_REVISION,
        source_sha256=hashlib.sha256(raw).hexdigest(),excluded_train_overlap=before-len(splits['train']),
        split_policy='native partitions; training copies of evaluation text removed',license='cc0-1.0'))


def evidence_inference():
    """Evidence Inference v2: full article + intervention/comparator/outcome → effect direction.

    Join annotations by PromptID AND PMCID, never by article alone. Use verified
    labels, exclude tied votes and upstream known-bad prompts. Evidence spans in
    metadata refer only to the same prompt and winning label. Full article text
    stays in the input so isolated snippets need not identify treatment arms.
    Native article train/validation/test splits are preserved. The upstream code
    and annotations have MIT terms; article-text rights vary and are not relabeled
    MIT. Mirror text/data licensing remains unspecified.
    """
    raw=urllib.request.urlopen(EVIDENCE_URL).read()
    digest=hashlib.sha256(raw).hexdigest()
    if digest != EVIDENCE_SHA256:
        raise ValueError('Evidence Inference archive changed; review before updating the pin')
    script=Path(hf_hub_download('bigbio/evidence_inference','evidence_inference.py',repo_type='dataset',
        revision='35dce6aba1b3eb9eb9af9bdc38ebeda73dad15b9')).read_text()
    skip=set()
    for node in ast.parse(script).body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in {
            'INCORRECT_PROMPT_IDS','QUESTIONABLE_PROMPT_IDS','SOMEWHAT_MALFORMED_PROMPT_IDS'}:
            skip.update(ast.literal_eval(node.value.args[0]))
    splits={s:[] for s in ('train','validation','test')}; counts=Counter();annotations=defaultdict(list)
    with tarfile.open(fileobj=io.BytesIO(raw)) as compressed:
        payload = {member.name:compressed.extractfile(member).read() for member in compressed if member.isfile()}
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        def read(name):
            return payload[name].decode('utf-8-sig')
        membership={}
        for split in splits:
            ids=set(read(f'splits/{split}_article_ids.txt').split())
            assert not ids.intersection(membership), 'Evidence Inference article overlap'
            membership.update({article:split for article in ids})
        for row in csv.DictReader(io.StringIO(read('annotations_merged.csv'))):
            if row['Valid Label'].lower() in ('true','1'):
                annotations[(row['PromptID'],row['PMCID'])].append(row)
        articles={}
        for prompt in csv.DictReader(io.StringIO(read('prompts_merged.csv'))):
            pid,article=prompt['PromptID'],prompt['PMCID']
            votes=Counter(a['Label Code'] for a in annotations[(pid,article)])
            if int(pid) in skip or not votes or article not in membership:
                counts['excluded/invalid_or_unassigned_prompt'] += 1
                continue
            ranked=votes.most_common()
            if ranked[0][0] not in ('-1','0','1') or len(ranked)>1 and ranked[0][1]==ranked[1][1]:
                counts['excluded/tied_or_invalid_labels'] += 1
                continue
            label=ranked[0][0]
            if article not in articles:
                articles[article]=read(f'txt_files/PMC{article}.txt')
            text=articles[article];spans=set()
            for annotation in annotations[(pid,article)]:
                start,end=int(annotation['Evidence Start']),int(annotation['Evidence End'])
                if annotation['Label Code']==label and annotation['Valid Reasoning'].lower() in ('true','1') and 0<=start<end<=len(text):
                    spans.add((start,end))
            question=f"Intervention: {prompt['Intervention']}\nComparator: {prompt['Comparator']}\nOutcome: {prompt['Outcome']}"
            meta=dict(id=pid,split_group_id=article,pmcid=article,prompt_id=pid,votes=dict(votes),
                evidence_spans=[dict(start=a,end=b,text=text[a:b]) for a,b in sorted(spans)])
            splits[membership[article]].append(dict(sentence1=text,sentence2=question,labels=label,
                metadata=json.dumps(meta,sort_keys=True)))
            counts['retained/'+label] += 1
    return write_mirror('evidence-inference',splits,'string',dict(conversion_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_url=EVIDENCE_URL,
        archive_sha256=digest,reviewed_exclusions_revision='35dce6aba1b3eb9eb9af9bdc38ebeda73dad15b9',
        known_bad_prompt_ids=sorted(skip),counts=dict(counts),split_policy='native article partitions',
        label_policy='majority verified labels; ties excluded',license='unspecified'))
