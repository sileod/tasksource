"""Pinned human linguistic annotations; preserve native spans, inventories and partitions."""
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
from urllib.request import urlopen

from datasets import Features, Sequence, Value
from huggingface_hub import hf_hub_download

from .relevance import text_key, write_mirror
from .stance import _holdouts

WORDNET_SHA = '9fc5cbd1bc358d793364dc4659e239bb91bfdcdf9ff5e30bac76e7bb317c438d'
STREUSLE_REVISION = '8ba61fe4f216e7967500a862554a4fff79d25f5d'
SPIDER_REVISION = '20060fadce13ab8a88c24d60dc7ad39e3eced695'
SEMCOR_REVISION = '2db8cc29752777441ed3bed7ca97352171059550'


def _download(name, url, sha=None):
    path=Path('build/source-archives/linguistic')/hashlib.sha256(url.encode()).hexdigest()[:16]/name
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(urlopen(url,timeout=180).read())
    raw=path.read_bytes()
    if sha and hashlib.sha256(raw).hexdigest()!=sha:
        raise ValueError(f'Source checksum mismatch: {name}')
    return raw


def _row(text,label,identity,group,**meta):
    return dict(sentence1=text,labels=label,metadata=json.dumps(dict(id=identity,split_group_id=group,**meta),sort_keys=True))


def _write(name,splits,report,mc=False):
    fields=dict(inputs=Value('string'),choices_list=Sequence(Value('string')),labels=Value('int64')) if mc else dict(sentence1=Value('string'),labels=Value('string'))
    features=Features(dict(**fields,metadata=Value('string')))
    report['conversion_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    splits={s:rows for s,rows in _holdouts(splits,report).items() if rows}
    return write_mirror(name,splits,'int64' if mc else 'string',report,features)


def mark_tokens(tokens, indices):
    indices=set(indices)
    if not indices or min(indices)<0 or max(indices)>=len(tokens):raise ValueError('Invalid annotated token span')
    return ' '.join(('[TARGET] '+t+' [/TARGET]') if i in indices else t for i,t in enumerate(tokens))


def streusle_rows(row):
    tokens=[t['word'] for t in row['toks']]
    for expression in [*row['swes'].values(),*row['smwes'].values()]:
        label=expression['ss']
        if not label or not label.startswith(('n.','v.')):continue
        indices=[i-1 for i in expression['toknums']]
        yield _row(mark_tokens(tokens,indices),label,row['sent_id']+':'+','.join(map(str,indices)),
                   row['sent_id'].rsplit('-',1)[0],token_indices=indices,lemma=expression['lexlemma'],part_of_speech=label[0])


def streusle():
    """STREUSLE native noun/verb supersenses, including discontinuous lexical expressions.

    Original distribution: https://github.com/nert-nlp/streusle.
    Native UD train/dev/test and review IDs are retained. Adposition role/function
    distinctions are omitted from these noun/verb views. Dataset terms: CC BY-SA 4.0;
    original README records permission to redistribute source review sentences.
    """
    splits={};hashes={}
    for split,native in [('train','train'),('validation','dev'),('test','test')]:
        file=f'{native}/streusle.ud_{native}.json'
        raw=_download('streusle-'+native+'.json',f'https://raw.githubusercontent.com/nert-nlp/streusle/{STREUSLE_REVISION}/{file}')
        hashes[file]=hashlib.sha256(raw).hexdigest()
        splits[split]=[r for row in json.loads(raw) for r in streusle_rows(row)]
    return _write('streusle',splits,dict(source='https://github.com/nert-nlp/streusle',revision=STREUSLE_REVISION,source_hashes=hashes,license='cc-by-sa-4.0',split_policy='native review partitions'))


def wordnet_inventory(raw):
    archive=tarfile.open(fileobj=io.BytesIO(raw));glosses={};inventory=defaultdict(dict);keys={}
    for pos,file in enumerate(('noun','verb','adj','adv'),1):
        for line in archive.extractfile('wordnet-1.6/dict/data.'+file).read().decode('latin1').splitlines():
            if line[:1].isdigit():
                glosses[pos,line.split()[0]]=line.split('|',1)[1].strip()
    for line in archive.extractfile('wordnet-1.6/dict/index.sense').read().decode('latin1').splitlines():
        key,offset,*_=line.split();lemma,lexsn=key.split('%');pos=int(lexsn[0]);pos=3 if pos==5 else pos
        identity=(pos,offset);inventory[lemma,pos][identity]=glosses[identity];keys[key]=identity
    return inventory,keys


def semcor():
    """SemCor native sense keys joined to the matching Princeton WordNet 1.6 glosses.

    Original distribution: https://wordnet.princeton.edu/semcor.
    Pinned faithful CSV copy: https://huggingface.co/datasets/thesofakillers/SemCor.
    Uses only one native gold sense and at least two distinct lemma/POS senses;
    unresolved keys, multiple senses and ambiguous duplicate glosses are excluded.
    All Brown sections remain training-only, not fabricated evaluation partitions.
    Terms: original Princeton WordNet/SemCor license; see upstream LICENSE.
    """
    raw=_download('wordnet16.tar.gz','https://wordnetcode.princeton.edu/1.6/wn16.unix.tar.gz',WORDNET_SHA)
    inventory,keys=wordnet_inventory(raw);rows=[];excluded=Counter();hashes={}
    for section in ('brown1','brown2','brownv'):
        path=hf_hub_download('thesofakillers/SemCor',f'data/{section}-00000-of-00001.csv',repo_type='dataset',revision=SEMCOR_REVISION)
        hashes[section]=hashlib.sha256(Path(path).read_bytes()).hexdigest();sentences=defaultdict(list)
        with open(path) as source:
            for r in csv.DictReader(source):sentences[r['tagfile'],r['snum']].append(r)
        for (document,sentence),words in sentences.items():
            tokens=[r['value'].replace('_',' ') for r in words]
            for i,r in enumerate(words):
                if r['cmd']!='done' or r['lemma']=='None' or r['lexsn']=='None':continue
                key=r['lemma']+'%'+r['lexsn'];gold=keys.get(key)
                if gold is None:excluded['unresolved_or_multiple_senses']+=1;continue
                senses=inventory[r['lemma'],gold[0]];choices=list(senses.values())
                if len(choices)<2 or len(choices)!=len(set(choices)):
                    excluded['single_sense_or_duplicate_gloss']+=1;continue
                meta=json.dumps(dict(id=f'{document}:{sentence}:{i}',split_group_id=document,wordnet_version='1.6',sense_key=key,token_index=i),sort_keys=True)
                rows.append(dict(inputs=mark_tokens(tokens,[i]),choices_list=choices,labels=list(senses).index(gold),metadata=meta))
    return _write('semcor',dict(train=rows),dict(source='https://wordnet.princeton.edu/semcor',hub_source='thesofakillers/SemCor',
        revision=SEMCOR_REVISION,inventory_url='https://wordnetcode.princeton.edu/1.6/wn16.unix.tar.gz',inventory_sha256=WORDNET_SHA,
        source_hashes=hashes,excluded=dict(excluded),license='wordnet',split_policy='native training corpus, no evaluation data'),mc=True)


def spider_rows(row,schema,projection,excluded):
    sql=row['sql'];selected=sql['select'][1]
    if len(selected)!=1 or any(sql.get(op) for op in ('union','intersect','except')):
        excluded['multiple_outputs_or_set_operation']+=1;return
    aggregation,unit=selected[0];operation,column,other=unit
    if operation!=0 or other is not None or column[0]!=0:
        excluded['arithmetic_or_nested_aggregation']+=1;return
    choices=['all rows (*)' if table<0 else schema['table_names_original'][table]+'.'+name
             for table,name in schema['column_names_original']]
    columns='\n'.join(f'{name} ({kind})' for name,kind in zip(choices,schema['column_types']))
    context='Question: '+row['question']+'\nDatabase columns:\n'+columns
    identity=text_key(row['db_id']+'\n'+row['question']);meta=dict(database_id=row['db_id'],query_sha256=text_key(row['query']))
    if projection=='aggregation':
        if aggregation not in range(6):raise ValueError('Unknown Spider aggregation')
        yield _row(context,('none','maximum','minimum','count','sum','average')[aggregation],identity,row['db_id'],**meta)
    elif not aggregation and column[1]>0 and len(choices)==len(set(choices)):
        # Only unaggregated, single-column outputs; COUNT(*) has equivalent column implementations.
        yield dict(inputs=context,choices_list=choices[1:],labels=column[1]-1,
                   metadata=json.dumps(dict(id=identity,split_group_id=row['db_id'],**meta),sort_keys=True))
    else:excluded['aggregated_or_wildcard_column']+=1


def spider():
    """Spider original parsed SQL and database schemas, single-output decision views.

    Source: https://huggingface.co/datasets/minktn/spider-data, original distribution
    https://github.com/taoyds/spider. Native 7,000-question train and database-disjoint
    dev are retained. No hidden test or WikiSQL-derived train_others is included.
    Gold SQL never enters inputs. Arithmetic/set operations and multi-output
    queries are excluded rather than choosing an arbitrary label. CC BY-SA 4.0.
    """
    repo='minktn/spider-data';files={};hashes={}
    for file in ('tables','train_spider','dev'):
        path=hf_hub_download(repo,'spider_data/'+file+'.json',repo_type='dataset',revision=SPIDER_REVISION)
        raw=Path(path).read_bytes();hashes[file]=hashlib.sha256(raw).hexdigest();files[file]=json.loads(raw)
    schemas={r['db_id']:r for r in files['tables']};output={};reports={}
    for projection in ('aggregation','column'):
        excluded=Counter();splits={}
        for split,file in [('train','train_spider'),('validation','dev')]:
            splits[split]=[r for row in files[file] for r in spider_rows(row,schemas[row['db_id']],projection,excluded)]
        reports[projection]=dict(source='https://github.com/taoyds/spider',hub_source=repo,revision=SPIDER_REVISION,
            source_hashes=hashes,excluded=dict(excluded),license='cc-by-sa-4.0',split_policy='native database partitions')
        output[projection]=_write('spider-'+projection,splits,reports[projection],mc=projection=='column')
    directory=Path('build/spider-release');directory.mkdir(parents=True,exist_ok=True)
    (directory/'provenance.json').write_text(json.dumps(dict(source='https://github.com/taoyds/spider',
        hub_source=repo,revision=SPIDER_REVISION,configs=reports),indent=2)+'\n')
    return output
