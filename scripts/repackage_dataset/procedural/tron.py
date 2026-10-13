"""Freeze pinned TRON generators into verified static multiple-choice examples."""
from collections import Counter
import importlib
import importlib.metadata
import io
import os
import random
import re
import sys

import numpy as np

from .common import checkout, seed_for, verified_choices, write_static

SOURCE = 'https://github.com/YangTianze009/TRON'
REVISION = '934f3928b7311ec70c080206d8199bbf3697b6c7'


def clean_question(question):
    # Remove explicit output-format clauses, never a question or domain rule.
    return re.sub(r'(?:^|(?<=[.!?])\s+)(?:'
                  r'Numeric (?:answer|value) in <answer>.*?|'
                  r'Place (?:the |your )?(?:numeric |final )?answer (?:in|inside) <answer>.*?|'
                  r'Answer with (?:a |the )?(?:single |one )?letter[^\n]*|'
                  r'(?:Pick the correct letter|Select the correct option)[^\n]*|'
                  r'Your answer should be one of the options letters only:[^\n]*|Think step by step[^\n]*)$',
                  '',question,flags=re.M).strip()


def native_mc(question, gold, verify):
    markers = list(re.finditer(r'^[ \t]*([A-Ha-h])[.):]\s*',question,re.M))
    if not markers:
        start = re.search(r'^\(A\)',question,re.M)
        if start:
            markers = list(re.compile(r'\(([A-Ha-h])\)\s*').finditer(question,start.start()))
    letters = [m[1].upper() for m in markers]
    if not markers or letters != list('ABCDEFGH'[:len(markers)]) or gold.upper() not in letters:
        raise ValueError('Native letter answer without explicit unique text choices')
    choices = [clean_question(question[m.end():markers[i+1].start() if i+1<len(markers) else None]).strip()
               for i,m in enumerate(markers)]
    if len(choices)<2 or any(not c for c in choices) or len({c.casefold() for c in choices})!=len(choices):
        raise ValueError('Invalid native choices')
    scores = [verify(letter) for letter in letters]
    label = letters.index(gold.upper())
    if scores != [int(i==label) for i in range(len(choices))]:raise ValueError('Ambiguous native gold')
    return clean_question(question[:markers[0].start()]),choices,label,dict(native_option_keys=letters,verifier_scores=scores)


def tron_mc(max_rows=5000, max_rows_eval=100, seed=42, task_ids=(), levels=(0,3,6,9), pool_size=12):
    """TRON static MC: native text options or verifier-rejected same-environment answers.

    Fresh, deterministic train/validation seeds; no model-derived labels or cross-task
    distractors. Generation failures, opaque image-choice tasks and insufficient
    distractor diversity are reported. Original generated PNGs are preserved in grouped
    image rows. Source: https://github.com/YangTianze009/TRON (research-use terms).
    """
    root, archive_sha = checkout('YangTianze009/TRON',REVISION)
    sys.path.insert(0,str(root))
    module = importlib.import_module('environments')
    if not str(module.__file__).startswith(str(root)):raise ValueError('A different environments package is loaded')
    names = sorted(task_ids or module.ENVIRONMENTS)
    if set(names)-module.ENVIRONMENTS.keys():raise ValueError('Unknown TRON environment')
    if not levels or any(level not in range(10) for level in levels):raise ValueError('TRON levels must be 0–9')
    os.environ['RLVE_AUGMENT']='0';os.environ['RLVE_MAX_IMAGE_PIXELS']='0'
    def generate(task, split, index, purpose):
        instance_seed = seed_for('tron',seed,task,split,index,purpose)
        random.seed(instance_seed);np.random.seed(instance_seed % 2**32)
        env = module.ENVIRONMENTS[task]()
        if not env.generate(instance_seed,{'level':levels[index%len(levels)]}):raise ValueError('Generator returned failure')
        return env,instance_seed
    splits,excluded = {},[]
    for split,limit in (('validation',max_rows_eval),('train',max_rows)):
        rows = [];splits[split] = rows
        for task_index,task in enumerate(names):
            quota = limit//len(names)+int(task_index<limit%len(names))
            donors = None
            for index in range(quota):
                try:
                    env,instance_seed = generate(task,split,index,'example')
                    gold,original = str(env._answer).strip(),env.get_instruction()
                    def verify(answer):
                        wrapped = (f'<answer>{answer}</answer>',f'\\boxed{{{answer}}}',f'Final answer: {answer}')[instance_seed%3]
                        result = env.verify(wrapped)
                        if result['format_score']!=1:raise ValueError('Verifier could not parse a choice')
                        return result['accuracy']
                    if len(gold)==1 and gold.upper() in 'ABCDEFGH':
                        question,choices,label,alignment = native_mc(original,gold,verify)
                        method = 'native_text_choices'
                    else:
                        if donors is None:
                            donors = []
                            for j in range(pool_size):
                                other,other_seed = generate(task,split,j,'distractor_pool')
                                answer = str(other._answer).strip()
                                if len(answer)==1 and answer.upper() in 'ABCDEFGH':continue
                                donors.append((answer,f'{task}:{other_seed}'))
                        choices,label,alignment = verified_choices(gold,donors,verify,instance_seed)
                        question,method = clean_question(original),'same_task_generated_answers'
                    if not question:raise ValueError('Empty question')
                    if any(token in text for text in [question,*choices] for token in ('<answer>','\\boxed','Final answer:')):
                        raise ValueError('Unrecognized output-format instruction')
                    image = env.render();buffer = io.BytesIO();image.save(buffer,format='PNG')
                    rows.append(dict(id=f'tron:{task}:{instance_seed}',image=buffer.getvalue(),inputs=question,
                        choices_list=choices,labels=label,metadata=dict(task=task,bucket=module.BUCKET[task],
                            seed=instance_seed,level=env.parameter['level'],source_revision=REVISION,
                            original_question=original,source_answer=gold,choice_method=method,**alignment)))
                except Exception as error:
                    excluded.append(dict(task=task,split=split,index=index,reason=f'{type(error).__name__}: {error}'))
            print('TRON',split,task,len(rows),flush=True)
    report = dict(source=SOURCE,revision=REVISION,source_archive_sha256=archive_sha,license='research use',license_url=SOURCE+'#license',
        configuration=dict(seed=seed,task_ids=names,levels=list(levels),pool_size=pool_size,max_rows=max_rows,max_rows_eval=max_rows_eval),
        split_policy='Disjoint generated seed namespaces; held-out image copies excluded from training.',
        annotation_origin='procedural',generation_exclusions=excluded,
        exclusion_counts=dict(Counter(r['reason'] for r in excluded)),
        rendering_versions={p:importlib.metadata.version(p) for p in ('Pillow','matplotlib','numpy')},
        generator_environment={k:v for k,v in os.environ.items() if k.startswith(('RLVE_','STANDALONE_'))})
    return write_static('tron-mcq',splits,report,tron_mc)
