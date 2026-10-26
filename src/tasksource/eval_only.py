"""Evaluation-only annotations, kept separate from the training catalogs.

These tasks are useful benchmarks rather than candidate training sources. Each
annotation has a REASONS entry and remains available through its normal loader:

    from tasksource import eval_only
    benchmark = eval_only.mmlu

Retrieval bundles are available through load_bright(), load_followir() and
load_trec_rag_2025(); they keep queries, corpora/candidates and native qrels separate.

They are not included in list_tasks(). Source splits and annotations are retained
as originally defined; benchmark-specific loader limitations still apply.
"""
import re
import json
import string

from datasets import ClassLabel, Dataset, DatasetDict, Features, Value, load_dataset

from .access import parse_var_name
from .metadata import (bigbench_discriminative_english, blimp_hard as blimp_hard_configs,
                       imppres_presupposition, imppres_implicature)
from .metadata.configs import MMLU
from .preprocess import Preprocessing, Classification, MultipleChoice, cat, constant, get, name
from .preprocess import ExtractiveQA
from .metadata.source_revisions import SOURCE_REVISIONS


def _mrqa_answer_key(text):
    # Official MRQA/SQuAD comparison: uncased, no ASCII punctuation/articles,
    # collapsed whitespace. This verifies a supplied occurrence; it never
    # searches for or relocates one.
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def _mrqa_eval_answers(row):
    """Verify gold occurrences under MRQA's documented answer normalization.

    Ends are inclusive. Return the exact source slice, not the normalized
    answer string. Reject occurrences whose normalized gold text disagrees.
    """
    detected = row['detected_answers']
    if len(detected['text']) != len(detected['char_spans']):
        raise ValueError('MRQA answer/span lengths differ')
    texts, starts, dropped = [], [], 0
    for text, spans in zip(detected['text'], detected['char_spans']):
        if len(spans['start']) != len(spans['end']):
            raise ValueError('MRQA start/end lengths differ')
        for start, end in zip(spans['start'], spans['end']):
            mention = row['context'][start:end + 1]
            if (0 <= start <= end < len(row['context']) and mention.strip()
                    and _mrqa_answer_key(mention) == _mrqa_answer_key(text)):
                texts.append(mention); starts.append(start)
            else:
                dropped += 1
    return {'checked_answers': {'text': texts, 'answer_start': starts}, 'dropped_occurrences': dropped}


def _mrqa_eval_split(dataset, subset):
    rows = dataset['test'].filter(lambda value: value == subset, input_columns='subset')
    checked = rows.map(_mrqa_eval_answers)
    kept = checked.filter(lambda row: bool(row['checked_answers']['text']))
    print(f'MRQA {subset}: excluded {len(rows) - len(kept)} unlocated positive rows; '
          f'{sum(checked["dropped_occurrences"])} mismatched occurrences')
    return type(dataset)(test=kept)


def _mrqa_eval(subset):
    # As for PROST below, the loader's train slot is an evaluation interface.
    # Select ONLY the source test split, never the in-domain train/dev rows.
    return ExtractiveQA(dataset_name='mrqa-workshop/mrqa', task_id=f'mrqa-eval/{subset}',
        splits=('test', None, None), complete_splits=False,
        pre_process=lambda ds: _mrqa_eval_split(ds, subset), answers='checked_answers',
        metadata=lambda row: {'id': row['qid'], 'mrqa_subset': row['subset'], 'source_split': 'test',
                              'dropped_occurrences': row['dropped_occurrences']},
        load_dataset_kwargs={'revision': SOURCE_REVISIONS['mrqa-workshop/mrqa'],
                             'data_files': {'test': 'plain_text/test-*.parquet'},
                             'verification_mode': 'no_checks'})


# AbstRCT (Mayer, Cabrio and Villata, ECAI 2020) cross-domain transfer test sets: no training abstracts.
# Same tasksource/abstrct-spans mirror and schema as abstrct_neoplasm in extractive_tasks.py.
abstrct_glaucoma = ExtractiveQA(dataset_name='tasksource/abstrct-spans', config_name='glaucoma',
    task_id='abstrct-eval/glaucoma', splits=('test', None, None), complete_splits=False,
    metadata='metadata', max_mismatch=0,
    load_dataset_kwargs={'revision': SOURCE_REVISIONS['tasksource/abstrct-spans']})
abstrct_mixed = ExtractiveQA(dataset_name='tasksource/abstrct-spans', config_name='mixed',
    task_id='abstrct-eval/mixed', splits=('test', None, None), complete_splits=False,
    metadata='metadata', max_mismatch=0,
    load_dataset_kwargs={'revision': SOURCE_REVISIONS['tasksource/abstrct-spans']})

mrqa_bioasq = _mrqa_eval('BioASQ')
mrqa_drop = _mrqa_eval('DROP')
mrqa_duorc = _mrqa_eval('DuoRC.ParaphraseRC')
mrqa_race = _mrqa_eval('RACE')
mrqa_relation_extraction = _mrqa_eval('RelationExtraction')
mrqa_textbookqa = _mrqa_eval('TextbookQA')

# MMLU is an evaluation benchmark; no train split
mmlu = MultipleChoice('question',labels='answer',choices_list='choices',splits=['validation','dev','test'],
    dataset_name="tasksource/mmlu",
    config_name=MMLU
)

# PROST explicitly provides test examples only; keep it out of training discovery.
# The annotation's internal train slot is a benchmark interface, not source training data.
prost = MultipleChoice(cat(['context','ex_question']), choices=['A','B','C','D'], labels='label',
    dataset_name='corypaik/prost', load_dataset_kwargs={'revision':'refs/convert/parquet'},
    splits=['test',None,None], complete_splits=False)

# QA4MRE main tasks (CLEF 2011-2013): the campaigns' test questions with gold answers; no training questions exist.
_QA4MRE_REVISION = 'a6ba3ed66833455732993c7d2daeb2f397785be0'


def _qa4mre_rows(dataset):
    def options(row):
        ids, texts = row['answer_options']['answer_id'], row['answer_options']['answer_str']
        return {'options': texts, 'label': ids.index(row['correct_answer_id'])}
    return dataset.map(options)


def _qa4mre(year):
    return MultipleChoice(cat(['document_str', 'question_str']), choices_list='options', labels='label',
        dataset_name='community-datasets/qa4mre', config_name=f'{year}.main.EN', task_id=f'qa4mre/{year}',
        load_dataset_kwargs={'revision': _QA4MRE_REVISION}, pre_process=_qa4mre_rows, preserve_choices=True,
        splits=['train', None, None], complete_splits=False)


qa4mre_2011 = _qa4mre(2011)
qa4mre_2012 = _qa4mre(2012)
qa4mre_2013 = _qa4mre(2013)

# BLiMP is an evaluation benchmark (test-only minimal pairs)
blimp_hard = MultipleChoice(inputs=constant(''),
    choices=['sentence_good','sentence_bad'],
    labels=constant(0),
    dataset_name="blimp",
    config_name=blimp_hard_configs # tasks where GPT2 is at least 10% below  human accuracy
)

# BIG-bench is an evaluation suite
bigbench = MultipleChoice(
    'inputs',
    choices_list='multiple_choice_targets',
    labels=lambda x:x['multiple_choice_scores'].index(1) if 1 in x['multiple_choice_scores'] else -1,
    dataset_name='tasksource/bigbench',
    config_name=bigbench_discriminative_english - {"social_i_qa","intersect_geometry"} # english multiple choice tasks, minus duplicates
)

# test-only diagnostic set, labels masked
glue___ax = Classification(sentence1="premise", sentence2="hypothesis", labels="label", splits=["test", None, None])

# diagnostic benchmark
glue__diagnostics = Classification("premise","hypothesis","label",
    dataset_name="pietrolesci/glue_diagnostics",splits=["test",None,None])

# adversarial evaluation benchmark, validation only
adv_glue___adv_sst2 = Classification(sentence1="sentence", labels="label", splits=["validation", None, None])

adv_glue___adv_qqp = Classification(sentence1="question1", sentence2="question2", labels="label", splits=["validation", None, None])

adv_glue___adv_mnli = Classification(sentence1="premise", sentence2="hypothesis", labels="label", splits=["validation", None, None])

adv_glue___adv_mnli_mismatched = Classification(sentence1="premise", sentence2="hypothesis", labels="label", splits=["validation", None, None])

adv_glue___adv_qnli = Classification(sentence1="question", labels="label", splits=["validation", None, None])

adv_glue___adv_rte = Classification(sentence1="sentence1", sentence2="sentence2", labels="label", splits=["validation", None, None])

# MBIB is an evaluation benchmark
mbib_cognitive_bias = Classification('text',labels=name('label',['not cognitive-bias','cognitive-bias']), dataset_name='mediabiasgroup/mbib-base', config_name='cognitive-bias')

mbib_fake_news = Classification('text',labels=name('label',['not fake-news','fake-news']), dataset_name='mediabiasgroup/mbib-base', config_name='fake-news')

mbib_gender_bias = Classification('text',labels=name('label',['not gender-bias','gender-bias']), dataset_name='mediabiasgroup/mbib-base', config_name='gender-bias')

mbib_hate_speech = Classification('text',labels=name('label',['not hate-speech','hate-speech']), dataset_name='mediabiasgroup/mbib-base', config_name='hate-speech')

mbib_linguistic_bias = Classification('text',labels=name('label',['not linguistic-bias','linguistic-bias']), dataset_name='mediabiasgroup/mbib-base', config_name='linguistic-bias')

mbib_political_bias = Classification('text',labels=name('label',['not political-bias','political-bias']), dataset_name='mediabiasgroup/mbib-base', config_name='political-bias')

mbib_racial_bias = Classification('text',labels=name('label',['not racial-bias','racial-bias']), dataset_name='mediabiasgroup/mbib-base', config_name='racial-bias')

mbib_text_level_bias = Classification('text',labels=name('label',['not text-level-bias','text-level-bias']), dataset_name='mediabiasgroup/mbib-base', config_name='text-level-bias')

# SuperTweetEval is an evaluation benchmark
ste_wic = Classification(cat("text_1","text_2"),
    lambda x:f"{x['target']} means the same thing in these texts",
    "gold_label_binary",
    dataset_name="cardiffnlp/super_tweeteval", config_name="tempo_wic",splits=['train','validation',None])

ste_nerd = Classification("text",
    lambda x:f"definition of {x['target']} here is 'x{['definition']}'",
    "gold_label_binary",
    dataset_name="cardiffnlp/super_tweeteval", config_name="tweet_nerd",splits=['train','validation',None])

ste_sim = Classification("text_1","text_2",lambda x:x['gold_score']/5,
    dataset_name="cardiffnlp/super_tweeteval",config_name="tweet_similarity",splits=['train','validation',None])

ste_intimacy = Classification("text_1",labels=lambda x:x['gold_score']/5,
    dataset_name="cardiffnlp/super_tweeteval",config_name="tweet_intimacy")

# Inverse Scaling Prize sets are evaluation probes with few-shot prompts baked in
neqa = MultipleChoice('prompt',choices_list='classes',labels="answer_index",
    dataset_name="inverse-scaling/NeQA")

quote_repetition = MultipleChoice('prompt',choices_list='classes',labels="answer_index",
    dataset_name="inverse-scaling/quote-repetition")

redefine_math = MultipleChoice('prompt',choices_list='classes',labels="answer_index",
    dataset_name="inverse-scaling/redefine-math")

# model-written-evals and TruthfulQA are evaluation suites
model_written_evals = MultipleChoice('question', choices_list=lambda x: [x['answer_matching_behavior'].strip(), x['answer_not_matching_behavior'].strip()], labels=constant(0),  
    dataset_name="Anthropic/model-written-evals")

truthful_qa___multiple_choice = MultipleChoice(
    "question",
    choices_list=get.mc1_targets.choices,
    labels=constant(0)
)

# ACES is a challenge set for evaluating translation metrics
aces_ranking = MultipleChoice("source",choices=['good-translation','incorrect-translation'],labels=constant(0), question="Which is the correct translation?", dataset_name='nikitam/ACES', config_name='ACES', task_id='ACES/ranking')

def _aces_phenomena_labels(dataset):
    # The catalog samples before fixing string labels; build the ontology from
    # the full source so rare phenomena in dev/test are not silently invalid.
    names = sorted(set(dataset["train"]["phenomena"]))
    return dataset.cast_column("phenomena", ClassLabel(names=names))

aces_phenomena = Classification('source','incorrect-translation','phenomena',
    dataset_name='nikitam/ACES', config_name='ACES',
    task_id='ACES/phenomena', pre_process=_aces_phenomena_labels)

# IMPPRES is a diagnostic set for NLI pragmatics; its only source split is labeled train.
def _imppres_post_process(ds,prefix=''):
    # imppres entailment definition is either purely semantic or purely pragmatic
    # because of that, we assign differentiate the labels from anli/mnli notation
    return ds.cast_column('labels', ClassLabel(
    names=[f'{prefix}_entailment',f'{prefix}_neutral',f'{prefix}_contradiction']))

imppres__presupposition = Classification("premise","hypothesis","gold_label",
    dataset_name="tasksource/imppres", config_name=imppres_presupposition,
    post_process=lambda x: _imppres_post_process(x,'presupposition'))

imppres__prag = Classification("premise","hypothesis","gold_label_prag",
    dataset_name="tasksource/imppres", config_name=imppres_implicature,
    post_process=lambda x: _imppres_post_process(x,'pragmatic'))

imppres__log = Classification("premise","hypothesis","gold_label_log",
    dataset_name="tasksource/imppres", config_name=imppres_implicature,
    post_process=lambda x: _imppres_post_process(x,'logical'))

def _cladder_context(x):
    """The causal graph decides the answer, but tasksource/cladder keeps it in `reasoning`."""
    return f'{x["reasoning"]["step0"]} Causal graph: {x["reasoning"]["step1"]}.\n{x["given_info"]}'

cladder = Classification(_cladder_context, "question", "answer", dataset_name="tasksource/cladder",
    # backdoor-adjustment rows carry no graph at all: identical texts get opposite answers
    pre_process=lambda ds: ds.filter(lambda x: bool(x["reasoning"]["step1"])))


# Retrieval benchmarks retain their native corpus and judgments: no synthetic negatives.
def load_bright(domain='biology'):
    """BRIGHT evaluation queries (with gold/excluded IDs) and the matching document corpus."""
    repo='xlangai/BRIGHT'
    return DatasetDict({key:load_dataset(repo,config,revision=SOURCE_REVISIONS[repo],
        data_files={domain:f'{config}/{domain}-*.parquet'},split=domain,
        verification_mode='no_checks')  # The source card describes all twelve domain splits.
        for key,config in (('queries','examples'),('corpus','documents'))})


def load_followir(subset='robust04'):
    """Human FollowIR evaluation, including original/changed instructions and qrels.

    Scores retain their native grades (News21 is not binary). No synthetic train data.
    """
    if subset not in ('robust04','core17','news21'):
        raise ValueError('FollowIR subset must be robust04, core17 or news21')
    repo=f'jhu-clsp/{subset}-instructions'
    return DatasetDict({config:load_dataset(repo,config,revision=SOURCE_REVISIONS[repo],split=split)
        for config,split in (('queries','queries'),('corpus','corpus'),('qrels_og','test'),
                            ('qrels_changed','test'),('top_ranked','top_ranked'))})


def load_trec_rag_2025():
    """Official top-100 requests and manual 0–4 test qrels; unjudged candidates stay unjudged."""
    candidates=SOURCE_REVISIONS['github:castorini/ragnarok_data']
    judgments=SOURCE_REVISIONS['github:castorini/anserini-tools']
    return DatasetDict(
        requests=load_dataset('json',split='test',data_files={'test':
            f'https://media.githubusercontent.com/media/castorini/ragnarok_data/{candidates}/rag25/'
            'retrieve_results/MISC/retrieve_results_rankqwen3_32b.rag25_top100.jsonl'}),
        qrels=load_dataset('csv',split='test',delimiter=' ',
            column_names=['query_id','iteration','document_id','grade'],
            features=Features({k:Value('int64' if k=='grade' else 'string')
                               for k in ('query_id','iteration','document_id','grade')}),
            data_files={'test':f'https://raw.githubusercontent.com/castorini/anserini-tools/{judgments}/qrels/qrels.rag25.test.txt'}))


GARAGE_LABELS = ('ANSWER-THE-QUESTION','RELATED-INFORMATION','UNKNOWN','OUTDATED')


def _garage_records(row):
    columns=('evidence_correct','evidence_relevant','evidence_cited')
    if any(len(row[k])!=len(row['grounding']) for k in columns):
        raise ValueError(f"GaRAGe passage/annotation length mismatch: {row['sample_id']}")
    for i,passage in enumerate(row['grounding']):
        label=row['evidence_correct'][i]
        if not label:continue  # Blank annotations are not negative judgments.
        if label not in GARAGE_LABELS:raise ValueError(f'Unknown GaRAGe evidence type: {label}')
        citations=[(k,v) for k,v in passage.items() if k.startswith('cite_') and v is not None]
        if len(citations)!=1 or not citations[0][1].strip():
            raise ValueError(f"GaRAGe requires exactly one passage text: {row['sample_id']}:{i}")
        citation,text=citations[0]
        metadata=dict(id=f"{row['sample_id']}:{i}",sample_id=row['sample_id'],passage_index=i,
            split_group_id=row['sample_id'],source_split='train',citation=citation,
            **{k:row[k][i] for k in ('evidence_relevant','evidence_cited')},
            passage_date=passage.get('date'),passage_age=passage.get('age'),provider=passage.get('provider'),
            **{k:v for k,v in row.items() if k.startswith('question_') and k!='question'})
        date=passage.get('date')
        context=f"Question date: {row['question_date']}\nQuestion: {row['question']}"
        if date and date!='not defined':context+=f"\nPassage date: {date}"
        yield dict(sentence1=context+f'\nPassage: {text.strip()}',labels=label,
            metadata=json.dumps(metadata,ensure_ascii=False,sort_keys=True))


def _garage_rows(dataset):
    features=Features({k:Value('string') for k in ('sentence1','labels','metadata')})
    return DatasetDict({split:Dataset.from_list([r for row in rows for r in _garage_records(row)],features=features)
                        for split,rows in dataset.items()})


garage__answer_support = Classification('sentence1',labels=lambda r:'yes' if r['labels']=='ANSWER-THE-QUESTION' else 'other evidence',
    dataset_name='AmazonScience/GaRAGe',task_id='garage/answer-support',complete_splits=False,
    pre_process=_garage_rows,metadata='metadata',label_values={'yes':'yes','other evidence':'other evidence'},
    question='Is this passage annotated as answering the question, rather than related, unknown or outdated evidence?',
    load_dataset_kwargs={'revision':SOURCE_REVISIONS['AmazonScience/GaRAGe']})
garage__evidence_type = Classification('sentence1',labels='labels',
    dataset_name='AmazonScience/GaRAGe',task_id='garage/evidence-type',complete_splits=False,
    pre_process=_garage_rows,metadata='metadata',label_values={label:label for label in GARAGE_LABELS},
    question='Classify the passage as answering the question, related information, unknown, or outdated at the question date.',
    load_dataset_kwargs={'revision':SOURCE_REVISIONS['AmazonScience/GaRAGe']})

REASONS = {
    'garage__answer_support': 'GaRAGe is a human-annotated RAG evaluation benchmark; the Hub train split is not a designed training partition',
    'garage__evidence_type': 'GaRAGe is a human-annotated RAG evaluation benchmark; the Hub train split is not a designed training partition',
    'abstrct_glaucoma': 'AbstRCT cross-domain transfer test set (Mayer et al. train on neoplasm only); no training abstracts',
    'abstrct_mixed': 'AbstRCT cross-domain transfer test set (Mayer et al. train on neoplasm only); no training abstracts',
    **{f'mrqa_{name}': 'MRQA out-of-domain subset has source test rows only; no genuine training split'
       for name in ('bioasq', 'drop', 'duorc', 'race', 'relation_extraction', 'textbookqa')},
    **{f'qa4mre_{year}': 'QA4MRE main-task questions are CLEF campaign test sets (120-284 per year); no training questions'
       for year in (2011, 2012, 2013)},
    'prost': 'PROST is a test-only zero-shot physical-reasoning probe; its publisher explicitly provides no training data',
    'cladder': 'CLadder is a causal-reasoning evaluation benchmark, not a training source',
    'mmlu': 'MMLU is an evaluation benchmark; no train split',
    'blimp_hard': 'BLiMP is an evaluation benchmark (test-only minimal pairs)',
    'bigbench': 'BIG-bench is an evaluation suite',
    'glue___ax': 'test-only diagnostic set, labels masked',
    'glue__diagnostics': 'diagnostic benchmark',
    'adv_glue___adv_sst2': 'adversarial evaluation benchmark, validation only',
    'adv_glue___adv_qqp': 'adversarial evaluation benchmark, validation only',
    'adv_glue___adv_mnli': 'adversarial evaluation benchmark, validation only',
    'adv_glue___adv_mnli_mismatched': 'adversarial evaluation benchmark, validation only',
    'adv_glue___adv_qnli': 'adversarial evaluation benchmark, validation only',
    'adv_glue___adv_rte': 'adversarial evaluation benchmark, validation only',
    'mbib_cognitive_bias': 'MBIB is an evaluation benchmark',
    'mbib_fake_news': 'MBIB is an evaluation benchmark',
    'mbib_gender_bias': 'MBIB is an evaluation benchmark',
    'mbib_hate_speech': 'MBIB is an evaluation benchmark',
    'mbib_linguistic_bias': 'MBIB is an evaluation benchmark',
    'mbib_political_bias': 'MBIB is an evaluation benchmark',
    'mbib_racial_bias': 'MBIB is an evaluation benchmark',
    'mbib_text_level_bias': 'MBIB is an evaluation benchmark',
    'ste_wic': 'SuperTweetEval is an evaluation benchmark',
    'ste_nerd': 'SuperTweetEval is an evaluation benchmark',
    'ste_sim': 'SuperTweetEval is an evaluation benchmark',
    'ste_intimacy': 'SuperTweetEval is an evaluation benchmark',
    'model_written_evals': "persona evaluations; the gold 'matching behavior' is often the undesired one (power-seeking, sycophancy)",
    'truthful_qa___multiple_choice': 'TruthfulQA is an evaluation benchmark; validation split only',
    'neqa': 'Inverse Scaling Prize evaluation probe; few-shot prompt baked into the inputs',
    'quote_repetition': 'Inverse Scaling Prize evaluation probe; few-shot prompt baked into the inputs',
    'redefine_math': 'Inverse Scaling Prize evaluation probe; "Q: ... A:" prompt baked into the inputs',
    'aces_ranking': 'ACES is a challenge set for evaluating translation metrics',
    'aces_phenomena': 'ACES is a challenge set for evaluating translation metrics',
    'imppres__presupposition': 'IMPPRES is a diagnostic set for NLI pragmatics, not training data',
    'imppres__prag': 'IMPPRES is a diagnostic set for NLI pragmatics, not training data',
    'imppres__log': 'IMPPRES is a diagnostic set for NLI pragmatics, not training data',
}

# Considered evaluation sources without an annotation yet.
NOT_ANNOTATED = {
    "demelin/wino_x": "machine translation evaluation set; script-only loader",
}

# Unset names default from the variable name, as in list_tasks.
for _key, _task in list(globals().items()):
    if isinstance(_task, Preprocessing):
        _dataset, _config, _ = parse_var_name(_key.strip())
        _task.dataset_name = _task.dataset_name or _dataset
        _task.config_name = _task.config_name or _config
