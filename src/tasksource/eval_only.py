"""Evaluation-only annotations, kept separate from the training catalogs.

These tasks are useful benchmarks rather than candidate training sources. Each
annotation has a REASONS entry and remains available through its normal loader:

    from tasksource import eval_only
    benchmark = eval_only.mmlu

They are not included in list_tasks(). Source splits and annotations are retained
as originally defined; benchmark-specific loader limitations still apply.
"""
from datasets import ClassLabel

from .access import parse_var_name
from .metadata import (bigbench_discriminative_english, blimp_hard as blimp_hard_configs,
                       imppres_presupposition, imppres_implicature)
from .metadata.configs import MMLU
from .preprocess import Preprocessing, Classification, MultipleChoice, cat, constant, get, name

# MMLU is an evaluation benchmark; no train split
mmlu = MultipleChoice('question',labels='answer',choices_list='choices',splits=['validation','dev','test'],
    dataset_name="tasksource/mmlu",
    config_name=MMLU
)

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

REASONS = {
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
