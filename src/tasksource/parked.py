"""Parked tasks: annotations kept out of the task list, each with its reason.

Tasksource is for training, so evaluation benchmarks, duplicates and unsound
or unloadable sources are parked here instead of being deleted. Each entry in
PARKED has a kind (see KINDS) and a reason. Parked tasks are not listed by
``list_tasks`` but still load directly:

    from tasksource import parked
    dataset = parked.glue___ax.load()
    benchmarks = parked.by_kind("evaluation")  # {name: task}

To revive one, move it back to tasks.py and drop its PARKED entry.
"""
from datasets import get_dataset_config_names

from .access import parse_var_name
from .metadata import bigbench_discriminative_english, blimp_hard as blimp_hard_configs
from .preprocess import Preprocessing, cat, constant, get, regen, name, Classification, TokenClassification, MultipleChoice

# MMLU is an evaluation benchmark; no train split
mmlu = MultipleChoice('question',labels='answer',choices_list='choices',splits=['validation','dev','test'],
    dataset_name="tasksource/mmlu",
    config_name=get_dataset_config_names("tasksource/mmlu")
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

# duplicate of glue/rte
super_glue___rte = Classification(sentence1="premise", sentence2="hypothesis", labels="label")

# same pairs as sick/entailment_AB
sick__entailment_BA = Classification('sentence_A','sentence_B','entailment_BA')

# generated labels not sound enough
gpt3_nli = Classification("text_a","text_b","label",dataset_name="pietrolesci/gpt3_nli")

# overlaps FEVER-based NLI tasks
enfever_nli = Classification("evidence","claim","label", dataset_name="ctu-aic/enfever_nli")

# diagnostic benchmark
glue__diagnostics = Classification("premise","hypothesis","label",
    dataset_name="pietrolesci/glue_diagnostics",splits=["test",None,None])

# silver labels; paws/labeled_final is used
paws___unlabeled_final = Classification("sentence1", "sentence2", "label")

# duplicate of glue/qqp
quora = Classification(get.questions.text[0], get.questions.text[1], 'is_duplicate')

# not loadable
tner___tweebank_ner    = TokenClassification(tokens="tokens", labels="tags")

# covered by math_qa
aqua_rat___tokenized = MultipleChoice("question",choices_list="options",labels=lambda x:"ABCDE".index(x['correct']))

# claim-only FEVER; the verdict needs evidence
fever___v1_0 = Classification(sentence1="claim", labels="label", splits=["train", "paper_dev", "paper_test"], dataset_name="fever", config_name="v1.0")
fever___v2_0 = Classification(sentence1="claim", labels="label", splits=[None, "validation", None], dataset_name="fever", config_name="v2.0")

# duplicate of glue/mnli
multi_nli = Classification(sentence1="premise", sentence2="hypothesis", labels="label", splits=["train", "validation_matched", None]) #glue

# covered by hyperpartisan_news
hyperpartisan_news_detection___byarticle = Classification(sentence1="text", labels="hyperpartisan", splits=["train", None, None])
hyperpartisan_news_detection___bypublisher = Classification(sentence1="text", labels="hyperpartisan", splits=["train","validation", None])

# covered by go_emotions/simplified
go_emotions___raw = Classification(sentence1="text", splits=["train", None, None])

# duplicate of super_glue/boolq
boolq = Classification(sentence1="question", splits=["train", "validation", None])

# adversarial evaluation benchmark, validation only
adv_glue___adv_sst2 = Classification(sentence1="sentence", labels="label", splits=["validation", None, None])
adv_glue___adv_qqp = Classification(sentence1="question1", sentence2="question2", labels="label", splits=["validation", None, None])
adv_glue___adv_mnli = Classification(sentence1="premise", sentence2="hypothesis", labels="label", splits=["validation", None, None])
adv_glue___adv_mnli_mismatched = Classification(sentence1="premise", sentence2="hypothesis", labels="label", splits=["validation", None, None])
adv_glue___adv_qnli = Classification(sentence1="question", labels="label", splits=["validation", None, None])
adv_glue___adv_rte = Classification(sentence1="sentence1", sentence2="sentence2", labels="label", splits=["validation", None, None])

# missing files
species_800 = TokenClassification(tokens="tokens", labels="ner_tags", config_name=["species_800"])

# horoscope is not predictable from text
blog_authorship_corpus__horoscope = Classification(sentence1="text",labels="horoscope")

# in bigbench, too heavy (100GB)
code_x_glue_cc_clone_detection_big_clone_bench = Classification("func1", "func2", "label")

# constant label, not a real task
code_x_glue_cc_code_refinement = MultipleChoice(
    constant(""), choices=["buggy","fixed"], labels=constant(0),
    config_name="medium")

# every option is a valid answer; gold is only the most popular
proto_qa = MultipleChoice(
    "question",
    choices_list=lambda x:x['answer-clusters']['answers'],
    labels=lambda x: x['answer-clusters']['count'].index(max(x['answer-clusters']['count'])),
    config_name='proto_qa'
)

# HC3 human answers are PTB-tokenized (a trivial shortcut); script-only loader
def _preprocess_chatgpt_detection(ex):
     import random
     label=random.random()<0.5
     ex['label']=int(label)
     ex['answer']=[str(ex['human_answers'][0]),str(ex['chatgpt_answers'][0])][label]
     return ex
chatgpt_detection = Classification("question","answer","label",
    dataset_name = 'Hello-SimpleAI/HC3', config_name="all",
    pre_process=lambda dataset:dataset.map(_preprocess_chatgpt_detection))


# unclear label semantics
attempto_nli = Classification("premise","hypothesis",
    lambda x:f'race-{x["race_label"]}',
    dataset_name="sileod/attempto-nli")

# regression target; acceptability is covered by other tasks
mega_acceptability = Classification("sentence",labels="average",
    dataset_name='tasksource/mega-acceptability-v2')

# MBIB is an evaluation benchmark
mbib_cognitive_bias = Classification('text',labels=name('label',['not cognitive-bias','cognitive-bias']), dataset_name='mediabiasgroup/mbib-base', config_name='cognitive-bias')
mbib_fake_news = Classification('text',labels=name('label',['not fake-news','fake-news']), dataset_name='mediabiasgroup/mbib-base', config_name='fake-news')
mbib_gender_bias = Classification('text',labels=name('label',['not gender-bias','gender-bias']), dataset_name='mediabiasgroup/mbib-base', config_name='gender-bias')
mbib_hate_speech = Classification('text',labels=name('label',['not hate-speech','hate-speech']), dataset_name='mediabiasgroup/mbib-base', config_name='hate-speech')
mbib_linguistic_bias = Classification('text',labels=name('label',['not linguistic-bias','linguistic-bias']), dataset_name='mediabiasgroup/mbib-base', config_name='linguistic-bias')
mbib_political_bias = Classification('text',labels=name('label',['not political-bias','political-bias']), dataset_name='mediabiasgroup/mbib-base', config_name='political-bias')
mbib_racial_bias = Classification('text',labels=name('label',['not racial-bias','racial-bias']), dataset_name='mediabiasgroup/mbib-base', config_name='racial-bias')
mbib_text_level_bias = Classification('text',labels=name('label',['not text-level-bias','text-level-bias']), dataset_name='mediabiasgroup/mbib-base', config_name='text-level-bias')

# summary-only, the source document is missing; script-only loader
xsum_factuality = Classification("summary",labels="is_factual")

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

# too long
lex_glue___ecthr_a = Classification(sentence1="text", labels="labels",dataset_name="coastalcph/lex_glue",config_name="ecthr_a")
lex_glue___ecthr_b = Classification(sentence1="text", labels="labels")

# merges of NLI tasks already included
nli_l2 = Classification("sentence1","sentence2","labels",
    dataset_name="tasksource/merged-2l-nli")
nli_l3 =  Classification("sentence1","sentence2","labels",
    dataset_name="tasksource/merged-3l-nli")

# too long
ecthr_cases___alleged_violation_prediction = Classification(labels="labels", dataset_name="ecthr_cases", config_name="alleged-violation-prediction")
ecthr_cases___violation_prediction = Classification(labels="labels", dataset_name="ecthr_cases", config_name="violation-prediction")

# source discontinued; see argument_feedback in tasks.py
effective_feedback_student_writing = Classification("discourse_text",
    labels="discourse_effectiveness", dataset_name="YaHi/EffectiveFeedbackStudentWriting")


# labels are model confidence scores, nearly all above 0.99
has_part = Classification("arg1","arg2", labels="score", splits=["train", None, None])

# label semantics (1-6) are undocumented
recast___recast_kg_relations = Classification(sentence1="context", sentence2="hypothesis", labels="label",
    dataset_name="tasksource/recast", config_name="recast_kg_relations")

# dependency relation labels depend on the head word, which the task does not show
from .multilingual_tasks import _udep_cast_label_sequence, all as _all_configs  # noqa: E402
udep__deprel_multilingual = TokenClassification('tokens', 'deprel',
    pre_process=lambda ds: _udep_cast_label_sequence(ds, 'deprel'),
    **_all_configs('universal-dependencies/universal_dependencies'))

# xglue's NER and POS configs repackage CoNLL-2002/2003 and Universal Dependencies (tasks listed
# elsewhere) and exist only behind a loading script
xglue__ner = TokenClassification("words", "ner", dataset_name="microsoft/xglue", config_name="ner")
xglue__pos = TokenClassification("words", "pos", dataset_name="microsoft/xglue", config_name="pos")

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
from datasets import ClassLabel  # noqa: E402
aces_ranking = MultipleChoice("source",choices=['good-translation','incorrect-translation'],labels=constant(0), question="Which is the correct translation?", dataset_name='nikitam/ACES', config_name='ACES', task_id='ACES/ranking')
def _aces_phenomena_labels(dataset):
    # The catalog samples before fixing string labels; build the ontology from
    # the full source so rare phenomena in dev/test are not silently invalid.
    names = sorted(set(dataset["train"]["phenomena"]))
    return dataset.cast_column("phenomena", ClassLabel(names=names))

aces_phenomena = Classification('source','incorrect-translation','phenomena',
    dataset_name='nikitam/ACES', config_name='ACES',
    task_id='ACES/phenomena', pre_process=_aces_phenomena_labels)

# the entailment direction of sick/label, which is listed
sick__entailment_AB = Classification('sentence_A','sentence_B','entailment_AB', dataset_name="tasksource/sick")

KINDS = {
    "evaluation": "evaluation benchmark: useful for evaluation, kept out of training",
    "duplicate": "duplicates or is covered by a listed task",
    "unsound": "labels or inputs do not support the task as annotated",
    "impractical": "inputs too long or data too heavy",
    "unavailable": "source no longer loads",
    "todo": "worth adding, not annotated yet",
}

PARKED = {
    'xglue__ner': ('duplicate', 'repackages CoNLL-2002/2003 NER, which conll2002 (es, nl) and conll2003 (en) cover; the German part is not openly licensed; script-only'),
    'xglue__pos': ('duplicate', 'repackages Universal Dependencies POS, which udep__pos covers; script-only'),
    'udep__deprel_multilingual': ('unsound', 'all-language variant of udep__deprel; relation labels depend on the head word, which the task does not show'),
    'has_part': ('unsound', 'labels are model confidence scores, nearly all above 0.99'),
    'recast___recast_kg_relations': ('unsound', 'label semantics (1-6) are undocumented'),
    'mmlu': ('evaluation', 'MMLU is an evaluation benchmark; no train split'),
    'blimp_hard': ('evaluation', 'BLiMP is an evaluation benchmark (test-only minimal pairs)'),
    'bigbench': ('evaluation', 'BIG-bench is an evaluation suite'),
    'glue___ax': ('evaluation', 'test-only diagnostic set, labels masked'),
    'super_glue___rte': ('duplicate', 'duplicate of glue/rte'),
    'sick__entailment_BA': ('duplicate', 'restates sick/label as B-to-A entailment'),
    'gpt3_nli': ('unsound', 'generated labels not sound enough'),
    'enfever_nli': ('duplicate', 'overlaps FEVER-based NLI tasks'),
    'glue__diagnostics': ('evaluation', 'diagnostic benchmark'),
    'paws___unlabeled_final': ('unsound', 'silver labels; paws/labeled_final is used'),
    'quora': ('duplicate', 'duplicate of glue/qqp'),
    'tner___tweebank_ner': ('unavailable', 'not loadable'),
    'aqua_rat___tokenized': ('duplicate', 'covered by math_qa'),
    'fever___v1_0': ('unsound', 'claim-only FEVER; the verdict needs evidence'),
    'fever___v2_0': ('unsound', 'claim-only FEVER; the verdict needs evidence'),
    'multi_nli': ('duplicate', 'duplicate of glue/mnli'),
    'hyperpartisan_news_detection___byarticle': ('duplicate', 'covered by hyperpartisan_news'),
    'hyperpartisan_news_detection___bypublisher': ('duplicate', 'covered by hyperpartisan_news'),
    'go_emotions___raw': ('duplicate', 'covered by go_emotions/simplified'),
    'boolq': ('duplicate', 'duplicate of super_glue/boolq'),
    'adv_glue___adv_sst2': ('evaluation', 'adversarial evaluation benchmark, validation only'),
    'adv_glue___adv_qqp': ('evaluation', 'adversarial evaluation benchmark, validation only'),
    'adv_glue___adv_mnli': ('evaluation', 'adversarial evaluation benchmark, validation only'),
    'adv_glue___adv_mnli_mismatched': ('evaluation', 'adversarial evaluation benchmark, validation only'),
    'adv_glue___adv_qnli': ('evaluation', 'adversarial evaluation benchmark, validation only'),
    'adv_glue___adv_rte': ('evaluation', 'adversarial evaluation benchmark, validation only'),
    'species_800': ('unavailable', 'missing files'),
    'blog_authorship_corpus__horoscope': ('unsound', 'horoscope is not predictable from text'),
    'code_x_glue_cc_clone_detection_big_clone_bench': ('impractical', 'in bigbench, too heavy (100GB)'),
    'code_x_glue_cc_code_refinement': ('unsound', 'constant label, not a real task'),
    'proto_qa': ('unsound', 'every option is a valid answer; gold is only the most popular'),
    'chatgpt_detection': ('unsound', 'HC3 human answers are PTB-tokenized (a trivial shortcut); script-only loader'),
    'attempto_nli': ('unsound', 'unclear label semantics'),
    'mega_acceptability': ('duplicate', 'regression target; acceptability is covered by other tasks'),
    'mbib_cognitive_bias': ('evaluation', 'MBIB is an evaluation benchmark'),
    'mbib_fake_news': ('evaluation', 'MBIB is an evaluation benchmark'),
    'mbib_gender_bias': ('evaluation', 'MBIB is an evaluation benchmark'),
    'mbib_hate_speech': ('evaluation', 'MBIB is an evaluation benchmark'),
    'mbib_linguistic_bias': ('evaluation', 'MBIB is an evaluation benchmark'),
    'mbib_political_bias': ('evaluation', 'MBIB is an evaluation benchmark'),
    'mbib_racial_bias': ('evaluation', 'MBIB is an evaluation benchmark'),
    'mbib_text_level_bias': ('evaluation', 'MBIB is an evaluation benchmark'),
    'xsum_factuality': ('unsound', 'summary-only, the source document is missing; script-only loader'),
    'ste_wic': ('evaluation', 'SuperTweetEval is an evaluation benchmark'),
    'ste_nerd': ('evaluation', 'SuperTweetEval is an evaluation benchmark'),
    'ste_sim': ('evaluation', 'SuperTweetEval is an evaluation benchmark'),
    'ste_intimacy': ('evaluation', 'SuperTweetEval is an evaluation benchmark'),
    'lex_glue___ecthr_a': ('impractical', 'too long'),
    'lex_glue___ecthr_b': ('impractical', 'too long'),
    'nli_l2': ('duplicate', 'merges of NLI tasks already included'),
    'nli_l3': ('duplicate', 'merges of NLI tasks already included'),
    'ecthr_cases___alleged_violation_prediction': ('impractical', 'too long'),
    'ecthr_cases___violation_prediction': ('impractical', 'too long'),
    'model_written_evals': ('evaluation', "persona evaluations; the gold 'matching behavior' is often the undesired one (power-seeking, sycophancy)"),
    'truthful_qa___multiple_choice': ('evaluation', 'TruthfulQA is an evaluation benchmark; validation split only'),
    'neqa': ('evaluation', 'Inverse Scaling Prize evaluation probe; few-shot prompt baked into the inputs'),
    'quote_repetition': ('evaluation', 'Inverse Scaling Prize evaluation probe; few-shot prompt baked into the inputs'),
    'redefine_math': ('evaluation', 'Inverse Scaling Prize evaluation probe; "Q: ... A:" prompt baked into the inputs'),
    'aces_ranking': ('evaluation', 'ACES is a challenge set for evaluating translation metrics'),
    'aces_phenomena': ('evaluation', 'ACES is a challenge set for evaluating translation metrics'),
    'sick__entailment_AB': ('duplicate', 'restates sick/label as A-to-B entailment'),
    'effective_feedback_student_writing': ('unavailable', 'source discontinued; see argument_feedback in tasks.py'),
}
REASONS = {key: reason for key, (_, reason) in PARKED.items()}

# Candidates considered but never annotated.
NOT_ANNOTATED = {
    "ccdv/patent-classification": ("todo", "abstract to patent section; not annotated yet"),
    "clue/clue cmnli": ("duplicate", "machine-translated MNLI; XNLI covers Chinese"),
    "demelin/wino_x": ("evaluation", "machine translation evaluation set; script-only loader"),
    "dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset": ("unsound", "machine-translated; many translations degenerate"),
    "ylacombe/xsum_factuality": ("unavailable", "no longer on the Hub"),
}


def by_kind(kind):
    """Parked tasks of one kind, e.g. ``by_kind("evaluation")`` for evaluation benchmarks."""
    assert kind in KINDS, f"kind must be one of {list(KINDS)}"
    return {key: globals()[key] for key, (task_kind, _) in PARKED.items() if task_kind == kind}

# Unset names default from the variable name, as in list_tasks.
for _key, _task in list(globals().items()):
    if isinstance(_task, Preprocessing):
        _dataset, _config, _ = parse_var_name(_key.strip())
        _task.dataset_name = _task.dataset_name or _dataset
        _task.config_name = _task.config_name or _config
