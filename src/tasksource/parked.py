"""Parked tasks: source or annotation issues that keep tasks out of training.

Each entry in PARKED has a kind (see KINDS) and a reason. Parked tasks are
not listed by ``list_tasks`` but their annotations remain available directly:

    from tasksource import parked
    dataset = parked.quora.load()
    duplicates = parked.by_kind("duplicate")

Evaluation benchmarks live separately in eval_only.py. To revive a parked
annotation, move it back to its task catalog and remove its PARKED entry.
"""
from .access import parse_var_name
from .metadata.configs import UNIVERSAL_DEPENDENCIES
from .preprocess import Preprocessing, VisualClassification, constant, get, Classification, TokenClassification, MultipleChoice

# duplicate of glue/rte
super_glue___rte = Classification(sentence1="premise", sentence2="hypothesis", labels="label")

# same pairs as sick/entailment_AB
sick__entailment_BA = Classification('sentence_A','sentence_B','entailment_BA')

# generated labels not sound enough
gpt3_nli = Classification("text_a","text_b","label",dataset_name="pietrolesci/gpt3_nli")

# overlaps FEVER-based NLI tasks
enfever_nli = Classification("evidence","claim","label", dataset_name="ctu-aic/enfever_nli")

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

# summary-only, the source document is missing; script-only loader
xsum_factuality = Classification("summary",labels="is_factual")

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
from .multilingual_tasks import _udep_cast_label_sequence  # noqa: E402
udep__deprel_multilingual = TokenClassification('tokens', 'deprel',
    pre_process=lambda ds: _udep_cast_label_sequence(ds, 'deprel'),
    dataset_name='universal-dependencies/universal_dependencies', config_name=UNIVERSAL_DEPENDENCIES)

# xglue's NER and POS configs repackage CoNLL-2002/2003 and Universal Dependencies (tasks listed
# elsewhere) and exist only behind a loading script
xglue__ner = TokenClassification("words", "ner", dataset_name="microsoft/xglue", config_name="ner")
xglue__pos = TokenClassification("words", "pos", dataset_name="microsoft/xglue", config_name="pos")

# the entailment direction of sick/label, which is listed
sick__entailment_AB = Classification('sentence_A','sentence_B','entailment_AB', dataset_name="tasksource/sick")

# prompt-injection aggregates
prompt_injection_threat_matrix = Classification(
    "text", labels="label",
    dataset_name="neuralchemy/prompt-injection-Threat-Matrix", config_name="binary",
    question="Is this prompt malicious or a prompt-injection attempt?",
    label_values={0: "benign", 1: "malicious"})

prompt_injection_geekyrakshit = Classification(
    "prompt", labels="label",
    dataset_name="geekyrakshit/prompt-injection-dataset",
    question="Is this prompt a prompt-injection attempt?",
    label_values={0: "benign", 1: "injection"})

# Caption-derived labels are not reliable image-grounded training supervision.
# Neutral evaluation pairs were reannotated, but the original train is uncorrected.
# https://openaccess.thecvf.com/content/ICCV2021/papers/Kayser_E-ViL_A_Dataset_and_Benchmark_for_Natural_Language_Explanations_in_ICCV_2021_paper.pdf
snli_ve = VisualClassification(
    images=lambda x: [x['image']], inputs='sentence', labels='gold_label',
    dataset_name='pingzhili/snli-ve', task_id='snli-ve', metadata=lambda x: {'image_id': x['Flickr30K_ID']},
    question='Does the image entail, contradict, or leave the statement neutral?',
    label_values={'entailment': 'entailment', 'neutral': 'neutral', 'contradiction': 'contradiction'},
    load_dataset_kwargs={'revision': '176e5ba43a2219043ebdaa057d9aaf42d2f34dc8'},
)

KINDS = {
    "duplicate": "duplicates or is covered by a listed task",
    "unsound": "labels or inputs do not support the task as annotated",
    "impractical": "inputs too long or data too heavy",
    "unavailable": "source no longer loads",
    "todo": "worth adding, not annotated yet",
}

PARKED = {
    "snli_ve": ("unsound", "caption-derived SNLI labels are not image-grounded; neutral evaluation pairs were reannotated, but this training source is uncorrected; hypothesis-only artifacts remain"),
    'xglue__ner': ('duplicate', 'repackages CoNLL-2002/2003 NER, which conll2002 (es, nl) and conll2003 (en) cover; the German part is not openly licensed; script-only'),
    'xglue__pos': ('duplicate', 'repackages Universal Dependencies POS, which udep__pos covers; script-only'),
    'udep__deprel_multilingual': ('unsound', 'all-language variant of udep__deprel; relation labels depend on the head word, which the task does not show'),
    'has_part': ('unsound', 'labels are model confidence scores, nearly all above 0.99'),
    'recast___recast_kg_relations': ('unsound', 'label semantics (1-6) are undocumented'),
    'super_glue___rte': ('duplicate', 'duplicate of glue/rte'),
    'sick__entailment_BA': ('duplicate', 'restates sick/label as B-to-A entailment'),
    'gpt3_nli': ('unsound', 'generated labels not sound enough'),
    'enfever_nli': ('duplicate', 'overlaps FEVER-based NLI tasks'),
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
    'species_800': ('unavailable', 'missing files'),
    'blog_authorship_corpus__horoscope': ('unsound', 'horoscope is not predictable from text'),
    'code_x_glue_cc_clone_detection_big_clone_bench': ('impractical', 'in bigbench, too heavy (100GB)'),
    'code_x_glue_cc_code_refinement': ('unsound', 'constant label, not a real task'),
    'chatgpt_detection': ('unsound', 'HC3 human answers are PTB-tokenized (a trivial shortcut); script-only loader'),
    'attempto_nli': ('unsound', 'unclear label semantics'),
    'mega_acceptability': ('duplicate', 'regression target; acceptability is covered by other tasks'),
    'xsum_factuality': ('unsound', 'summary-only, the source document is missing; script-only loader'),
    'lex_glue___ecthr_a': ('impractical', 'too long'),
    'lex_glue___ecthr_b': ('impractical', 'too long'),
    'nli_l2': ('duplicate', 'merges of NLI tasks already included'),
    'nli_l3': ('duplicate', 'merges of NLI tasks already included'),
    'ecthr_cases___alleged_violation_prediction': ('impractical', 'too long'),
    'ecthr_cases___violation_prediction': ('impractical', 'too long'),
    'sick__entailment_AB': ('duplicate', 'restates sick/label as A-to-B entailment'),
    'effective_feedback_student_writing': ('unavailable', 'source discontinued; see argument_feedback in tasks.py'),
    'prompt_injection_threat_matrix': ('unavailable', 'gated or removed: the Hub reports the dataset missing'),
    'prompt_injection_geekyrakshit': ('duplicate', 'aggregate of deepset/prompt-injections and xTRam1 (listed); 90% of its test rows are in its own train'),
}
REASONS = {key: reason for key, (_, reason) in PARKED.items()}

# Candidates considered but never annotated.
NOT_ANNOTATED = {
    "ccdv/patent-classification": ("todo", "abstract to patent section; not annotated yet"),
    "clue/clue cmnli": ("duplicate", "machine-translated MNLI; XNLI covers Chinese"),
    "dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset": ("unsound", "machine-translated; many translations degenerate"),
    "ylacombe/xsum_factuality": ("unavailable", "no longer on the Hub"),
}


def by_kind(kind):
    """Parked tasks of one kind, e.g. ``by_kind("duplicate")``."""
    assert kind in KINDS, f"kind must be one of {list(KINDS)}"
    return {key: globals()[key] for key, (task_kind, _) in PARKED.items() if task_kind == kind}

# Unset names default from the variable name, as in list_tasks.
for _key, _task in list(globals().items()):
    if isinstance(_task, Preprocessing):
        _dataset, _config, _ = parse_var_name(_key.strip())
        _task.dataset_name = _task.dataset_name or _dataset
        _task.config_name = _task.config_name or _config
