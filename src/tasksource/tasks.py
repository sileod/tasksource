from .preprocess import cat, get, regen, name, constant, Classification, TokenClassification, MultipleChoice
from .metadata import udep_en_configs
from datasets import get_dataset_config_names, Sequence, ClassLabel, Dataset, DatasetDict, Features, Value
import html
from collections import Counter
import random
import re

# the Hub's automatic parquet export of script-only datasets (same splits and features)
PARQUET = "refs/convert/parquet"

def _single_label(dataset, column, empty=None):
    """Keep rows of a multi-label column with exactly one label (or none, named ``empty``)."""
    names = dataset["train"].features[column].feature.names + ([empty] if empty else [])
    keep = (lambda labels: len(labels) <= 1) if empty else (lambda labels: len(labels) == 1)
    dataset = dataset.filter(lambda x: keep(x[column]))
    return dataset.map(lambda x: {column: x[column][0] if x[column] else len(names) - 1},
                       features=Features({**dataset["train"].features, column: ClassLabel(names=names)}))

# Integer encodings documented by the corresponding source dataset cards.
NLI_LABEL_VALUES = {0: "entailment", 1: "neutral", 2: "contradiction"}
ENTAILMENT_LABEL_VALUES = {0: "not-entailed", 1: "entailed"}

# variable name: dataset___config__task

###################### NLI/paraphrase ###############################

glue___mnli = Classification(sentence1="premise", sentence2="hypothesis", labels="label", splits=["train", None, "validation_matched"])
glue___qnli = Classification("question","sentence", labels="label")
glue___rte = Classification(sentence1="sentence1", sentence2="sentence2", labels="label")
glue___wnli = Classification(sentence1="sentence1", sentence2="sentence2", labels="label")

glue___mrpc = Classification(sentence1="sentence1", sentence2="sentence2", labels="label")
glue___qqp = Classification(sentence1="question1", sentence2="question2", labels="label")
glue___stsb = Classification(sentence1="sentence1", sentence2="sentence2", labels="label",
    question="How similar are the two sentences, from 0 (unrelated) to 5 (equivalent)?")

super_glue___boolq = Classification(sentence1="question", labels="label")
boolq_passage = Classification("passage", "question", labels="label", # reading-comprehension variant
    dataset_name="super_glue", config_name="boolq", task_id="super_glue/boolq_passage")
super_glue___cb = Classification(sentence1="premise", sentence2="hypothesis", labels="label")
super_glue___multirc = Classification(
    cat(["paragraph", "question"]),
    'answer',
    labels=name('label', ['incorrect answer', 'correct answer'])
)
super_glue___wic = Classification(
    sentence1=lambda x: f"Word: {x['word']}\n{x['sentence1']}",
    sentence2="sentence2",
    labels=name('label', ['different meaning', 'same meaning'])
)
super_glue___axg = Classification(sentence1="premise", sentence2="hypothesis", labels="label", splits=["test", None, None])


anli__a1 = Classification('premise','hypothesis','label', splits=['train_r1','dev_r1','test_r1'])
anli__a2 = Classification('premise','hypothesis','label', splits=['train_r2','dev_r2','test_r2'])
anli__a3 = Classification('premise','hypothesis','label', splits=['train_r3','dev_r3','test_r3'])


babi_nli = Classification("premise", "hypothesis", "label",
    dataset_name="tasksource/babi_nli",
    config_name=sorted(set(get_dataset_config_names("tasksource/babi_nli"))-{"agents-motivations"})
) # agents-motivations task is not as clear-cut as the others


sick__label         = Classification('sentence_A','sentence_B','label', dataset_name="tasksource/sick")
sick__relatedness   = Classification('sentence_A','sentence_B','relatedness_score', dataset_name="tasksource/sick",
    question="How related are the two sentences, from 1 (unrelated) to 5 (very related)?")


def remove_neg_1(dataset):
    return dataset.filter(lambda x:x['labels']!=-1)

def _parse_jeggers_riddle_choices(batch):
    import ast
    choices = batch["choices"]
    if isinstance(choices, str):
        try:
            choices = ast.literal_eval(choices)
        except Exception:
            choices = [choices]
    stripped = [
        c.split(": ", 1)[1] if isinstance(c, str) and ": " in c else c
        for c in choices
    ]
    return {"choices": {"text": stripped, "label": [None] * len(stripped)}}

def _strip_option_prefix(text):
    import re
    if isinstance(text, str):
        return re.sub(r'^[A-E][\.\):]\s*', '', text).strip()
    return text

def _concat_splits_to_train(dataset, splits=("train", "test")):
    """Merge listed splits into train (reproduces single-pool sources)."""
    from datasets import concatenate_datasets, DatasetDict
    parts = [dataset[s] for s in splits if s in dataset]
    merged = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    out = DatasetDict({"train": merged})
    for k in dataset:
        if k not in splits:
            out[k] = dataset[k]
    return out

def _logiqa_options(example):
    import ast
    opts = example.get("options")
    if isinstance(opts, str):
        try:
            opts = ast.literal_eval(opts)
        except Exception:
            opts = [opts]
    opts = [_strip_option_prefix(o) for o in opts]
    ans = example.get("answer", "A")
    if isinstance(ans, int):
        label = ans
    else:
        ans = str(ans).strip().rstrip('.').upper()
        label = ord(ans[0]) - ord('A') if ans and ans[0] in "ABCD" else 0
    return {"options": opts, "correct_option": label, "query": example.get("question", "")}

snli = Classification(sentence1="premise", sentence2="hypothesis", labels="label",
    post_process=remove_neg_1)

scitail = Classification("sentence1","sentence2","gold_label",config_name="snli_format")

hans = Classification(sentence1="sentence1", sentence2="sentence2", labels="labels",
    dataset_name="tasksource/hans")

wanli = Classification('premise','hypothesis','gold', dataset_name="alisawuffles/WANLI")

recast_nli = Classification(sentence1="context", sentence2="hypothesis", labels="label", dataset_name="tasksource/recast",
    config_name=['recast_puns', 'recast_factuality', 'recast_verbnet',
    'recast_verbcorner', 'recast_ner', 'recast_sentiment', 'recast_megaveridicality'])


probability_words_nli = Classification(sentence1="context", sentence2="hypothesis", labels="label",
    dataset_name="sileod/probability_words_nli", 
    config_name=["reasoning_1hop","reasoning_2hop","usnli"])

nan_nli = Classification("premise", "hypothesis", "label", dataset_name="joey234/nan-nli")

nli_fever = Classification("premise","hypothesis","label",
    dataset_name="pietrolesci/nli_fever", splits=["train","dev",None])

breaking_nli = Classification("sentence1","sentence2","label",
    dataset_name="pietrolesci/breaking_nli", splits=["full",None,None],
    label_values=NLI_LABEL_VALUES)

conj_nli = Classification("premise","hypothesis","label",post_process=remove_neg_1,
    dataset_name="pietrolesci/conj_nli",splits=['train','dev',None],
    label_values=NLI_LABEL_VALUES)

fracas = Classification("premise","hypothesis","label",
    dataset_name="pietrolesci/fracas", label_values=NLI_LABEL_VALUES)

dialogue_nli = Classification("sentence1","sentence2","label",
    dataset_name="pietrolesci/dialogue_nli", label_values=NLI_LABEL_VALUES)

mpe_nli = Classification("premise","hypothesis","label",
    dataset_name="pietrolesci/mpe",
    splits=["train","dev","test"], label_values=NLI_LABEL_VALUES)

dnc_nli = Classification("context","hypothesis","label",
    dataset_name="pietrolesci/dnc", label_values=ENTAILMENT_LABEL_VALUES)


recast_white__fnplus = Classification("text","hypothesis","label",
    dataset_name="pietrolesci/recast_white",splits=['fnplus',None,None],
    label_values=ENTAILMENT_LABEL_VALUES)
recast_white__sprl = Classification("text","hypothesis","label",
    dataset_name="pietrolesci/recast_white",splits=['sprl',None,None],
    label_values=ENTAILMENT_LABEL_VALUES)
recast_white__dpr = Classification("text","hypothesis","label",
    dataset_name="pietrolesci/recast_white",splits=['dpr',None,None],
    label_values=ENTAILMENT_LABEL_VALUES)

joci = Classification("context","hypothesis",
    labels=lambda x: [None, "impossible", "technically possible", "plausible", "likely", "very likely"][x["original_label"]],
    pre_process=lambda ds:ds.filter(lambda x:x['original_label']!=0),
    dataset_name="pietrolesci/joci",splits=['full',None,None])


robust_nli__IS_CS = Classification("premise","hypothesis","label",
	dataset_name="pietrolesci/robust_nli", splits=["IS_CS",None,None], label_values=NLI_LABEL_VALUES)
robust_nli__LI_LI = Classification("premise","hypothesis","label",
	dataset_name="pietrolesci/robust_nli", splits=["LI_LI",None,None], label_values=NLI_LABEL_VALUES)
robust_nli__ST_WO = Classification("premise","hypothesis","label",
	dataset_name="pietrolesci/robust_nli", splits=["ST_WO",None,None], label_values=NLI_LABEL_VALUES)
robust_nli__PI_SP = Classification("premise","hypothesis","label",
	dataset_name="pietrolesci/robust_nli", splits=["PI_SP",None,None], label_values=NLI_LABEL_VALUES)
robust_nli__PI_CD = Classification("premise","hypothesis","label",
	dataset_name="pietrolesci/robust_nli", splits=["PI_CD",None,None], label_values=NLI_LABEL_VALUES)
robust_nli__ST_SE = Classification("premise","hypothesis","label",
	dataset_name="pietrolesci/robust_nli", splits=["ST_SE",None,None], label_values=NLI_LABEL_VALUES)
robust_nli__ST_NE = Classification("premise","hypothesis","label",
	dataset_name="pietrolesci/robust_nli", splits=["ST_NE",None,None], label_values=NLI_LABEL_VALUES)
robust_nli__ST_LM = Classification("premise","hypothesis","label",
	dataset_name="pietrolesci/robust_nli", splits=["ST_LM",None,None], label_values=NLI_LABEL_VALUES)
robust_nli_is_sd = Classification("premise","hypothesis","label",
    dataset_name="pietrolesci/robust_nli_is_sd",
    label_values={0: "non-entailment", 1: "entailment"})
robust_nli_li_ts = Classification("premise","hypothesis","label",
    dataset_name="pietrolesci/robust_nli_li_ts",
    label_values={0: "non-contradiction", 1: "contradiction"})

gen_debiased_nli__snli_seq_z = Classification("premise","hypothesis","label",
	dataset_name="pietrolesci/gen_debiased_nli", splits=["snli_seq_z",None,None], label_values=NLI_LABEL_VALUES)
gen_debiased_nli__snli_z_aug = Classification("premise","hypothesis","label",
	dataset_name="pietrolesci/gen_debiased_nli", splits=["snli_z_aug",None,None], label_values=NLI_LABEL_VALUES)
gen_debiased_nli__snli_par_z = Classification("premise","hypothesis","label",
	dataset_name="pietrolesci/gen_debiased_nli", splits=["snli_par_z",None,None], label_values=NLI_LABEL_VALUES)
gen_debiased_nli__mnli_par_z = Classification("premise","hypothesis","label",
	dataset_name="pietrolesci/gen_debiased_nli", splits=["mnli_par_z",None,None], label_values=NLI_LABEL_VALUES)
gen_debiased_nli__mnli_z_aug = Classification("premise","hypothesis","label",
	dataset_name="pietrolesci/gen_debiased_nli", splits=["mnli_z_aug",None,None], label_values=NLI_LABEL_VALUES)
gen_debiased_nli__mnli_seq_z = Classification("premise","hypothesis","label",
	dataset_name="pietrolesci/gen_debiased_nli", splits=["mnli_seq_z",None,None], label_values=NLI_LABEL_VALUES)

add_one_rte = Classification("premise","hypothesis","label",
    dataset_name="pietrolesci/add_one_rte",splits=["train","dev","test"],
    label_values=ENTAILMENT_LABEL_VALUES)

hlgd = Classification("headline_a", "headline_b", labels="label", dataset_name="tasksource/hlgd")

paws___labeled_final   = Classification("sentence1", "sentence2", name('label',['not_paraphrase','paraphrase']))
paws___labeled_swap    = Classification("sentence1", "sentence2", name('label',['not_paraphrase','paraphrase']), splits=["train", None, None])

medical_questions_pairs = Classification("question_1","question_2", name("label",['not similar','similar']))
 
###################### Token Classification #########################

# data-only mirror with the original columns, ClassLabel names and splits
conll2003__pos_tags   = TokenClassification(tokens="tokens", labels='pos_tags', dataset_name="tomaarsen/conll2003")
conll2003__chunk_tags = TokenClassification(tokens="tokens", labels='chunk_tags', dataset_name="tomaarsen/conll2003")
conll2003__ner_tags   = TokenClassification(tokens="tokens", labels='ner_tags', dataset_name="tomaarsen/conll2003")


######################## Multiple choice ###########################


fig_qa = MultipleChoice(
    "startphrase",
    choices=["ending1","ending2"],
    labels="labels",
    dataset_name="nightingal3/fig-qa",
    splits=["train","validation",None]
)


cos_e = MultipleChoice('question',
    choices_list='choices',
    labels= lambda x: x['choices_list'].index(x['answer']),
    config_name='v1.0')

cosmos_qa = MultipleChoice(cat(['context','question']),regen('answer[0-3]'),'label',
    dataset_name="Samsoup/cosmos_qa")

dream = MultipleChoice(
    lambda x:"\n".join(x['dialogue']+[x['question']]),
    choices_list='choice',
    labels=lambda x:x['choices_list'].index(x['answer']),
    dataset_name="dataset-org/dream", task_id="dream",
    load_dataset_kwargs=dict(revision=PARQUET, data_dir="plain_text"))

openbookqa = MultipleChoice(
    'question_stem',
    choices_list=get.choices.text,
    labels='answerKey'
)

qasc = MultipleChoice(
    'question',
    choices_list=get.choices.text,
    labels=lambda x: "ABCDEFGH".index(x['answerKey']),
    splits=['train','validation',None]
    
)

quartz = MultipleChoice(
    'question',
    choices_list=get.choices.text,
    labels='answerKey'
)
quail = MultipleChoice(
    cat(['context','question']),
    choices_list='answers',
    labels='correct_answer_id' 
)

head_qa___en = MultipleChoice("qtext",
    choices_list = lambda x:[a['atext'] for a in x["answers"]],
    labels = lambda x:[a['aid'] for a in x["answers"]].index(x["ra"]),
    dataset_name="EleutherAI/headqa", config_name="en",
    task_id="head_qa/en"
)


sciq = MultipleChoice(
    'question',
    ['correct_answer']+regen('distractor[1-3]'),
    labels=constant(0))

social_i_qa = MultipleChoice(
    cat(['context','question']),
    ['answerA','answerB','answerC'],
    'label',
    dataset_name="tasksource/social_i_qa")

wiki_hop___original = MultipleChoice(  # query is "relation subject"
    lambda x: (lambda r, s: f"What is the {r.replace('_', ' ')} of {s}?")(*x['query'].split(' ', 1)),
    choices_list='candidates',
    labels=lambda x:x['choices_list'].index(x["answer"]),
    dataset_name="MoE-UNC/wikihop", config_name="default",
    task_id="wiki_hop/original")

wiqa = MultipleChoice('question_stem',
    choices_list = lambda x: x['choices']['text'],
    labels='answer_label_as_choice',
    dataset_name="tasksource/wiqa")

piqa = MultipleChoice('goal', choices=['sol1','sol2'], labels='label',
    dataset_name="baber/piqa")

def _hellaswag_text(text):
    # WikiHow items carry [header]/[title]/[step] tags; same cleanup as lm-eval-harness
    text = re.sub(r"\[.*?\]", "", text.strip().replace(" [title]", ". "))
    return re.sub(r"\s+", " ", text).strip()

hellaswag = MultipleChoice(lambda x: _hellaswag_text(x['ctx_a']),
    # ctx_b is the lowercased start of the sentence each ending completes
    choices_list=lambda x: [_hellaswag_text(f'{x["ctx_b"][:1].upper()}{x["ctx_b"][1:]} {e}') for e in x["endings"]],
    labels='label', splits=['train','validation',None])

def _copa_input(x):
    ask = "What was the cause of this?" if x["question"] == "cause" else "What happened as a result?"
    return f"{x['premise']} {ask}"

super_glue___copa = MultipleChoice(_copa_input,['choice1','choice2'],'label')

balanced_copa = MultipleChoice(_copa_input,['choice1','choice2'],'label',
    dataset_name="pkavumba/balanced-copa")

e_care = MultipleChoice(_copa_input,['choice1','choice2'],'label',
    dataset_name="12ml/e-CARE")

art = MultipleChoice(
    lambda x: f"Beginning: {x['observation_1']}\nEnding: {x['observation_2']}",
    ['hypothesis_1','hypothesis_2'], question="What happened in between?",
    labels=lambda x:x['label']-1,
    splits=['train','validation',None]
)


winogrande = MultipleChoice('sentence',['option1','option2'],'answer',config_name='winogrande_xl',
    splits=['train','validation',None])

codah = MultipleChoice('question_propmt',choices_list='candidate_answers',labels='correct_answer_idx',config_name='codah')

ai2_arc__challenge = MultipleChoice('question',
    choices_list=get.choices.text,  
    labels=lambda x: get.choices.label(x).index(x["answerKey"]),
    config_name=["ARC-Challenge","ARC-Easy"])

definite_pronoun_resolution = MultipleChoice(
    inputs=lambda x: f"{x['sentence']}\nWho or what does \"{x['pronoun']}\" refer to?",
    choices_list='candidates',
    labels="label",
    splits=['train',None,'test'])

swag___regular=MultipleChoice(cat(["sent1","sent2"]),regen("ending[0-3]"),"label")

def _split_choices(s):
    import re
    return [x.rstrip(', ') for x in re.split(r'[a-e] \) (.*?)',s) if x.strip(', ')]

math_qa = MultipleChoice(
    'Problem', 
    choices_list = lambda x: _split_choices(x['options']),
    labels = lambda x:'abcde'.index(x['correct']),
    dataset_name="tasksource/math_qa"
)


######################## Classification (other) ########################
glue___cola = Classification(sentence1="sentence", labels="label")
glue___sst2 = Classification(sentence1="sentence", labels="label")

def _utilitarianism_comparisons(dataset):
    # Reproduce the source builder's seeded orientation without its global RNG
    # mutation or its accidental CSV-header example (index + 1).
    def orient(row, index):
        label = random.Random(index + 1).randint(0, 1)
        pair = [row["baseline"], row["less_pleasant"]]
        return {
            "comparison": f'"{pair[1 - label]}" is better than "{pair[label]}"',
            "label": label,
        }
    return dataset.map(orient, with_indices=True)

utilitarianism = Classification(
    "comparison", labels="label", dataset_name="csv", task_id="utilitarianism",
    load_dataset_kwargs={"data_files": {
        "train": "hf://datasets/hendrycks/ethics/data/utilitarianism/train.csv",
        "test": "hf://datasets/hendrycks/ethics/data/utilitarianism/test.csv",
    }}, pre_process=_utilitarianism_comparisons,
    label_values={0: "false", 1: "true"})

amazon_counterfactual = Classification(
    "text", labels="label_text",
    dataset_name="mteb/amazon_counterfactual",
    config_name="en")

insincere_questions = Classification(
    "text", labels="label_text",
    dataset_name="SetFit/insincere-questions")

toxic_conversations = Classification(
    "text", labels="label_text",
    dataset_name="SetFit/toxic_conversations")

turingbench = Classification("Generation",labels="label",
    dataset_name="csv", task_id="TuringBench",
    load_dataset_kwargs={"data_files": {
        "train": "hf://datasets/jana4/turingbench-humanized/TuringBench/AA/train.csv",
        "validation": "hf://datasets/jana4/turingbench-humanized/TuringBench/AA/valid.csv",
        "test": "hf://datasets/jana4/turingbench-humanized/TuringBench/AA/test.csv",
    }}, splits=["train","validation",None])


trec = Classification(sentence1="text", labels="fine_label",
    dataset_name="tasksource/trec")

tals_vitaminc = Classification('claim','evidence','label', dataset_name="tals/vitaminc")

hope_edi = Classification(
    "text", labels="label", splits=["train", "validation", None],
    dataset_name="csv", task_id="hope_edi/english",
    load_dataset_kwargs={
        "data_files": {
            "train": "https://drive.google.com/uc?id=1ydsOTvBZXKqcRvXawOuePrJ99slOEbkk&export=download&confirm=t",
            "validation": "https://drive.google.com/uc?id=1pvpPA97kybx5IyotR9HNuqP4T5ktEtr4&export=download&confirm=t",
        },
        "delimiter": "\t",
        "column_names": ["text", "label", "dummy"],
    },
)


rumoureval_2019 = Classification(
    sentence1="source_text",
    sentence2="reply_text",
    labels="label", dataset_name="csv",
    task_id="rumoureval_2019/RumourEval2019",
    load_dataset_kwargs={"data_files": {
        "train": "hf://datasets/strombergnlp/rumoureval_2019/rumoureval2019_train.csv",
        "validation": "hf://datasets/strombergnlp/rumoureval_2019/rumoureval2019_val.csv",
        "test": "hf://datasets/strombergnlp/rumoureval_2019/rumoureval2019_test.csv",
    }},
    # filter before fix_labels, otherwise None becomes a class name
    pre_process=lambda ds: ds.filter(lambda x: x['label'] is not None and x['reply_text'] is not None)
)

ethos = Classification(sentence1="text", labels=name("label", ["no hate speech", "hate speech"]),
    splits=["train", None, None], dataset_name="SetFit/ethos_binary",
    task_id="ethos/binary",
    pre_process=lambda ds: _concat_splits_to_train(ds, ("train", "test")))
_ETHOS_ASPECTS = {
    "violence": "Does this comment incite violence?",
    "directed_vs_generalized": "Is this comment directed at a specific person rather than a group?",
    "gender": "Does this comment attack people for their gender?",
    "race": "Does this comment attack people for their race?",
    "national_origin": "Does this comment attack people for their national origin?",
    "disability": "Does this comment attack people for a disability?",
    "religion": "Does this comment attack people for their religion?",
    "sexual_orientation": "Does this comment attack people for their sexual orientation?",
}

def _ethos_aspects(dataset):
    # one yes/no question per hateful comment and aspect; aspects are rater shares, keep clear cases
    def rows(split):
        for x in split:
            for aspect, question in _ETHOS_ASPECTS.items():
                if not 0.2 < x[aspect] < 0.5:
                    yield dict(comment=x["comment"], question=question, answer=["no", "yes"][x[aspect] >= 0.5])
    return DatasetDict({name: Dataset.from_generator(rows, gen_kwargs=dict(split=split)) for name, split in dataset.items()})

ethos___multilabel = Classification("comment", "question", "answer", dataset_name="tasksource/ethos",
    config_name="multilabel", pre_process=_ethos_aspects)

tweet_eval = Classification(sentence1="text", labels="label",
    config_name=["emoji", "emotion", "hate", "irony", "offensive", "sentiment"])

_STANCE_TOPICS = dict(abortion="abortion", atheism="atheism", climate="climate change", feminist="feminism",
                      Hillary="Hillary Clinton")

def stance_kwargs(topic):
    return {
        "sentence1": "text",
        "question": f"What stance does the tweet take on {_STANCE_TOPICS[topic]}?",
        "labels": "label", 
        "config_name": f"stance_{topic.lower()}",
        "dataset_name": "tweet_eval"
    }

tweet_eval_abortion = Classification(**stance_kwargs("abortion"))
tweet_eval_atheism  = Classification(**stance_kwargs("atheism"))
tweet_eval_climate  = Classification(**stance_kwargs("climate"))
tweet_eval_feminist = Classification(**stance_kwargs("feminist"))
tweet_eval_hillary  = Classification(**stance_kwargs("Hillary"))


discovery = Classification("sentence1", "sentence2", labels="label", config_name=["discovery"])

pragmeval_1 = Classification("sentence",labels="label",
    dataset_name="pragmeval",
    config_name= ["switchboard","mrda","verifiability"])

pragmeval_2 = Classification("sentence1","sentence2",labels="label",
    dataset_name="pragmeval",
    config_name= ["emergent", "gum", "pdtb", "persuasiveness-claimtype", "persuasiveness-premisetype", "sarcasm","stac"])

# low/high scales over the same inputs: the label names the scale
def _pragmeval_scale(config, scale, *inputs):
    return Classification(*inputs, labels="label", dataset_name="pragmeval", config_name=config, task_id=f"pragmeval/{config}",
        label_values={0: f"low {scale}", 1: f"high {scale}"})

pragmeval__emobank_arousal = _pragmeval_scale("emobank-arousal", "emotional arousal", "sentence")
pragmeval__emobank_dominance = _pragmeval_scale("emobank-dominance", "dominance (sense of control)", "sentence")
pragmeval__emobank_valence = _pragmeval_scale("emobank-valence", "valence (pleasantness)", "sentence")
pragmeval__squinky_formality = _pragmeval_scale("squinky-formality", "formality", "sentence")
pragmeval__squinky_implicature = _pragmeval_scale("squinky-implicature", "implicature (implied beyond what is said)", "sentence")
pragmeval__squinky_informativeness = _pragmeval_scale("squinky-informativeness", "informativeness", "sentence")
pragmeval__persuasiveness_eloquence = _pragmeval_scale("persuasiveness-eloquence", "eloquence", "sentence1", "sentence2")
pragmeval__persuasiveness_relevance = _pragmeval_scale("persuasiveness-relevance", "relevance", "sentence1", "sentence2")
pragmeval__persuasiveness_specificity = _pragmeval_scale("persuasiveness-specificity", "specificity", "sentence1", "sentence2")
pragmeval__persuasiveness_strength = _pragmeval_scale("persuasiveness-strength", "argument strength", "sentence1", "sentence2")

silicone = Classification("Utterance",labels="Label",
    dataset_name="tasksource/silicone",
    config_name=['dyda_da', 'dyda_e', 'maptask', 'meld_e', 'meld_s', 'oasis', 'sem'] # +['swda', 'mrda'] # in pragmeval
)

_IEMOCAP = dict(ang='anger', dis='disgust', exc='excitement', fea='fear', fru='frustration', hap='happiness',
    neu='neutral', oth='other', sad='sadness', sur='surprise')
silicone___iemocap = Classification("Utterance", labels=lambda x: _IEMOCAP[x['Emotion']],
    dataset_name="tasksource/silicone",
    # xxx marks utterances without annotator agreement (24% of train)
    pre_process=lambda ds: ds.filter(lambda x: x['Emotion'] in _IEMOCAP))

lex_glue___eurlex = Classification(sentence1="text", labels="labels") 
# Supreme Court Database issue areas 1-13 (14, private action, is absent)
lex_glue___scotus = Classification(sentence1="text", labels="label", label_values=dict(enumerate([
    "criminal procedure", "civil rights", "first amendment", "due process", "privacy", "attorneys", "unions",
    "economic activity", "judicial power", "federalism", "interstate relations", "federal taxation", "miscellaneous"])))
lex_glue___ledgar = Classification(sentence1="text", labels="label")
# single-label rows only; an unlabeled clause is fair (the large majority)
lex_glue___unfair_tos = Classification(sentence1="text", labels="labels",
    pre_process=lambda ds: _single_label(ds, "labels", empty="fair clause"),
    question="Which kind of unfair term, if any, does this terms-of-service clause contain?")
lex_glue___case_hold = MultipleChoice("context", choices_list='endings', labels="label")

# langcodes names ISO codes; WiLI also uses a few Wikipedia codes langcodes misreads
_WIKIPEDIA_LANGUAGES = {"roa-tara": "Tarantino", "map-bms": "Banyumasan", "be-tarask": "Belarusian (Taraškievica)"}
def language_name(code):
    import langcodes
    return _WIKIPEDIA_LANGUAGES.get(code) or langcodes.get(code).display_name()

language_identification = Classification("text",labels=lambda x: language_name(x["labels"]), question="What language is this text in?",
    dataset_name="papluca/language-identification")

################ Automatically generated (verified)##########

imdb = Classification(sentence1="text", labels="label", splits=["train", None, "test"])

rotten_tomatoes = Classification(sentence1="text", labels="label")

ag_news = Classification(sentence1="text", labels="label", splits=["train", None, "test"])

yelp_review_full = Classification(sentence1=lambda x: x["text"].replace("\\n", "\n"), labels="label",  # newlines are escaped in the source
    label_values={stars - 1: f"{stars} star{'s' if stars != 1 else ''}" for stars in range(1, 6)},
    splits=["train", None, "test"], config_name=["yelp_review_full"])

financial_phrasebank = Classification(sentence1="text", labels="label", splits=["train", None, None],
    dataset_name="ghbacct/financial-phrasebank-all-agree-classification",
    task_id="financial_phrasebank/sentences_allagree",
    pre_process=lambda ds: _concat_splits_to_train(ds, ("train", "test")))

poem_sentiment = Classification(sentence1="verse_text", labels="label")

emotion = Classification(sentence1="text", labels="label", dataset_name="dair-ai/emotion")

dbpedia_14 = Classification(sentence1="content", labels="label", splits=["train", None, "test"], config_name=["dbpedia_14"])

amazon_polarity = Classification(sentence1="content", labels="label", splits=["train", None, "test"], config_name=["amazon_polarity"])

app_reviews = Classification("review", labels="star", splits=["train", None, None],
    label_values={star: f"{star} star" + "s" * (star > 1) for star in range(1, 6)})


hate_speech18 = Classification(sentence1="text", labels="label", splits=["train", None, None],
    dataset_name="tasksource/hate_speech18",
    # idk/skip are skipped items; relation is hate only in context of neighbouring sentences
    pre_process=lambda ds: ds.filter(lambda x: x["label"] in (0, 1)),
    post_process=lambda ds: ds.cast_column("labels", ClassLabel(names=["noHate", "hate"])))

sms_spam = Classification(sentence1="sms", labels="label", splits=["train", None, None])

# meanGrade averages five 0-3 funniness grades
humicroedit___subtask_1 = Classification(lambda x: f"Original: {x['headline']}", lambda x: f"Edited: {x['edited']}",
    labels=lambda x: int(x["meanGrade"] + 0.5),
    label_values={0: "not funny", 1: "slightly funny", 2: "moderately funny", 3: "funny"},
    dataset_name="tasksource/humicroedit", config_name="subtask-1")
def _humicroedit(i):  # the headline with its <word/> replaced by the edit
    return lambda x: re.sub(r"<[^>]*/>", x[f"edit{i}"], x[f"original{i}"])
humicroedit___subtask_2 = Classification(_humicroedit(1), _humicroedit(2),
    labels="label", question="Which edited headline is funnier?",
    label_values={0: "equally funny", 1: "first headline", 2: "second headline"},
    dataset_name="tasksource/humicroedit", config_name="subtask-2")

snips_built_in_intents = Classification(sentence1="text", labels="label", splits=["train", None, None])

banking77 = Classification(sentence1="text", labels="label", splits=["train", None, "test"])

hate_speech_offensive = Classification(sentence1="tweet", labels="class", splits=["train", None, None])

yahoo_answers_topics = Classification(
    "question_title","question_content",labels="topic")

# popularity bands from the dataset's BigQuery thresholds on score, favorites and views
stackoverflow_questions=Classification("title","body",labels="label",
    dataset_name="pacovaldez/stackoverflow-questions", label_values={
        0: "very popular question", 1: "popular question", 2: "somewhat popular question", 3: "unpopular question"})


hyperpartisan_news = Classification(
    "text",
    labels=lambda x: {'true':'hyperpartisan','false':'not_hyperpartisan'}.get(x["label"]),
    dataset_name="zapsdcn/hyperpartisan_news")

scierc = Classification("text",labels="label",dataset_name="zapsdcn/sciie")
citation_intent = Classification("text",labels="label",dataset_name="zapsdcn/citation_intent")

go_emotions___simplified = Classification(sentence1="text", labels="labels",
    pre_process=lambda ds: _single_label(ds, "labels"))  # 84% of comments have one emotion


scicite = Classification(sentence1="string", labels="label",dataset_name="tasksource/scicite")

liar = Classification(sentence1="statement", labels="label",
    dataset_name="tasksource/liar")

# the source relation codes, spelled out (EVALution names are already readable)
LEXICAL_RELATIONS = {"attri": "attribute", "coord": "co-hyponym", "COORD": "co-hyponym", "sibl": "co-hyponym",
    "hyper": "hypernym", "HYPER": "hypernym", "hypo": "hyponym", "mero": "meronym",
    "random": "unrelated", "RANDOM": "unrelated", "false": "unrelated"}
LEXICAL_RELATION_QUESTION = "How is the second word related to the first?"

relbert_lexical_relation_classification = Classification(sentence1="head", sentence2="tail",
 labels=lambda x: LEXICAL_RELATIONS.get(x["relation"], x["relation"]), question=LEXICAL_RELATION_QUESTION,
 dataset_name="json",
 config_name=["BLESS","EVALution","K&H+N","ROOT09"],
 task_id="lexical_relation_classification/{config_name}",
 load_dataset_kwargs={"data_files": {
     "train": "hf://datasets/relbert/lexical_relation_classification/dataset/{config_name}/train.jsonl",
     "validation": "hf://datasets/relbert/lexical_relation_classification/dataset/{config_name}/val.jsonl",
     "test": "hf://datasets/relbert/lexical_relation_classification/dataset/{config_name}/test.jsonl",
 }})

# CogALexV has train/test but no validation file in the source repository.
# Establish its complete ontology before source-row sampling (rare relations
# can otherwise be absent from a small train sample but present in test).
COGALEXV_RELATIONS = {
    "ANT": "antonym relation", "HYPER": "hypernym relation",
    "PART_OF": "part-of relation", "RANDOM": "unrelated words",
    "SYN": "synonym relation",
}

def _cogalexv_relations(dataset):
    dataset = dataset.map(lambda row: {
        "relation": COGALEXV_RELATIONS[row["relation"]]})
    return dataset.cast_column(
        "relation", ClassLabel(names=list(COGALEXV_RELATIONS.values())))

relbert_cogalexv = Classification(
 sentence1="head", sentence2="tail", labels="relation", dataset_name="json", question=LEXICAL_RELATION_QUESTION,
 task_id="lexical_relation_classification/CogALexV",
 load_dataset_kwargs={"data_files": {
     "train": "hf://datasets/relbert/lexical_relation_classification/dataset/CogALexV/train.jsonl",
     "test": "hf://datasets/relbert/lexical_relation_classification/dataset/CogALexV/test.jsonl",
 }}, pre_process=_cogalexv_relations)


def _probing(config, readable):
    # SentEval probing labels are codes (NN, PAST, O/I...); spell them out
    def pre_process(dataset):
        names = dataset["train"].features["label"].names
        return dataset.cast_column("label", ClassLabel(names=[readable(name) for name in names]))
    return Classification("sentence", labels="label", dataset_name="tasksource/linguisticprobing",
        config_name=config, pre_process=pre_process)

_SENTENCE_LENGTH = ["5-8 words", "9-12 words", "13-16 words", "17-20 words", "21-25 words", "26-28 words"]
linguisticprobing___subj_number = _probing("subj_number", {"NN": "singular subject", "NNS": "plural subject"}.get)
linguisticprobing___obj_number = _probing("obj_number", {"NN": "singular object", "NNS": "plural object"}.get)
linguisticprobing___past_present = _probing("past_present", {"PAST": "past tense", "PRES": "present tense"}.get)
linguisticprobing___sentence_length = _probing("sentence_length", lambda code: _SENTENCE_LENGTH[int(code)])
linguisticprobing___top_constituents = _probing("top_constituents",
    lambda code: "other constituents" if code == "OTHER" else "constituents " + code.replace("_", " "))
linguisticprobing___tree_depth = _probing("tree_depth", lambda code: code.replace("depth_", "parse tree depth "))
linguisticprobing___coordination_inversion = _probing("coordination_inversion",
    {"O": "original clause order", "I": "inverted clause order"}.get)
linguisticprobing___odd_man_out = _probing("odd_man_out", {"O": "original sentence", "C": "one word replaced"}.get)
linguisticprobing___bigram_shift = _probing("bigram_shift", {"O": "original word order", "I": "two adjacent words swapped"}.get)

crowdflower = Classification("text", labels="label",
 splits=["train", None, None], dataset_name="tasksource/crowdflower",
 config_name=['sentiment_nuclear_power',
            'tweet_global_warming',
            'airline-sentiment',
            'corporate-messaging',
            'economic-news',
            'political-media-audience',
            'political-media-bias',
            'political-media-message',
            'text_emotion']
)

def _ethics_binary_label(x):
    value = str(x["label"])
    if value in {"0", "acceptable"}:
        return "acceptable"
    return "unacceptable"

def _ethics_virtue_first(x):
    return x["scenario"].rsplit(" [SEP] ", 1)[0]

def _ethics_virtue_second(x):
    return x["scenario"].rsplit(" [SEP] ", 1)[1]

ethics___commonsense = Classification(
    sentence1="input", labels=_ethics_binary_label,
    dataset_name="csv", task_id="ethics/commonsense",
    load_dataset_kwargs={"data_files": {
        "train": "hf://datasets/hendrycks/ethics/data/commonsense/train.csv",
        "validation": "hf://datasets/hendrycks/ethics/data/commonsense/test.csv",
        "test": "hf://datasets/hendrycks/ethics/data/commonsense/test_hard.csv",
    }})
ethics___deontology = Classification(
    sentence1="scenario", sentence2="excuse", labels=name("label", ["unreasonable", "reasonable"]),
    dataset_name="csv", task_id="ethics/deontology",
    load_dataset_kwargs={"data_files": {
        "train": "hf://datasets/hendrycks/ethics/data/deontology/train.csv",
        "validation": "hf://datasets/hendrycks/ethics/data/deontology/test.csv",
        "test": "hf://datasets/hendrycks/ethics/data/deontology/test_hard.csv",
    }})
ethics___justice = Classification(
    sentence1="scenario", labels=name("label", ["unreasonable", "reasonable"]),
    dataset_name="csv", task_id="ethics/justice",
    load_dataset_kwargs={"data_files": {
        "train": "hf://datasets/hendrycks/ethics/data/justice/train.csv",
        "validation": "hf://datasets/hendrycks/ethics/data/justice/test.csv",
        "test": "hf://datasets/hendrycks/ethics/data/justice/test_hard.csv",
    }})
ethics___virtue = Classification(
    sentence1=_ethics_virtue_first, sentence2=_ethics_virtue_second,
    labels=name("label", ["trait not shown", "trait shown"]),
    dataset_name="hendrycks/ethics", config_name="default", task_id="ethics/virtue",
    load_dataset_kwargs=dict(revision=PARQUET, data_dir="virtue"))

def _emocontext_text(x):
    return " ".join(x[field] for field in ("turn1", "turn2", "turn3") if x[field] is not None)

emo = Classification(sentence1=_emocontext_text,
    labels=name("label", ["others", "happy", "sad", "angry"]),
    splits=["train", None, "test"],
    dataset_name="oneonlee/cleansed_emocontext", task_id="emo/emo2019")

# rating is the share of 5 raters who found the query a well-formed question; keep clear cases
google_wellformed_query = Classification("content", question="Is this search query a well-formed question?",
    labels=lambda x: ["not well-formed", "well-formed"][x["rating"] >= 0.8],
    pre_process=lambda ds: ds.filter(lambda x: not 0.2 < x["rating"] < 0.8),
    dataset_name="tasksource/google_wellformed_query")

tweets_hate_speech_detection = Classification(sentence1="tweet", labels="label", splits=["train", None, None])



wnut_17 = TokenClassification(tokens="tokens", labels="ner_tags", dataset_name="flaitenberger/wnut_17",
    task_id="wnut_17/wnut_17")  # data-only mirror

ncbi_disease = TokenClassification(tokens="tokens", labels="ner_tags", dataset_name="ncbi/ncbi_disease",
    task_id="ncbi_disease/ncbi_disease", load_dataset_kwargs=dict(revision=PARQUET, data_dir="ncbi_disease"))

acronym_identification = TokenClassification(labels="labels", tokens="tokens",
    dataset_name="amirveyseh/acronym_identification", task_id="acronym_identification")

jnlpba = TokenClassification(tokens="tokens", labels="ner_tags", splits=["train", "validation", None],
    dataset_name="jnlpba/jnlpba", task_id="jnlpba/jnlpba", load_dataset_kwargs=dict(revision=PARQUET, data_dir="jnlpba"))


# the parquet export lost the tag names; these are the ones the dataset card documents
_ONTONOTES_TAGS = ["O"] + [f"{p}-{t}" for t in ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "DATE", "TIME",
    "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"] for p in "BI"]
SpeedOfMagic_ontonotes_english = TokenClassification(tokens="tokens", labels="ner_tags", dataset_name="SpeedOfMagic/ontonotes_english",
    task_id="ontonotes_english/SpeedOfMagic--ontonotes_english", load_dataset_kwargs=dict(revision=PARQUET),
    pre_process=lambda ds: ds.cast_column("ner_tags", Sequence(ClassLabel(names=_ONTONOTES_TAGS))))

blog_authorship_corpus__gender    = Classification(sentence1="text",labels="gender", question="What is the blogger's gender?",
    dataset_name="tasksource/blog_authorship_corpus")
blog_authorship_corpus__age       = Classification(sentence1="text", question="What is the blogger's age group?",
    labels=lambda x: "13-17" if x["age"] <= 17 else "23-27" if x["age"] <= 27 else "33-48",  # the corpus age groups
    dataset_name="tasksource/blog_authorship_corpus")
blog_authorship_corpus__job       = Classification(sentence1="text",labels="topic", question="In which industry does the blogger work?",
    dataset_name="tasksource/blog_authorship_corpus",
    pre_process=lambda ds: _cast_blog_topics(ds))

def _cast_blog_topics(dataset):
    labels = sorted(set(dataset["train"]["topic"]))
    return DatasetDict({
        split: rows.cast_column("topic", ClassLabel(names=labels))
        for split, rows in dataset.items()
    })

launch_open_question_type = Classification(sentence1="question", labels="resolve_type", dataset_name="Korea-MES/open_question_type")

health_fact = Classification(sentence1="claim", labels="label",
    pre_process=lambda ds: ds.filter(lambda x: x['label'] not in {-1}),
    dataset_name="marcov/health_fact_promptsource", task_id="health_fact")

commonsense_qa = MultipleChoice(
    "question",
    choices_list=get.choices.text,
    labels=lambda x: "ABCDE".index(x["answerKey"]),
    splits=["train","validation",None]
)
mc_taco = Classification(
    lambda x: f'{x["sentence"]}\n{x["question"]}', "answer",
    labels="label", question="Is this answer plausible?",
    splits=[ "validation",None,"test"],
    dataset_name="marcov/mc_taco_promptsource", task_id="mc_taco"
)

ade_corpus_v2___Ade_corpus_v2_classification = Classification("text",labels="label")

discosense = MultipleChoice("context",choices=regen(r"option_[0-3]"),labels="label",
    dataset_name="json", task_id="discosense",
    load_dataset_kwargs={"data_files": {
        "train": "https://raw.githubusercontent.com/prajjwal1/discosense/main/data/discosense_train.json",
        "test": "https://raw.githubusercontent.com/prajjwal1/discosense/main/data/discosense_test.json",
    }})
    
circa = Classification(
    sentence1=cat(["context","question-X"]),
    sentence2="answer-Y",
    labels="goldstandard2", post_process=remove_neg_1)

code_x_glue_cc_defect_detection = Classification("func", labels=lambda x: ["no defect", "defect"][int(x["target"])],
    dataset_name="google/code_x_glue_cc_defect_detection")


phrase_similarity = Classification(
    sentence1=lambda x: f"Phrase: {x['phrase1']}\n{x['sentence1']}",
    sentence2=lambda x: f"Phrase: {x['phrase2']}\n{x['sentence2']}",
    labels=name('label', ['different meaning', 'same meaning']),
    dataset_name="Deehan1866/processed_phrase_similarity",
    task_id="phrase_similarity"
)

exaggeration_detection = Classification(
    sentence1="press_release_conclusion",
    sentence2="abstract_conclusion",
    labels="exaggeration_label", 
    dataset_name="copenlu/scientific-exaggeration-detection"
)
quarel = Classification(
    "question",
    labels=lambda x: "AB"[x["answer_index"]]
)

mwong_fever_evidence_related = Classification(sentence1="claim", sentence2="evidence", labels=name("labels",['unrelated','related']),
    splits=["train", "valid", "test"], dataset_name="mwong/fever-evidence-related")

numer_sense = Classification("sentence", labels="target", dataset_name="tasksource/numer_sense")

def _dynasent_ternary(dataset):
    return dataset.filter(
        lambda row: row["gold_label"] in {"positive", "negative", "neutral"}
    )

dynasent___r1 = Classification(
    "sentence", labels="gold_label", dataset_name="tasksource/dynasent",
    config_name="r1", task_id="dynasent/dynabench.dynasent.{config_name}.all/{config_name}",
    pre_process=_dynasent_ternary)
dynasent___r2 = Classification(
    "sentence", labels="gold_label", dataset_name="tasksource/dynasent",
    config_name="r2", task_id="dynasent/dynabench.dynasent.{config_name}.all/{config_name}",
    pre_process=_dynasent_ternary)

sarcasm_news = Classification("headline", labels=name("is_sarcastic", ["not sarcastic", "sarcastic"]),
    dataset_name="raquiba/Sarcasm_News_Headline")

sem_eval_2010_task_8 = Classification("sentence",labels="relation")

auditor_review = Classification(sentence1="sentence",
    labels=name("label",['negative','neutral','positive']),
    dataset_name="demo-org/auditor_review")

medmcqa = MultipleChoice("question", choices=regen('op[a-d]'),labels='cop')


def _clear_disagreement(dataset):
    # binary_disagreement flags any dissent (one of three annotators); keep clear cases
    # and one row per text (SBIC repeats texts once per annotator)
    def keep(split):
        rows = split.to_pandas().drop_duplicates("text")
        rows = rows[(rows.disagreement_rate == 0) | (rows.disagreement_rate >= 0.5)]
        return Dataset.from_pandas(rows[["text", "disagreement_rate"]], preserve_index=False)
    return DatasetDict({name: keep(split) for name, split in dataset.items()})

def _disagreement(dataset_name, question):
    return Classification("text", question=question,
        labels=lambda x: ["annotators agree", "annotators disagree"][x["disagreement_rate"] > 0],
        dataset_name=dataset_name, pre_process=_clear_disagreement)

dynasent_disagreement = _disagreement("RuyuanWan/Dynasent_Disagreement",
    "Would annotators disagree about the sentiment of this text?")
politeness_disagreement = _disagreement("RuyuanWan/Politeness_Disagreement",
    "Would annotators disagree about the politeness of this text?")
sbic_disagreement = _disagreement("RuyuanWan/SBIC_Disagreement",
    "Would annotators disagree about whether this text is offensive?")
schem_disagreement = _disagreement("RuyuanWan/SChem_Disagreement",
    "Would annotators disagree about whether this rule of thumb is acceptable?")
dilemmas_disagreement = _disagreement("RuyuanWan/Dilemmas_Disagreement",
    "Would annotators disagree about which of these two actions is less ethical?")

logiqa = MultipleChoice(
    cat(["context","query"]),
    choices_list = 'options',
    labels = "correct_option",
    dataset_name="fireworks-ai/logiqa",
    pre_process=lambda ds: ds.map(_logiqa_options)
)


wiki_qa = Classification("question","answer", name("label",['False','True']), question="Does this sentence answer the question?")

cycic_classification = Classification("question",labels=name("correct_answer",['False','True']),
    dataset_name = "tasksource/cycic_classification")
cycic_mc = MultipleChoice("question", choices=regen(r"answer_option[0-4]"), labels="correct_answer",
    dataset_name = "tasksource/cycic_multiplechoice")


sts_companion = Classification("sentence1","sentence2","label",
    dataset_name="tasksource/sts-companion")

commonsense_qa_2 = Classification("question",labels="answer",
    dataset_name="tasksource/commonsense_qa_2.0")

ling_nli = Classification("premise","hypothesis","label",dataset_name="tasksource/lingnli")

monotonicity_entailment = Classification("sentence1", "sentence2", "gold_label",    
    dataset_name="tasksource/monotonicity-entailment")

arct = MultipleChoice(cat(["reason","claim"]),choices=["warrant0","warrant1"],
    labels="correctLabelW0orW1", dataset_name="tasksource/arct")

scinli = Classification("sentence1", "sentence2", labels="label",
    post_process=lambda x:x.shuffle(seed=0),
    dataset_name="tasksource/scinli")

naturallogic = Classification(" sent1 "," sent2 "," new_label ",dataset_name="tasksource/naturallogic")

onestop_qa = MultipleChoice(cat(["paragraph","question"]),choices_list="answers",
    labels=constant(0))

moral_stories = MultipleChoice(cat(["situation","intention"]),
    choices=['moral_action',"immoral_action"],labels=constant(0),
    dataset_name="LabHC/moral_stories", task_id="moral_stories/full")

def _prost_label(x):
    value = str(x["label"]).strip()
    return "ABCD".index(value) if value in "ABCD" else int(value)

prost = MultipleChoice(cat(["context","ex_question"]), choices=['A','B','C','D'],
    labels=_prost_label, dataset_name="json", task_id="prost",
    load_dataset_kwargs={"data_files":
        "hf://datasets/corypaik/prost/data/default.jsonl"})

dyna_hate = Classification("text",labels="label",dataset_name="tasksource/dynahate",splits=['train',None,None])

syntactic_augmentation_nli = Classification('sentence1',"sentence2","gold_label",dataset_name="tasksource/syntactic-augmentation-nli")

autotnli = Classification("premises", "hypothesis", "label", dataset_name="tasksource/autotnli")

conqada = Classification("sentence1","sentence2","label",dataset_name="lasha-nlp/CONDAQA",
    pre_process = lambda ds:ds.filter(lambda x:x['label'] in {"DON'T KNOW","YES","NO"})
)

def _webgpt_question_text(row):
    import ast
    value = row["question"]
    question = ast.literal_eval(value) if isinstance(value, str) else value
    return question["full_text"]

webgpt_comparisons = MultipleChoice(
    _webgpt_question_text, choices=['answer_0','answer_1'],
    labels=lambda x:int(float(x['score_1']) > 0), question="Which answer did the human rater prefer?",
    dataset_name="heegyu/webgpt_comparisons_ko", task_id="webgpt_comparisons",
    # score_1 == 0 is a tie (27% of rows), which the label would read as answer_0 winning
    pre_process=lambda ds: ds.filter(lambda x: float(x['score_1']) != 0
        and str(x['answer_0']).strip() and str(x['answer_1']).strip()))

synthetic_instruct = MultipleChoice('prompt', choices=['chosen', 'rejected'],
    labels=constant(0), question="Which response is better?", dataset_name="Dahoas/synthetic-instruct-gptj-pairwise")

scruples = Classification("text",labels="binarized_label", question="Was the author in the right or in the wrong?",dataset_name="tasksource/scruples")

wouldyourather = MultipleChoice(constant(''), choices=['option_a','option_b'], question="Which would most people rather do?",
    labels= lambda x: int(x['votes_a']<x['votes_b']),
    # clear majorities only: at least 100 votes and twice as many for the winner
    pre_process=lambda ds: ds.filter(lambda x: x['votes_a'] + x['votes_b'] >= 100
                                     and max(x['votes_a'], x['votes_b']) >= 2 * min(x['votes_a'], x['votes_b'])),
    dataset_name="tasksource/wouldyourather")


defeasible_nli = Classification(cat(["Premise","Hypothesis"]),"Update",labels="UpdateType",
    dataset_name="tasksource/defeasible-nli",config_name=['atomic', 'snli'])

defeasible_nli_social = Classification("Hypothesis", "Update", labels="UpdateType",
    dataset_name="tasksource/defeasible-nli", config_name='social')  # social has no premise

help_nli = Classification("ori_sentence","new_sentence","gold_label",
    dataset_name="tasksource/help-nli")
    
nli_veridicality_transitivity = Classification("sentence1","sentence2","gold_label",
    dataset_name="tasksource/nli-veridicality-transitivity")

lonli = Classification("premise","hypothesis","label",
    dataset_name="tasksource/lonli")

dadc_limit = Classification("sentence1","sentence2","label",
    dataset_name="tasksource/dadc-limit-nli")

flute = Classification("premise","hypothesis","label",
    dataset_name="ColumbiaNLP/FLUTE")

strategy_qa = Classification('question',labels='answer',
    dataset_name="tasksource/strategy-qa",splits=['train',None,None])

summarize_from_feedback = MultipleChoice(get.info.post,
    choices_list=lambda x: [x['summaries'][0]['text'],x['summaries'][1]['text']],
    labels="choice", question="Which summary did the human rater prefer?",
    dataset_name="vwxyzjn/summarize_from_feedback_oai_preprocessing",
    task_id="summarize_from_feedback/comparisons",
    pre_process = lambda ds:ds.filter(lambda x: type(get.info.post(x))==str)
)

folio = Classification("premises","conclusion",
    labels=lambda x:{'False':'contradiction','True':'entailment', 'Uncertain':'neutral'}.get(x["label"]),
    dataset_name="tasksource/folio")

tomi_nli = Classification("premise","hypothesis","label",
    dataset_name="tasksource/tomi-nli")

avicenna = Classification("Premise 1","Premise 2","Syllogistic relation", question="Do the two premises form a syllogism?",
    dataset_name="tasksource/avicenna")

shp = MultipleChoice(
    lambda x: f"r/{x['domain'].rsplit('_', 1)[0]}: {x['history']}", question="Which reply did readers prefer?",
    choices=['human_ref_A','human_ref_B'],
    labels=lambda x: 1 - x['labels'],  # labels is 1 when A is preferred
    # the SHP authors recommend a score ratio of at least 2; closer pairs are near ties
    pre_process=lambda ds: ds.filter(lambda x: x['score_ratio'] >= 2),
    dataset_name="stanfordnlp/SHP")

medqa_usmle = MultipleChoice('sent1',choices=regen('ending[0-3]'),labels='label',
    dataset_name="GBaker/MedQA-USMLE-4-options-hf")

wikimedqa = MultipleChoice("text",choices=regen(r"option_[0-7]"),labels='label',
    dataset_name="sileod/wikimedqa",
    config_name=["medwiki"])

def _cicero_input(x):
    dialogue = "\n".join(x['Dialogue'])
    question = x['Question'].replace("target", "the target utterance")
    return f"{dialogue}\n\nTarget utterance: {x['Target']}\n{question}"

cicero = MultipleChoice(_cicero_input,
    choices_list="Choices", labels=lambda x:x['Human Written Answer'][0],
    dataset_name="declare-lab/cicero")

creak = Classification("sentence",labels="label",
    dataset_name='amydeng2000/CREAK')

mutual = MultipleChoice("article",choices_list="options",
    labels=lambda x: "ABCD".index(x['answers']),
    dataset_name="tasksource/mutual",splits=["train",None,None])

puzzte = Classification("puzzle_text","question","answer",
    dataset_name="tasksource/puzzte",
    # "non-entailment" overlaps the specific contradiction/unknown labels
    pre_process=lambda ds: ds.filter(lambda x: x["answer"] != "non-entailment"))

implicatures = MultipleChoice(cat(['context','response'],"\n"),
    choices=['correct_implicature','incorrect_implicature'],
    labels=constant(0),
    dataset_name='tasksource/implicatures')

race = MultipleChoice(cat(['question','article'],'\n'), choices_list='options',
    labels=lambda x:'ABCDE'.index(x['answer']),
    config_name=['middle','high'])

race_c = MultipleChoice(cat(['question','article'],'\n'),choices_list='option',labels='label',
    dataset_name='tasksource/race-c')

spartqa_yn=Classification("story","question",lambda x: {"DK": "don't know"}.get(x["answer"], x["answer"]),
    dataset_name="tasksource/spartqa-yn")

spartqa_mc=MultipleChoice(cat(["story","question"]),choices_list="candidate_answers",labels="answer",
    dataset_name="tasksource/spartqa-mchoice")

temporal_nli = Classification("Premise","Hypothesis","Label",
    dataset_name="tasksource/temporal-nli")

riddle_sense = MultipleChoice("question", choices_list=get.choices.text,
    labels=lambda x : "ABCDE".index(x['answerKey']),
    dataset_name="jeggers/riddle_sense",
    pre_process=lambda ds: ds.map(_parse_jeggers_riddle_choices))

clcd = Classification(
    "sentence1","sentence2","label",
    dataset_name="tasksource/clcd-english")

TWENTYQUESTIONS_ANSWERS = [
    "never", "rarely", "sometimes", "usually", "always", "irrelevant",
]

def _twentyquestions_answers(dataset):
    dataset = dataset.filter(lambda row: row["answer"] is not None)
    return dataset.cast_column("answer", ClassLabel(names=TWENTYQUESTIONS_ANSWERS))

twentyquestions = Classification(
    lambda row: f"Subject: {row['subject']}\nQuestion: {row['question']}",
    labels="answer", dataset_name="tasksource/twentyquestions",
    pre_process=_twentyquestions_answers)

reclor = MultipleChoice(cat(["context","question"]),choices_list="answers",labels="label",
    dataset_name="tasksource/reclor",splits=['train','validation',None])

c_aug_imdb = Classification("Text",labels="Sentiment",
    dataset_name='tasksource/counterfactually-augmented-imdb')

c_aug_snli = Classification("sentence1","sentence2","gold_label",
    dataset_name='tasksource/counterfactually-augmented-snli')

cnli = Classification("premise","hypothesis","label",
    dataset_name='tasksource/cnli')

perturbed_boolq = Classification("question",labels="hard_label",
    dataset_name='tasksource/boolq-natural-perturbations')


graded_acceptability = Classification("text",labels="normalized_score",
    question="How acceptable is this sentence, from 0 (unacceptable) to 1 (acceptable)?",
    dataset_name="tasksource/acceptability-prediction")

equate = Classification("sentence1","sentence2","gold_label",
    dataset_name='tasksource/equate')

science_qa = MultipleChoice("question",choices_list="choices",labels="answer",
    dataset_name="tasksource/ScienceQA_text_only")

ekar=MultipleChoice("question",choices_list=get.choices.text, question="Which pair is related in the same way?",
    labels=lambda x:"ABCD".index(x['answerKey']),
dataset_name="Jiangjie/ekar_english")

implicit_hate = Classification("post",labels="class",
    dataset_name="tasksource/implicit-hate-stg1")

nli_unambiguity = Classification("premise","hypothesis","gini",
    question="How much would annotators agree on the inference, from 0 (evenly split) to 1 (unanimous)?",
    dataset_name="tasksource/chaos-mnli-ambiguity")

headline_cause = Classification('left_title', 'right_title', 'label', dataset_name='json', task_id='headline_cause/en_simple',
    load_dataset_kwargs={"data_files": {split: f"hf://datasets/IlyaGusev/headline_cause/en/simple/{name}.jsonl"
                                        for split, name in [("train", "train"), ("validation", "val"), ("test", "test")]}},
    label_values={0: "no causal link", 1: "first headline caused the second", 2: "second headline caused the first"})

logiqa_2 = Classification("premise","hypothesis","label",dataset_name="tasksource/logiqa-2.0-nli")

_oasst = dict(dataset_name="tasksource/oasst2_dense_flat",
    pre_process = lambda ds:ds.filter(lambda x:x['lang']=='en'))

oasst1__quality = Classification("parent_text","text",labels="quality",**_oasst,
    question="How good is the reply, from 0 (low quality) to 1 (high quality)?")
oasst1__toxicity = Classification("parent_text","text",labels="toxicity",**_oasst,
    question="How toxic is the reply, from 0 (not toxic) to 1 (very toxic)?")
oasst1__helpfulness = Classification("parent_text","text",labels="helpfulness",**_oasst,
    question="How helpful is the reply, from 0 (unhelpful) to 1 (helpful)?")

mindgames = Classification("premise","hypothesis","label",dataset_name="sileod/mindgames")

def _udep_deprel_pre_process(dataset):
    labels = sorted({
        label
        for split in dataset.values()
        for sequence in split["deprel"]
        for label in sequence
    })
    return DatasetDict({
        name: split.cast_column("deprel", Sequence(ClassLabel(names=labels)))
        for name, split in dataset.items()
    })

udep__deprel = TokenClassification(
    "tokens", "deprel",
    config_name=udep_en_configs,
    dataset_name="universal-dependencies/universal_dependencies",
    pre_process=_udep_deprel_pre_process)

ambient= Classification("premise","hypothesis","hypothesis_ambiguous",dataset_name="tasksource/ambient",
    question="Is the hypothesis ambiguous?")

path_naturalness = MultipleChoice(constant(''),choices=['choice1','choice2'],labels="label",
    question="Which chain of relations is more natural?",
    dataset_name="tasksource/path-naturalness-prediction")

def _civil(attribute, negative, positive, flag):
    # attributes are the share of raters who flagged the comment; keep clear cases
    return Classification("text", labels=lambda x: [negative, positive][x[attribute] >= 0.5],
        question=f"Would most raters flag this comment as {flag}?",
        pre_process=lambda ds: ds.filter(lambda x: not 0.1 <= x[attribute] < 0.5), dataset_name="google/civil_comments")

civil_comments__toxicity = _civil("toxicity", "not toxic", "toxic", "toxic")
civil_comments__severe_toxicity = _civil("severe_toxicity", "not severely toxic", "severely toxic", "severely toxic")
civil_comments__obscene = _civil("obscene", "not obscene", "obscene", "obscene")
civil_comments__threat = _civil("threat", "no threat", "threat", "a threat")
civil_comments__insult = _civil("insult", "not insulting", "insulting", "insulting")
civil_comments__identity_attack = _civil("identity_attack", "no identity attack", "identity attack", "an identity attack")
civil_comments__sexual_explicit = _civil("sexual_explicit", "not sexually explicit", "sexually explicit", "sexually explicit")

cloth = MultipleChoice("sentence", choices_list=lambda x:[x["answer"]]+x["distractors"],labels=constant(0), dataset_name="AndyChiang/cloth")
dgen  = MultipleChoice("sentence", choices_list=lambda x:[x["answer"]]+x["distractors"],labels=constant(0), dataset_name="AndyChiang/dgen")

i2d2 = Classification("sentence1",labels=name('label',['False','True']), dataset_name="tasksource/I2D2")

arg_me = Classification(
    'argument', 'conclusion', 'stance', dataset_name="webis/args_me", task_id="args_me",
    load_dataset_kwargs=dict(revision=PARQUET, data_dir="corpus"))  # one argument per row
valueeval_stance = Classification(
    "Premise", "Conclusion", "Stance", dataset_name="csv",
    task_id="Touche23-ValueEval",
    load_dataset_kwargs={"data_files": {
        "train": "https://zenodo.org/records/7879430/files/arguments-training.tsv",
        "validation": "https://zenodo.org/records/7879430/files/arguments-validation.tsv",
        "test": "https://zenodo.org/records/7879430/files/arguments-test.tsv",
    }, "delimiter": "\t"})
starcon = Classification('argument','topic','label',dataset_name="tasksource/starcon")

banking77 = Classification("text",labels="label",dataset_name="legacy-datasets/banking77")

it_support_tickets = Classification(
    "text", labels="label", dataset_name="tasksource/it-support-tickets",
    splits=["train", None, "test"])
    
control = Classification('premise','hypothesis',"label",dataset_name="tasksource/ConTRoL-nli")
tracie = Classification("premise","hypothesis","answer",dataset_name='tasksource/tracie')
sherliic = Classification("premise","hypothesis","label",dataset_name='tasksource/sherliic')

sen_making__1 = MultipleChoice(constant(''), choices=['sentence0','sentence1'],labels='false',
    question="Which statement makes sense?",
    dataset_name="tasksource/sen-making")

sen_making__2 = MultipleChoice(lambda x: [x['sentence0'],x['sentence1']][x['false']],
    question="Why is this statement implausible?", choices=['A','B','C'],labels=lambda x: 'ABC'.index(x['reason']), dataset_name="tasksource/sen-making")

winowhy = Classification('sentence', lambda x: f'In "{x["wnli_sent1"]}", {x["wnli_sent2"]}',
    labels=name('label',['False','True']), question="Is this explanation correct?", dataset_name="tasksource/winowhy")


robustLR = Classification("context","statement","label", dataset_name="tasksource/robustLR")

cluttr = Classification("story", "query", "label", dataset_name="tasksource/clutrr")

logical_fallacy = Classification("source_article", labels="logical_fallacies", dataset_name="tasksource/logical-fallacy")

parade = Classification("Definition1","Definition2", labels=name('Binary labels',["not-paraphrase","paraphrase"]), dataset_name="tasksource/parade")

cladder = Classification("given_info", "question", "answer",dataset_name="tasksource/cladder")

subjectivity = Classification("Sentence",labels=lambda x: {"OBJ": "objective", "SUBJ": "subjective"}[x["Label"]],dataset_name="tasksource/subjectivity")

moh   = Classification("context","expression","label", dataset_name="tasksource/MOH")
vuac  = Classification("context","expression","label", dataset_name="tasksource/VUAC")
trofi = Classification(
    "context", "expression", "label", dataset_name="parquet", task_id="TroFi",
    load_dataset_kwargs={"data_files": {
        "train": "hf://datasets/tasksource/TroFi/data/train-00000-of-00001-67b67b8474db644d.parquet",
        "test": "hf://datasets/tasksource/TroFi/data/test-00000-of-00001-a467035ce73d87fe.parquet",
    }}, splits=['train', None, 'test'])

sharc_classification = Classification("snippet",
    lambda x: "\n".join(part for part in (x["scenario"], x["question"], x["history"]) if part),
    labels="label", dataset_name="tasksource/sharc")

conceptrules_v2 = Classification("context", "text", "label", question="Is the statement true given the context?", dataset_name="tasksource/conceptrules_v2")

scidtb = Classification("unit1_txt","unit2_txt","label", dataset_name="multilingual-discourse-hub/disrpt",config_name='eng.dep.scidtb.rels')

chunking = TokenClassification("tokens","chunk_tags", dataset_name="eriktks/conll2000", task_id="conll2000",
    load_dataset_kwargs=dict(revision=PARQUET))

few_nerd = TokenClassification("tokens","fine_ner_tags",dataset_name="DFKI-SLT/few-nerd",config_name='supervised')
finer = TokenClassification('tokens','ner_tags',dataset_name='nlpaueb/finer-139',
    load_dataset_kwargs=dict(revision=PARQUET, data_dir="finer-139"))

label_nli = Classification("premise","hypothesis","labels",dataset_name='tasksource/zero-shot-label-nli')

com2sense = Classification("sent",labels="label",dataset_name="tasksource/com2sense",splits=['train',"validation",None])

scone = Classification('sentence1_edited','sentence2_edited','gold_label_edited',dataset_name="tasksource/scone")

winodict = MultipleChoice(cat(['definition','sentence']),['option1','option2'],'label',dataset_name='tasksource/winodict')

fool_me_twice = Classification(
    lambda x: " ".join(a['text'] for a in x['gold_evidence']),
    'text', 'label', dataset_name='tasksource/fool-me-twice')

monli = Classification("sentence1","sentence2","gold_label", dataset_name="tasksource/monli")

causality = Classification('premise','hypothesis','relation', dataset_name='tasksource/corr2cause')

lsat = MultipleChoice(cat(['passage','question']), choices_list='references',labels='gold_index',dataset_name='lighteval/lsat_qa',config_name='all')

apt = Classification('text_a','text_b',name('labels',['not_paraphrase','paraphrase']),dataset_name='tasksource/apt')


financial_sentiment = Classification("text",labels=name('label',['Bearish','Bullish','Neutral']),
    dataset_name="zeroshot/twitter-financial-news-sentiment")

def _icl_rand(x):
    import random
    return random.Random(x['sentence1'][:50]).randint(0,1) #deterministic label for each input

icl = Classification("inputs", lambda x: x['symbols'][_icl_rand(x)],
    labels=lambda x: str(x['symbols'][_icl_rand(x)]==x['targets']), question="Is this the right label for the last input?",
    dataset_name="tasksource/icl-symbol-tuning-instruct",
    pre_process=lambda ds:ds.filter(lambda x:len(x['inputs'])<500*4), # 500 tokens of 4 char 
)

space_nli = Classification("premises","hypothesis","label",dataset_name="tasksource/SpaceNLI")

propsegment = Classification("hypothesis","premise",
    labels = lambda x:{'n':'neutral','e':'entailment','c':'contradiction'}[x['label']],
    dataset_name="json", task_id="propsegment/nli",
    load_dataset_kwargs={"data_files": {
        "train": "https://raw.githubusercontent.com/schen149/PropSegmEnt/main/propnli.train.jsonl",
        "validation": "https://raw.githubusercontent.com/schen149/PropSegmEnt/main/propnli.dev.jsonl",
        "test": "https://raw.githubusercontent.com/schen149/PropSegmEnt/main/propnli.test.jsonl",
    }})

hatemoji = Classification('text',labels=name("label_gold", ['not-hate-speech','hate-speech']),
    dataset_name="HannahRoseKirk/HatemojiBuild")

regset = Classification("context",labels="answer", question="Does the string match the regular expression?",dataset_name='tasksource/regset')

def _esci_product(x): # product_text writes missing fields as 'None' lines
    fields = ['product_title','product_brand','product_color','product_description','product_bullet_point']
    return "\n".join(str(x[f]) for f in fields if x[f] not in (None, "", "None"))

esci = Classification('query',_esci_product,'esci_label',
    dataset_name="tasksource/esci",
    pre_process=lambda ds:ds.filter(lambda x:x['product_locale']=='us'))

def _preprocess_chatbot_arena(ds):
    ds=ds.filter(lambda x:x['winner'] in ["model_a","model_b"])
    ds=ds.filter(lambda x:x['language']=="English")

    def _unroll(x):
        # single-turn: the prompt is the state and the replies are the options
        single = x['turn'] == 1
        f=lambda x:"\n".join([f"{turn['role']}:\n{turn['content']}" for turn in x])
        x['prompt'] = x['conversation_a'][0]['content'] if single else ""
        x['conversation_a'] = x['conversation_a'][1]['content'] if single else f(x['conversation_a'])
        x['conversation_b'] = x['conversation_b'][1]['content'] if single else f(x['conversation_b'])
        return x
    ds=ds.map(_unroll)
    return ds

chatbot_arena = MultipleChoice("prompt",
    choices=["conversation_a","conversation_b"],
    labels=lambda x: ["model_a","model_b"].index(x["winner"]), question="Which assistant did the user prefer?",
    dataset_name="lmsys/chatbot_arena_conversations",
    pre_process=_preprocess_chatbot_arena)

dnd_intent = Classification("examples",labels="label_names",
    dataset_name='neurae/dnd_style_intents')

fld = Classification("context","hypothesis", "proof_label",
    dataset_name="hitachi-nlp/FLD.v2",config_name="default")

flds = Classification("context","hypothesis", "proof_label",
    dataset_name="hitachi-nlp/FLD.v2",config_name="star")

sdoh_nli = Classification("premise","hypothesis",labels=lambda x:{True:"entailment",False:"not_entailment"}[x['label']],
    dataset_name="tasksource/SDOH-NLI")

scifact_entailment = Classification(lambda x:"\n".join(x["abstract"]),"claim",
    labels=lambda x:x['verdict'].replace('NEI','NEUTRAL').lower(),
    dataset_name="tasksource/scifact_entailment")

feasibilityQA = Classification(cat(['knowledge','premise']),'hypothesis','binary_classification_label',
    dataset_name="tasksource/feasibilityQA")
                               
simple_pair = Classification("premise","hypothesis","label", dataset_name="tasksource/simple_pair")
adjective_scale_probe = Classification("premise","hypothesis","label", dataset_name="tasksource/AdjectiveScaleProbe-nli")
repectively_nli = Classification("premise","hypothesis","label",dataset_name="tasksource/resnli")

spartun=MultipleChoice(cat(["story","question"]),choices_list="candidate_answers",
    labels=lambda x: [c.lower() for c in x['choices_list']].index(x["answer"][0].lower()),
    pre_process=lambda ds:ds.filter(lambda x:len(x['answer'])==1),
    dataset_name="tasksource/SpaRTUN")

resq=MultipleChoice(cat(["story","question"]),choices_list="candidate_answers",
    labels=lambda x: [c.lower() for c in x['choices_list']].index(x["answer"][0].lower()),
    pre_process=lambda ds:ds.filter(lambda x:len(x['answer'])==1),
    dataset_name="tasksource/ReSQ")

semantic_fragments_nli = Classification("sentence1","sentence2","gold_label",
    dataset_name="tasksource/semantic_fragments_nli")

moritz_zs_nli = Classification('text','hypothesis','labels',
    pre_process=lambda ds:ds.filter(lambda x:x['task_name'] not in  ["mnli", "anli", "fevernli", "wanli", "lingnli"]),
    dataset_name="MoritzLaurer/dataset_train_nli"
) 

stepgame = Classification('story','question','label',dataset_name="tasksource/stepgame")

def _nlgraph_binarize(x):
    a=x['answer'].lower()
    if "yes" in a: return "True"
    if "no" in a: return "False"
    assert False

nlgraph = Classification('question',labels=_nlgraph_binarize,
    pre_process=lambda ds:ds.filter(lambda x:x['task'] in "connectivity cycle hamilton"),
    dataset_name="tasksource/nlgraph")

oasst_rlhf = MultipleChoice("prompt",choices=['chosen','rejected'],labels=constant(0), question="Which reply is better?",
    dataset_name="tasksource/oasst2_pairwise_rlhf_reward")

def _hh_split(ds):
    # chosen/rejected repeat the whole dialogue; keep it once and compare the final replies
    marker = "\n\nAssistant:"
    ds = ds.filter(lambda x: x["chosen"][:x["chosen"].rfind(marker)] == x["rejected"][:x["rejected"].rfind(marker)])
    return ds.map(lambda x: {"dialogue": x["chosen"][:x["chosen"].rfind(marker)].strip(),
        "chosen_reply": x["chosen"][x["chosen"].rfind(marker) + len(marker):].strip(),
        "rejected_reply": x["rejected"][x["rejected"].rfind(marker) + len(marker):].strip()})

anthropic_rlhf_helpfulness = MultipleChoice("dialogue",
    ['chosen_reply','rejected_reply'], constant(0), pre_process=_hh_split, question="Which next assistant reply is more helpful?",
    dataset_name="tasksource/hh-rlhf",config_name=["helpful-base", "helpful-online", "helpful-rejection-sampled"])

anthropic_rlhf_harmless = MultipleChoice("dialogue",
    ['chosen_reply','rejected_reply'], constant(0), pre_process=_hh_split, question="Which next assistant reply is more harmless?",
    dataset_name="tasksource/hh-rlhf",config_name="harmless-base")

ruletaker = Classification(
    "context", "question", question="Does the statement follow from the context? What is not explicitly stated as true is considered false.",
    labels="label", dataset_name="tasksource/ruletaker")

para_rules = Classification(
    "context", "question", question="Is the statement true? What is not explicitly stated as true is considered false.", labels=name("label",["False","True"]),
    dataset_name="qbao775/PARARULE-Plus")

proofwriter_deduction = Classification("theory","question","answer",
    dataset_name="tasksource/proofwriter") #open world assumption

logical_entailment = Classification("A","B","label",dataset_name='tasksource/logical-entailment')

nope = Classification('premise','hypothesis',
    labels=lambda x:dict(E='entailment',N='neutral',C='contradiction').get(x['label'],x['label']),
    dataset_name='tasksource/nope')

logicNLI = Classification('premise','hypothesis','label',dataset_name='tasksource/LogicNLI')

contract_nli__seg = Classification("premise","hypothesis","label", dataset_name="tasksource/contract-nli",config_name="contractnli_a")

contract_nli__full = Classification("premise","hypothesis","label", dataset_name="tasksource/contract-nli",config_name="contractnli_b")

nli4ct = Classification(lambda x: "\n".join(x['Primary_evidence']),'Statement',"Label",
    dataset_name="AshtonIsNotHere/nli4ct_semeval2024",splits=['train','dev',None])

lsat_ar = MultipleChoice(
    cat(['context','question']),
    choices_list='answers',labels="label",
     dataset_name="tasksource/lsat-ar")
    
lsat_rc = MultipleChoice(
    cat(['context','question']),
    choices_list='answers',labels="label",
     dataset_name="tasksource/lsat-rc")
    
biosift_nli = Classification("Abstract","Hypothesis",
    labels=lambda x: {True:"entailment",False:"not-entailment"}[bool(x['Entailment'])],
    dataset_name="AshtonIsNotHere/biosift-nli")

brainteasers = MultipleChoice("question",
    choices_list=lambda x:eval(x["choice_list"]),
    labels="label",
    dataset_name="tasksource/brainteasers",config_name=['WP','SP'])

# toxicity_human is a 1-5 mean rating; the ambiguous middle is dropped
toxigen = Classification("text", labels=lambda x: ["benign", "toxic"][x["toxicity_human"] >= 4],
    pre_process=lambda ds: ds.filter(lambda x: not 2 < x["toxicity_human"] < 4),
    dataset_name="skg/toxigen-data", config_name="annotated")

def _support_shift_name(shift):
    if shift == 0:
        return "no change in support"
    direction = "increases" if shift > 0 else "decreases"
    amount = abs(shift)
    return f"support {direction} by {amount} point{'s' if amount != 1 else ''}"

persuasiveness = Classification(
    "claim", "argument", labels="persuasiveness_metric",
    dataset_name="Anthropic/persuasion",
    label_values={shift: _support_shift_name(shift) for shift in range(-2, 6)})


ambigNQ = Classification("question",labels=lambda x:{True:"ambiguous", False:"not ambiguous"}.get(x["ambig"]),
    dataset_name="erbacher/AmbigNQ-clarifying-question")

siga_nli = Classification("premise","statement","label",dataset_name="tasksource/SIGA-nli")

unigram_fol = Classification("premise","hypothesis","label",dataset_name='unigram/FOL-nli')

gs_goal = MultipleChoice(lambda x: f"Step: {x['sent2']}\nGoal:", regen("ending[0-3]"), "label",
        dataset_name="tasksource/goal-step-wikihow", config_name="goal")

gs_step = MultipleChoice(lambda x: f"Goal: {x['sent2']}\nStep:", regen("ending[0-3]"), "label",
        dataset_name="tasksource/goal-step-wikihow", config_name="step")

gs_order = MultipleChoice("sent2",regen("ending[0-1]"),"label",
        dataset_name="tasksource/goal-step-wikihow",config_name="order")

paradise = MultipleChoice("sent2",regen("ending[0-3]"),"label",
      dataset_name="GGLab/PARADISE")

docnli = Classification("premise","hypothesis","label",dataset_name="tasksource/doc-nli")

mctest_nli = Classification("premise","hypothesis","label",dataset_name="tasksource/mctest-nli")

patent_phrase_similarity = Classification("anchor","target","label",dataset_name="tasksource/patent-phrase-similarity")

nlsat = Classification('sentence',labels='label',dataset_name="tasksource/natural-language-satisfiability")

idioms_nli = Classification('premise','hypothesis','label',dataset_name="tasksource/idioms-nli")

lifeycle_entailment = Classification("premise","hypothesis","label",dataset_name='tasksource/lifecycle-entailment')


# modern classification / relation extraction datasets

# safety / prompt-injection classifier training datasets

prompt_injection_xtram = Classification(
    "text", labels="label",
    dataset_name="xTRam1/safe-guard-prompt-injection",
    question="Is this prompt a prompt-injection attempt?",
    label_values={0: "benign", 1: "injection"})

prompt_injection_deepset = Classification(
    "text", labels="label",
    dataset_name="deepset/prompt-injections",
    question="Is this prompt a prompt-injection attempt?",
    label_values={0: "benign", 1: "injection"})

prompt_injection_slabs = Classification(
    "text", labels="label",
    dataset_name="S-Labs/prompt-injection-dataset",
    question="Is this prompt a prompt-injection attempt?",
    label_values={0: "benign", 1: "injection"})

prompt_injection_neuralchemy = Classification(
    "text", labels="label",
    dataset_name="neuralchemy/Prompt-injection-dataset", config_name="full",
    question="Is this prompt malicious or a prompt-injection attempt?",
    label_values={0: "benign", 1: "malicious"})

prompt_shield = Classification(
    "prompt", labels="label",
    dataset_name="hendzh/PromptShield",
    question="Is this prompt a prompt-injection attempt?",
    label_values={0: "benign", 1: "injection"})

shell_safety = Classification(
    lambda x: f"Session context: {x['session_context']}\nCommand: {x['command']}",
    labels="label", dataset_name="tomngdev/shell-safety-v2",
    question="What safety decision should be made before running this shell command?")

agent_action_safety = Classification(
    lambda x: "\n".join(
        f"{field}: {x[field]}" for field in
        ["original_goal", "user_message", "context", "constraints", "conversation", "action"]
        if x.get(field) not in (None, "", [])
    ),
    labels=name("is_safe", ["unsafe", "safe"]),
    dataset_name="json", task_id="agent_action_safety",
    splits=["train", "validation", None],
    load_dataset_kwargs={"data_files": {
        "train": "hf://datasets/karanxa/agent-action-safety-dataset/train.jsonl",
        "validation": "hf://datasets/karanxa/agent-action-safety-dataset/val.jsonl",
    }},
    question="Is this proposed agent action safe in the given context?")

shell_risk = Classification(
    "command", labels="label", label_values={0: "not risky", 1: "risky"},
    dataset_name="kontext-security/ShellRisk-Bench",
    question="Is this shell command risky?")

wildguardmix__prompt_harm = Classification(
    "prompt", labels="prompt_harm_label",
    dataset_name="bogdanminko/wildguardmix-cleaned",
    question="Is this user prompt harmful?")

wildguardmix__response_harm = Classification(
    "prompt", "response", labels="response_harm_label",
    dataset_name="bogdanminko/wildguardmix-cleaned",
    question="Is the assistant response harmful?")

wildguardmix__response_refusal = Classification(
    "prompt", "response", labels="response_refusal_label",
    dataset_name="bogdanminko/wildguardmix-cleaned",
    question="Does the assistant response refuse the request?")

beavertails_safety = Classification(
    "prompt", "response", labels=name("is_safe", ["unsafe", "safe"]),
    dataset_name="PKU-Alignment/BeaverTails",
    splits=["330k_train", None, "330k_test"],
    question="Is the assistant response safe?")


toxic_chat__toxicity = Classification(
    "user_input", labels=name("toxicity", ["not toxic", "toxic"]), question="Is this user prompt toxic?",
    dataset_name="lmsys/toxic-chat", config_name="toxicchat0124",
    splits=["train", None, "test"])

toxic_chat__jailbreaking = Classification(
    "user_input", labels=name("jailbreaking", ["not jailbreak", "jailbreak"]),
    question="Is this user prompt a jailbreak attempt?",
    dataset_name="lmsys/toxic-chat", config_name="toxicchat0124",
    splits=["train", None, "test"])

clinc_oos = Classification(
    "text", labels="intent",
    dataset_name="clinc/clinc_oos", config_name="plus")

def _records(x):
    if not isinstance(x, dict):
        return x
    return [dict(zip(x, values)) for values in zip(*(x[k] for k in x))]

def _fewrel_relation_match(dataset):
    rows = {}
    for split in ["train_wiki", "val_wiki", "val_nyt"]:
        data = dataset[split]
        relations = {}
        for x in data:
            relations.setdefault(
                x["relation"],
                x["names"][0] if x["names"] and x["names"][0] else x["relation"],
            )
        relation_ids = sorted(relations)
        next_relation = {
            r: relation_ids[(i + 1) % len(relation_ids)]
            for i, r in enumerate(relation_ids)
        }
        examples = []
        for x in data:
            text = " ".join(x["tokens"])
            relation = relations[x["relation"]]
            negative = relations[next_relation[x["relation"]]]
            examples += [
                {"text": text, "relation": relation, "label": 1},
                {"text": text, "relation": negative, "label": 0},
            ]
        rows[split] = Dataset.from_list(examples).cast_column(
            "label", ClassLabel(names=["negative", "positive"]))
    return DatasetDict(rows)

fewrel = Classification(
    "text", "relation", "label",
    dataset_name="tasksource/few_rel", config_name="default",
    splits=["train_wiki", "val_wiki", "val_nyt"],
    pre_process=_fewrel_relation_match)

def _yufei_docred_to_columnar(dataset):
    """Map YufeiHFUT raw labels ({h,t,r}) to the columnar form _docred_relations expects."""
    out = {}
    for split in dataset:
        def convert(x):
            labels = x["labels"] or []
            return {"labels": {
                "head": [r["h"] for r in labels],
                "tail": [r["t"] for r in labels],
                "relation_id": [r["r"] for r in labels],
                "relation_text": [r["r"] for r in labels],
            }}
        out[split] = dataset[split].map(convert)
    return DatasetDict(out)

def _docred_relations(dataset):
    rows = {}
    for split in ["train_annotated", "validation"]:
        examples = []
        for x in dataset[split]:
            text = "\n".join(" ".join(sent) for sent in x["sents"])
            entities = [
                " / ".join(dict.fromkeys(mention["name"] for mention in entity))
                for entity in x["vertexSet"]
            ]
            for relation in _records(x["labels"]):
                examples.append({
                    "text": text,
                    "entity_pair": f'{entities[relation["head"]]} -> {entities[relation["tail"]]}',
                    "relation": relation["relation_text"] or relation["relation_id"],
                })
        rows[split] = Dataset.from_list(examples)
    relation_names = sorted({
        relation for split_rows in rows.values() for relation in split_rows["relation"]
    })
    return DatasetDict({
        split: split_rows.cast_column("relation", ClassLabel(names=relation_names))
        for split, split_rows in rows.items()
    })

docred = Classification(
    "text", "entity_pair", "relation",
    dataset_name="json", task_id="docred",
    splits=["train_annotated", "validation", None],
    load_dataset_kwargs={"data_files": {
        "train_annotated": "hf://datasets/YufeiHFUT/DocRED_origin/train_annotated.json",
        "validation": "hf://datasets/YufeiHFUT/DocRED_origin/dev.json",
    }},
    pre_process=lambda ds: _docred_relations(_yufei_docred_to_columnar(ds)))

def _chemprot_relations(dataset):
    rows = {}
    for split in ["train", "validation", "test"]:
        examples = []
        for x in dataset[split]:
            entities = {
                entity["id"]: entity["text"]
                for entity in _records(x["entities"])
            }
            for relation in _records(x["relations"]):
                examples.append({
                    "text": x["text"],
                    "entity_pair": f'{entities[relation["arg1"]]} -> {entities[relation["arg2"]]}',
                    "relation": relation["type"],
                })
        rows[split] = Dataset.from_list(examples)
    return DatasetDict(rows)

chemprot = Classification(
    "text", "entity_pair", "relation",
    dataset_name="bigbio/chemprot", config_name="chemprot_full_source",
    pre_process=_chemprot_relations)

pku_saferlhf__helpfulness = MultipleChoice(
    "prompt", choices=["response_0", "response_1"], labels="better_response_id",
    question="Which response is more helpful?",
    dataset_name="PKU-Alignment/PKU-SafeRLHF")

pku_saferlhf__safety = MultipleChoice(
    "prompt", choices=["response_0", "response_1"], labels="safer_response_id",
    question="Which response is safer?",
    dataset_name="PKU-Alignment/PKU-SafeRLHF")

_HELPSTEER_SCALES = dict(helpfulness=("not helpful", "extremely helpful"),
    correctness=("mostly incorrect", "fully correct and complete"), coherence=("incoherent", "perfectly clear"),
    complexity=("basic competency", "deep domain expertise"), verbosity=("very terse", "very verbose"))

def _helpsteer(attribute, dataset_name):
    low, high = _HELPSTEER_SCALES[attribute]
    return Classification("prompt", "response", name(attribute, [f"0: {low}", "1", "2", "3", f"4: {high}"]),
        dataset_name=dataset_name, question=f"How would you rate the {attribute} of the response?")

helpsteer__helpfulness = _helpsteer("helpfulness", "nvidia/HelpSteer")
helpsteer__correctness = _helpsteer("correctness", "nvidia/HelpSteer")
helpsteer__coherence = _helpsteer("coherence", "nvidia/HelpSteer")
helpsteer__complexity = _helpsteer("complexity", "nvidia/HelpSteer")
helpsteer__verbosity = _helpsteer("verbosity", "nvidia/HelpSteer")

helpsteer_2__helpfulness = _helpsteer("helpfulness", "nvidia/HelpSteer2")
helpsteer_2__correctness = _helpsteer("correctness", "nvidia/HelpSteer2")
helpsteer_2__coherence = _helpsteer("coherence", "nvidia/HelpSteer2")
helpsteer_2__complexity = _helpsteer("complexity", "nvidia/HelpSteer2")
helpsteer_2__verbosity = _helpsteer("verbosity", "nvidia/HelpSteer2")

def render_dialogue(turns):
    return "\n\n".join(f"{turn['role'].capitalize()}: {turn['content']}" for turn in turns)

helpsteer_3___preference = MultipleChoice(lambda x: render_dialogue(x['context']), question="Which next assistant reply is better?",
    choices=["response1", "response2"], labels=lambda x: int(x["overall_preference"] > 0),
    pre_process=lambda ds: ds.filter(lambda x: x["overall_preference"] != 0),  # 0 is a tie
    dataset_name="nvidia/HelpSteer3", config_name="preference")

helpsteer_3___principle = Classification(
    lambda x: f"{render_dialogue(x['context'])}\n\nAssistant: {x['response']}",
    lambda x: f"Does the assistant reply satisfy this principle: {x['principle']}?",
    labels="fulfilment", dataset_name="nvidia/HelpSteer3", config_name="principle")

helpsteer_3___edit_quality = MultipleChoice(
    lambda x: f"{render_dialogue(x['context'])}\n\nOriginal reply: {x['original_response']}",
    question="Which edit improves the reply?",
    choices=["good_edited_response", "bad_edited_response"], labels=constant(0),
    dataset_name="nvidia/HelpSteer3", config_name="edit_quality")

HELPFULNESS = ["not helpful", "slightly helpful", "partially helpful", "mostly helpful", "perfectly helpful"]
_HELPFULNESS = re.compile(r"^\W*The response is (not|slightly|partially|mostly|perfectly) helpful", re.I)

def _helpsteer3_feedback(dataset):
    # each annotator critique opens with a helpfulness level; keep the majority level
    def rows(split):
        for x in split:
            for i in ("1", "2"):
                levels = [m.group(1).lower() for m in map(_HELPFULNESS.match, x[f"feedback{i}"]) if m]
                level, votes = Counter(levels).most_common(1)[0] if levels else (None, 0)
                if votes >= 2:
                    yield dict(context=x["context"], response=x[f"response{i}"],
                               helpfulness=HELPFULNESS.index(f"{level} helpful"))
    return DatasetDict({name: Dataset.from_generator(rows, gen_kwargs=dict(split=split))
                        .cast_column("helpfulness", ClassLabel(names=HELPFULNESS))
                        for name, split in dataset.items()})

helpsteer_3___feedback = Classification(
    lambda x: f"{render_dialogue(x['context'])}\n\nAssistant: {x['response']}",
    labels="helpfulness", pre_process=_helpsteer3_feedback, dataset_name="nvidia/HelpSteer3", config_name="feedback",
    question="How helpful is the assistant reply?")

msci_nli = Classification('sentence1','sentence2','label',dataset_name='sadat2307/MSciNLI')


ultrafeedback = MultipleChoice("question", choices=['response_j','response_k'],labels=constant(0), question="Which response is better?", dataset_name="pushpdeep/UltraFeedback-paired")

# PRM800K math solutions: chosen solutions are human-validated and correct, rejected ones flawed and wrong;
# the step config compares next steps after a prefix that reached a verified answer
prm800k_dpo___solution = MultipleChoice("prompt", choices=["chosen", "rejected"], labels=constant(0),
    question="Which solution is correct?", dataset_name="tasksource/prm800k_dpo", config_name="solution",
    splits=["train", None, None])  # the source splits follow MATH; its test split is a benchmark
prm800k_dpo___step = MultipleChoice("prompt", choices=["chosen", "rejected"], labels=constant(0),
    question="Which next step is correct?", dataset_name="tasksource/prm800k_dpo", config_name="step",
    splits=["train", None, None])  # the source splits follow MATH; its test split is a benchmark

essay_scoring = Classification("full_text", labels="score", question="What holistic score does this student essay deserve?",
    dataset_name='tasksource/AES2-essay-scoring',
    label_values={score: f"{score} out of 6" for score in range(1, 7)})

argument_feedback = Classification(lambda x: f"{x['discourse_type']}: {x['discourse_text']}",
    question="How effective is this element of the student's argument?",
    labels="discourse_effectiveness", dataset_name="tasksource/argument-feedback")

# analytic scores from 1 to 5 in half points, averaged over raters; rounded half up
eg = lambda x: Classification("full_text", question=f"What {x} score does this English learner essay deserve?",
    labels=lambda y: f"{int(y[x] + 0.5)} out of 5", dataset_name="tasksource/english-grading")
grading__cohesion = eg('cohesion')
grading__syntax = eg('syntax')
grading__vocabulary = eg('vocabulary')
grading__phraseology = eg('phraseology')
grading__grammar = eg('grammar')
grading__conventions = eg('conventions')

wice = Classification(lambda x: "\n".join(x['evidence']),'claim','label',
    dataset_name='tasksource/wice')

hover = Classification("evidence","claim","label",
    dataset_name="Dzeniks/hover",
    label_values={0: "supports the claim", 1: "refutes the claim"})

hover__nli = Classification("evidence","claim",name("label",["entailment","neutral","contradiction"]),
    dataset_name="Dzeniks/hover-3way")

tasksource_dpo = MultipleChoice("prompt",choices=['chosen','rejected'],labels=constant(0), question="Which response is better?",
    dataset_name="tasksource/tasksource_dpo_pairs")

seahorse = Classification('article',cat(["summary", "question"]),'answer',
    dataset_name="tasksource/seahorse_summarization_evaluation")

mip = Classification(lambda x: x["prompt"].split("\nProvide no explanation")[0].strip(),  # drop the answer-format instruction
    labels=lambda x: x["y"].rstrip(".").lower(),
    dataset_name="sileod/missing-item-prediction",config_name="contrastive")

jigsaw_toxicity = Classification('comment_text',labels=name("toxic",["not toxic","toxic"]),
    dataset_name="tasksource/jigsaw_toxicity")

pol_nli = Classification("premise","hypothesis",labels=name('entailment',['entailment','not_entailment']),
    dataset_name="mlburnham/Pol_NLI")

synthetic_retrieval_nli = Classification('premise','hypothesis','label',dataset_name='tasksource/synthetic-retrieval-NLI',
    config_name=["binary","count","position"],
    pre_process=lambda ds:ds.filter(lambda x:x['n']<=2048))

def _html_text(html_text):
    text = re.sub(r"<br\s*/?>|</(?:p|h\d|li|pre|div)>", "\n", html_text)
    text = html.unescape(re.sub(r"<[^>]+>", "", text))
    return re.sub(r"\n\s*\n+", "\n", text).strip()

issue_similarity = Classification(lambda x: _html_text(x["text1"]), lambda x: _html_text(x["text2"]), "label",
    dataset_name="WhereIsAI/github-issue-similarity",
    label_values={0: "dissimilar issues", 1: "similar issues"})
