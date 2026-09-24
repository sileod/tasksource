from .preprocess import cat, get,name, regen, constant, Classification, TokenClassification, MultipleChoice
from .tasks import _copa_input
from datasets import get_dataset_config_names, ClassLabel, Dataset, DatasetDict, concatenate_datasets, Sequence

def all(dataset_name):
    try:
        config_name=get_dataset_config_names(dataset_name)
    except Exception as e:
        print(dataset_name,e)
        config_name=None
    return dict(dataset_name=dataset_name, config_name=config_name)

def concatenate_configs(dataset):
    return DatasetDict(train=concatenate_datasets(list(dataset.values())))

# english tasks (few, to keep balance between languages)

moritz_xnli = Classification("premise","hypothesis",name("label",["entailment", "neutral","contradiction"]), 
    pre_process=concatenate_configs, 
    dataset_name="MoritzLaurer/multilingual-NLI-26lang-2mil7")

xnli = Classification("premise", "hypothesis", "label",
    dataset_name="facebook/xnli", config_name="en", task_id="xnli")

americas_nli = Classification("premise","hypothesis","label",config_name="all_languages")

stsb_multi_mt = Classification("sentence1", "sentence2",
    lambda x: float(x["similarity_score"]/5),
    **all('stsb_multi_mt'))

pawsx = Classification("sentence1","sentence2",name('label',['not_paraphrase','paraphrase']), **all('paws-x'))

MIAM_DIHANA_LABELS = [
    "Afirmacion", "Apertura", "Cierre", "Confirmacion", "Espera",
    "Indefinida", "Negacion", "No_entendido", "Nueva_consulta",
    "Pregunta", "Respuesta",
]

def _miam_preprocess(dataset):
    dataset = dataset.rename_column("Dialogue_Act", "Label")
    return dataset.cast_column("Label", ClassLabel(names=MIAM_DIHANA_LABELS))

miam = Classification(
    "Utterance", labels="Label", dataset_name="csv", task_id="miam",
    load_dataset_kwargs={"data_files": {
        "train": "https://raw.githubusercontent.com/eusip/MIAM/main/dihana/train.csv",
        "validation": "https://raw.githubusercontent.com/eusip/MIAM/main/dihana/dev.csv",
        "test": "https://raw.githubusercontent.com/eusip/MIAM/main/dihana/test.csv",
    }},
    pre_process=_miam_preprocess)

def _xstance_question(x):
    lang = x.get("language") or "en"
    return x.get("question_" + lang) or x.get("question_en") or ""

xstance = Classification(_xstance_question, "comment", "stance_label",
    dataset_name="michiel/xstance", task_id="x-stance")


def _offenseval_mapping(dataset_name, config_name, task_id):
    return dict(
        sentence1=lambda x: str(x["text"]),
        labels=name("subtask_a", ['not offensive', 'offensive']),
        pre_process=lambda ds: ds.filter(lambda x: x['subtask_a'] in [0, 1]),
        dataset_name=dataset_name, config_name=config_name, task_id=task_id)

offenseval_ar = Classification(**_offenseval_mapping('khalidalt/offenseval_2020_ar', None, 'offenseval_2020/ar'))
offenseval_da = Classification(**_offenseval_mapping('tasksource/offenseval_2020', 'da', 'offenseval_2020/da'))
offenseval_gr = Classification(**_offenseval_mapping('tasksource/offenseval_2020', 'gr', 'offenseval_2020/gr'))
offenseval_tr = Classification(**_offenseval_mapping('tasksource/offenseval_2020', 'tr', 'offenseval_2020/tr'))

offenseval_dravidian = Classification("text",labels="label",config_name=['kannada','malayalam','tamil'])

mlma_hate = Classification("tweet", labels=lambda x:x["sentiment"].split('_'),
    dataset_name="nedjmaou/MLMA_hate_speech")

qam = Classification("question","answer","label", dataset_name="tasksource/xglue",config_name="qam")

#x_sum_factuality = Classification("summary","generated_summary","label", dataset_name="ylacombe/xsum_factuality")

def _x_fact_labels(dataset):
    # Use the full source train ontology before bounded sampling; "other" is
    # rare enough to disappear from small train samples while remaining in dev.
    names = sorted(set(dataset["train"]["label"]))
    return dataset.cast_column("label", ClassLabel(names=names))

x_fact = Classification(
    'evidence', 'claim', 'label', dataset_name="tasksource/x-fact",
    splits=["train", "dev", "test"], pre_process=_x_fact_labels)

xgluenc = Classification('text', labels='label_text',
    dataset_name="SetFit/xglue_nc", task_id="xglue/nc")
xglue___qadsm = Classification('query','ad_description','relevance_label',
    dataset_name="tasksource/xglue", config_name="qadsm")
xglue___qam = Classification('question','answer','label',
    dataset_name="tasksource/xglue", config_name="qam")
xglue___wpr = Classification('query','web_page_snippet','relavance_label',
    dataset_name="tasksource/xglue", config_name="wpr") # relavance_label : sic

xlwic = Classification(
    sentence1=cat(["target_word","context_1"], " : "),
    sentence2=cat(["target_word","context_2"], " : "),
    labels='label',dataset_name="tasksource/xlwic",config_name=['xlwic_de_de','xlwic_it_it','xlwic_fr_fr','xlwic_en_ko'])

#[ "spam", "fails_task", "lang_mismatch", "pii", "not_appropriate", "hate_speech", "sexual_content", "quality", "toxicity", "humor", "helpfulness", "creativity", "violence" ]

oasst1__quality = Classification("parent_text","text",labels="quality", dataset_name="tasksource/oasst1_dense_flat",
    pre_process = lambda ds:ds.remove_columns('labels'))
oasst1__toxicity = Classification("parent_text","text",labels="toxicity", dataset_name="tasksource/oasst1_dense_flat",
    pre_process = lambda ds:ds.remove_columns('labels'))
oasst1__helpfulness = Classification("parent_text","text",labels="helpfulness", dataset_name="tasksource/oasst1_dense_flat",
    pre_process = lambda ds:ds.remove_columns('labels'))


language_identification = Classification("text",labels="labels", dataset_name="papluca/language-identification")
wili_2018_langid = Classification("sentence",labels="label",dataset_name="wili_2018")

exams = MultipleChoice(get.question.stem, choices_list=get.question.choices.text,
    labels=lambda x:'ABCDE'.index(x['answerKey']),
    dataset_name="exams", config_name='multilingual',
    pre_process=lambda ds:ds.filter(lambda x:  x['answerKey'] in "ABCDE"))

xcsr = MultipleChoice(lambda x: x['question']['stem'].strip() or 'Most plausible:', # X-CODAH stems are empty
    choices_list=get.question.choices.text,
    labels=lambda x:'ABCDE'.index(x['answerKey']),
    **all('xcsr'))

xcopa = MultipleChoice(_copa_input,choices=['choice1','choice2'],labels="label",
    **all('xcopa'))

#xstory = MultipleChoice(constant(''),choices=["text_right_ending","text_wrong_ending"],labels=constant(0), **all("juletxara/xstory_cloze"))

xstory = MultipleChoice(lambda x: "\n".join([x[f'input_sentence_{i}'] for i in range(1,5)]),
    choices=["sentence_quiz1","sentence_quiz2"],labels=constant(0), **all("juletxara/xstory_cloze"))


xglue_ner = TokenClassification("words","ner", dataset_name="xglue",config_name="ner")
xglue_pos = TokenClassification("words","pos", dataset_name="xglue",config_name="pos")

#disrpt_23 = Classification("unit1_sent", "unit2_sent", "label",**all("multilingual-discourse-hub/disrpt"))

def _udep_cast_label_sequence(dataset, column):
    label_names = sorted({
        label
        for split in dataset.values()
        for sequence in split[column]
        for label in sequence
    })
    return DatasetDict({
        name: split.cast_column(column, Sequence(ClassLabel(names=label_names)))
        for name, split in dataset.items()
    })

udep__pos = TokenClassification(
    'tokens', 'upos',
    pre_process=lambda ds: _udep_cast_label_sequence(ds, 'upos'),
    **all('universal-dependencies/universal_dependencies'))

def udep_post_process(ds):
    return _udep_cast_label_sequence(ds, 'labels')

#udep__deprel = TokenClassification('tokens',lambda x:[udep_labels.index(a) for a in x['deprel']],
#    **all('universal_dependencies'),post_process=udep_post_process)

oasst_rlhf = MultipleChoice("prompt",choices=['chosen','rejected'],labels=constant(0),
    dataset_name="tasksource/oasst1_pairwise_rlhf_reward")

sentiment = Classification(
    "text", labels="label", dataset_name="csv", config_name=None,
    task_id="multilingual-sentiments/all",
    load_dataset_kwargs={"data_files": {
        "train": "https://raw.githubusercontent.com/tyqiangz/multilingual-sentiment-datasets/main/data/all/train.csv",
        "validation": "https://raw.githubusercontent.com/tyqiangz/multilingual-sentiment-datasets/main/data/all/valid.csv",
        "test": "https://raw.githubusercontent.com/tyqiangz/multilingual-sentiment-datasets/main/data/all/test.csv",
    }},
    pre_process=lambda ds: ds.filter(lambda x: "amazon_reviews" not in x['source']))
_TWEET_SENTIMENT_LANGS = [
    "arabic", "english", "french", "german", "hindi", "italian",
    "portuguese", "spanish",
]
tweet_sentiment = Classification(
    "text", labels=lambda x: ["negative", "neutral", "positive"][int(x["label"])],
    dataset_name="json", task_id="tweet_sentiment_multilingual",
    load_dataset_kwargs={"data_files": {
        split: [
            f"hf://datasets/mteb/tweet_sentiment_multilingual/{split}/{lang}.jsonl.gz"
            for lang in _TWEET_SENTIMENT_LANGS
        ]
        for split in ("train", "validation", "test")
    }})
review_sentiment = Classification(
    "review_body", labels="stars", dataset_name="goosmanlei/amazon_reviews_multi",
    config_name="all_languages",
    label_values={stars: f"{stars} star{'s' if stars != 1 else ''}" for stars in range(1, 6)})
emotion = Classification("text",labels="emotion",dataset_name="tasksource/universal-joy")
# in mms

def _mms_label_filter(dataset):
    valid = {"negative", "neutral", "positive"}
    dataset = dataset.filter(
        lambda x: x["label"] in [-1, 0, 1] or x["label"] in valid
    )

    def normalize(row):
        if row["label"] in valid:
            return {"label": row["label"]}
        return {
            "label": ["negative", "neutral", "positive"][int(row["label"]) + 1]
        }

    return dataset.map(normalize)

mms_sentiment = Classification(
    "text", labels="label", dataset_name="csv", task_id="mms",
    load_dataset_kwargs={
        "data_files": {"train": "hf://datasets/Brand24/mms/data/**/*.tsv"},
        "delimiter": "\t",
        "column_names": ["label", "text", "cleanlab_self_confidence"],
        "streaming": True,
    },
    pre_process=_mms_label_filter)

mapa_fine = TokenClassification("tokens","coarse_grained",dataset_name='joelito/mapa')
mapa_corase = TokenClassification("tokens","fine_grained",dataset_name='joelito/mapa')

aces_ranking = MultipleChoice("source",choices=['good-translation','incorrect-translation'],labels=constant(0), dataset_name='nikitam/ACES', config_name='ACES', task_id='ACES/ranking')
def _aces_phenomena_labels(dataset):
    # The catalog samples before fixing string labels; build the ontology from
    # the full source so rare phenomena in dev/test are not silently invalid.
    names = sorted(set(dataset["train"]["phenomena"]))
    return dataset.cast_column("phenomena", ClassLabel(names=names))

aces_phenomena = Classification('source','incorrect-translation','phenomena',
    dataset_name='nikitam/ACES', config_name='ACES',
    task_id='ACES/phenomena', pre_process=_aces_phenomena_labels)

amazon_intent = Classification("text",labels="label",
    dataset_name='mteb/MassiveIntentClassification', config_name="en",
    task_id="massive")


# modern multilingual classification / reward datasets

masakhanews = Classification(
    "headline", labels="category",
    **all("masakhane/masakhanews"))

nusax_sentiment = Classification(
    "text", labels=name("label", ["negative", "neutral", "positive"]),
    dataset_name="mteb/NusaX-senti",
    config_name=["ace", "ban", "bbc", "bjn", "bug", "eng", "ind", "jav",
                 "mad", "min", "nij", "sun"])

afrisenti = Classification(
    "text", labels=name("label", ["positive", "neutral", "negative"]),
    dataset_name="mteb/AfriSentiClassification",
    task_id="AfriSenti-twitter-sentiment/{config_name}",
    # Oromo and Tigrinya have no train split. Keep this list static so importing
    # the task catalog does not require executing the dataset's legacy script.
    config_name=["amh", "arq", "ary", "hau", "ibo", "kin", "pcm", "por",
                 "swa", "tso", "twi", "yor"])

def _helpsteer3_context(x):
    return "\n".join(
        f'{message["role"]}: {message["content"]}'
        for message in x["context"]
    )

helpsteer3 = MultipleChoice(
    _helpsteer3_context,
    choices=["response1", "response2"],
    labels=lambda x: int(x["overall_preference"] > 0),
    dataset_name="nvidia/HelpSteer3", config_name="preference",
    pre_process=lambda ds: ds.filter(lambda x: x["overall_preference"] != 0))


#    dataset_name='glue',config_name=['ocnli','afqmc'])

tidy_as2=Classification("Question","Sentence","Label",dataset_name='tasksource/tydi-as2-balanced') 

multiconer = TokenClassification("tokens","ner_tags_index", **all("MultiCoNER/multiconer_v2"))

mtop = Classification("question",labels="intent", dataset_name="tasksource/mtop")

mlabel_nli = Classification("premise","hypothesis","labels",dataset_name="tasksource/multilingual-zero-shot-label-nli")

#wino_x
# clue, klue, indic_glue
# SMS_Spam_Multilingual_Collection_Dataset
