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
    **all('PhilipMay/stsb_multi_mt'))

pawsx = Classification("sentence1","sentence2",name('label',['not_paraphrase','paraphrase']), **all('google-research-datasets/paws-x'))

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
    sentence1=lambda x: f"Word: {x['target_word']}\n{x['context_1']}",
    sentence2="context_2",
    labels='label',dataset_name="tasksource/xlwic",config_name=['xlwic_de_de','xlwic_it_it','xlwic_fr_fr','xlwic_en_ko'])

oasst1__quality = Classification("parent_text","text",labels="quality", dataset_name="tasksource/oasst1_dense_flat",
    question="How good is the reply, from 0 (low quality) to 1 (high quality)?",
    pre_process = lambda ds:ds.remove_columns('labels'))
oasst1__toxicity = Classification("parent_text","text",labels="toxicity", dataset_name="tasksource/oasst1_dense_flat",
    question="How toxic is the reply, from 0 (not toxic) to 1 (very toxic)?",
    pre_process = lambda ds:ds.remove_columns('labels'))
oasst1__helpfulness = Classification("parent_text","text",labels="helpfulness", dataset_name="tasksource/oasst1_dense_flat",
    question="How helpful is the reply, from 0 (unhelpful) to 1 (helpful)?",
    pre_process = lambda ds:ds.remove_columns('labels'))


language_identification = Classification("text",labels="labels", dataset_name="papluca/language-identification")
wili_2018_langid = Classification("sentence",labels="label",dataset_name="wili_2018")

exams = MultipleChoice(get.question.stem, choices_list=get.question.choices.text,
    labels=lambda x:'ABCDE'.index(x['answerKey']),
    dataset_name="exams", config_name='multilingual',
    pre_process=lambda ds:ds.filter(lambda x:  x['answerKey'] in "ABCDE"))

_xcsr = all('INK-USC/xcsr')
_xcsr_fields = dict(choices_list=get.question.choices.text, labels=lambda x:'ABCDE'.index(x['answerKey']), dataset_name=_xcsr['dataset_name'])
xcsr = MultipleChoice(get.question.stem, **_xcsr_fields,
    config_name=[c for c in _xcsr['config_name'] or [] if c.startswith('X-CSQA')])
xcsr_codah = MultipleChoice(constant(''), question="Which sentence is most plausible?", **_xcsr_fields,  # X-CODAH stems are empty
    config_name=[c for c in _xcsr['config_name'] or [] if c.startswith('X-CODAH')])

xcopa = MultipleChoice(_copa_input,choices=['choice1','choice2'],labels="label",
    **all('cambridgeltl/xcopa'))

xstory = MultipleChoice(lambda x: "\n".join([x[f'input_sentence_{i}'] for i in range(1,5)]),
    choices=["sentence_quiz1","sentence_quiz2"],labels=constant(0), **all("juletxara/xstory_cloze"))



# DISRPT discourse relations between two units. Relation inventories differ per corpus, so each is its
# own task; keep corpora with at least 1,600 training pairs, minus the English ones tasksource already
# has (scidtb, and STAC via pragmeval) and por.pdtb.crpc (no label column)
disrpt = Classification("unit1_txt", "unit2_txt", "label", dataset_name="multilingual-discourse-hub/disrpt",
    config_name=["deu.rst.pcc.rels", "eus.rst.ert.rels", "fas.rst.prstc.rels", "fra.sdrt.annodis.rels",
                 "nld.rst.nldt.rels", "por.rst.cstn.rels", "rus.rst.rrt.rels", "spa.rst.rststb.rels",
                 "tha.pdtb.tdtb.rels", "zho.rst.gcdt.rels"])

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

oasst_rlhf = MultipleChoice("prompt",choices=['chosen','rejected'],labels=constant(0), question="Which reply is better?",
    dataset_name="tasksource/oasst1_pairwise_rlhf_reward")

# the tweet sources duplicate tweet_sentiment_multilingual and amazon_reviews_multi is its own task
sentiment = Classification("text", labels="label", dataset_name="tasksource/multilingual-sentiments",
    task_id="multilingual-sentiments/all",
    pre_process=lambda ds: ds.filter(lambda x: x["source"] in {"indonlue/smsa", "malaya"}))
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
# corpora bundled in MMS that tasksource already loads on their own
_MMS_DUPLICATES = ("_multilan_amazon", "en_financial_phrasebank", "en_poem_sentiment", "en_silicone_",
                   "en_semeval_2017", "ar_semeval_2017", "de_sb10k", "it_evalita2016", "pt_tweet_sent_br")
_MMS_LANGUAGES = ["ar", "bg", "bs", "cs", "de", "el", "en", "es", "fa", "fr", "he", "hi", "hr", "hu", "it",
                  "lv", "pl", "pt", "ru", "sk", "sl", "sq", "sr", "sv", "th", "ur", "zh"]  # ja is only duplicates
_MMS_MAX_ROWS_PER_SOURCE = 5_000  # one task over all languages, capped per corpus (en_amazon alone has 1.7M rows)

def _mms_sources(dataset):
    def keep(split):
        seen = {}
        def keep_row(x):
            name = x["original_dataset"]
            if any(duplicate in name for duplicate in _MMS_DUPLICATES):
                return False
            seen[name] = seen.get(name, 0) + 1
            return seen[name] <= (_MMS_MAX_ROWS_PER_SOURCE if split == "train" else _MMS_MAX_ROWS_PER_SOURCE // 10)
        return keep_row
    return type(dataset)({split: rows.filter(keep(split)) for split, rows in dataset.items()})

mms_sentiment = Classification("text", labels="label", dataset_name="parquet", task_id="mms",
    load_dataset_kwargs={"data_files": {split: [f"hf://datasets/tasksource/mms/{language}/{split}-*.parquet"
                                                for language in _MMS_LANGUAGES]
                                        for split in ("train", "validation", "test")}},
    pre_process=_mms_sources)

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

# CLUE: natively Chinese sets (CMNLI is machine-translated MNLI and XNLI covers it); test labels are hidden
clue___ocnli = Classification("sentence1", "sentence2", "label", dataset_name="clue/clue", config_name="ocnli",
    splits=["train", "validation", None])
clue___afqmc = Classification("sentence1", "sentence2", "label", dataset_name="clue/clue", config_name="afqmc",
    splits=["train", "validation", None], label_values={0: "different meaning", 1: "same meaning"})
clue___tnews = Classification("sentence", labels="label", dataset_name="clue/clue", config_name="tnews",
    splits=["train", "validation", None], label_values=dict(enumerate([
        "story", "culture", "entertainment", "sports", "finance", "real estate", "cars", "education", "technology",
        "military", "travel", "world", "stocks", "agriculture", "games"])))

# KLUE: natively Korean sets; test labels are hidden
klue___nli = Classification("premise", "hypothesis", "label", dataset_name="klue/klue", config_name="nli",
    splits=["train", "validation", None])
klue___ynat = Classification("title", labels="label", dataset_name="klue/klue", config_name="ynat",
    splits=["train", "validation", None])
klue___sts = Classification("sentence1", "sentence2", labels=lambda x: x["labels"]["binary-label"],
    dataset_name="klue/klue", config_name="sts", splits=["train", "validation", None],
    label_values={0: "not paraphrases", 1: "paraphrases"})

# IndicGLUE: natively labeled Indic sets (COPA/WNLI are translations, CSQA is test-only)
def _used_label_names(dataset):
    # the shared ClassLabel lists every family's classes; keep the ones present
    names = dataset["train"].features["label"].names
    return dataset.map(lambda x: {"label_text": names[x["label"]]})

indic_glue__sentiment = Classification("text", labels="label_text", dataset_name="ai4bharat/indic_glue",
    config_name=["actsa-sc.te", "iitp-mr.hi", "iitp-pr.hi", "inltkh.te"], pre_process=_used_label_names)
def _indic_glue_files(configs):
    # one task per family across languages, so a family is not weighted by its language count
    return {"data_files": {split: [f"hf://datasets/ai4bharat/indic_glue/{config}/{split}-*.parquet" for config in configs]
                           for split in ("train", "validation", "test")}}

indic_glue__news = Classification("text", labels="label", dataset_name="ai4bharat/indic_glue",
    config_name=["bbca.hi", "sna.bn"])
indic_glue__headlines = Classification("text", labels="label_text", dataset_name="parquet", task_id="indic_glue/inltkh",
    load_dataset_kwargs=_indic_glue_files(["inltkh.gu", "inltkh.ml", "inltkh.mr", "inltkh.ta"]),  # inltkh.te is sentiment
    pre_process=_used_label_names)
indic_glue___md__discourse_mode = Classification("sentence", labels="discourse_mode",
    dataset_name="ai4bharat/indic_glue", config_name="md.hi")
indic_glue__section_title = MultipleChoice("sectionText", question="Which title fits this section?",
    choices=["titleA", "titleB", "titleC", "titleD"],
    labels=lambda x: ["titleA", "titleB", "titleC", "titleD"].index(x["correctTitle"]),  # names the gold column
    dataset_name="parquet", task_id="indic_glue/wstp", load_dataset_kwargs=_indic_glue_files(
        [f"wstp.{language}" for language in ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]]))

tidy_as2=Classification("Question","Sentence","Label",dataset_name='tasksource/tydi-as2-balanced') 

# the Hub's parquet export of the script-only dataset; the MULTI config is the union of the others
multiconer = TokenClassification("tokens", "ner_tags_index", dataset_name="parquet", task_id="multiconer_v2/{config_name}",
    config_name=["Bangla (BN)", "Chinese (ZH)", "English (EN)", "Farsi (FA)", "French (FR)", "German (DE)", "Hindi (HI)",
                 "Italian (IT)", "Portuguese (PT)", "Spanish (ES)", "Swedish (SV)", "Ukrainian (UK)"],
    load_dataset_kwargs={"data_files": {split: f"hf://datasets/MultiCoNER/multiconer_v2@refs%2Fconvert%2Fparquet/{{config_name}}/{split}/*.parquet"
                                        for split in ("train", "validation", "test")}})

mtop = Classification("question",labels="intent", dataset_name="tasksource/mtop")

mlabel_nli = Classification("premise","hypothesis","labels",dataset_name="tasksource/multilingual-zero-shot-label-nli")

