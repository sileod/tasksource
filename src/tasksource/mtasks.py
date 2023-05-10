from .preprocess import cat, get, regen, constant, Classification, TokenClassification, MultipleChoice
from .metadata import udep_labels
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

xnli = Classification("premise", "hypothesis", "label", **all("metaeval/xnli"))

americas_nli = Classification("premise","hypothesis","label",config_name="all_languages")

moritz_xnli = Classification("premise","hypothesis","label", 
    pre_process=concatenate_configs, dataset_name="MoritzLaurer/multilingual-NLI-26lang-2mil7")

stsb_multi_mt = Classification("sentence1", "sentence2",
    lambda x: float(x["similarity_score"]/5),
    **all('stsb_multi_mt'))

pawsx = Classification("sentence1","sentence2","label", **all('paws-x'))

miam = Classification("Utterance",labels="Label", **all('miam'))

xstance = Classification("question", "comment", "label",
    **all("strombergnlp/x-stance"))

sentiment = Classification("text",labels="label",
    dataset_name="tyqiangz/multilingual-sentiments",config_name="all",
    pre_process=lambda ds:ds.filter(lambda x: "amazon_reviews" not in x['source'])    
)

emotion = Classification("text",labels="emotion",dataset_name="metaeval/universal-joy")

review_sentiment = Classification("review_body",labels="stars",
    dataset_name="amazon_reviews_multi",config_name="all_languages")

tweet_sentiment = Classification("text", labels="label",
    **all('cardiffnlp/tweet_sentiment_multilingual'))

offenseval = Classification(lambda x: str(x["text"]), labels="subtask_a",
    dataset_name='strombergnlp/offenseval_2020',
    config_name=["ar","da","gr","tr"])

offenseval_dravidian = Classification("text",labels="label",config_name=['kannada','malayalam','tamil'])

mlma_hate = Classification("tweet", labels="sentiment",
    dataset_name="nedjmaou/MLMA_hate_speech")


qam = Classification("question","answer","label", dataset_name="xglue",config_name="qam")

x_sum_factuality = Classification("summary","generated_summary","label",
    dataset_name="ylacombe/xsum_factuality")

x_fact = Classification('evidence','claim','label', dataset_name="metaeval/x-fact")

xglue___nc = Classification('news_body',labels='news_category')
xglue___qadsm = Classification('query','ad_description','relevance_label')
xglue___qam = Classification('question','answer','label')
xglue___wpr = Classification('query','web_page_snippet','relavance_label') # relavance_label : sic

xlwic = Classification(
    sentence1=cat(["target_word","context_1"], " : "),
    sentence2=cat(["target_word","context_2"], " : "),
    labels='label',dataset_name="pasinit/xlwic",config_name=['xlwic_de_de','xlwic_it_it','xlwic_fr_fr','xlwic_en_ko'])



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

xcsr = MultipleChoice(get.question.stem, choices_list=get.question.choices.text,
    labels=lambda x:'ABCDE'.index(x['answerKey']),
    **all('xcsr'))

xcopa = MultipleChoice("premise",choices=['choice1','choice2'],labels="label",
    **all('xcopa'))

xstory = MultipleChoice(constant(''),choices=["text_right_ending","text_wrong_ending"],labels=constant(0),
    **all("juletxara/xstory_cloze"))

anthropic_rlhf = MultipleChoice(constant(''), ['chosen','rejected'], constant(0),
    dataset_name="Anthropic/hh-rlhf")

xglue_ner = TokenClassification("words","ner", dataset_name="xglue",config_name="ner")
xglue_pos = TokenClassification("words","pos", dataset_name="xglue",config_name="pos")

disrpt_23 = Classification("unit1_sent", "unit2_sent", "label",**all("metaeval/disrpt"))

udep__pos = TokenClassification('tokens','upos', **all('universal_dependencies'))

def udep_post_process(ds):
    return ds.cast_column('labels', Sequence(ClassLabel(names=udep_labels)))

#udep__deprel = TokenClassification('tokens',lambda x:[udep_labels.index(a) for a in x['deprel']],
#    **all('universal_dependencies'),post_process=udep_post_process)

oasst_rlhf = MultipleChoice("prompt",choices=['chosen','rejected'],labels=constant(0),
    dataset_name="tasksource/oasst1_pairwise_rlhf_reward")

#Classification(
#    dataset_name='glue',config_name=['ocnli','afqmc'])

# 
#wino_x
# clue, klue, indic_glue
# SMS_Spam_Multilingual_Collection_Dataset
