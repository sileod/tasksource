from .preprocess import cat, get, regen, constant, Classification, TokenClassification, MultipleChoice
from .metadata import bigbench_discriminative_english, blimp_hard, imppres_presupposition, imppres_implicature
from datasets import get_dataset_config_names, ClassLabel, Dataset, DatasetDict

def all(dataset_name):
    return dict(dataset_name=dataset_name, config_name=get_dataset_config_names(dataset_name))


# english tasks (few, to keep balance between languages)
from .tasks import anli__a1, anthropic_rlhf, dyna_hate, dynasent__r1

xnli = Classification("premise","hypothesis","label",
    **all("MoritzLaurer/multilingual-NLI-26lang-2mil7")) 

stsb_multi_mt = Classification("sentence1","sentence2","similarity_score",
    **all('stsb_multi_mt'))

pawsx = Classification("sentence1","sentence2","label",
    **all('paws-x'))

xstance = Classification("question","comment","label",
    **all("strombergnlp/x-stance"))

miam = Classification("Utterance",labels="label",
    **all('miam'))

rumoureval_2019 = Classification("source_text","reply_text","label",
    **all("strombergnlp/rumoureval_2019"))

tweet_sentiment = Classification("text",labels="label",
    **all('cardiffnlp/tweet_sentiment_multilingual'))

offenseval = Classification("text",labels="subtask_a",
    dataset_name='strombergnlp/offenseval_2020',
    config_name=["ar","da","gr","tr"])

disrpt_23 = Classification("unit1_sent","unit2_sent","label",
    **all("metaeval/dsr"))

ner = TokenClassification("words","ner",
    dataset_name="xglue",config_name="ner")

mlma_hate = Classification("tweet","sentiment",
    dataset_name="nedjmaou/MLMA_hate_speech")

xcopa = MultipleChoice("premise",choices=['choice1','choice2'],labels="label",
    **all('xcopa'))

# disrpt
# exams
#wino_x
# clue, klue, indic_glue

