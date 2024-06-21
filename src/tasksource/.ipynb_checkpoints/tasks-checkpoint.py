from .preprocess import cat, get, regen, name, constant, Classification, TokenClassification, MultipleChoice
from .metadata import bigbench_discriminative_english, blimp_hard, imppres_presupposition, imppres_implicature, udep_en_configs, udep_en_labels
from datasets import get_dataset_config_names, Sequence, ClassLabel, Dataset, DatasetDict

# variable name: dataset___config__task

###################### NLI/paraphrase ###############################

glue___mnli = Classification(sentence1="premise", sentence2="hypothesis", labels="label", splits=["train", None, "validation_matched"])
glue___qnli = Classification("question","sentence", labels="label")
glue___rte = Classification(sentence1="sentence1", sentence2="sentence2", labels="label")
glue___wnli = Classification(sentence1="sentence1", sentence2="sentence2", labels="label")
#glue___ax = Classification(sentence1="premise", sentence2="hypothesis", labels="label", splits=["test", None, None]) # fully masked

glue___mrpc = Classification(sentence1="sentence1", sentence2="sentence2", labels="label")
glue___qqp = Classification(sentence1="question1", sentence2="question2", labels="label")
glue___stsb = Classification(sentence1="sentence1", sentence2="sentence2", labels="label")

super_glue___boolq = Classification(sentence1="question", labels="label")
super_glue___cb = Classification(sentence1="premise", sentence2="hypothesis", labels="label")
super_glue___multirc = Classification(
    cat(["paragraph", "question"]),
    'answer',
    labels='label'
)
#super_glue___rte = Classification(sentence1="premise", sentence2="hypothesis", labels="label") # in glue
super_glue___wic = Classification(
    sentence1=cat(["word","sentence1"], " : "),
    sentence2=cat(["word","sentence2"], " : "),
    labels='label'
)
super_glue___axg = Classification(sentence1="premise", sentence2="hypothesis", labels="label", splits=["test", None, None])


anli__a1 = Classification('premise','hypothesis','label', splits=['train_r1','dev_r1','test_r1'])
anli__a2 = Classification('premise','hypothesis','label', splits=['train_r2','dev_r2','test_r2'])
anli__a3 = Classification('premise','hypothesis','label', splits=['train_r3','dev_r3','test_r3'])


babi_nli = Classification("premise", "hypothesis", "label",
    dataset_name="tasksource/babi_nli",
    config_name=set(get_dataset_config_names("tasksource/babi_nli"))-{"agents-motivations"}
) # agents-motivations task is not as clear-cut as the others


sick__label         = Classification('sentence_A','sentence_B','label')
sick__relatedness   = Classification('sentence_A','sentence_B','relatedness_score')
sick__entailment_AB = Classification('sentence_A','sentence_B','entailment_AB')
#sick__entailment_BA = Classification('sentence_A','sentence_B','entailment_BA')

def remove_neg_1(dataset):
    return dataset.filter(lambda x:x['labels']!=-1)

snli = Classification(sentence1="premise", sentence2="hypothesis", labels="label",
    post_process=remove_neg_1)

scitail = Classification("sentence1","sentence2","gold_label",config_name="snli_format")

hans = Classification(sentence1="premise", sentence2="hypothesis", labels="label")

wanli = Classification('premise','hypothesis','gold', dataset_name="alisawuffles/WANLI")

recast_nli = Classification(sentence1="context", sentence2="hypothesis", labels="label", dataset_name="tasksource/recast",
    config_name=['recast_kg_relations', 'recast_puns', 'recast_factuality', 'recast_verbnet',
    'recast_verbcorner', 'recast_ner', 'recast_sentiment', 'recast_megaveridicality'])


probability_words_nli = Classification(sentence1="context", sentence2="hypothesis", labels="label",
    dataset_name="sileod/probability_words_nli", 
    config_name=["reasoning_1hop","reasoning_2hop","usnli"])

nan_nli = Classification("premise", "hypothesis", "label", dataset_name="joey234/nan-nli", config_name="joey234--nan-nli")

nli_fever = Classification("premise","hypothesis","label",
    dataset_name="pietrolesci/nli_fever", splits=["train","dev",None])

breaking_nli = Classification("sentence1","sentence2","label",
    dataset_name="pietrolesci/breaking_nli", splits=["full",None,None])

conj_nli = Classification("premise","hypothesis","label",post_process=remove_neg_1,
    dataset_name="pietrolesci/conj_nli",splits=['train','dev',None])

fracas = Classification("premise","hypothesis","label",
    dataset_name="pietrolesci/fracas")

dialogue_nli = Classification("sentence1","sentence2","label",
    dataset_name="pietrolesci/dialogue_nli")   

mpe_nli = Classification("premise","hypothesis","label",
    dataset_name="pietrolesci/mpe",
    splits=["train","dev","test"])  

dnc_nli = Classification("context","hypothesis","label",
    dataset_name="pietrolesci/dnc")

# gpt3_nli = Classification("text_a","text_b","label",dataset_name="pietrolesci/gpt3_nli") # not sound enough

recast_white__fnplus = Classification("text","hypothesis","label",
    dataset_name="pietrolesci/recast_white",splits=['fnplus',None,None])
recast_white__sprl = Classification("text","hypothesis","label",
    dataset_name="pietrolesci/recast_white",splits=['sprl',None,None])
recast_white__dpr = Classification("text","hypothesis","label",
    dataset_name="pietrolesci/recast_white",splits=['dpr',None,None])

joci = Classification("context","hypothesis",
    labels=lambda x: [None, "impossible", "technically possible", "plausible", "likely", "very likely"][x["original_label"]],
    pre_process=lambda ds:ds.filter(lambda x:x['original_label']!=0),
    dataset_name="pietrolesci/joci",splits=['full',None,None])

#enfever_nli = Classification("evidence","claim","label", dataset_name="ctu-aic/enfever_nli")

robust_nli__IS_CS = Classification("premise","hypothesis","label",
	dataset_name="pietrolesci/robust_nli", splits=["IS_CS",None,None])
robust_nli__LI_LI = Classification("premise","hypothesis","label",
	dataset_name="pietrolesci/robust_nli", splits=["LI_LI",None,None])
robust_nli__ST_WO = Classification("premise","hypothesis","label",
	dataset_name="pietrolesci/robust_nli", splits=["ST_WO",None,None])
robust_nli__PI_SP = Classification("premise","hypothesis","label",
	dataset_name="pietrolesci/robust_nli", splits=["PI_SP",None,None])
robust_nli__PI_CD = Classification("premise","hypothesis","label",
	dataset_name="pietrolesci/robust_nli", splits=["PI_CD",None,None])
robust_nli__ST_SE = Classification("premise","hypothesis","label",
	dataset_name="pietrolesci/robust_nli", splits=["ST_SE",None,None])
robust_nli__ST_NE = Classification("premise","hypothesis","label",
	dataset_name="pietrolesci/robust_nli", splits=["ST_NE",None,None])
robust_nli__ST_LM = Classification("premise","hypothesis","label",
	dataset_name="pietrolesci/robust_nli", splits=["ST_LM",None,None])
robust_nli_is_sd = Classification("premise","hypothesis","label",
    dataset_name="pietrolesci/robust_nli_is_sd")
robust_nli_li_ts = Classification("premise","hypothesis","label",
    dataset_name="pietrolesci/robust_nli_li_ts")

gen_debiased_nli__snli_seq_z = Classification("premise","hypothesis","label",
	dataset_name="pietrolesci/gen_debiased_nli", splits=["snli_seq_z",None,None])
gen_debiased_nli__snli_z_aug = Classification("premise","hypothesis","label",
	dataset_name="pietrolesci/gen_debiased_nli", splits=["snli_z_aug",None,None])
gen_debiased_nli__snli_par_z = Classification("premise","hypothesis","label",
	dataset_name="pietrolesci/gen_debiased_nli", splits=["snli_par_z",None,None])
gen_debiased_nli__mnli_par_z = Classification("premise","hypothesis","label",
	dataset_name="pietrolesci/gen_debiased_nli", splits=["mnli_par_z",None,None])
gen_debiased_nli__mnli_z_aug = Classification("premise","hypothesis","label",
	dataset_name="pietrolesci/gen_debiased_nli", splits=["mnli_z_aug",None,None])
gen_debiased_nli__mnli_seq_z = Classification("premise","hypothesis","label",
	dataset_name="pietrolesci/gen_debiased_nli", splits=["mnli_seq_z",None,None])

add_one_rte = Classification("premise","hypothesis","label",
    dataset_name="pietrolesci/add_one_rte",splits=["train","dev","test"])

def _imppres_post_process(ds,prefix=''):
    # imppres entailment definition is either purely semantic or purely pragmatic
    # because of that, we assign differentiate the labels from anli/mnli notation
    return ds.cast_column('labels', ClassLabel(
    names=[f'{prefix}_entailment',f'{prefix}_neutral',f'{prefix}_contradiction']))

imppres__presupposition = imppres__prag = Classification("premise","hypothesis","gold_label",
    dataset_name="tasksource/imppres", config_name=imppres_presupposition,
    post_process=_imppres_post_process)

imppres__prag = Classification("premise","hypothesis","gold_label_prag",
    dataset_name="tasksource/imppres", config_name=imppres_implicature,
    post_process=lambda x: _imppres_post_process(x,'pragmatic'))

imppres__log = Classification("premise","hypothesis","gold_label_log",
    dataset_name="tasksource/imppres", config_name=imppres_implicature,
    post_process=lambda x: _imppres_post_process(x,'logical'))


#glue__diagnostics = Classification("premise","hypothesis","label",
#    dataset_name="pietrolesci/glue_diagnostics",splits=["test",None,None])

hlgd = Classification("headline_a", "headline_b", labels="label")

paws___labeled_final   = Classification("sentence1", "sentence2", name('label',['not_paraphrase','paraphrase']))
paws___labeled_swap    = Classification("sentence1", "sentence2", name('label',['not_paraphrase','paraphrase']), splits=["train", None, None])
#paws___unlabeled_final = Classification("sentence1", "sentence2", "label")

#quora = Classification(get.questions.text[0], get.questions.text[1], 'is_duplicate') # in glue
medical_questions_pairs = Classification("question_1","question_2", name("label",['not similar','similar']))
 
###################### Token Classification #########################

conll2003__pos_tags   = TokenClassification(tokens="tokens", labels='pos_tags')
conll2003__chunk_tags = TokenClassification(tokens="tokens", labels='chunk_tags')
conll2003__ner_tags   = TokenClassification(tokens="tokens", labels='ner_tags')

#tner___tweebank_ner    = TokenClassification(tokens="tokens", labels="tags")

######################## Multiple choice ###########################


model_written_evals = MultipleChoice('question', choices=['answer_matching_behavior','answer_not_matching_behavior'], labels=constant(0),  
    dataset_name="Anthropic/model-written-evals")

truthful_qa___multiple_choice = MultipleChoice(
    "question",
    choices_list=get.mc1_targets.choices,
    labels=constant(0)
)

fig_qa = MultipleChoice(
    "startphrase",
    choices=["ending1","ending2"],
    labels="labels",
    dataset_name="nightingal3/fig-qa",
    splits=["train","validation",None]
)

bigbench = MultipleChoice(
    'inputs',
    choices_list='multiple_choice_targets',
    labels=lambda x:x['multiple_choice_scores'].index(1) if 1 in ['multiple_choice_scores'] else -1,
    dataset_name='tasksource/bigbench',
    config_name=bigbench_discriminative_english - {"social_i_qa","intersect_geometry"} # english multiple choice tasks, minus duplicates
)
#"goal_step_wikihow"

blimp_hard = MultipleChoice(inputs=constant(''),
    choices=['sentence_good','sentence_bad'],
    labels=constant(0),
    dataset_name="blimp",
    config_name=blimp_hard # tasks where GPT2 is at least 10% below  human accuracy
)

cos_e = MultipleChoice('question',
    choices_list='choices',
    labels= lambda x: x['choices_list'].index(x['answer']),
    config_name='v1.0')

cosmos_qa = MultipleChoice(cat(['context','question']),regen('answer[0-3]'),'label')

dream = MultipleChoice(
    lambda x:"\n".join(x['dialogue']+[x['question']]),
    choices_list='choice',
    labels=lambda x:x['choices_list'].index(x['answer'])
)

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
    labels = lambda x:[a['aid'] for a in x["answers"]].index(x["ra"])
)


sciq = MultipleChoice(
    'question',
    ['correct_answer']+regen('distractor[1-3]'),
    labels=constant(0))

social_i_qa = MultipleChoice(
    'question',
    ['answerA','answerB','answerC'],
    'label')

wiki_hop___original = MultipleChoice(
    'question', 
    choices_list='candidates',
    labels=lambda x:x['choices_list'].index(x["answer"]))

wiqa = MultipleChoice('question_stem',
    choices_list = lambda x: x['choices']['text'],
    labels='answer_label_as_choice')

piqa = MultipleChoice('goal', choices=['sol1','sol2'], labels='label')

hellaswag = MultipleChoice('ctx_a',
    choices_list=lambda x: [f'{x["ctx_b"]}{e}' for e in x["endings"]],
    labels='label', splits=['train','validation',None])

super_glue___copa = MultipleChoice('premise',['choice1','choice2'],'label')

balanced_copa = MultipleChoice('premise',['choice1','choice2'],'label',
    dataset_name="pkavumba/balanced-copa")

e_care = MultipleChoice('premise',['choice1','choice2'],'label',
    dataset_name="12ml/e-CARE")

art = MultipleChoice(cat(['hypothesis_1','hypothesis_2']),
    ['observation_1','observation_2'],
    labels=lambda x:x['label']-1,
    splits=['train','validation',None]
)


mmlu = MultipleChoice('question',labels='answer',choices_list='choices',splits=['validation','dev','test'],
    dataset_name="tasksource/mmlu",
    config_name=get_dataset_config_names("tasksource/mmlu")
)

winogrande = MultipleChoice('sentence',['option1','option2'],'answer',config_name='winogrande_xl',
    splits=['train','validation',None])

codah = MultipleChoice('question_propmt',choices_list='candidate_answers',labels='correct_answer_idx',config_name='codah')

ai2_arc__challenge = MultipleChoice('question',
    choices_list=get.choices.text,  
    labels=lambda x: get.choices.label(x).index(x["answerKey"]),
    config_name=["ARC-Challenge","ARC-Easy"])

definite_pronoun_resolution = MultipleChoice(
    inputs=cat(["sentence","pronoun"],' : '),
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
    labels = lambda x:'abcde'.index(x['correct'])   
)

#aqua_rat___tokenized = MultipleChoice("question",choices_list="options",labels=lambda x:"ABCDE".index(x['correct'])) in math_qa


######################## Classification (other) ########################
glue___cola = Classification(sentence1="sentence", labels="label")
glue___sst2 = Classification(sentence1="sentence", labels="label")

utilitarianism = Classification("comparison",labels="label",
dataset_name="metaeval/utilitarianism")

amazon_counterfactual = Classification(
    "text", labels="label",
    dataset_name="mteb/amazon_counterfactual",
    config_name="en")

insincere_questions = Classification(
    "text", labels="label_text",
    dataset_name="SetFit/insincere-questions")

toxic_conversations = Classification(
    "text", labels="label",
    dataset_name="SetFit/toxic_conversations")

turingbench = Classification("Generation",labels="label",
    dataset_name="turingbench/TuringBench",
    splits=["train","validation",None])


trec = Classification(sentence1="text", labels="fine_label")

tals_vitaminc = Classification('claim','evidence','label', dataset_name="tals/vitaminc", config_name="tals--vitaminc")

hope_edi = Classification("text", labels="label", splits=["train", "validation", None], config_name=["english"])

#fever___v1_0 = Classification(sentence1="claim", labels="label", splits=["train", "paper_dev", "paper_test"], dataset_name="fever", config_name="v1.0")
#fever___v2_0 = Classification(sentence1="claim", labels="label", splits=[None, "validation", None], dataset_name="fever", config_name="v2.0")

rumoureval_2019 = Classification(
    sentence1="source_text",
    sentence2=lambda x: str(x["reply_text"]),
    labels="label", dataset_name="strombergnlp/rumoureval_2019", config_name="RumourEval2019",
    post_process=lambda ds:ds.filter(lambda x:x['labels']!=None)    
)

ethos___binary = Classification(sentence1="text", labels="label", splits=["train", None, None])
ethos___multilabel = Classification(
    'text',
    labels=lambda x: [x[c] for c in
    ['violence', 'gender', 'race', 'national_origin', 'disability', 'religion', 'sexual_orientation','directed_vs_generalized']
    ],
    splits=["train", None, None]
)

tweet_eval = Classification(sentence1="text", labels="label",
    config_name=["emoji", "emotion", "hate", "irony", "offensive", "sentiment"])

def stance_kwargs(topic):
    return {
        "sentence1": constant(f'Topic: {topic}. \n Opinion:\n'), 
        "sentence2": "text", 
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
    config_name= ["emobank-arousal", "emobank-dominance", "emobank-valence", "squinky-formality", "squinky-implicature", 
    "squinky-informativeness","switchboard","mrda","verifiability"])

pragmeval_2 = Classification("sentence1","sentence2",labels="label",
    dataset_name="pragmeval",
    config_name= ["emergent", "gum", "pdtb", "persuasiveness-claimtype", 
    "persuasiveness-eloquence", "persuasiveness-premisetype", "persuasiveness-relevance", "persuasiveness-specificity", 
    "persuasiveness-strength", "sarcasm","stac"])

silicone = Classification("Utterance",labels="Label",
    config_name=['dyda_da', 'dyda_e', 'iemocap', 'maptask', 'meld_e', 'meld_s', 'oasis', 'sem'] # +['swda', 'mrda'] # in pragmeval
)

lex_glue___eurlex = Classification(sentence1="text", labels="labels") 
lex_glue___scotus = Classification(sentence1="text", labels="label")
lex_glue___ledgar = Classification(sentence1="text", labels="label")
lex_glue___unfair_tos = Classification(sentence1="text", labels="labels")
lex_glue___case_hold = MultipleChoice("context", choices_list='endings', labels="label")

language_identification = Classification("text",labels="labels", dataset_name="papluca/language-identification")

################ Automatically generated (verified)##########

imdb = Classification(sentence1="text", labels="label", splits=["train", None, "test"])

rotten_tomatoes = Classification(sentence1="text", labels="label")

ag_news = Classification(sentence1="text", labels="label", splits=["train", None, "test"])

yelp_review_full = Classification(sentence1="text", labels="label", splits=["train", None, "test"], config_name=["yelp_review_full"])

financial_phrasebank = Classification(sentence1="sentence", labels="label", splits=["train", None, None],
    config_name=["sentences_allagree"])

poem_sentiment = Classification(sentence1="verse_text", labels="label")

#emotion = Classification(sentence1="text", labels="label") # file not found

dbpedia_14 = Classification(sentence1="content", labels="label", splits=["train", None, "test"], config_name=["dbpedia_14"])

amazon_polarity = Classification(sentence1="content", labels="label", splits=["train", None, "test"], config_name=["amazon_polarity"])

app_reviews = Classification("review", labels="star", splits=["train", None, None])

# multi_nli = Classification(sentence1="premise", sentence2="hypothesis", labels="label", splits=["train", "validation_matched", None]) #glue

hate_speech18 = Classification(sentence1="text", labels="label", splits=["train", None, None])

sms_spam = Classification(sentence1="sms", labels="label", splits=["train", None, None])

humicroedit___subtask_1 = Classification("original", "edit", labels="meanGrade", dataset_name="humicroedit", config_name="subtask-1")
humicroedit___subtask_2 = Classification(
    sentence1=cat(['original1','edit1'],' : '),
    sentence2=cat(['original2','edit2'],' : '),
    labels="label", dataset_name="humicroedit", config_name="subtask-2")

snips_built_in_intents = Classification(sentence1="text", labels="label", splits=["train", None, None])

banking77 = Classification(sentence1="text", labels="label", splits=["train", None, "test"])

hate_speech_offensive = Classification(sentence1="tweet", labels="class", splits=["train", None, None])

yahoo_answers_topics = Classification(
    "question_title","question_content",labels="topic")

stackoverflow_questions=Classification("title","body",labels="label",
    dataset_name="pacovaldez/stackoverflow-questions")

#hyperpartisan_news_detection___byarticle = Classification(sentence1="text", labels="hyperpartisan", splits=["train", None, None]) # files too heavy
#hyperpartisan_news_detection___bypublisher = Classification(sentence1="text", labels="hyperpartisan", splits=["train","validation", None]) # files too heavy
hyperpartisan_news = Classification("text",labels="label",dataset_name="zapsdcn/hyperpartisan_news")

scierc = Classification("text",labels="label",dataset_name="zapsdcn/sciie")
citation_intent = Classification("text",labels="label",dataset_name="zapsdcn/citation_intent")

#go_emotions___raw = Classification(sentence1="text", splits=["train", None, None])
go_emotions___simplified = Classification(sentence1="text", labels="labels")

#boolq = Classification(sentence1="question", splits=["train", "validation", None]) # in superglue

#ecthr_cases___alleged_violation_prediction = Classification(labels="labels", dataset_name="ecthr_cases", config_name="alleged-violation-prediction")
#ecthr_cases___violation_prediction = Classification(labels="labels", dataset_name="ecthr_cases", config_name="violation-prediction")
#   too long

scicite = Classification(sentence1="string", labels="label",dataset_name="allenai/scicite")

liar = Classification(sentence1="statement", labels="label")

relbert_lexical_relation_classification = Classification(sentence1="head", sentence2="tail", labels="relation",
 dataset_name="relbert/lexical_relation_classification",
 config_name=["BLESS","CogALexV","EVALution","K&H+N","ROOT09"])


linguisticprobing = Classification("sentence", labels="label", dataset_name="tasksource/linguisticprobing", 
    config_name=['subj_number',
                'obj_number',
                'past_present',
                'sentence_length',
                'top_constituents',
                'tree_depth',
                'coordination_inversion',
                'odd_man_out',
                'bigram_shift']#+['word_content'] #too many labels 
)

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

ethics___commonsense = Classification(sentence1="text", labels="label", dataset_name="metaeval/ethics", config_name="commonsense")
ethics___deontology = Classification(sentence1="text", labels="label", dataset_name="metaeval/ethics", config_name="deontology")
ethics___justice = Classification(sentence1="text", labels="label", dataset_name="metaeval/ethics", config_name="justice")
ethics___virtue = Classification(sentence1="sentence1", sentence2="sentence2", labels="label", dataset_name="metaeval/ethics", config_name="virtue")

emo = Classification(sentence1="text", labels="label", splits=["train", None, "test"], config_name=["emo2019"])

google_wellformed_query = Classification(sentence1="content", labels="rating")

tweets_hate_speech_detection = Classification(sentence1="tweet", labels="label", splits=["train", None, None])

#adv_glue___adv_sst2 = Classification(sentence1="sentence", labels="label", splits=["validation", None, None])
#adv_glue___adv_qqp = Classification(sentence1="question1", sentence2="question2", labels="label", splits=["validation", None, None])
#adv_glue___adv_mnli = Classification(sentence1="premise", sentence2="hypothesis", labels="label", splits=["validation", None, None])
#adv_glue___adv_mnli_mismatched = Classification(sentence1="premise", sentence2="hypothesis", labels="label", splits=["validation", None, None])
#adv_glue___adv_qnli = Classification(sentence1="question", labels="label", splits=["validation", None, None])
#adv_glue___adv_rte = Classification(sentence1="sentence1", sentence2="sentence2", labels="label", splits=["validation", None, None])

has_part = Classification("arg1","arg2", labels="score", splits=["train", None, None])

wnut_17 = TokenClassification(tokens="tokens", labels="ner_tags", config_name=["wnut_17"])

ncbi_disease = TokenClassification(tokens="tokens", labels="ner_tags", config_name=["ncbi_disease"])

acronym_identification = TokenClassification(labels="labels", tokens="tokens")

jnlpba = TokenClassification(tokens="tokens", labels="ner_tags", splits=["train", "validation", None], config_name=["jnlpba"])

#species_800 = TokenClassification(tokens="tokens", labels="ner_tags", config_name=["species_800"]) missing files

SpeedOfMagic_ontonotes_english = TokenClassification(tokens="tokens", labels="ner_tags", dataset_name="SpeedOfMagic/ontonotes_english", config_name="SpeedOfMagic--ontonotes_english")

blog_authorship_corpus__gender    = Classification(sentence1="text",labels="gender")
blog_authorship_corpus__age       = Classification(sentence1="text",labels="age")
#blog_authorship_corpus__horoscope = Classification(sentence1="text",labels="horoscope")
blog_authorship_corpus__job       = Classification(sentence1="text",labels="job")

launch_open_question_type = Classification(sentence1="question", labels="resolve_type", dataset_name="launch/open_question_type")

health_fact = Classification(sentence1="claim", labels="label",
    pre_process = lambda ds:ds.filter(lambda x:x['label'] not in {-1})
)

commonsense_qa = MultipleChoice(
    "question",
    choices_list=get.choices.text,
    labels=lambda x: "ABCDE".index(x["answerKey"]),
    splits=["train","validation",None]
)
mc_taco = Classification(
    lambda x: f'{x["sentence"]} {x["question"]} {x["answer"]}',
    labels="label",
    splits=[ "validation",None,"test"]
)

ade_corpus_v2___Ade_corpus_v2_classification = Classification("text",labels="label")

discosense = MultipleChoice("context",choices=regen("option\_[0-3]"),labels="label",
    dataset_name="prajjwal1/discosense")
    
circa = Classification(
    sentence1=cat(["context","question-X"]),
    sentence2="answer-Y",
    labels="goldstandard2", post_process=remove_neg_1)

#code_x_glue_cc_defect_detection = Classification("func", labels="target")

#code_x_glue_cc_clone_detection_big_clone_bench = Classification("func1", "func2", "label") # in bigbench + too heavy (100g)

#code_x_glue_cc_code_refinement = MultipleChoice(
#    constant(""), choices=["buggy","fixed"], labels=constant(0),
#    config_name="medium")

#effective_feedback_student_writing = Classification("discourse_text", 
#labels="discourse_effectiveness",dataset_name="YaHi/EffectiveFeedbackStudentWriting")
# discontinued /!\

#promptSentiment = Classification("text",labels="label",dataset_name="Ericwang/promptSentiment")
#promptNLI = Classification("premise","hypothesis",labels="label",dataset_name="Ericwang/promptNLI")
#promptSpoke = Classification("text",labels="label",dataset_name="Ericwang/promptSpoke")
#promptProficiency = Classification("text",labels="label",dataset_name="Ericwang/promptProficiency")
#promptGrammar = Classification("text",labels="label",dataset_name="Ericwang/promptGrammar")
#promptCoherence = Classification("text",labels="label",dataset_name="Ericwang/promptCoherence")

phrase_similarity = Classification(
    sentence1=cat(["phrase1","sentence1"], " : "),
    sentence2=cat(["phrase2","sentence2"], " : "),
    labels='label',
    dataset_name="PiC/phrase_similarity"
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

mwong_fever_evidence_related = Classification(sentence1="claim", sentence2="evidence", labels="labels", splits=["train", "valid", "test"], dataset_name="mwong/fever-evidence-related", config_name="mwong--fever-related")

numer_sense = Classification("sentence",labels="target",splits=["train",None,None])

dynasent__r1 = Classification("sentence", labels="gold_label", 
    dataset_name="dynabench/dynasent", config_name="dynabench.dynasent.r1.all")
dynasent__r2 = Classification("sentence", labels="gold_label", 
    dataset_name="dynabench/dynasent", config_name="dynabench.dynasent.r2.all")

sarcasm_news = Classification("headline", labels="is_sarcastic",
    dataset_name="raquiba/Sarcasm_News_Headline")

sem_eval_2010_task_8 = Classification("sentence",labels="relation")

demo_org_auditor_review = Classification(sentence1="sentence", labels="label", splits=["train", None, "test"], dataset_name="demo-org/auditor_review", config_name="demo-org--auditor_review")

medmcqa = MultipleChoice("question", choices=regen('op[a-d]'),labels='cop')


dynasent_disagreement    = Classification("text", labels="binary_disagreement", dataset_name="RuyuanWan/Dynasent_Disagreement")
politeness_disagreement  = Classification("text", labels="binary_disagreement", dataset_name="RuyuanWan/Politeness_Disagreement")
sbic_disagreement        = Classification("text", labels="binary_disagreement", dataset_name="RuyuanWan/SBIC_Disagreement")
schem_disagreement       = Classification("text", labels="binary_disagreement", dataset_name="RuyuanWan/SChem_Disagreement")
dilemmas_disagreement    = Classification("text", labels="binary_disagreement", dataset_name="RuyuanWan/Dilemmas_Disagreement")

logiqa = MultipleChoice(
    cat(["context","query"]),
    choices_list = 'options',
    labels = "correct_option",
    dataset_name="lucasmccabe/logiqa"
)

#proto_qa = MultipleChoice(
#    "question",
#    choices_list=lambda x:x['answer-clusters']['answers'],
#    labels=lambda x: x['answer-clusters']['count'].index(max(x['answer-clusters']['count'])),
#    config_name='proto_qa'
#)

wiki_qa = Classification("question","answer", name("label",['False','True']))

cycic_classification = Classification("question",labels=name("correct_answer",['False','True']),
    dataset_name = "tasksource/cycic_classification")
cycic_mc = MultipleChoice("question", choices=regen('answer\_option[0-4]'), labels="correct_answer",
    dataset_name = "tasksource/cycic_multiplechoice")


def _preprocess_chatgpt_detection(ex):
    import random
    label=random.random()<0.5
    ex['label']=int(label)
    ex['answer']=[str(ex['human_answers'][0]),str(ex['chatgpt_answers'][0])][label]
    return ex

#chatgpt_detection = Classification("question","answer","label",
#    dataset_name = 'Hello-SimpleAI/HC3', config_name="all",
#    pre_process=lambda dataset:dataset.map(_preprocess_chatgpt_detection))

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
    dataset_name="demelin/moral_stories", config_name="full")

prost = MultipleChoice(cat(["context","ex_question"]), choices=['A','B','C','D'],labels="label",
    dataset_name="corypaik/prost")

dyna_hate = Classification("text",labels="label",dataset_name="aps/dynahate",splits=['train',None,None])

syntactic_augmentation_nli = Classification('sentence1',"sentence2","gold_label",dataset_name="metaeval/syntactic-augmentation-nli")

autotnli = Classification("premises", "hypothesis", "label", dataset_name="tasksource/autotnli")
#equate = Classification("sentence1", "sentence2", "gold_label",dataset_name="metaeval/equate")

conqada = Classification("sentence1","sentence2","label",dataset_name="lasha-nlp/CONDAQA",
    pre_process = lambda ds:ds.filter(lambda x:x['label'] in {"DON'T KNOW","YES","NO"})
)

webgbpt_comparisons = MultipleChoice(get.question.full_text, choices=['answer_0','answer_1'],
    labels=lambda x:int(x['score_1']>0),
    dataset_name="openai/webgpt_comparisons")

synthetic_instruct = MultipleChoice('prompt', choices=['chosen', 'rejected'],
    labels=constant(0), dataset_name="Dahoas/synthetic-instruct-gptj-pairwise")

scruples = Classification("text",labels="binarized_label",dataset_name="metaeval/scruples")

wouldyourather = MultipleChoice(constant('Most people would rather:'), choices=['option_a','option_b'],
    labels= lambda x: int(x['votes_a']<x['votes_b']),
    dataset_name="metaeval/wouldyourather")

#attempto_nli = Classification("premise","hypothesis",
#    lambda x:f'race-{x["race_label"]}',
#    dataset_name="sileod/attempto-nli")

defeasible_nli = Classification(cat(["Premise","Hypothesis"]),"Update",labels="UpdateType",
    dataset_name="metaeval/defeasible-nli",config_name=['atomic', 'snli'])

#defeasible_nli_social = Classification(cat(["SocialChemROT","Hypothesis"]),"Update",labels="UpdateType",
#    dataset_name="metaeval/defeasible-nli",config_name='social')

help_nli = Classification("ori_sentence","new_sentence","gold_label",
    dataset_name="tasksource/help-nli")
    
nli_veridicality_transitivity = Classification("sentence1","sentence2","gold_label",
    dataset_name="metaeval/nli-veridicality-transitivity")

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
    labels="choice",
    dataset_name="openai/summarize_from_feedback", config_name="comparisons",
    pre_process = lambda ds:ds.filter(lambda x: type(get.info.post(x))==str)
)

folio = Classification("premises","conclusion",
    labels=lambda x:{'False':'contradiction','True':'entailment', 'Uncertain':'neutral'}.get(x["label"]),
    dataset_name="tasksource/folio")

tomi_nli = Classification("premise","hypothesis","label",
    dataset_name="tasksource/tomi-nli")

avicenna = Classification("Premise 1","Premise 2","Syllogistic relation",
    dataset_name="tasksource/avicenna")

shp = MultipleChoice("history",
    choices=['human_ref_A','human_ref_B'],
    labels="labels",
    dataset_name="stanfordnlp/SHP")

medqa_usmle = MultipleChoice('sent1',choices=regen('ending[0-3]'),labels='label',
    dataset_name="GBaker/MedQA-USMLE-4-options-hf")

wikimedqa = MultipleChoice("text",choices=regen('option\_[0-7]'),labels='label',
    dataset_name="sileod/wikimedqa",
    config_name=["medwiki"])

cicero = MultipleChoice(lambda x: " ".join(x['Dialogue']),
    choices_list="Choices", labels=lambda x:x['Human Written Answer'][0],
    dataset_name="declare-lab/cicero")

creak = Classification("sentence",labels="label",
    dataset_name='amydeng2000/CREAK')

mutual = MultipleChoice("article",choices_list="options",
    labels=lambda x: "ABCD".index(x['answers']),
    dataset_name="tasksource/mutual",splits=["train",None,None])

neqa = MultipleChoice('prompt',choices_list='classes',labels="answer_index",
    dataset_name="inverse-scaling/NeQA")
quote_repetition = MultipleChoice('prompt',choices_list='classes',labels="answer_index",
    dataset_name="inverse-scaling/quote-repetition")
redefine_math = MultipleChoice('prompt',choices_list='classes',labels="answer_index",
    dataset_name="inverse-scaling/redefine-math")

puzzte = Classification("puzzle_text","question","answer",
    dataset_name="tasksource/puzzte")

implicatures = MultipleChoice(cat(['context','response'],"\n"),
    choices=['correct_implicature','incorrect_implicature'],
    labels=constant(0),
    dataset_name='tasksource/implicatures')

race = MultipleChoice(cat(['question','article'],'\n'), choices_list='options',
    labels=lambda x:'ABCDE'.index(x['answer']),
    config_name=['middle','high'])

race_c = MultipleChoice(cat(['question','article'],'\n'),choices_list='option',labels='label',
    dataset_name='tasksource/race-c')

spartqa_yn=Classification("story","question","answer",
    dataset_name="tasksource/spartqa-yn")

spartqa_mc=MultipleChoice(cat(["story","question"]),choices_list="candidate_answers",labels="answer",
    dataset_name="tasksource/spartqa-mchoice")

temporal_nli = Classification("Premise","Hypothesis","Label",
    dataset_name="tasksource/temporal-nli")

riddle_sense = MultipleChoice("question", choices_list=get.choices.text, 
    labels=lambda x : "ABCDE".index(x['answerKey']))

clcd = Classification(
    "sentence1","sentence2","label",
    dataset_name="tasksource/clcd-english")

twentyquestions = Classification("question","subject","answer",dataset_name="maximedb/twentyquestions")

reclor = MultipleChoice(cat(["context","question"]),choices_list="answers",labels="label",
    dataset_name="metaeval/reclor",splits=['train','validation',None])

c_aug_imdb = Classification("Text",labels="Sentiment",
    dataset_name='tasksource/counterfactually-augmented-imdb')

c_aug_snli = Classification("sentence1","sentence2","gold_label",
    dataset_name='tasksource/counterfactually-augmented-snli')

cnli = Classification("premise","hypothesis","label",
    dataset_name='metaeval/cnli')

perturbed_boolq = Classification("question",labels="hard_label",
    dataset_name='tasksource/boolq-natural-perturbations')

#mega_acceptability = Classification("sentence",labels="average",
#    dataset_name='metaeval/mega-acceptability-v2')

graded_acceptability = Classification("text",labels="normalized_score",
    dataset_name="metaeval/acceptability-prediction")

equate = Classification("sentence1","sentence2","gold_label",
    dataset_name='metaeval/equate')

science_qa = MultipleChoice("question",choices_list="choices",labels="answer",
    dataset_name="tasksource/ScienceQA_text_only")

ekar=MultipleChoice("question",choices_list=get.choices.text,
    labels=lambda x:"ABCD".index(x['answerKey']),
dataset_name="Jiangjie/ekar_english")

implicit_hate = Classification("post",labels="class",
    dataset_name="tasksource/implicit-hate-stg1")

nli_unambiguity = Classification("premise","hypothesis","gini",
    dataset_name="metaeval/chaos-mnli-ambiguity")

headline_cause = Classification('left_title','right_title','label',
    dataset_name='IlyaGusev/headline_cause',config_name='en_simple')

logiqa_2 = Classification("premise","hypothesis","label",dataset_name="tasksource/logiqa-2.0-nli")

_oasst = dict(dataset_name="tasksource/oasst2_dense_flat",
    pre_process = lambda ds:ds.filter(lambda x:x['lang']=='en'))

oasst1__quality = Classification("parent_text","text",labels="quality",**_oasst)
oasst1__toxicity = Classification("parent_text","text",labels="toxicity",**_oasst)
oasst1__helpfulness = Classification("parent_text","text",labels="helpfulness",**_oasst)

mindgames = Classification("premise","hypothesis","label",dataset_name="sileod/mindgames")

def _udep_post_process(ds):
    return ds.cast_column('labels', Sequence(ClassLabel(names=udep_en_labels)))

udep__deprel = TokenClassification('tokens',lambda x:[udep_en_labels.index(a) for a in x['deprel']],
    config_name=udep_en_configs,dataset_name="universal_dependencies",post_process=_udep_post_process)

ambient= Classification("premise","hypothesis","hypothesis_ambiguous",dataset_name="metaeval/ambient")

path_naturalness = MultipleChoice(constant(""),choices=['choice1','choice2'],labels="label",
    dataset_name="metaeval/path-naturalness-prediction")

civil_comments__toxicity = Classification("text",labels="toxicity")
civil_comments__severe_toxicity = Classification("text",labels="severe_toxicity")
civil_comments__obscene = Classification("text",labels="obscene")
civil_comments__threat = Classification("text",labels="threat")
civil_comments__insult = Classification("text",labels="insult")
civil_comments__identity_attack = Classification("text",labels="identity_attack")
civil_comments__sexual_explicit = Classification("text",labels="sexual_explicit")

cloth = MultipleChoice("sentence", choices_list=lambda x:[x["answer"]]+x["distractors"],labels=constant(0), dataset_name="AndyChiang/cloth")
dgen  = MultipleChoice("sentence", choices_list=lambda x:[x["answer"]]+x["distractors"],labels=constant(0), dataset_name="AndyChiang/dgen")

i2d2 = Classification("sentence1",labels=name('label',['False','True']), dataset_name="tasksource/I2D2")

arg_me = Classification('argument','conclusion','stance', dataset_name="webis/args_me")
valueeval_stance = Classification("Premise","Conclusion","Stance", dataset_name="webis/Touche23-ValueEval")
starcon = Classification('argument','topic','label',dataset_name="tasksource/starcon")

banking77 = Classification("text",labels="label",dataset_name="PolyAI/banking77")
    
control = Classification('premise','hypothesis',"label",dataset_name="tasksource/ConTRoL-nli")
tracie = Classification("premise","hypothesis","answer",dataset_name='tasksource/tracie')
sherliic = Classification("premise","hypothesis","label",dataset_name='tasksource/sherliic')

sen_making__1 = MultipleChoice(constant('Chose most plausible:'), choices=['sentence0','sentence1'],labels='false', 
    dataset_name="tasksource/sen-making")

sen_making__2 = MultipleChoice(lambda x: [x['sentence0'],x['sentence1']][x['false']] + '\n is not plausible because :',
    choices=['A','B','C'],labels=lambda x: 'ABC'.index(x['reason']), dataset_name="tasksource/sen-making")

winowhy = Classification('sentence', lambda x: f'In "{x["wnli_sent1"]}", {x["wnli_sent2"]}',
    labels=name('label',['False','True']), dataset_name="tasksource/winowhy")

#for CFG in "cognitive-bias", "fake-news", "gender-bias", "hate-speech", "linguistic-bias", "political-bias", "racial-bias", "text-level-bias":
#    print(f"mbib__{CFG.replace('-','_')} = Classification('text',labels=name('label',['not {CFG}','{CFG}']), dataset_name='mediabiasgroup/mbib-base', config_name='{CFG}')")

"""
mbib_cognitive_bias	= Classification('text',labels=name('label',['not cognitive-bias','cognitive-bias']), dataset_name='mediabiasgroup/mbib-base', config_name='cognitive-bias')
mbib_fake_news	= Classification('text',labels=name('label',['not fake-news','fake-news']), dataset_name='mediabiasgroup/mbib-base', config_name='fake-news')
mbib_gender_bias	= Classification('text',labels=name('label',['not gender-bias','gender-bias']), dataset_name='mediabiasgroup/mbib-base', config_name='gender-bias')
mbib_hate_speech	= Classification('text',labels=name('label',['not hate-speech','hate-speech']), dataset_name='mediabiasgroup/mbib-base', config_name='hate-speech')
mbib_linguistic_bias	= Classification('text',labels=name('label',['not linguistic-bias','linguistic-bias']), dataset_name='mediabiasgroup/mbib-base', config_name='linguistic-bias')
mbib_political_bias	= Classification('text',labels=name('label',['not political-bias','political-bias']), dataset_name='mediabiasgroup/mbib-base', config_name='political-bias')
mbib_racial_bias	= Classification('text',labels=name('label',['not racial-bias','racial-bias']), dataset_name='mediabiasgroup/mbib-base', config_name='racial-bias')
mbib_text_level_bias	= Classification('text',labels=name('label',['not text-level-bias','text-level-bias']), dataset_name='mediabiasgroup/mbib-base', config_name='text-level-bias')
"""

robustLR = Classification("context","statement","label", dataset_name="tasksource/robustLR")

cluttr = Classification("story","query", "target_text",dataset_name="CLUTRR/v1", config_name="gen_train234_test2to10")

logical_fallacy = Classification("source_article", labels="logical_fallacies", dataset_name="tasksource/logical-fallacy")

parade = Classification("Definition1","Definition2", labels=name('Binary labels',["not-paraphrase","paraphrase"]), dataset_name="tasksource/parade")

cladder = Classification("given_info", "question", "answer",dataset_name="tasksource/cladder")

subjectivity = Classification("Sentence",labels="Label",dataset_name="tasksource/subjectivity")

moh   = Classification("context","expression","label", dataset_name="tasksource/MOH")
vuac  = Classification("context","expression","label", dataset_name="tasksource/VUAC")
trofi = Classification("context","expression","label", dataset_name="tasksource/TroFi", splits=['train',None,'test'])

sharc_classification = Classification("snippet", lambda x:f'{x["scenario"]}\n{x["question"]}',
    labels=lambda x:x["answer"] if x['answer'] in  {"Yes","No","Irrelevant"} else "Clarification needed",
    dataset_name='sharc_modified',config_name='mod')

conceptrules_v2 = Classification("context", "text", "label", dataset_name="tasksource/conceptrules_v2")

scidtb = Classification("unit1_txt","unit2_txt","label", dataset_name="metaeval/disrpt",config_name='eng.dep.scidtb.rels')

chunking = TokenClassification("tokens","chunk_tags", dataset_name="conll2000")

few_nerd = TokenClassification("tokens","fine_ner_tags",dataset_name="DFKI-SLT/few-nerd",config_name='supervised')
finer = TokenClassification('tokens','ner_tags',dataset_name='nlpaueb/finer-139')

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

#xsum_factuality = Classification("summary",labels="is_factual")

financial_sentiment = Classification("text",labels=name('label',['Bearish','Bullish','Neutral']),
    dataset_name="zeroshot/twitter-financial-news-sentiment")

def _icl_rand(x):
    import random
    return random.Random(x['sentence1'][:50]).randint(0,1) #deterministic label for each input

icl = Classification("inputs", lambda x: x['symbols'][_icl_rand(x)],
    labels=lambda x: str(x['symbols'][_icl_rand(x)]==x['targets']),
    dataset_name="tasksource/icl-symbol-tuning-instruct",
    pre_process=lambda ds:ds.filter(lambda x:len(x['inputs'])<500*4), # 500 tokens of 4 char 
)

space_nli = Classification("premises","hypothesis","label",dataset_name="tasksource/SpaceNLI")

propsegment = Classification("hypothesis","premise",
    labels = lambda x:{'n':'neutral','e':'entailment','c':'contradiction'}[x['label']],
    dataset_name="sihaochen/propsegment",config_name='nli')

hatemoji = Classification('text',labels=name("label_gold", ['not-hate-speech','hate-speech']),
    dataset_name="HannahRoseKirk/HatemojiBuild")

regset = Classification("context",labels="answer",dataset_name='tasksource/regset')

esci = Classification('query','product_text','esci_label',
    dataset_name="tasksource/esci",
    pre_process=lambda ds:ds.filter(lambda x:x['product_locale']=='us'))

def _preprocess_chatbot_arena(ds):
    ds=ds.filter(lambda x:x['winner'] in ["model_a","model_b"])
    ds=ds.filter(lambda x:x['language']=="English")

    def _unroll(x):
        f=lambda x:"\n".join([f"{turn['role']}:\n{turn['content']}" for turn in x])
        x['conversation_a'] = f(x['conversation_a'])
        x['conversation_b'] = f(x['conversation_b'])
        return x
    ds=ds.map(_unroll)
    return ds

chatbot_arena = MultipleChoice(constant(""),
    choices=["conversation_a","conversation_b"],
    labels=lambda x: ["model_a","model_b"].index(x["winner"]),
    dataset_name="lmsys/chatbot_arena_conversations",
    pre_process=_preprocess_chatbot_arena)

dnd_intent = Classification("examples",labels="label_names",
    dataset_name='neurae/dnd_style_intents')

fld = Classification("context","hypothesis", "proof_label",
    dataset_name="hitachi-nlp/FLD.v2",config_name="default")

flds = Classification("context","hypothesis", "proof_label",
    dataset_name="hitachi-nlp/FLD.v2",config_name="star")

sdoh_nli = Classification("premise","hypothesis",labels=lambda x:{True:"entailment",False:"not-entailment"}[x['label']],
    dataset_name="tasksource/SDOH-NLI")

scifact_entailment = Classification(lambda x:"\n".join(x["abstract"]),"claim",
    labels=lambda x:x['verdict'].replace('NEI','NEUTRAL').lower(),
    dataset_name="allenai/scifact_entailment")

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

oasst_rlhf = MultipleChoice("prompt",choices=['chosen','rejected'],labels=constant(0),
    dataset_name="tasksource/oasst2_pairwise_rlhf_reward")

anthropic_rlhf_helpfulness = MultipleChoice(constant('Most helpful assistant answer:'), ['chosen','rejected'], constant(0),
    dataset_name="tasksource/hh-rlhf",config_name=["helpful-base", "helpful-online", "helpful-rejection-sampled"])

anthropic_rlhf_harmless = MultipleChoice(constant('Most harmless assistant answer:'), ['chosen','rejected'], constant(0),
    dataset_name="tasksource/hh-rlhf",config_name="harmless-base")

ruletaker = Classification(
    lambda x: 'What is not explicitly stated as true is considered false. \n' +x["context"], #closed world assumption
    "question","label",dataset_name="tasksource/ruletaker")

para_rules = Classification(
    lambda x: 'What is not explicitly stated as true is considered false. \n' +x["context"], #closed world assumption
    "question", labels=name("label",["False","True"]),
    dataset_name="qbao775/PARARULE-Plus")

proofwriter_deduction = Classification("theory","question","answer",
    dataset_name="tasksource/proofwriter") #open world assumption

logical_entailment = Classification("A","B","label",dataset_name='tasksource/logical-entailment')

nope = Classification('premise','hypothesis',
    labels=lambda x:dict(E='entailment',N='neutral',C='contradiction').get(x['label'],x['label']),
    dataset_name='tasksource/nope')

logicNLI = Classification('premise','hypothesis','label',dataset_name='tasksource/LogicNLI')

contract_nli = Classification("premise","hypothesis","label", dataset_name="kiddothe2b/contract-nli",config_name="contractnli_a")

nli4ct = Classification(lambda x: "\n".join(x['Primary_evidence']),'Statement',
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

#GATED !
#toxigen = Classification("text",labels="toxicity_human", dataset_name="skg/toxigen-data")

persuasiveness = Classification("claim","argument",labels="persuasiveness_metric",dataset_name="Anthropic/persuasion")

#ste_wic = Classification(cat("text_1","text_2"),
#    lambda x:f"{x['target']} means the same thing in these texts",
#    "gold_label_binary",
#    dataset_name="cardiffnlp/super_tweeteval", config_name="tempo_wic",splits=['train','validation',None])

#ste_nerd = Classification("text",
#    lambda x:f"definition of {x['target']} here is 'x{['definition']}'",
#    "gold_label_binary",
#    dataset_name="cardiffnlp/super_tweeteval", config_name="tweet_nerd",splits=['train','validation',None])
 
#ste_sim = Classification("text_1","text_2",lambda x:x['gold_score']/5,
#    dataset_name="cardiffnlp/super_tweeteval",config_name="tweet_similarity",splits=['train','validation',None])

#ste_intimacy = Classification("text_1",labels=lambda x:x['gold_score']/5,
#    dataset_name="cardiffnlp/super_tweeteval",config_name="tweet_intimacy")

#ccdv/patent-classification|abstract text label

ambigNQ = Classification("question",labels=lambda x:{True:"ambiguous", False:"not ambiguous"}.get(x["ambig"]),
    dataset_name="erbacher/AmbigNQ-clarifying-question")

siga_nli = Classification("premise","statement","label",dataset_name="tasksource/SIGA-nli")

unigram_fol = Classification("premise","hypothesis","label",dataset_name='unigram/FOL-nli')

#gs_goal = MultipleChoice("sent2",regen("ending[0-3]"),"label",
#        dataset_name="tasksource/goal-step-wikihow",config_name="goal")

#gs_step = MultipleChoice("sent2",regen("ending[0-3]"),"label",
#        dataset_name="tasksource/goal-step-wikihow",config_name="step")

gs_order = MultipleChoice("sent2",regen("ending[0-1]"),"label",
        dataset_name="tasksource/goal-step-wikihow",config_name="order")

paradise = MultipleChoice("sent2",regen("ending[0-1]"),"label",
      dataset_name="GGLab/PARADISE")

docnli = Classification("premise","hypothesis","label",dataset_name="tasksource/doc-nli")

mctest_nli = Classification("premise","hypothesis","label",dataset_name="tasksource/mctest-nli")

patent_phrase_similarity = Classification("anchor","target","label",dataset_name="tasksource/patent-phrase-similarity")

nlsat = Classification('sentence',labels='label',dataset_name="tasksource/natural-language-satisfiability")

idioms_nli = Classification('premise','hypothesis','label',dataset_name="tasksource/idioms-nli")

lifeycle_entailment = Classification("premise","hypothesis","label",dataset_name='tasksource/lifecycle-entailment')

helpsteer_2__helpfulness = Classification("prompt","response","helpfulness",dataset_name="nvidia/HelpSteer2")
helpsteer_2__correctness = Classification("prompt", "response", "correctness", dataset_name="nvidia/HelpSteer2")
helpsteer_2__coherence = Classification("prompt", "response", "coherence", dataset_name="nvidia/HelpSteer2")
helpsteer_2__complexity = Classification("prompt", "response", "complexity", dataset_name="nvidia/HelpSteer2")
helpsteer_2__verbosity = Classification("prompt", "response", "verbosity", dataset_name="nvidia/HelpSteer2")

helpsteer__helpfulness = Classification("prompt", "response", "helpfulness", dataset_name="nvidia/HelpSteer")
helpsteer__correctness = Classification("prompt", "response", "correctness", dataset_name="nvidia/HelpSteer")
helpsteer__coherence = Classification("prompt", "response", "coherence", dataset_name="nvidia/HelpSteer")
helpsteer__complexity = Classification("prompt", "response", "complexity", dataset_name="nvidia/HelpSteer")
helpsteer__verbosity = Classification("prompt", "response", "verbosity", dataset_name="nvidia/HelpSteer")

msci_nli = Classification('sentence1','sentence2','label',dataset_name='sadat2307/MSciNLI')

lex_glue___ecthr_a = Classification(sentence1="text", labels="labels",dataset_name="coastalcph/lex_glue",config_name="ecthr_a") # too long
lex_glue___ecthr_b = Classification(sentence1="text", labels="labels") # too long

ultrafeedback = MultipleChoice("question", choices=['response_j','response_k'],labels=constant(0), dataset_name="pushpdeep/UltraFeedback-paired")

essay_scoring = Classification("full_text",labels="score",dataset_name='tasksource/AES2-essay-scoring')

argument_feedback = Classification("discourse_text",labels="discourse_effectiveness", dataset_name="tasksource/argument-feedback")

eg = lambda x: Classification("full_text", labels=lambda y:int(y[x]), dataset_name="tasksource/english-grading")
grading__cohesion = eg('cohesion')
grading__syntax = eg('syntax')
grading__vocabulary = eg('vocabulary')
grading__phraseology = eg('phraseology')
grading__grammar = eg('grammar')
grading__conventions = eg('conventions')
