from .preprocess import cat, get, regen, constant, Classification, TokenClassification, MultipleChoice
from .metadata import bigbench_discriminative_english, blimp_hard, imppres_presupposition, imppres_implicature
from datasets import get_dataset_config_names, ClassLabel

# variable name: dataset___config__task

###################### NLI/paraphrase ###############################


anli__a1 = Classification('premise','hypothesis','label', splits=['train_r1','dev_r1','test_r1'])
anli__a2 = Classification('premise','hypothesis','label', splits=['train_r2','dev_r2','test_r2'])
anli__a3 = Classification('premise','hypothesis','label', splits=['train_r3','dev_r3','test_r3'])


babi_nli = Classification("premise", "hypothesis", "label",
    dataset_name="metaeval/babi_nli",
    config_name=set(get_dataset_config_names("metaeval/babi_nli"))-{"agents-motivations"}
) # agents-motivations task is not as clear-cut as the others


sick__label         = Classification('sentence_A','sentence_B','label')
sick__relatedness   = Classification('sentence_A','sentence_B','relatedness_score')
sick__entailment_AB = Classification('sentence_A','sentence_B','entailment_AB')
sick__entailment_BA = Classification('sentence_A','sentence_B','entailment_BA')

snli = Classification(sentence1="premise", sentence2="hypothesis", labels="label")

scitail = Classification("sentence1","sentence2","gold_label",config_name="snli_format")

hans = Classification(sentence1="premise", sentence2="hypothesis", labels="label")

wanli = Classification('premise','hypothesis','gold', dataset_name="alisawuffles/WANLI")

recast = Classification(sentence1="context", sentence2="hypothesis", labels="label", dataset_name="metaeval/recast",
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

conj_nli = Classification("premise","hypothesis","label",
    dataset_name="pietrolesci/conj_nli")

fracas = Classification("premise","hypothesis","label",
    dataset_name="pietrolesci/fracas")

dialogue_nli = Classification("sentence1","sentence2","label",
    dataset_name="pietrolesci/dialogue_nli")   

mpe_nli = Classification("premise","hypothesis","label",
    dataset_name="pietrolesci/mpe",
    splits=["train","dev","test"])  

dnc_nli = Classification("context","hypothesis","label",
    dataset_name="pietrolesci/dnc")

gpt3_nli = Classification("text_a","text_b","label",dataset_name="pietrolesci/gpt3_nli")

recast_white__fnplus = Classification("text","hypothesis","label",
    dataset_name="pietrolesci/recast_white",splits=['fnplus',None,None])
recast_white__sprl = Classification("text","hypothesis","label",
    dataset_name="pietrolesci/recast_white",splits=['sprl',None,None])
recast_white__dpr = Classification("text","hypothesis","label",
    dataset_name="pietrolesci/recast_white",splits=['dpr',None,None])

joci = Classification("context","hypothesis","label", dataset_name="pietrolesci/joci",splits=['full',None,None])

#enfever_nli = Classification("evidence","claim","label", dataset_name="ctu-aic/enfever_nli")

contrast_nli = Classification("premise", "hypothesis",	"label",dataset_name="martn-nguyen/contrast_nli")

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

def imppres_post_process(ds,prefix=''):
    # imppres entailment definition is either purely semantic or purely pragmatic
    # because of that, we assign differentiate the labels from anli/mnli notation
    return ds.cast_column('labels', ClassLabel(
    names=[f'imppres{prefix}_entailment',f'imppres{prefix}_neutral',f'imppres{prefix}_contradiction']))

imppres__presupposition = imppres__prag = Classification("premise","hypothesis","gold_label",
    dataset_name="metaeval/imppres", config_name=imppres_presupposition,
    post_process=imppres_post_process)

imppres__prag = Classification("premise","hypothesis","gold_label_prag",
    dataset_name="metaeval/imppres", config_name=imppres_implicature,
    post_process=lambda x: imppres_post_process(x,'_prag'))

imppres__log = Classification("premise","hypothesis","gold_label_log",
    dataset_name="metaeval/imppres", config_name=imppres_implicature,
    post_process=lambda x: imppres_post_process(x,'_log'))


glue__diagnostics = Classification("premise","hypothesis","label",
    dataset_name="pietrolesci/glue_diagnostics",splits=["test",None,None])

hlgd = Classification("headline_a", "headline_b", labels="label")

paws___labeled_final   = Classification("sentence1", "sentence2", "label")
paws___labeled_swap    = Classification("sentence1", "sentence2", "label", splits=["train", None, None])
#paws___unlabeled_final = Classification("sentence1", "sentence2", "label")

quora = Classification(get.questions.text[0], get.questions.text[1], 'is_duplicate')
medical_questions_pairs = Classification("question_1","question_2", "label")
 
###################### Token Classification #########################

conll2003__pos_tags   = TokenClassification(tokens="tokens", labels='pos_tags')
conll2003__chunk_tags = TokenClassification(tokens="tokens", labels='chunk_tags')
conll2003__ner_tags   = TokenClassification(tokens="tokens", labels='ner_tags')

#tner___tweebank_ner    = TokenClassification(tokens="tokens", labels="tags")

######################## Multiple choice ###########################

anthropic_rlhf = MultipleChoice(constant(''), ['chosen','rejected'], constant(0),
    dataset_name="Anthropic/hh-rlhf")

model_written_evals = MultipleChoice('question', ['answer_matching_behavior','answer_not_matching_behavior'], constant(0),
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
    config_name=bigbench_discriminative_english - {"social_iqa"} # english multiple choice tasks, minus duplicates
)

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

#race___middle = MultipleChoice('question', choices_list='options', labels='answer')
#race___high   = MultipleChoice('question', choices_list='options', labels='answer')
# too long input


sciq = MultipleChoice(
    'question',
    ['correct_answer']+regen('distractor[1-3]'),
    labels=constant(0))

social_i_qa = MultipleChoice(
    'question',
    ['answerA','answerB','answerC'],
    'label')

wiki_hop = MultipleChoice(
    'question', 
    choices_list='candidates',
    labels=lambda x:x['choices_list'].index(x["answer"]))

wiqa = MultipleChoice('question_stem',
    choices_list = lambda x: x['choices']['text'],
    labels='answer_label_as_choice')

piqa = MultipleChoice('goal', ['sol1','sol2'], 'label')

hellaswag = MultipleChoice('ctx_a',
    choices_list=lambda x: [f'{x["ctx_b"]}{e}' for e in x["endings"]],
    labels='label', splits=['train','validation',None])

super_glue___copa = MultipleChoice('premise',['choice1','choice2'],'label')

art = MultipleChoice(cat(['hypothesis_1','hypothesis_2']),
    ['observation_1','observation_2'],
    labels=lambda x:x['label']-1,
    splits=['train','validation',None]
)


hendrycks_test = MultipleChoice('question',labels='answer',choices_list='choices',splits=['test','dev','validation'],
config_name=get_dataset_config_names("hendrycks_test")
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

swag=MultipleChoice(cat(["sent1","sent2"]),regen("ending[0-3]"),"label")

def split_choices(s):
    import re
    return [x.rstrip(', ') for x in re.split(r'[a-e] \) (.*?)',s) if x.strip(', ')]

math_qa = MultipleChoice(
    'Problem', 
    choices_list = lambda x: split_choices(x['options']),
    labels = lambda x:'abcde'.index(x['correct'])   
)


######################## Classification (other) ########################

utilitarianism = Classification("comparison",labels="label",
dataset_name="metaeval/utilitarianism")

amazon_counterfactual = Classification(
    "text", labels="label",
    dataset_name="mteb/amazon_counterfactual",
    config_name="en")

insincere_questions = Classification(
    "text", labels="label",
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

glue___cola = Classification(sentence1="sentence", labels="label")
glue___sst2 = Classification(sentence1="sentence", labels="label")
glue___mrpc = Classification(sentence1="sentence1", sentence2="sentence2", labels="label")
glue___qqp = Classification(sentence1="question1", sentence2="question2", labels="label")
glue___stsb = Classification(sentence1="sentence1", sentence2="sentence2", labels="label")
glue___mnli = Classification(sentence1="premise", sentence2="hypothesis", labels="label", splits=["train", None, "validation_matched"])
glue___qnli = Classification(sentence1="question", labels="label")
glue___rte = Classification(sentence1="sentence1", sentence2="sentence2", labels="label")
glue___wnli = Classification(sentence1="sentence1", sentence2="sentence2", labels="label")
#glue___ax = Classification(sentence1="premise", sentence2="hypothesis", labels="label", splits=["test", None, None]) # fully masked


super_glue___boolq = Classification(sentence1="question", labels="label")
super_glue___cb = Classification(sentence1="premise", sentence2="hypothesis", labels="label")
super_glue___multirc = Classification(sentence1="question", labels="label")
#super_glue___rte = Classification(sentence1="premise", sentence2="hypothesis", labels="label") # in glue
super_glue___wic = Classification(
    sentence1=cat(["word","sentence1"], " : "),
    sentence2=cat(["word","sentence2"], " : "),
    labels='label'
)
super_glue___axg = Classification(sentence1="premise", sentence2="hypothesis", labels="label", splits=["test", None, None])

tweet_eval = Classification(sentence1="text", labels="label", config_name=["emoji", "emotion", "hate", "irony", "offensive", "sentiment", "stance_abortion", "stance_atheism", "stance_climate", "stance_feminist", "stance_hillary"])

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

silicone = Classification("Uterance",labels="Label",
    config_name=['dyda_da', 'dyda_e', 'iemocap', 'maptask', 'meld_e', 'meld_s', 'oasis', 'sem'] # +['swda', 'mrda'] # in pragmeval
)

#lex_glue___ecthr_a = Classification(sentence1="text", labels="labels") # too long
#lex_glue___ecthr_b = Classification(sentence1="text", labels="labels") # too long
lex_glue___eurlex = Classification(sentence1="text", labels="labels") 
lex_glue___scotus = Classification(sentence1="text", labels="label")
lex_glue___ledgar = Classification(sentence1="text", labels="label")
lex_glue___unfair_tos = Classification(sentence1="text", labels="labels")
lex_glue___case_hold = MultipleChoice("context", choices_list='endings', labels="label")



################ Automatically generated (verified)##########

imdb = Classification(sentence1="text", labels="label", splits=["train", None, "test"])

#trec = Classification(sentence1="text", labels="fine_label", splits=["train", None, "test"])

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

#yahoo_answers_topics = Classification(splits=["train", None, "test"], config_name=["yahoo_answers_topics"]) #trivial


hyperpartisan_news_detection___byarticle = Classification(sentence1="text", labels="hyperpartisan", splits=["train", None, None])
hyperpartisan_news_detection___bypublisher = Classification(sentence1="text", labels="hyperpartisan", splits=["train","validation", None])

#go_emotions___raw = Classification(sentence1="text", splits=["train", None, None])
go_emotions___simplified = Classification(sentence1="text", labels="labels")

#boolq = Classification(sentence1="question", splits=["train", "validation", None]) # in superglue

#ecthr_cases___alleged_violation_prediction = Classification(labels="labels", dataset_name="ecthr_cases", config_name="alleged-violation-prediction")
#ecthr_cases___violation_prediction = Classification(labels="labels", dataset_name="ecthr_cases", config_name="violation-prediction")
#   too long

scicite = Classification(sentence1="string", labels="label")

liar = Classification(sentence1="statement", labels="label")

relbert_lexical_relation_classification = Classification(sentence1="head", sentence2="tail", labels="relation",
 dataset_name="relbert/lexical_relation_classification",
 config_name=["BLESS","CogALexV","EVALution","K&H+N","ROOT09"])


metaeval_linguisticprobing = Classification("sentence", labels="label", dataset_name="metaeval/linguisticprobing", 
    config_name=['subj_number',
                'word_content',
                'obj_number',
                'past_present',
                'sentence_length',
                'top_constituents',
                'tree_depth',
                'coordination_inversion',
                'odd_man_out',
                'bigram_shift']
)

metaeval_crowdflower = Classification("text", labels="label",
 splits=["train", None, None], dataset_name="metaeval/crowdflower",
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

metaeval_ethics___commonsense = Classification(sentence1="text", labels="label", dataset_name="metaeval/ethics", config_name="commonsense")
metaeval_ethics___deontology = Classification(sentence1="text", labels="label", dataset_name="metaeval/ethics", config_name="deontology")
metaeval_ethics___justice = Classification(sentence1="text", labels="label", dataset_name="metaeval/ethics", config_name="justice")
#metaeval_ethics___utilitarianism = Classification(sentence1="text", labels="label", dataset_name="metaeval/ethics", config_name="utilitarianism")
metaeval_ethics___virtue = Classification(sentence1="sentence1", sentence2="sentence2", labels="label", dataset_name="metaeval/ethics", config_name="virtue")

emo = Classification(sentence1="text", labels="label", splits=["train", None, "test"], config_name=["emo2019"])

google_wellformed_query = Classification(sentence1="content", labels="rating")

tweets_hate_speech_detection = Classification(sentence1="tweet", labels="label", splits=["train", None, None])

adv_glue___adv_sst2 = Classification(sentence1="sentence", labels="label", splits=["validation", None, None])
adv_glue___adv_qqp = Classification(sentence1="question1", sentence2="question2", labels="label", splits=["validation", None, None])
adv_glue___adv_mnli = Classification(sentence1="premise", sentence2="hypothesis", labels="label", splits=["validation", None, None])
adv_glue___adv_mnli_mismatched = Classification(sentence1="premise", sentence2="hypothesis", labels="label", splits=["validation", None, None])
adv_glue___adv_qnli = Classification(sentence1="question", labels="label", splits=["validation", None, None])
adv_glue___adv_rte = Classification(sentence1="sentence1", sentence2="sentence2", labels="label", splits=["validation", None, None])


has_part = Classification("arg1","arg2", labels="score", splits=["train", None, None])

wnut_17 = TokenClassification(tokens="tokens", labels="ner_tags", config_name=["wnut_17"])

ncbi_disease = TokenClassification(tokens="tokens", labels="ner_tags", config_name=["ncbi_disease"])

acronym_identification = TokenClassification(labels="labels", tokens="tokens")

jnlpba = TokenClassification(tokens="tokens", labels="ner_tags", splits=["train", "validation", None], config_name=["jnlpba"])

species_800 = TokenClassification(tokens="tokens", labels="ner_tags", config_name=["species_800"])

SpeedOfMagic_ontonotes_english = TokenClassification(tokens="tokens", labels="ner_tags", dataset_name="SpeedOfMagic/ontonotes_english", config_name="SpeedOfMagic--ontonotes_english")


blog_authorship_corpus__gender    = Classification(sentence1="text",labels="gender", splits=["train", "validation", None])
blog_authorship_corpus__age       = Classification(sentence1="text",labels="age", splits=["train", "validation", None])
blog_authorship_corpus__horoscope = Classification(sentence1="text",labels="horoscope", splits=["train", "validation", None])
blog_authorship_corpus__job       = Classification(sentence1="text",labels="job", splits=["train", "validation", None])

launch_open_question_type = Classification(sentence1="question", labels="resolve_type", dataset_name="launch/open_question_type")

health_fact = Classification(sentence1="claim", labels="label")

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
    labels="goldstandard2")

code_x_glue_cc_defect_detection = Classification("func", labels="target")
code_x_glue_cc_clone_detection_big_clone_bench = Classification("func1", "func2", "label")
code_x_glue_cc_code_refinement = MultipleChoice(
    constant(""), choices=["buggy","fixed"], labels=constant(0),
    config_name="medium")
effective_feedback_student_writing = Classification("discourse_text", labels="discourse_effectiveness",dataset_name="YaHi/EffectiveFeedbackStudentWriting")

promptSentiment = Classification("text",labels="label",dataset_name="Ericwang/promptSentiment")
promptNLI = Classification("premise","hypothesis",labels="label",dataset_name="Ericwang/promptNLI")
promptSpoke = Classification("text",labels="label",dataset_name="Ericwang/promptSpoke")
promptProficiency = Classification("text",labels="label",dataset_name="Ericwang/promptProficiency")
promptGrammar = Classification("text",labels="label",dataset_name="Ericwang/promptGrammar")
promptCoherence = Classification("text",labels="label",dataset_name="Ericwang/promptCoherence")

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
    labels="answer_index"
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

###END
################### END OF SUPPORT ######################

# RuyuanWan/SChem_Disagreement

crows_pairs = Classification(splits=["test", None, None], config_name=["crows_pairs"]) # not a task we want to perform well on

jigsaw_unintended_bias = Classification(sentence1="comment_text", labels="rating", splits=["train", None, None])
jigsaw_toxicity_pred = Classification(sentence1="comment_text", splits=["train", None, "test"])
# has to be downloaded

# proto_qa
tab_fact = Classification(labels="label", config_name=["tab_fact", "blind_test"])

tab_fact___blind_test = Classification(splits=["test", None, None])

movie_rationales = Classification(labels="label")
#conllpp = TokenClassification(tokens="tokens", labels="pos_tags", config_name=["conllpp"])
#tner_tweetner7 = TokenClassification(tokens="tokens", labels="tags", splits=[None, None, None], dataset_name="tner/tweetner7", config_name="tweetner7")
#tner_ontonotes5 = TokenClassification(tokens="tokens", labels="tags", dataset_name="tner/ontonotes5", config_name="ontonotes5")


#strombergnlp_twitter_pos___foster = TokenClassification(tokens="tokens", labels="pos_tags", splits=[None, "validation", "test"], dataset_name="strombergnlp/twitter_pos", config_name="foster")
#strombergnlp_twitter_pos___ritter = TokenClassification(tokens="tokens", labels="pos_tags", dataset_name="strombergnlp/twitter_pos", config_name="ritter")
#strombergnlp_named_timexes = TokenClassification(tokens="tokens", labels="ntimex_tags", splits=["train", None, "test"], dataset_name="strombergnlp/named_timexes", config_name="named-timexes")


biosses = Classification(sentence1="sentence1", sentence2="sentence2", labels="score", splits=["train", None, None])

sem_eval_2014_task_1 = Classification(sentence1="premise", sentence2="hypothesis")

gutenberg_time = Classification(splits=["train", None, None], config_name=["gutenberg"])


clinc_oos = Classification(sentence1="text", config_name=["small", "imbalanced", "plus"])

nlu_evaluation_data = Classification(sentence1="text", labels="label", splits=["train", None, None])

newspop = Classification(splits=["train", None, None])

hatexplain = TokenClassification()

bing_coronavirus_query_set = Classification(splits=["train", None, None], config_name=["country_2020-09-01_2020-09-30"])

stereoset = Classification(sentence1="context", splits=["validation", None, None], config_name=["intersentence", "intrasentence"])

swda = Classification(sentence1="text")


#conceptnet5 = Classification(sentence1="sentence", splits=["train", None, None], config_name=["conceptnet5", "omcs_sentences_free", "omcs_sentences_more"])

#conceptnet5___omcs_sentences_free = Classification(sentence1="sentence", splits=["train", None, None])
#conceptnet5___omcs_sentences_more = Classification(sentence1="sentence", splits=["train", None, None])

ucberkeley_dlab_measuring_hate_speech = Classification(splits=["train", None, None], dataset_name="ucberkeley-dlab/measuring-hate-speech", config_name="ucberkeley-dlab--measuring-hate-speech")

eurlex = Classification(sentence1="text", config_name=["eurlex57k"])

fhamborg_news_sentiment_newsmtsc___mt = Classification(sentence1="sentence", dataset_name="fhamborg/news_sentiment_newsmtsc", config_name="mt")
fhamborg_news_sentiment_newsmtsc___rw = Classification(sentence1="sentence", dataset_name="fhamborg/news_sentiment_newsmtsc", config_name="rw")

prachathai67k = Classification(config_name=["prachathai67k"])

cardiffnlp_tweet_topic_multi = Classification(sentence1="text", labels="label", splits=[None, None, None], dataset_name="cardiffnlp/tweet_topic_multi", config_name="tweet_topic_multi")

datacommons_factcheck = Classification(labels="review_rating", splits=["train", None, None], config_name=["fctchk_politifact_wapo", "weekly_standard"])

scifact___corpus = Classification(splits=["train", None, None])
scifact___claims = Classification(sentence1="claim")

coastalcph_fairlex___ecthr = Classification(sentence1="text", labels="labels", dataset_name="coastalcph/fairlex", config_name="ecthr")
coastalcph_fairlex___scotus = Classification(sentence1="text", labels="label", dataset_name="coastalcph/fairlex", config_name="scotus")
coastalcph_fairlex___fscs = Classification(sentence1="text", labels="label", dataset_name="coastalcph/fairlex", config_name="fscs")
coastalcph_fairlex___cail = Classification(sentence1="text", labels="label", dataset_name="coastalcph/fairlex", config_name="cail")

peer_read = Classification(config_name=["parsed_pdfs", "reviews"])

per_sent = Classification(splits=["train", "validation", None])

diplomacy_detection = Classification()

demo_org_auditor_review = Classification(sentence1="sentence", labels="label", splits=["train", None, "test"], dataset_name="demo-org/auditor_review", config_name="demo-org--auditor_review")

Abirate_english_quotes = Classification(splits=["train", None, None], dataset_name="Abirate/english_quotes", config_name="Abirate--english_quotes")

sled_umich_TRIP = Classification(labels="label", splits=[None, None, None], dataset_name="sled-umich/TRIP")

GonzaloA_fake_news = Classification(sentence1="text", labels="label", dataset_name="GonzaloA/fake_news", config_name="GonzaloA--fake_news")

consumer_finance_complaints = Classification(splits=["train", None, None], dataset_name="consumer-finance-complaints")

ohsumed = Classification(splits=["train", None, "test"], config_name=["ohsumed"])

hate_offensive = Classification(sentence1="tweet", labels="label", splits=["train", None, None])

fake_news_english = Classification(splits=[None, None, None])

coarse_discourse = Classification(splits=["train", None, None])

Paul_hatecheck = Classification(splits=["test", None, None], dataset_name="Paul/hatecheck", config_name="Paul--hatecheck")

hippocorpus = Classification(splits=["train", None, None])

multi_nli_mismatch = Classification(sentence1="premise", sentence2="hypothesis", labels="label", splits=["train", "validation", None])

time_dial = Classification(splits=["test", None, None])

NbAiLab_norec_agg = Classification(sentence1="text", labels="label", dataset_name="NbAiLab/norec_agg")

DanL_scientific_challenges_and_directions_dataset = Classification(sentence1="text", labels="label", splits=["train", "dev", "test"], dataset_name="DanL/scientific-challenges-and-directions-dataset", config_name="DanL--scientific-challenges-and-directions-dataset")

CyranoB_polarity = Classification(sentence1="content", labels="label", splits=["train", None, "test"], dataset_name="CyranoB/polarity", config_name="CyranoB--polarity")

bhavnicksm_sentihood = Classification(dataset_name="bhavnicksm/sentihood", config_name="bhavnicksm--sentihood")

cardiffnlp_tweet_topic_single = Classification(sentence1="text", labels="label", splits=[None, None, None], dataset_name="cardiffnlp/tweet_topic_single", config_name="tweet_topic_single")

OxAISH_AL_LLM_wiki_toxic = Classification(sentence1="comment_text", labels="label", dataset_name="OxAISH-AL-LLM/wiki_toxic")

carblacac_twitter_sentiment_analysis = Classification(sentence1="text", dataset_name="carblacac/twitter-sentiment-analysis")

bdotloh_empathetic_dialogues_contexts = Classification(dataset_name="bdotloh/empathetic-dialogues-contexts", config_name="bdotloh--empathetic-dialogues-contexts")

zeroshot_twitter_financial_news_sentiment = Classification(splits=["train", "validation", None], dataset_name="zeroshot/twitter-financial-news-sentiment", config_name="zeroshot--twitter-financial-news-sentiment")

pacovaldez_stackoverflow_questions = Classification(dataset_name="pacovaldez/stackoverflow-questions", config_name="pacovaldez--stackoverflow-questions")

PolyAI_banking77 = Classification(sentence1="text", labels="label", splits=["train", None, "test"], dataset_name="PolyAI/banking77")

FinanceInc_auditor_sentiment = Classification(sentence1="sentence", labels="label", splits=["train", None, "test"], dataset_name="FinanceInc/auditor_sentiment", config_name="demo-org--auditor_review")

Tidrael_tsl_news = Classification(labels="label", splits=["train", None, "test"], dataset_name="Tidrael/tsl_news", config_name="plain_text")

okite97_news_data = Classification(splits=["train", None, "test"], dataset_name="okite97/news-data", config_name="okite97--news-data")

mwong_fever_claim_related = Classification(sentence1="claim", sentence2="evidence", labels="labels", splits=["train", None, "test"], dataset_name="mwong/fever-claim-related", config_name="mwong--fever-claim-related")

llangnickel_long_covid_classification_data = Classification(splits=["train", None, "test"], dataset_name="llangnickel/long-covid-classification-data", config_name="llangnickel--long-covid-classification-data")

rungalileo_20_Newsgroups_Fixed = Classification(splits=["train", None, "test"], dataset_name="rungalileo/20_Newsgroups_Fixed", config_name="rungalileo--20_Newsgroups_Fixed")

merve_poetry = Classification(splits=["train", None, None], dataset_name="merve/poetry", config_name="merve--poetry")

DFKI_SLT_tacred___original = Classification(labels="relation", dataset_name="DFKI-SLT/tacred", config_name="original")
DFKI_SLT_tacred___revised = Classification(labels="relation", dataset_name="DFKI-SLT/tacred", config_name="revised")

valurank_News_Articles_Categorization = Classification(splits=["train", None, None], dataset_name="valurank/News_Articles_Categorization", config_name="valurank--News_Articles_Categorization")

arize_ai_ecommerce_reviews_with_language_drift = Classification(sentence1="text", labels="label", splits=["training", "validation", None], dataset_name="arize-ai/ecommerce_reviews_with_language_drift")

copenlu_fever_gold_evidence = Classification(dataset_name="copenlu/fever_gold_evidence", config_name="copenlu--fever_gold_evidence")

qanastek_Biosses_BLUE = Classification(sentence1="sentence1", sentence2="sentence2", labels="score", dataset_name="qanastek/Biosses-BLUE", config_name="biosses")

arize_ai_movie_reviews_with_context_drift = Classification(splits=["train", "validation", None], dataset_name="arize-ai/movie_reviews_with_context_drift", config_name="arize-ai--movie_reviews_with_context_drift")

launch_ampere = Classification(dataset_name="launch/ampere", config_name="launch--ampere")

jpwahle_etpc = Classification(splits=[None, None, None], dataset_name="jpwahle/etpc")

climatebert_environmental_claims = Classification(dataset_name="climatebert/environmental_claims", config_name="climatebert--environmental_claims")

KheemDH_data = Classification(splits=["train", None, None], dataset_name="KheemDH/data", config_name="KheemDH--data")

pacovaldez_stackoverflow_questions_2016 = Classification(dataset_name="pacovaldez/stackoverflow-questions-2016", config_name="pacovaldez--stackoverflow-questions-2016")

zeroshot_twitter_financial_news_topic = Classification(splits=["train", "validation", None], dataset_name="zeroshot/twitter-financial-news-topic", config_name="zeroshot--twitter-financial-news-topic")

copenlu_sufficient_facts___fever = Classification(sentence1="claim", sentence2="evidence", splits=["test", None, None], dataset_name="copenlu/sufficient_facts", config_name="fever")
copenlu_sufficient_facts___hover = Classification(sentence1="claim", sentence2="evidence", splits=["test", None, None], dataset_name="copenlu/sufficient_facts", config_name="hover")
copenlu_sufficient_facts___vitaminc = Classification(sentence1="claim", sentence2="evidence", splits=["test", None, None], dataset_name="copenlu/sufficient_facts", config_name="vitaminc")

demo_org_diabetes = Classification(splits=["train", None, None], dataset_name="demo-org/diabetes", config_name="demo-org--diabetes")

bergr7_weakly_supervised_ag_news = Classification(dataset_name="bergr7/weakly_supervised_ag_news", config_name="bergr7--weakly_supervised_ag_news")

frankier_cross_domain_reviews = Classification(sentence1="text", labels="rating", splits=["train", None, "test"], dataset_name="frankier/cross_domain_reviews")

peixian_rtGender___annotations = Classification(splits=["train", None, None], dataset_name="peixian/rtGender", config_name="annotations")
peixian_rtGender___posts = Classification(splits=[None, None, None], dataset_name="peixian/rtGender", config_name="posts")
peixian_rtGender___responses = Classification(splits=[None, None, None], dataset_name="peixian/rtGender", config_name="responses")

valurank_Adult_content_dataset = Classification(splits=[None, None, None], dataset_name="valurank/Adult-content-dataset")

DeveloperOats_DBPedia_Classes = Classification(dataset_name="DeveloperOats/DBPedia_Classes", config_name="DeveloperOats--DBPedia_Classes")

jakartaresearch_semeval_absa___laptop = Classification(sentence1="text", splits=["train", "validation", None], dataset_name="jakartaresearch/semeval-absa", config_name="laptop")
jakartaresearch_semeval_absa___restaurant = Classification(sentence1="text", splits=["train", "validation", None], dataset_name="jakartaresearch/semeval-absa", config_name="restaurant")

copenlu_citeworth = Classification(dataset_name="copenlu/citeworth", config_name="copenlu--citeworth")

fkdosilovic_docee_event_classification = Classification(splits=["train", None, "test"], dataset_name="fkdosilovic/docee-event-classification", config_name="fkdosilovic--docee-event-classification")

julien_c_reactiongif = Classification(splits=["train", None, None], dataset_name="julien-c/reactiongif", config_name="julien-c--reactiongif")

peixian_equity_evaluation_corpus = Classification(sentence1="sentence", splits=["train", None, None], dataset_name="peixian/equity_evaluation_corpus", config_name="first_domain")

valurank_hate_multi = Classification(sentence1="text", labels="label", splits=["train", None, None], dataset_name="valurank/hate-multi", config_name="valurank--hate-multi")

valurank_news_12factor = Classification(splits=["train", None, None], dataset_name="valurank/news-12factor", config_name="valurank--news-12factor")

valurank_offensive_multi = Classification(sentence1="text", labels="label", splits=["train", None, None], dataset_name="valurank/offensive-multi", config_name="valurank--offensive-multi")

webimmunization_COVID_19_vaccine_attitude_tweets = Classification(splits=["train", None, None], dataset_name="webimmunization/COVID-19-vaccine-attitude-tweets", config_name="webimmunization--COVID-19-vaccine-attitude-tweets")

projecte_aina_gencata = Classification(labels="label", dataset_name="projecte-aina/gencata")

mwong_climate_evidence_related = Classification(sentence1="claim", sentence2="evidence", labels="labels", splits=["train", None, "test"], dataset_name="mwong/climate-evidence-related", config_name="mwong--climate-evidence-related")

mwong_climate_claim_related = Classification(sentence1="claim", sentence2="evidence", labels="labels", splits=["train", None, "test"], dataset_name="mwong/climate-claim-related", config_name="mwong--climate-claim-related")

mwong_climatetext_claim_related_evaluation = Classification(sentence1="claim", sentence2="evidence", labels="labels", splits=["test", None, None], dataset_name="mwong/climatetext-claim-related-evaluation", config_name="mwong--climatetext-claim-related-evaluation")

mwong_climatetext_evidence_related_evaluation = Classification(sentence1="claim", sentence2="evidence", labels="labels", splits=["test", None, None], dataset_name="mwong/climatetext-evidence-related-evaluation", config_name="mwong--climatetext-evidence-related-evaluation")

mwong_climatetext_climate_evidence_claim_related_evaluation = Classification(sentence1="claim", sentence2="evidence", labels="labels", splits=["test", None, None], dataset_name="mwong/climatetext-climate_evidence-claim-related-evaluation", config_name="mwong--climatetext-climate_evidence-claim-related-evaluation")

mwong_climatetext_claim_climate_evidence_related_evaluation = Classification(sentence1="claim", sentence2="evidence", labels="labels", splits=["test", None, None], dataset_name="mwong/climatetext-claim-climate_evidence-related-evaluation", config_name="mwong--climatetext-claim-climate_evidence-related-evaluation")

mwong_climatetext_evidence_claim_pair_related_evaluation = Classification(sentence1="claim", sentence2="evidence", labels="labels", splits=["test", None, None], dataset_name="mwong/climatetext-evidence-claim-pair-related-evaluation", config_name="mwong--climatetext-evidence-claim-pair-related-evaluation")

mwong_climatetext_claim_evidence_pair_related_evaluation = Classification(sentence1="claim", sentence2="evidence", labels="labels", splits=["test", None, None], dataset_name="mwong/climatetext-claim-evidence-pair-related-evaluation", config_name="mwong--climatetext-claim-evidence-pair-related-evaluation")

BritishLibraryLabs_web_archive_classification = Classification(splits=["train", None, None], dataset_name="BritishLibraryLabs/web_archive_classification")

Filippo_osdg_cd = Classification(sentence1="text", labels="label", splits=["train", None, None], dataset_name="Filippo/osdg_cd", config_name="main_config")

pile_of_law_eoir_privacy___all = Classification(sentence1="text", labels="label", splits=["train", "validation", None], dataset_name="pile-of-law/eoir_privacy", config_name="all")
pile_of_law_eoir_privacy___eoir_privacy = Classification(sentence1="text", labels="label", splits=["train", "validation", None], dataset_name="pile-of-law/eoir_privacy", config_name="eoir_privacy")

morteza_cogtext = Classification(splits=["train", None, None], dataset_name="morteza/cogtext", config_name="morteza--cogtext")

florentgbelidji_edmunds_car_ratings = Classification(splits=["train", None, None], dataset_name="florentgbelidji/edmunds-car-ratings", config_name="florentgbelidji--edmunds-car-ratings")

rajistics_auditor_review = Classification(sentence1="sentence", labels="label", splits=["train", None, "test"], dataset_name="rajistics/auditor_review", config_name="rajistics--auditor_review")

fever_feverous = Classification(sentence1="claim", sentence2="evidence", labels="label", dataset_name="fever/feverous")

launch_reddit_qg = Classification(sentence1="question", labels="score", dataset_name="launch/reddit_qg")

#story_cloze = MultipleChoice(splits=[None, "validation", "test"], config_name=["2016"])

winograd_wsc = MultipleChoice(inputs="text", labels="label", splits=["test", None, None], config_name=["wsc285", "wsc273"])

mwsc = MultipleChoice(inputs="sentence")

asnq = MultipleChoice(inputs="question", labels="label", splits=["train", "validation", None])

eraser_multi_rc = MultipleChoice(labels="label")

medmcqa = MultipleChoice(inputs="question")

sileod_movie_recommendation = Classification(sentence1="question", labels="label", splits=["test", None, None], dataset_name="sileod/movie_recommendation")

nightingal3_fig_qa = MultipleChoice(dataset_name="nightingal3/fig-qa", config_name="nightingal3--fig-qa",
    splits=['train','validation',None])


sileod_discourse_marker_qa = Classification(sentence1="context", labels="label", splits=["test", None, None], dataset_name="sileod/discourse_marker_qa")

#md_gender_bias___gendered_words = Classification(splits=["train", None, None])
#md_gender_bias___name_genders = Classification(splits=[None, None, None])
#md_gender_bias___new_data = Classification(sentence1="text", labels="labels", splits=["train", None, None])
#md_gender_bias___funpedia = Classification(sentence1="text")
#md_gender_bias___image_chat = Classification()
#md_gender_bias___wizard = Classification(sentence1="text")
#md_gender_bias___convai2_inferred = Classification(sentence1="text")
#md_gender_bias___light_inferred = Classification(sentence1="text")
#md_gender_bias___opensubtitles_inferred = Classification(sentence1="text")
#md_gender_bias___yelp_inferred = Classification(sentence1="text")


patriziobellan_PET___token_classification = TokenClassification(tokens="tokens", splits=["test", None, None], dataset_name="patriziobellan/PET", config_name="token-classification")
patriziobellan_PET___relations_extraction = TokenClassification(tokens="tokens", labels="ner_tags", splits=["test", None, None], dataset_name="patriziobellan/PET", config_name="relations-extraction")

gap = TokenClassification()

bc2gm_corpus = TokenClassification(tokens="tokens", labels="ner_tags", config_name=["bc2gm_corpus"])

irc_disentangle___ubuntu = TokenClassification()
irc_disentangle___channel_two = TokenClassification(splits=[None, "dev", "test"])

numeric_fused_head___identification = TokenClassification(labels="label", tokens="tokens")
numeric_fused_head___resolution = TokenClassification(tokens="tokens")

tner_wnut2017 = TokenClassification(tokens="tokens", labels="tags", dataset_name="tner/wnut2017", config_name="wnut2017")

linnaeus = TokenClassification(tokens="tokens", labels="ner_tags", config_name=["linnaeus"])

sede = TokenClassification(config_name=["sede"])

tner_fin = TokenClassification(tokens="tokens", labels="tags", dataset_name="tner/fin", config_name="fin")

tner_bc5cdr = TokenClassification(tokens="tokens", labels="tags", dataset_name="tner/bc5cdr", config_name="bc5cdr")

tner_btc = TokenClassification(tokens="tokens", labels="tags", dataset_name="tner/btc", config_name="btc")

tner_mit_restaurant = TokenClassification(tokens="tokens", labels="tags", dataset_name="tner/mit_restaurant", config_name="mit_restaurant")

strombergnlp_broad_twitter_corpus = TokenClassification(tokens="tokens", labels="ner_tags", dataset_name="strombergnlp/broad_twitter_corpus", config_name="broad-twitter-corpus")

tner_bionlp2004 = TokenClassification(tokens="tokens", labels="tags", dataset_name="tner/bionlp2004", config_name="bionlp2004")

tner_mit_movie_trivia = TokenClassification(tokens="tokens", labels="tags", dataset_name="tner/mit_movie_trivia", config_name="mit_movie_trivia")

drAbreu_bc4chemd_ner = TokenClassification(tokens="tokens", labels="ner_tags", dataset_name="drAbreu/bc4chemd_ner", config_name="bc4chemd")

tner_tweebank_ner = TokenClassification(tokens="tokens", labels="tags", dataset_name="tner/tweebank_ner", config_name="tweebank_ner")

beki_privy___small = TokenClassification(tokens="tokens", labels="tags", dataset_name="beki/privy", config_name="small")
beki_privy___large = TokenClassification(tokens="tokens", labels="tags", dataset_name="beki/privy", config_name="large")

DTU54DL_common3k_train = TokenClassification(splits=["train", None, None], dataset_name="DTU54DL/common3k-train", config_name="DTU54DL--common3k-train")

DTU54DL_common_voice_test16k = TokenClassification(splits=["test", None, None], dataset_name="DTU54DL/common-voice-test16k", config_name="DTU54DL--common-voice-test16k")

adsabs_WIESP2022_NER = TokenClassification(tokens="tokens", labels="ner_tags", dataset_name="adsabs/WIESP2022-NER", config_name="fgrezes--WIESP2022-NER")

rungalileo_mit_movies_fixed_connll_format = TokenClassification(splits=["train", None, "test"], dataset_name="rungalileo/mit_movies_fixed_connll_format", config_name="rungalileo--mit_movies_fixed_connll_format")

DTU54DL_common_voice = TokenClassification(splits=["train", None, "test"], dataset_name="DTU54DL/common-voice", config_name="DTU54DL--common-voice")

bgstud_libri_whisper_raw = TokenClassification(dataset_name="bgstud/libri-whisper-raw", config_name="bgstud--libri-whisper-raw")

DTU54DL_common_native_proc = TokenClassification(labels="labels", splits=["train", None, "test"], dataset_name="DTU54DL/common-native-proc", config_name="DTU54DL--common-native-proc")

GateNLP_broad_twitter_corpus = TokenClassification(tokens="tokens", labels="ner_tags", dataset_name="GateNLP/broad_twitter_corpus", config_name="broad-twitter-corpus")

DTU54DL_common_voice_test3k = TokenClassification(splits=["test", None, None], dataset_name="DTU54DL/common-voice-test3k", config_name="DTU54DL--common-voice-test3k")

DTU54DL_common_proc_whisper = TokenClassification(labels="labels", splits=["train", None, "test"], dataset_name="DTU54DL/common-proc-whisper", config_name="DTU54DL--common-proc-whisper")

bgstud_libri = TokenClassification(splits=["test", None, None], dataset_name="bgstud/libri", config_name="bgstud--libri")

DTU54DL_commonvoice_accent_test = TokenClassification(splits=[None, "validation", "test"], dataset_name="DTU54DL/commonvoice_accent_test", config_name="DTU54DL--commonvoice_accent_test")

DFKI_SLT_scidtb = TokenClassification(splits=["train", "dev", "test"], dataset_name="DFKI-SLT/scidtb", config_name="SciDTB")

surrey_nlp_PLOD_filtered = TokenClassification(tokens="tokens", labels="pos_tags", dataset_name="surrey-nlp/PLOD-filtered", config_name="PLODfiltered")

strombergnlp_ipm_nel = TokenClassification(tokens="tokens", labels="ner_tags", splits=["train", None, None], dataset_name="strombergnlp/ipm_nel", config_name="ipm_nel")

ncats_EpiSet4NER_v2 = TokenClassification(tokens="tokens", labels="ner_tags", dataset_name="ncats/EpiSet4NER-v2", config_name="EpiSet4NER")

tner_ttc_dummy = TokenClassification(splits=[None, None, None], dataset_name="tner/ttc_dummy")

havens2_naacl2022 = TokenClassification(tokens="tokens", labels="ner_tags", dataset_name="havens2/naacl2022", config_name="nacl22")

DTU54DL_demo_common_whisper = TokenClassification(labels="labels", splits=["train", None, "test"], dataset_name="DTU54DL/demo-common-whisper", config_name="DTU54DL--demo-common-whisper")

surrey_nlp_PLOD_unfiltered = TokenClassification(tokens="tokens", labels="pos_tags", dataset_name="surrey-nlp/PLOD-unfiltered", config_name="PLODunfiltered")

strombergnlp_twitter_pos_vcb = TokenClassification(tokens="tokens", labels="pos_tags", splits=["train", None, None], dataset_name="strombergnlp/twitter_pos_vcb", config_name="twitter-pos-vcb")

wkrl_cord = TokenClassification(labels="labels", dataset_name="wkrl/cord", config_name="CORD")

arize_ai_xtreme_en = TokenClassification(labels="ner_tags", splits=["training", "validation", None], dataset_name="arize-ai/xtreme_en")