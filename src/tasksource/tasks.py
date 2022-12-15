from .preprocess import cat, get, regen, constant, Classification, TokenClassification, MultipleChoice

# variable name: dataset___config__task

###################### Natural language inference ###################

anli__r1 = Classification('premise','hypothesis','label', splits=['train_r1','dev_r1','test_r1'])
anli__r2 = Classification('premise','hypothesis','label', splits=['train_r2','dev_r2','test_r2'])
anli__r3 = Classification('premise','hypothesis','label', splits=['train_r3','dev_r3','test_r3'])

sick__label         = Classification('sentence_A','sentence_B','label')
sick__relatedness   = Classification('sentence_A','sentence_B','relatedness_score')
sick__entailment_AB = Classification('sentence_A','sentence_B','entailment_AB')
sick__entailment_BA = Classification('sentence_A','sentence_B','entailment_BA')

snli = Classification(sentence1="premise", sentence2="hypothesis", labels="label")

multi_nli = Classification(sentence1="premise", sentence2="hypothesis", labels="label")

wanli = Classification('premise','hypothesis','label', dataset_name="alisawuffles/WANLI")

metaeval___recast = Classification(sentence1="context", sentence2="hypothesis", labels="label", dataset_name="metaeval/recast")

joey234___nan_nli = Classification(dataset_name="joey234/nan-nli", config_name="joey234--nan-nli")


###################### Token Classification #########################

conll2003___pos_tags   = TokenClassification(tokens="tokens", labels='pos_tags')
conll2003___chunk_tags = TokenClassification(tokens="tokens", labels='chunk_tags')
conll2003___ner_tags   = TokenClassification(tokens="tokens", labels='ner_tags')

tner___tweebank_ner    = TokenClassification(tokens="tokens", labels="tags")

######################## Multiple choice ###########################

bigbench = MultipleChoice(
    'inputs',
    choices_list='multiple_choice_targets',
    labels=lambda x:x['multiple_choice_scores'].index(1))

cos_e = MultipleChoice('question',
    choices_list='choices',
    labels= lambda x: x['choices'].index(x['answer']))

cosmos_qa = MultipleChoice(cat('context','question'),regen('answer[0-3]'),'labels')

dream = MultipleChoice(
    lambda x:"\n".join(x['dialogue']+[x['question']]),
    choices_list='choice',
    labels=lambda x:x['choice'].index['answer']
)

openbookqa = MultipleChoice(
    'question_stem ',
    choices_list=get.choices.text,
    labels='label'
)

qasc = MultipleChoice(
    'question',
    choices_list=get.choices.text,
    labels='answerKey'
)

quartz=qasc

quail = MultipleChoice(
    cat('context','question'),
    choices_list='answers',
    labels='correct_answer_id' 
)

race= MultipleChoice(
    'question',
    choices_list='choices',
    labels='answer')

sciq = MultipleChoice(
    'question',
    ['correct_answer']+regen('distractor[1-3]'),
    labels=constant(0)
    )

social_i_qa = MultipleChoice(
    'question',
    ['answerA','answerB'],
        'label '
    )

wiki_hop = MultipleChoice(
    'question', 
    choices_list='candidates',
    labels=lambda x:x['candidates'].index('answer')
    )

wiqa = MultipleChoice('question_stem',
    choices_list = lambda x: x['choices']['text'],
    labels='answer_label_as_choice')

piqa = MultipleChoice('goal',['sol1','sol2'], 'label')

hellaswag = MultipleChoice('ctx_a',
                           choices_list=cat(['ctx_b','endings']),
                           labels='label')

super_glue___copa = MultipleChoice('premise',['choice1','choice2'],'label')

art = MultipleChoice(cat('hypothesis_1','hypothesis_2'),
    ['observation_1','observation_2'],
    labels='label')

blimp = MultipleChoice(inputs=constant(''),
    choices=['sentence_good','sentence_bad'],
    labels=constant(0))

hendrycks_test = MultipleChoice('question',labels='answer',choices_list='choices')

winogrande = MultipleChoice('sentence1',['option1','option2'],'answer')

quora = Classification(get.text[0], get.text[1], 'is_duplicate')

medical_questions_pairs = Classification("question_1","question_2", labels="label")

#codah, ai2_arc 

# definite_pronoun_resolution = TokenClassification(sentence1="sentence", labels="label")

swag=MultipleChoice(cat("sent1","sent2"),regen("ending[0-3]"),"label")


trec = Classification(sentence1="text", labels="fine_label")


###################### Automatically generated (verified)############


###################### Automatically generated ##################### 


glue___cola = Classification(sentence1="sentence", labels="label")
glue___sst2 = Classification(sentence1="sentence", labels="label")
glue___mrpc = Classification(sentence1="sentence1", sentence2="sentence2", labels="label")
glue___qqp = Classification(sentence1="question1", sentence2="question2", labels="label")
glue___stsb = Classification(sentence1="sentence1", sentence2="sentence2", labels="label")
glue___mnli = Classification(sentence1="premise", sentence2="hypothesis", labels="label")
glue___mnli_mismatched = Classification(sentence1="premise", sentence2="hypothesis", labels="label")
glue___mnli_matched = Classification(sentence1="premise", sentence2="hypothesis", labels="label")
glue___qnli = Classification(sentence1="question", labels="label")
glue___rte = Classification(sentence1="sentence1", sentence2="sentence2", labels="label")
glue___wnli = Classification(sentence1="sentence1", sentence2="sentence2", labels="label")
glue___ax = Classification(sentence1="premise", sentence2="hypothesis", labels="label")

tweet_eval = Classification(sentence1="text", labels="label")

imdb = Classification(sentence1="text", labels="label")

rotten_tomatoes = Classification(sentence1="text", labels="label")

ag_news = Classification(sentence1="text", labels="label")

yelp_review_full = Classification(sentence1="text", labels="label")

financial_phrasebank = Classification(sentence1="sentence", labels="label")

poem_sentiment = Classification(sentence1="verse_text", labels="label")

paws = Classification(sentence1="sentence1", sentence2="sentence2", labels="label")

emotion = Classification(sentence1="text", labels="label")

dbpedia_14 = Classification(sentence1="content", labels="label")

amazon_polarity = Classification(sentence1="content", labels="label")

snli = Classification(sentence1="premise", sentence2="hypothesis", labels="label")

app_reviews = Classification(labels="star")

swag___regular = Classification(labels="label")
swag___full = Classification()

hans = Classification(sentence1="premise", sentence2="hypothesis", labels="label")

multi_nli = Classification(sentence1="premise", sentence2="hypothesis", labels="label")

hate_speech18 = Classification(sentence1="text", labels="label")

lex_glue___ecthr_a = Classification(sentence1="text", labels="labels")
lex_glue___ecthr_b = Classification(sentence1="text", labels="labels")
lex_glue___eurlex = Classification(sentence1="text", labels="labels")
lex_glue___scotus = Classification(sentence1="text", labels="label")
lex_glue___ledgar = Classification(sentence1="text", labels="label")
lex_glue___unfair_tos = Classification(sentence1="text", labels="labels")
lex_glue___case_hold = Classification(sentence1="context", labels="label")

sms_spam = Classification(sentence1="sms", labels="label")

fever___v1_0 = Classification(labels="label", dataset_name="fever", config_name="v1.0")
fever___v2_0 = Classification(labels="label", dataset_name="fever", config_name="v2.0")
fever___wiki_pages = Classification(sentence1="text")

humicroedit___subtask_1 = Classification(labels="meanGrade", dataset_name="humicroedit", config_name="subtask-1")
humicroedit___subtask_2 = Classification(labels="label", dataset_name="humicroedit", config_name="subtask-2")

snips_built_in_intents = Classification(sentence1="text", labels="label")

banking77 = Classification(sentence1="text", labels="label")

hate_speech_offensive = Classification(sentence1="tweet", labels="class")

yahoo_answers_topics = Classification()

hyperpartisan_news_detection = Classification(sentence1="text", labels="hyperpartisan")

health_fact = Classification(labels="label")

ethos___binary = Classification(sentence1="text", labels="label")
ethos___multilabel = Classification(sentence1="text")

medical_questions_pairs = Classification(labels="label")

pragmeval___verifiability = Classification(sentence1="sentence", labels="label")
pragmeval___emobank_arousal = Classification(sentence1="sentence", labels="label", dataset_name="pragmeval", config_name="emobank-arousal")
pragmeval___switchboard = Classification(sentence1="sentence", labels="label")
pragmeval___persuasiveness_eloquence = Classification(sentence1="sentence1", sentence2="sentence2", labels="label", dataset_name="pragmeval", config_name="persuasiveness-eloquence")
pragmeval___mrda = Classification(sentence1="sentence", labels="label")
pragmeval___gum = Classification(sentence1="sentence1", sentence2="sentence2", labels="label")
pragmeval___emergent = Classification(sentence1="sentence1", sentence2="sentence2", labels="label")
pragmeval___persuasiveness_relevance = Classification(sentence1="sentence1", sentence2="sentence2", labels="label", dataset_name="pragmeval", config_name="persuasiveness-relevance")
pragmeval___persuasiveness_specificity = Classification(sentence1="sentence1", sentence2="sentence2", labels="label", dataset_name="pragmeval", config_name="persuasiveness-specificity")
pragmeval___persuasiveness_strength = Classification(sentence1="sentence1", sentence2="sentence2", labels="label", dataset_name="pragmeval", config_name="persuasiveness-strength")
pragmeval___emobank_dominance = Classification(sentence1="sentence", labels="label", dataset_name="pragmeval", config_name="emobank-dominance")
pragmeval___squinky_implicature = Classification(sentence1="sentence", labels="label", dataset_name="pragmeval", config_name="squinky-implicature")
pragmeval___sarcasm = Classification(sentence1="sentence1", sentence2="sentence2", labels="label")
pragmeval___squinky_formality = Classification(sentence1="sentence", labels="label", dataset_name="pragmeval", config_name="squinky-formality")
pragmeval___stac = Classification(sentence1="sentence1", sentence2="sentence2", labels="label")
pragmeval___pdtb = Classification(sentence1="sentence1", sentence2="sentence2", labels="label")
pragmeval___persuasiveness_premisetype = Classification(sentence1="sentence1", sentence2="sentence2", labels="label", dataset_name="pragmeval", config_name="persuasiveness-premisetype")
pragmeval___squinky_informativeness = Classification(sentence1="sentence", labels="label", dataset_name="pragmeval", config_name="squinky-informativeness")
pragmeval___persuasiveness_claimtype = Classification(sentence1="sentence1", sentence2="sentence2", labels="label", dataset_name="pragmeval", config_name="persuasiveness-claimtype")
pragmeval___emobank_valence = Classification(sentence1="sentence", labels="label", dataset_name="pragmeval", config_name="emobank-valence")

daily_dialog = Classification()

crows_pairs = Classification()

go_emotions___raw = Classification(sentence1="text")
go_emotions___simplified = Classification(sentence1="text", labels="labels")

boolq = Classification(sentence1="question")

movie_rationales = Classification(labels="label")

discovery = Classification(sentence1="sentence1", sentence2="sentence2", labels="label")

ecthr_cases___alleged_violation_prediction = Classification(labels="labels", dataset_name="ecthr_cases", config_name="alleged-violation-prediction")
ecthr_cases___violation_prediction = Classification(labels="labels", dataset_name="ecthr_cases", config_name="violation-prediction")

scicite = Classification(sentence1="string", labels="label")

tab_fact = Classification(labels="label")

tab_fact___blind_test = Classification()

liar = Classification(sentence1="context", labels="label")

biosses = Classification(sentence1="sentence1", sentence2="sentence2", labels="score")

sem_eval_2014_task_1 = Classification(sentence1="premise", sentence2="hypothesis")

gutenberg_time = Classification()

hlgd = Classification(labels="label")

clinc_oos = Classification(sentence1="text")

circa = Classification(sentence1="context")

nlu_evaluation_data = Classification(sentence1="text", labels="label")

newspop = Classification()

relbert_lexical_relation_classification___BLESS = Classification(sentence1="head", sentence2="tail", labels="relation", dataset_name="relbert/lexical_relation_classification", config_name="BLESS")
relbert_lexical_relation_classification___CogALexV = Classification(sentence1="head", sentence2="tail", labels="relation", dataset_name="relbert/lexical_relation_classification", config_name="CogALexV")
relbert_lexical_relation_classification___EVALution = Classification(sentence1="head", sentence2="tail", labels="relation", dataset_name="relbert/lexical_relation_classification", config_name="EVALution")
relbert_lexical_relation_classification___K_H_N = Classification(sentence1="head", sentence2="tail", labels="relation", dataset_name="relbert/lexical_relation_classification", config_name="K&H+N")
relbert_lexical_relation_classification___ROOT09 = Classification(sentence1="head", sentence2="tail", labels="relation", dataset_name="relbert/lexical_relation_classification", config_name="ROOT09")

emo = Classification(sentence1="text", labels="label")

md_gender_bias___gendered_words = Classification()
md_gender_bias___name_genders = Classification()
md_gender_bias___new_data = Classification(sentence1="text", labels="labels")
md_gender_bias___funpedia = Classification(sentence1="text")
md_gender_bias___image_chat = Classification()
md_gender_bias___wizard = Classification(sentence1="text")
md_gender_bias___convai2_inferred = Classification(sentence1="text")
md_gender_bias___light_inferred = Classification(sentence1="text")
md_gender_bias___opensubtitles_inferred = Classification(sentence1="text")
md_gender_bias___yelp_inferred = Classification(sentence1="text")

google_wellformed_query = Classification(sentence1="content", labels="rating")

tweets_hate_speech_detection = Classification(sentence1="tweet", labels="label")

hatexplain = Classification()

bing_coronavirus_query_set = Classification()

stereoset = Classification(sentence1="context")

swda = Classification(sentence1="text")

adv_glue___adv_sst2 = Classification(sentence1="sentence", labels="label")
adv_glue___adv_qqp = Classification(sentence1="question1", sentence2="question2", labels="label")
adv_glue___adv_mnli = Classification(sentence1="premise", sentence2="hypothesis", labels="label")
adv_glue___adv_mnli_mismatched = Classification(sentence1="premise", sentence2="hypothesis", labels="label")
adv_glue___adv_qnli = Classification(sentence1="question", labels="label")
adv_glue___adv_rte = Classification(sentence1="sentence1", sentence2="sentence2", labels="label")

conceptnet5 = Classification(sentence1="sentence")

hope_edi = Classification(sentence1="text", labels="label")

ucberkeley_dlab_measuring_hate_speech = Classification(dataset_name="ucberkeley-dlab/measuring-hate-speech", config_name="ucberkeley-dlab--measuring-hate-speech")

eurlex = Classification(sentence1="text")

fhamborg_news_sentiment_newsmtsc___mt = Classification(sentence1="sentence", dataset_name="fhamborg/news_sentiment_newsmtsc", config_name="mt")
fhamborg_news_sentiment_newsmtsc___rw = Classification(sentence1="sentence", dataset_name="fhamborg/news_sentiment_newsmtsc", config_name="rw")

prachathai67k = Classification()

cardiffnlp_tweet_topic_multi = Classification(sentence1="text", labels="label", dataset_name="cardiffnlp/tweet_topic_multi", config_name="tweet_topic_multi")

datacommons_factcheck = Classification(labels="review_rating")

scifact = Classification()

coastalcph_fairlex___ecthr = Classification(sentence1="text", labels="labels", dataset_name="coastalcph/fairlex", config_name="ecthr")
coastalcph_fairlex___scotus = Classification(sentence1="text", labels="label", dataset_name="coastalcph/fairlex", config_name="scotus")
coastalcph_fairlex___fscs = Classification(sentence1="text", labels="label", dataset_name="coastalcph/fairlex", config_name="fscs")
coastalcph_fairlex___cail = Classification(sentence1="text", labels="label", dataset_name="coastalcph/fairlex", config_name="cail")

peer_read = Classification()

metaeval_linguisticprobing___subj_number = Classification(sentence1="sentence", labels="label", dataset_name="metaeval/linguisticprobing", config_name="subj_number")
metaeval_linguisticprobing___word_content = Classification(sentence1="sentence", labels="label", dataset_name="metaeval/linguisticprobing", config_name="word_content")
metaeval_linguisticprobing___obj_number = Classification(sentence1="sentence", labels="label", dataset_name="metaeval/linguisticprobing", config_name="obj_number")
metaeval_linguisticprobing___past_present = Classification(sentence1="sentence", labels="label", dataset_name="metaeval/linguisticprobing", config_name="past_present")
metaeval_linguisticprobing___sentence_length = Classification(sentence1="sentence", labels="label", dataset_name="metaeval/linguisticprobing", config_name="sentence_length")
metaeval_linguisticprobing___top_constituents = Classification(sentence1="sentence", labels="label", dataset_name="metaeval/linguisticprobing", config_name="top_constituents")
metaeval_linguisticprobing___tree_depth = Classification(sentence1="sentence", labels="label", dataset_name="metaeval/linguisticprobing", config_name="tree_depth")
metaeval_linguisticprobing___coordination_inversion = Classification(sentence1="sentence", labels="label", dataset_name="metaeval/linguisticprobing", config_name="coordination_inversion")
metaeval_linguisticprobing___odd_man_out = Classification(sentence1="sentence", labels="label", dataset_name="metaeval/linguisticprobing", config_name="odd_man_out")
metaeval_linguisticprobing___bigram_shift = Classification(sentence1="sentence", labels="label", dataset_name="metaeval/linguisticprobing", config_name="bigram_shift")

jigsaw_unintended_bias = Classification(labels="rating")

per_sent = Classification()

jigsaw_toxicity_pred = Classification()

diplomacy_detection = Classification()

demo_org_auditor_review = Classification(sentence1="sentence", labels="label", dataset_name="demo-org/auditor_review", config_name="demo-org--auditor_review")

Abirate_english_quotes = Classification(dataset_name="Abirate/english_quotes", config_name="Abirate--english_quotes")

sled_umich_TRIP = Classification(labels="label", dataset_name="sled-umich/TRIP")

GonzaloA_fake_news = Classification(sentence1="text", labels="label", dataset_name="GonzaloA/fake_news", config_name="GonzaloA--fake_news")

consumer_finance_complaints = Classification(dataset_name="consumer-finance-complaints")

ohsumed = Classification()

hate_offensive = Classification(sentence1="tweet", labels="label")

fake_news_english = Classification()

blog_authorship_corpus = Classification(sentence1="text")

coarse_discourse = Classification()

Paul_hatecheck = Classification(dataset_name="Paul/hatecheck", config_name="Paul--hatecheck")

hippocorpus = Classification()

has_part = Classification(labels="score")

multi_nli_mismatch = Classification(sentence1="premise", sentence2="hypothesis", labels="label")

time_dial = Classification()

NbAiLab_norec_agg = Classification(sentence1="text", labels="label", dataset_name="NbAiLab/norec_agg")

DanL_scientific_challenges_and_directions_dataset = Classification(sentence1="text", labels="label", dataset_name="DanL/scientific-challenges-and-directions-dataset", config_name="DanL--scientific-challenges-and-directions-dataset")

CyranoB_polarity = Classification(sentence1="content", labels="label", dataset_name="CyranoB/polarity", config_name="CyranoB--polarity")

bhavnicksm_sentihood = Classification(dataset_name="bhavnicksm/sentihood", config_name="bhavnicksm--sentihood")

cardiffnlp_tweet_topic_single = Classification(sentence1="text", labels="label", dataset_name="cardiffnlp/tweet_topic_single", config_name="tweet_topic_single")

metaeval_recast___recast_kg_relations = Classification(sentence1="context", sentence2="hypothesis", labels="label", dataset_name="metaeval/recast", config_name="recast_kg_relations")
metaeval_recast___recast_puns = Classification(sentence1="context", sentence2="hypothesis", labels="label", dataset_name="metaeval/recast", config_name="recast_puns")
metaeval_recast___recast_factuality = Classification(sentence1="context", sentence2="hypothesis", labels="label", dataset_name="metaeval/recast", config_name="recast_factuality")
metaeval_recast___recast_verbnet = Classification(sentence1="context", sentence2="hypothesis", labels="label", dataset_name="metaeval/recast", config_name="recast_verbnet")
metaeval_recast___recast_verbcorner = Classification(sentence1="context", sentence2="hypothesis", labels="label", dataset_name="metaeval/recast", config_name="recast_verbcorner")
metaeval_recast___recast_ner = Classification(sentence1="context", sentence2="hypothesis", labels="label", dataset_name="metaeval/recast", config_name="recast_ner")
metaeval_recast___recast_sentiment = Classification(sentence1="context", sentence2="hypothesis", labels="label", dataset_name="metaeval/recast", config_name="recast_sentiment")
metaeval_recast___recast_megaveridicality = Classification(sentence1="context", sentence2="hypothesis", labels="label", dataset_name="metaeval/recast", config_name="recast_megaveridicality")

OxAISH_AL_LLM_wiki_toxic = Classification(labels="label", dataset_name="OxAISH-AL-LLM/wiki_toxic")

carblacac_twitter_sentiment_analysis = Classification(sentence1="text", dataset_name="carblacac/twitter-sentiment-analysis")

metaeval_crowdflower___sentiment_nuclear_power = Classification(sentence1="text", labels="label", dataset_name="metaeval/crowdflower", config_name="sentiment_nuclear_power")
metaeval_crowdflower___tweet_global_warming = Classification(sentence1="text", labels="label", dataset_name="metaeval/crowdflower", config_name="tweet_global_warming")
metaeval_crowdflower___airline_sentiment = Classification(sentence1="text", labels="label", dataset_name="metaeval/crowdflower", config_name="airline-sentiment")
metaeval_crowdflower___corporate_messaging = Classification(sentence1="text", labels="label", dataset_name="metaeval/crowdflower", config_name="corporate-messaging")
metaeval_crowdflower___economic_news = Classification(sentence1="text", labels="label", dataset_name="metaeval/crowdflower", config_name="economic-news")
metaeval_crowdflower___political_media_audience = Classification(sentence1="text", labels="label", dataset_name="metaeval/crowdflower", config_name="political-media-audience")
metaeval_crowdflower___political_media_bias = Classification(sentence1="text", labels="label", dataset_name="metaeval/crowdflower", config_name="political-media-bias")
metaeval_crowdflower___political_media_message = Classification(sentence1="text", labels="label", dataset_name="metaeval/crowdflower", config_name="political-media-message")
metaeval_crowdflower___text_emotion = Classification(sentence1="text", labels="label", dataset_name="metaeval/crowdflower", config_name="text_emotion")

metaeval_ethics___commonsense = Classification(sentence1="text", labels="label", dataset_name="metaeval/ethics", config_name="commonsense")
metaeval_ethics___deontology = Classification(sentence1="text", labels="label", dataset_name="metaeval/ethics", config_name="deontology")
metaeval_ethics___justice = Classification(sentence1="text", labels="label", dataset_name="metaeval/ethics", config_name="justice")
metaeval_ethics___utilitarianism = Classification(sentence1="text", labels="label", dataset_name="metaeval/ethics", config_name="utilitarianism")
metaeval_ethics___virtue = Classification(sentence1="sentence1", sentence2="sentence2", labels="label", dataset_name="metaeval/ethics", config_name="virtue")

copenlu_scientific_exaggeration_detection = Classification(dataset_name="copenlu/scientific-exaggeration-detection", config_name="copenlu--scientific-exaggeration-detection")

bdotloh_empathetic_dialogues_contexts = Classification(dataset_name="bdotloh/empathetic-dialogues-contexts", config_name="bdotloh--empathetic-dialogues-contexts")

zeroshot_twitter_financial_news_sentiment = Classification(dataset_name="zeroshot/twitter-financial-news-sentiment", config_name="zeroshot--twitter-financial-news-sentiment")

tals_vitaminc = Classification(dataset_name="tals/vitaminc", config_name="tals--vitaminc")

pacovaldez_stackoverflow_questions = Classification(dataset_name="pacovaldez/stackoverflow-questions", config_name="pacovaldez--stackoverflow-questions")

sagnikrayc_snli_cf_kaushik = Classification(sentence1="premise", sentence2="hypothesis", labels="label", dataset_name="sagnikrayc/snli-cf-kaushik", config_name="plain_text")

PolyAI_banking77 = Classification(sentence1="text", labels="label", dataset_name="PolyAI/banking77")

FinanceInc_auditor_sentiment = Classification(sentence1="sentence", labels="label", dataset_name="FinanceInc/auditor_sentiment", config_name="demo-org--auditor_review")

Tidrael_tsl_news = Classification(labels="label", dataset_name="Tidrael/tsl_news", config_name="plain_text")

okite97_news_data = Classification(dataset_name="okite97/news-data", config_name="okite97--news-data")

mwong_fever_claim_related = Classification(labels="labels", dataset_name="mwong/fever-claim-related", config_name="mwong--fever-claim-related")

joey234_nan_nli = Classification(dataset_name="joey234/nan-nli", config_name="joey234--nan-nli")

llangnickel_long_covid_classification_data = Classification(dataset_name="llangnickel/long-covid-classification-data", config_name="llangnickel--long-covid-classification-data")

rungalileo_20_Newsgroups_Fixed = Classification(dataset_name="rungalileo/20_Newsgroups_Fixed", config_name="rungalileo--20_Newsgroups_Fixed")

merve_poetry = Classification(dataset_name="merve/poetry", config_name="merve--poetry")

DFKI_SLT_tacred___original = Classification(labels="relation", dataset_name="DFKI-SLT/tacred", config_name="original")
DFKI_SLT_tacred___revised = Classification(labels="relation", dataset_name="DFKI-SLT/tacred", config_name="revised")

valurank_News_Articles_Categorization = Classification(dataset_name="valurank/News_Articles_Categorization", config_name="valurank--News_Articles_Categorization")

arize_ai_ecommerce_reviews_with_language_drift = Classification(sentence1="text", labels="label", dataset_name="arize-ai/ecommerce_reviews_with_language_drift")

copenlu_fever_gold_evidence = Classification(dataset_name="copenlu/fever_gold_evidence", config_name="copenlu--fever_gold_evidence")

qanastek_Biosses_BLUE = Classification(sentence1="sentence1", sentence2="sentence2", labels="score", dataset_name="qanastek/Biosses-BLUE", config_name="biosses")

arize_ai_movie_reviews_with_context_drift = Classification(dataset_name="arize-ai/movie_reviews_with_context_drift", config_name="arize-ai--movie_reviews_with_context_drift")

launch_ampere = Classification(dataset_name="launch/ampere", config_name="launch--ampere")

jpwahle_etpc = Classification(dataset_name="jpwahle/etpc", config_name="nan")

climatebert_environmental_claims = Classification(dataset_name="climatebert/environmental_claims", config_name="climatebert--environmental_claims")

KheemDH_data = Classification(dataset_name="KheemDH/data", config_name="KheemDH--data")

mwong_fever_evidence_related = Classification(labels="labels", dataset_name="mwong/fever-evidence-related", config_name="mwong--fever-related")

pacovaldez_stackoverflow_questions_2016 = Classification(dataset_name="pacovaldez/stackoverflow-questions-2016", config_name="pacovaldez--stackoverflow-questions-2016")

zeroshot_twitter_financial_news_topic = Classification(dataset_name="zeroshot/twitter-financial-news-topic", config_name="zeroshot--twitter-financial-news-topic")

copenlu_sufficient_facts___fever = Classification(dataset_name="copenlu/sufficient_facts", config_name="fever")
copenlu_sufficient_facts___hover = Classification(dataset_name="copenlu/sufficient_facts", config_name="hover")
copenlu_sufficient_facts___vitaminc = Classification(dataset_name="copenlu/sufficient_facts", config_name="vitaminc")

strombergnlp_rumoureval_2019 = Classification(labels="label", dataset_name="strombergnlp/rumoureval_2019", config_name="RumourEval2019")

demo_org_diabetes = Classification(dataset_name="demo-org/diabetes", config_name="demo-org--diabetes")

bergr7_weakly_supervised_ag_news = Classification(dataset_name="bergr7/weakly_supervised_ag_news", config_name="bergr7--weakly_supervised_ag_news")

frankier_cross_domain_reviews = Classification(sentence1="text", labels="rating", dataset_name="frankier/cross_domain_reviews")

peixian_rtGender___annotations = Classification(dataset_name="peixian/rtGender", config_name="annotations")
peixian_rtGender___posts = Classification(dataset_name="peixian/rtGender", config_name="posts")
peixian_rtGender___responses = Classification(dataset_name="peixian/rtGender", config_name="responses")

valurank_Adult_content_dataset = Classification(dataset_name="valurank/Adult-content-dataset", config_name="nan")

launch_open_question_type = Classification(sentence1="question", dataset_name="launch/open_question_type")

DeveloperOats_DBPedia_Classes = Classification(dataset_name="DeveloperOats/DBPedia_Classes", config_name="DeveloperOats--DBPedia_Classes")

jakartaresearch_semeval_absa___laptop = Classification(sentence1="text", dataset_name="jakartaresearch/semeval-absa", config_name="laptop")
jakartaresearch_semeval_absa___restaurant = Classification(sentence1="text", dataset_name="jakartaresearch/semeval-absa", config_name="restaurant")

copenlu_citeworth = Classification(dataset_name="copenlu/citeworth", config_name="copenlu--citeworth")

fkdosilovic_docee_event_classification = Classification(dataset_name="fkdosilovic/docee-event-classification", config_name="fkdosilovic--docee-event-classification")

julien_c_reactiongif = Classification(dataset_name="julien-c/reactiongif", config_name="julien-c--reactiongif")

peixian_equity_evaluation_corpus = Classification(sentence1="sentence", dataset_name="peixian/equity_evaluation_corpus", config_name="first_domain")

valurank_hate_multi = Classification(sentence1="text", labels="label", dataset_name="valurank/hate-multi", config_name="valurank--hate-multi")

valurank_news_12factor = Classification(dataset_name="valurank/news-12factor", config_name="valurank--news-12factor")

valurank_offensive_multi = Classification(sentence1="text", labels="label", dataset_name="valurank/offensive-multi", config_name="valurank--offensive-multi")

webimmunization_COVID_19_vaccine_attitude_tweets = Classification(dataset_name="webimmunization/COVID-19-vaccine-attitude-tweets", config_name="webimmunization--COVID-19-vaccine-attitude-tweets")

projecte_aina_gencata = Classification(labels="label", dataset_name="projecte-aina/gencata")

mwong_climate_evidence_related = Classification(labels="labels", dataset_name="mwong/climate-evidence-related", config_name="mwong--climate-evidence-related")

mwong_climate_claim_related = Classification(labels="labels", dataset_name="mwong/climate-claim-related", config_name="mwong--climate-claim-related")

mwong_climatetext_claim_related_evaluation = Classification(labels="labels", dataset_name="mwong/climatetext-claim-related-evaluation", config_name="mwong--climatetext-claim-related-evaluation")

mwong_climatetext_evidence_related_evaluation = Classification(labels="labels", dataset_name="mwong/climatetext-evidence-related-evaluation", config_name="mwong--climatetext-evidence-related-evaluation")

mwong_climatetext_climate_evidence_claim_related_evaluation = Classification(labels="labels", dataset_name="mwong/climatetext-climate_evidence-claim-related-evaluation", config_name="mwong--climatetext-climate_evidence-claim-related-evaluation")

mwong_climatetext_claim_climate_evidence_related_evaluation = Classification(labels="labels", dataset_name="mwong/climatetext-claim-climate_evidence-related-evaluation", config_name="mwong--climatetext-claim-climate_evidence-related-evaluation")

mwong_climatetext_evidence_claim_pair_related_evaluation = Classification(labels="labels", dataset_name="mwong/climatetext-evidence-claim-pair-related-evaluation", config_name="mwong--climatetext-evidence-claim-pair-related-evaluation")

mwong_climatetext_claim_evidence_pair_related_evaluation = Classification(labels="labels", dataset_name="mwong/climatetext-claim-evidence-pair-related-evaluation", config_name="mwong--climatetext-claim-evidence-pair-related-evaluation")

BritishLibraryLabs_web_archive_classification = Classification(dataset_name="BritishLibraryLabs/web_archive_classification")

Filippo_osdg_cd = Classification(sentence1="text", labels="label", dataset_name="Filippo/osdg_cd", config_name="main_config")

pile_of_law_eoir_privacy___all = Classification(sentence1="text", labels="label", dataset_name="pile-of-law/eoir_privacy", config_name="all")
pile_of_law_eoir_privacy___eoir_privacy = Classification(sentence1="text", labels="label", dataset_name="pile-of-law/eoir_privacy", config_name="eoir_privacy")

morteza_cogtext = Classification(dataset_name="morteza/cogtext", config_name="morteza--cogtext")

florentgbelidji_edmunds_car_ratings = Classification(dataset_name="florentgbelidji/edmunds-car-ratings", config_name="florentgbelidji--edmunds-car-ratings")

rajistics_auditor_review = Classification(sentence1="sentence", labels="label", dataset_name="rajistics/auditor_review", config_name="rajistics--auditor_review")

fever_feverous = Classification(labels="label", dataset_name="fever/feverous")

launch_reddit_qg = Classification(sentence1="question", labels="score", dataset_name="launch/reddit_qg")

story_cloze = MultipleChoice()

winograd_wsc = MultipleChoice(inputs="text", labels="label")

mwsc = MultipleChoice(inputs="sentence")

asnq = MultipleChoice(inputs="question", labels="label")

eraser_multi_rc = MultipleChoice(labels="label")

medmcqa = MultipleChoice(inputs="question")

sileod_movie_recommendation = Classification(sentence1="question", labels="label", dataset_name="sileod/movie_recommendation")

nightingal3_fig_qa = MultipleChoice(dataset_name="nightingal3/fig-qa", config_name="nightingal3--fig-qa")

sileod_wep_probes___reasoning_1hop = Classification(sentence1="context", sentence2="hypothesis", labels="label", dataset_name="sileod/wep-probes", config_name="reasoning_1hop")
sileod_wep_probes___reasoning_2hop = Classification(sentence1="context", sentence2="hypothesis", labels="label", dataset_name="sileod/wep-probes", config_name="reasoning_2hop")
sileod_wep_probes___usnli = Classification(sentence1="context", sentence2="hypothesis", labels="label", dataset_name="sileod/wep-probes", config_name="usnli")

sileod_discourse_marker_qa = Classification(sentence1="context", labels="label", dataset_name="sileod/discourse_marker_qa")

wnut_17 = TokenClassification(tokens="tokens", labels="ner_tags")

ncbi_disease = TokenClassification(tokens="tokens", labels="ner_tags")

acronym_identification = TokenClassification(labels="labels", tokens="tokens")

conllpp = TokenClassification(tokens="tokens", labels="pos_tags")

jnlpba = TokenClassification(tokens="tokens", labels="ner_tags")

species_800 = TokenClassification(tokens="tokens", labels="ner_tags")

patriziobellan_PET___token_classification = TokenClassification(tokens="tokens", dataset_name="patriziobellan/PET", config_name="token-classification")
patriziobellan_PET___relations_extraction = TokenClassification(tokens="tokens", labels="ner_tags", dataset_name="patriziobellan/PET", config_name="relations-extraction")

tner_tweetner7 = TokenClassification(tokens="tokens", labels="tags", dataset_name="tner/tweetner7", config_name="tweetner7")

tner_ontonotes5 = TokenClassification(tokens="tokens", labels="tags", dataset_name="tner/ontonotes5", config_name="ontonotes5")

gap = TokenClassification()

bc2gm_corpus = TokenClassification(tokens="tokens", labels="ner_tags")

irc_disentangle = TokenClassification()

# TODO numeric_fused_head___identification = TokenClassification(labels="label", tokens="tokens")
# TODO numeric_fused_head___resolution = TokenClassification(sentence1="head", tokens="tokens")

tner_wnut2017 = TokenClassification(tokens="tokens", labels="tags", dataset_name="tner/wnut2017", config_name="wnut2017")

linnaeus = TokenClassification(tokens="tokens", labels="ner_tags")

SpeedOfMagic_ontonotes_english = TokenClassification(tokens="tokens", labels="ner_tags", dataset_name="SpeedOfMagic/ontonotes_english", config_name="SpeedOfMagic--ontonotes_english")

sede = TokenClassification()

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

strombergnlp_twitter_pos___foster = TokenClassification(tokens="tokens", labels="pos_tags", dataset_name="strombergnlp/twitter_pos", config_name="foster")
strombergnlp_twitter_pos___ritter = TokenClassification(tokens="tokens", labels="pos_tags", dataset_name="strombergnlp/twitter_pos", config_name="ritter")

adsabs_WIESP2022_NER = TokenClassification(tokens="tokens", labels="ner_tags", dataset_name="adsabs/WIESP2022-NER", config_name="fgrezes--WIESP2022-NER")

rungalileo_mit_movies_fixed_connll_format = TokenClassification(dataset_name="rungalileo/mit_movies_fixed_connll_format", config_name="rungalileo--mit_movies_fixed_connll_format")

GateNLP_broad_twitter_corpus = TokenClassification(tokens="tokens", labels="ner_tags", dataset_name="GateNLP/broad_twitter_corpus", config_name="broad-twitter-corpus")

DFKI_SLT_scidtb = TokenClassification(dataset_name="DFKI-SLT/scidtb", config_name="SciDTB")

surrey_nlp_PLOD_filtered = TokenClassification(tokens="tokens", labels="pos_tags", dataset_name="surrey-nlp/PLOD-filtered", config_name="PLODfiltered")

strombergnlp_ipm_nel = TokenClassification(tokens="tokens", labels="ner_tags", dataset_name="strombergnlp/ipm_nel", config_name="ipm_nel")

ncats_EpiSet4NER_v2 = TokenClassification(tokens="tokens", labels="ner_tags", dataset_name="ncats/EpiSet4NER-v2", config_name="EpiSet4NER")

tner_ttc_dummy = TokenClassification(dataset_name="tner/ttc_dummy", config_name="nan")

havens2_naacl2022 = TokenClassification(tokens="tokens", labels="ner_tags", dataset_name="havens2/naacl2022", config_name="nacl22")

DTU54DL_demo_common_whisper = TokenClassification(labels="labels", dataset_name="DTU54DL/demo-common-whisper", config_name="DTU54DL--demo-common-whisper")

surrey_nlp_PLOD_unfiltered = TokenClassification(tokens="tokens", labels="pos_tags", dataset_name="surrey-nlp/PLOD-unfiltered", config_name="PLODunfiltered")

strombergnlp_twitter_pos_vcb = TokenClassification(tokens="tokens", labels="pos_tags", dataset_name="strombergnlp/twitter_pos_vcb", config_name="twitter-pos-vcb")

strombergnlp_named_timexes = TokenClassification(tokens="tokens", labels="ntimex_tags", dataset_name="strombergnlp/named_timexes", config_name="named-timexes")

wkrl_cord = TokenClassification(labels="labels", dataset_name="wkrl/cord", config_name="CORD")

arize_ai_xtreme_en = TokenClassification(labels="ner_tags", dataset_name="arize-ai/xtreme_en")