"""Canonical Hub ids for legacy dataset names without a namespace.

The Hub redirects these old names (``glue``) to their current repos
(``nyu-mll/glue``); huggingface_hub>=1 rejects names without a namespace.
Task annotations keep the short names, which also form the task ids.
"""

CANONICAL = {
    'ade_corpus_v2': 'ade-benchmark-corpus/ade_corpus_v2', 'ag_news': 'fancyzhx/ag_news', 'ai2_arc': 'allenai/ai2_arc',
    'amazon_polarity': 'fancyzhx/amazon_polarity', 'americas_nli': 'nala-cub/americas_nli', 'anli': 'facebook/anli',
    'app_reviews': 'sealuzh/app_reviews', 'art': 'allenai/art', 'circa': 'google-research-datasets/circa',
    'codah': 'jaredfern/codah', 'commonsense_qa': 'tau/commonsense_qa', 'cos_e': 'Salesforce/cos_e',
    'dbpedia_14': 'fancyzhx/dbpedia_14', 'definite_pronoun_resolution': 'community-datasets/definite_pronoun_resolution',
    'discovery': 'sileod/discovery', 'exams': 'mhardalov/exams', 'glue': 'nyu-mll/glue',
    'go_emotions': 'google-research-datasets/go_emotions', 'hate_speech_offensive': 'tdavidson/hate_speech_offensive',
    'hellaswag': 'Rowan/hellaswag', 'imdb': 'stanfordnlp/imdb', 'lex_glue': 'coastalcph/lex_glue',
    'medical_questions_pairs': 'curaihealth/medical_questions_pairs', 'medmcqa': 'openlifescienceai/medmcqa',
    'offenseval_dravidian': 'community-datasets/offenseval_dravidian', 'onestop_qa': 'malmaud/onestop_qa',
    'openbookqa': 'allenai/openbookqa', 'paws': 'google-research-datasets/paws',
    'poem_sentiment': 'google-research-datasets/poem_sentiment', 'pragmeval': 'sileod/pragmeval', 'qasc': 'allenai/qasc',
    'quail': 'textmachinelab/quail', 'quarel': 'community-datasets/quarel', 'quartz': 'allenai/quartz', 'race': 'ehovy/race',
    'rotten_tomatoes': 'cornell-movie-review-data/rotten_tomatoes', 'sciq': 'allenai/sciq', 'scitail': 'allenai/scitail',
    'sem_eval_2010_task_8': 'SemEvalWorkshop/sem_eval_2010_task_8', 'sms_spam': 'ucirvine/sms_spam',
    'snips_built_in_intents': 'sonos-nlu-benchmark/snips_built_in_intents', 'snli': 'stanfordnlp/snli',
    'super_glue': 'aps/super_glue', 'swag': 'allenai/swag', 'tweet_eval': 'cardiffnlp/tweet_eval',
    'tweets_hate_speech_detection': 'tweets-hate-speech-detection/tweets_hate_speech_detection',
    'wiki_qa': 'microsoft/wiki_qa', 'wili_2018': 'MartinThoma/wili_2018', 'winogrande': 'allenai/winogrande',
    'yahoo_answers_topics': 'community-datasets/yahoo_answers_topics', 'yelp_review_full': 'Yelp/yelp_review_full',
}
