"""Original Hub datasets behind tasksource copies, mirrors and raw-file loaders.

A model trained on tasksource tasks should credit both the repo the data was
loaded from and the original dataset (the ``datasets:`` model card field).
Keys are the loading repo (``dataset_name``) or, for tasks that read raw
files, the task id. Repos read through ``hf://datasets/...`` data files are
found automatically and need no entry. See ``tasksource.hub_datasets``.

Add an entry whenever a dataset is re-uploaded under tasksource/ (for example
by scripts/upload_repackaged.py) or a task switches to a mirror.
"""

ORIGINALS = {
    # re-uploaded under tasksource/
    "tasksource/blog_authorship_corpus": ["barilan/blog_authorship_corpus"],
    "tasksource/chaos-mnli-ambiguity": [],  # ChaosNLI is only on GitHub/Dropbox
    "tasksource/clutrr": ["CLUTRR/v1"],
    "tasksource/lewidi": [],  # the LeWiDi GitHub release; not on the Hub
    "tasksource/measuring-hate-speech-votes": ["ucberkeley-dlab/measuring-hate-speech"],
    "tasksource/contract-nli": ["kiddothe2b/contract-nli"],
    "tasksource/corr2cause": ["causal-nlp/corr2cause"],
    "tasksource/dynahate": ["aps/dynahate"],
    "tasksource/ethos": ["iamollas/ethos"],
    "tasksource/google_wellformed_query": ["google-research-datasets/google_wellformed_query"],
    "tasksource/hans": ["jhu-cogsci/hans"],
    "tasksource/hate_speech18": ["odegiber/hate_speech18"],
    "tasksource/hlgd": ["philippelaban/hlgd"],
    "tasksource/humicroedit": ["SemEvalWorkshop/humicroedit"],
    "tasksource/liar": ["ucsbai/liar"],
    "tasksource/mms": ["Brand24/mms"],
    "tasksource/multilingual-sentiments": ["tyqiangz/multilingual-sentiments"],
    "tasksource/math_qa": ["allenai/math_qa"],
    "tasksource/numer_sense": ["INK-USC/numer_sense"],
    "tasksource/prm800k_dpo": ["tasksource/PRM800K"],
    "tasksource/scicite": ["allenai/scicite"],
    "tasksource/scifact_entailment": ["allenai/scifact_entailment"],
    "tasksource/sharc": ["nikhilweee/sharc_modified"],
    "tasksource/sick": ["RobZamp/sick"],
    "tasksource/silicone": ["eusip/silicone"],
    "tasksource/social_i_qa": ["allenai/social_i_qa"],
    "tasksource/trec": ["CogComp/trec"],
    "tasksource/wiqa": ["allenai/wiqa"],
    "tasksource/xglue": ["microsoft/xglue"],
    "tasksource/xlwic": ["pasinit/xlwic"],
    # third-party mirrors of script-only or retired datasets
    "Deehan1866/processed_phrase_similarity": ["PiC/phrase_similarity"],
    "EleutherAI/headqa": ["dvilares/head_qa"],
    "Korea-MES/open_question_type": ["launch/open_question_type"],
    "LabHC/moral_stories": ["demelin/moral_stories"],
    "MoE-UNC/wikihop": ["QAngaroo/wiki_hop"],
    "Samsoup/cosmos_qa": ["allenai/cosmos_qa"],
    "tomaarsen/conll2003": ["eriktks/conll2003"],
    "flaitenberger/wnut_17": ["leondz/wnut_17"],
    "baber/piqa": ["ybisk/piqa"],
    "fireworks-ai/logiqa": ["lucasmccabe/logiqa"],
    "ghbacct/financial-phrasebank-all-agree-classification": ["takala/financial_phrasebank"],
    "goosmanlei/amazon_reviews_multi": ["defunct-datasets/amazon_reviews_multi"],
    "jeggers/riddle_sense": ["INK-USC/riddle_sense"],
    "legacy-datasets/banking77": ["PolyAI/banking77"],
    "marcov/health_fact_promptsource": ["ImperialCollegeLondon/health_fact"],
    "marcov/mc_taco_promptsource": ["CogComp/mc_taco"],
    "michiel/xstance": ["strombergnlp/x-stance"],
    "mteb/AfriSentiClassification": ["shmuhammad/AfriSenti-twitter-sentiment"],
    "mteb/MassiveIntentClassification": ["AmazonScience/massive"],
    "mteb/NusaX-senti": ["indonlp/NusaX-senti"],
    "oneonlee/cleansed_emocontext": ["SemEvalWorkshop/emo"],
    "vwxyzjn/summarize_from_feedback_oai_preprocessing": ["openai/summarize_from_feedback"],
    # raw-file loaders, keyed by task id
    "TuringBench": ["turingbench/TuringBench"],
    "discosense": ["prajjwal1/discosense"],
    "docred": ["thunlp/docred"],
    "dream": ["dataset-org/dream"],
    "hope_edi/english": ["dravidianlangtech/hope_edi"],
    "miam": ["PierreColombo/miam"],
    "propsegment/nli": ["sihaochen/propsegment"],
    "tweet_sentiment_multilingual": ["cardiffnlp/tweet_sentiment_multilingual"],
    "Touche23-ValueEval": ["webis/Touche23-ValueEval"],
}
