504 English tasks. Load one with `load_task(id)`; the annotations are in [tasks.py](src/tasksource/tasks.py), and tasks kept out on purpose are in [parked.py](src/tasksource/parked.py).

| id | type | dataset | question |
|---|---|---|---|
| glue/mnli | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| glue/qnli | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| glue/rte | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| glue/wnli | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| glue/mrpc | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| glue/qqp | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| glue/stsb | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) | How similar are the two sentences, from 0 (unrelated) to 5 (equivalent)? |
| super_glue/boolq | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| super_glue/boolq_passage | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| super_glue/cb | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| super_glue/multirc | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| super_glue/wic | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| super_glue/axg | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| anli/a1 | Classification | [facebook/anli](https://hf.co/datasets/facebook/anli) |  |
| anli/a2 | Classification | [facebook/anli](https://hf.co/datasets/facebook/anli) |  |
| anli/a3 | Classification | [facebook/anli](https://hf.co/datasets/facebook/anli) |  |
| babi_nli/counting | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| babi_nli/indefinite-knowledge | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| babi_nli/lists-sets | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| babi_nli/path-finding | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| babi_nli/positional-reasoning | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| babi_nli/simple-negation | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| babi_nli/size-reasoning | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| babi_nli/conjunction | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| babi_nli/three-arg-relations | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| babi_nli/three-supporting-facts | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| babi_nli/time-reasoning | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| babi_nli/two-arg-relations | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| babi_nli/single-supporting-fact | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| babi_nli/compound-coreference | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| babi_nli/basic-deduction | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| babi_nli/basic-coreference | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| babi_nli/two-supporting-facts | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| babi_nli/basic-induction | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| babi_nli/yes-no-questions | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| sick/label | Classification | [tasksource/sick](https://hf.co/datasets/tasksource/sick) |  |
| sick/relatedness | Classification | [tasksource/sick](https://hf.co/datasets/tasksource/sick) | How related are the two sentences, from 1 (unrelated) to 5 (very related)? |
| snli | Classification | [stanfordnlp/snli](https://hf.co/datasets/stanfordnlp/snli) |  |
| scitail/snli_format | Classification | [allenai/scitail](https://hf.co/datasets/allenai/scitail) |  |
| hans | Classification | [tasksource/hans](https://hf.co/datasets/tasksource/hans) |  |
| WANLI | Classification | [alisawuffles/WANLI](https://hf.co/datasets/alisawuffles/WANLI) |  |
| recast/recast_factuality | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| recast/recast_verbnet | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| recast/recast_puns | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| recast/recast_ner | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| recast/recast_sentiment | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| recast/recast_megaveridicality | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| recast/recast_verbcorner | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| probability_words_nli/usnli | Classification | [sileod/probability_words_nli](https://hf.co/datasets/sileod/probability_words_nli) |  |
| probability_words_nli/reasoning_2hop | Classification | [sileod/probability_words_nli](https://hf.co/datasets/sileod/probability_words_nli) |  |
| probability_words_nli/reasoning_1hop | Classification | [sileod/probability_words_nli](https://hf.co/datasets/sileod/probability_words_nli) |  |
| nan-nli | Classification | [joey234/nan-nli](https://hf.co/datasets/joey234/nan-nli) |  |
| nli_fever | Classification | [pietrolesci/nli_fever](https://hf.co/datasets/pietrolesci/nli_fever) |  |
| breaking_nli | Classification | [pietrolesci/breaking_nli](https://hf.co/datasets/pietrolesci/breaking_nli) |  |
| conj_nli | Classification | [pietrolesci/conj_nli](https://hf.co/datasets/pietrolesci/conj_nli) |  |
| fracas | Classification | [pietrolesci/fracas](https://hf.co/datasets/pietrolesci/fracas) |  |
| dialogue_nli | Classification | [pietrolesci/dialogue_nli](https://hf.co/datasets/pietrolesci/dialogue_nli) |  |
| mpe | Classification | [pietrolesci/mpe](https://hf.co/datasets/pietrolesci/mpe) |  |
| dnc | Classification | [pietrolesci/dnc](https://hf.co/datasets/pietrolesci/dnc) |  |
| recast_white/fnplus | Classification | [pietrolesci/recast_white](https://hf.co/datasets/pietrolesci/recast_white) |  |
| recast_white/sprl | Classification | [pietrolesci/recast_white](https://hf.co/datasets/pietrolesci/recast_white) |  |
| recast_white/dpr | Classification | [pietrolesci/recast_white](https://hf.co/datasets/pietrolesci/recast_white) |  |
| joci | Classification | [pietrolesci/joci](https://hf.co/datasets/pietrolesci/joci) |  |
| robust_nli/IS_CS | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| robust_nli/LI_LI | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| robust_nli/ST_WO | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| robust_nli/PI_SP | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| robust_nli/PI_CD | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| robust_nli/ST_SE | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| robust_nli/ST_NE | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| robust_nli/ST_LM | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| robust_nli_is_sd | Classification | [pietrolesci/robust_nli_is_sd](https://hf.co/datasets/pietrolesci/robust_nli_is_sd) |  |
| robust_nli_li_ts | Classification | [pietrolesci/robust_nli_li_ts](https://hf.co/datasets/pietrolesci/robust_nli_li_ts) |  |
| gen_debiased_nli/snli_seq_z | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| gen_debiased_nli/snli_z_aug | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| gen_debiased_nli/snli_par_z | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| gen_debiased_nli/mnli_par_z | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| gen_debiased_nli/mnli_z_aug | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| gen_debiased_nli/mnli_seq_z | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| add_one_rte | Classification | [pietrolesci/add_one_rte](https://hf.co/datasets/pietrolesci/add_one_rte) |  |
| imppres/presupposition_question_presupposition/presupposition | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| imppres/presupposition_possessed_definites_uniqueness/presupposition | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| imppres/presupposition_possessed_definites_existence/presupposition | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| imppres/presupposition_only_presupposition/presupposition | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| imppres/presupposition_cleft_uniqueness/presupposition | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| imppres/presupposition_cleft_existence/presupposition | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| imppres/presupposition_change_of_state/presupposition | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| imppres/presupposition_both_presupposition/presupposition | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| imppres/presupposition_all_n_presupposition/presupposition | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| imppres/implicature_numerals_2_3/prag | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| imppres/implicature_numerals_10_100/prag | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| imppres/implicature_modals/prag | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| imppres/implicature_gradable_verb/prag | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| imppres/implicature_gradable_adjective/prag | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| imppres/implicature_connectives/prag | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| imppres/implicature_quantifiers/prag | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| imppres/implicature_modals/log | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| imppres/implicature_gradable_verb/log | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| imppres/implicature_gradable_adjective/log | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| imppres/implicature_quantifiers/log | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| imppres/implicature_numerals_2_3/log | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| imppres/implicature_connectives/log | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| imppres/implicature_numerals_10_100/log | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| hlgd | Classification | [tasksource/hlgd](https://hf.co/datasets/tasksource/hlgd) |  |
| paws/labeled_final | Classification | [google-research-datasets/paws](https://hf.co/datasets/google-research-datasets/paws) |  |
| paws/labeled_swap | Classification | [google-research-datasets/paws](https://hf.co/datasets/google-research-datasets/paws) |  |
| medical_questions_pairs | Classification | [curaihealth/medical_questions_pairs](https://hf.co/datasets/curaihealth/medical_questions_pairs) |  |
| conll2003/pos_tags | TokenClassification | [tomaarsen/conll2003](https://hf.co/datasets/tomaarsen/conll2003) |  |
| conll2003/chunk_tags | TokenClassification | [tomaarsen/conll2003](https://hf.co/datasets/tomaarsen/conll2003) |  |
| conll2003/ner_tags | TokenClassification | [tomaarsen/conll2003](https://hf.co/datasets/tomaarsen/conll2003) |  |
| fig-qa | MultipleChoice | [nightingal3/fig-qa](https://hf.co/datasets/nightingal3/fig-qa) |  |
| cos_e/v1.0 | MultipleChoice | [Salesforce/cos_e](https://hf.co/datasets/Salesforce/cos_e) |  |
| cosmos_qa | MultipleChoice | [Samsoup/cosmos_qa](https://hf.co/datasets/Samsoup/cosmos_qa) |  |
| dream | MultipleChoice | [dataset-org/dream](https://hf.co/datasets/dataset-org/dream) |  |
| openbookqa | MultipleChoice | [allenai/openbookqa](https://hf.co/datasets/allenai/openbookqa) |  |
| qasc | MultipleChoice | [allenai/qasc](https://hf.co/datasets/allenai/qasc) |  |
| quartz | MultipleChoice | [allenai/quartz](https://hf.co/datasets/allenai/quartz) |  |
| quail | MultipleChoice | [textmachinelab/quail](https://hf.co/datasets/textmachinelab/quail) |  |
| head_qa/en | MultipleChoice | [EleutherAI/headqa](https://hf.co/datasets/EleutherAI/headqa) |  |
| sciq | MultipleChoice | [allenai/sciq](https://hf.co/datasets/allenai/sciq) |  |
| social_i_qa | MultipleChoice | [tasksource/social_i_qa](https://hf.co/datasets/tasksource/social_i_qa) |  |
| wiki_hop/original | MultipleChoice | [MoE-UNC/wikihop](https://hf.co/datasets/MoE-UNC/wikihop) |  |
| wiqa | MultipleChoice | [tasksource/wiqa](https://hf.co/datasets/tasksource/wiqa) |  |
| piqa | MultipleChoice | [baber/piqa](https://hf.co/datasets/baber/piqa) |  |
| hellaswag | MultipleChoice | [Rowan/hellaswag](https://hf.co/datasets/Rowan/hellaswag) |  |
| super_glue/copa | MultipleChoice | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| balanced-copa | MultipleChoice | [pkavumba/balanced-copa](https://hf.co/datasets/pkavumba/balanced-copa) |  |
| e-CARE | MultipleChoice | [12ml/e-CARE](https://hf.co/datasets/12ml/e-CARE) |  |
| art | MultipleChoice | [allenai/art](https://hf.co/datasets/allenai/art) | What happened in between? |
| winogrande/winogrande_xl | MultipleChoice | [allenai/winogrande](https://hf.co/datasets/allenai/winogrande) |  |
| codah/codah | MultipleChoice | [jaredfern/codah](https://hf.co/datasets/jaredfern/codah) |  |
| ai2_arc/ARC-Easy/challenge | MultipleChoice | [allenai/ai2_arc](https://hf.co/datasets/allenai/ai2_arc) |  |
| ai2_arc/ARC-Challenge/challenge | MultipleChoice | [allenai/ai2_arc](https://hf.co/datasets/allenai/ai2_arc) |  |
| definite_pronoun_resolution | MultipleChoice | [community-datasets/definite_pronoun_resolution](https://hf.co/datasets/community-datasets/definite_pronoun_resolution) |  |
| swag/regular | MultipleChoice | [allenai/swag](https://hf.co/datasets/allenai/swag) |  |
| math_qa | MultipleChoice | [tasksource/math_qa](https://hf.co/datasets/tasksource/math_qa) |  |
| glue/cola | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| glue/sst2 | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| utilitarianism | Classification | csv |  |
| amazon_counterfactual/en | Classification | [mteb/amazon_counterfactual](https://hf.co/datasets/mteb/amazon_counterfactual) |  |
| insincere-questions | Classification | [SetFit/insincere-questions](https://hf.co/datasets/SetFit/insincere-questions) |  |
| toxic_conversations | Classification | [SetFit/toxic_conversations](https://hf.co/datasets/SetFit/toxic_conversations) |  |
| TuringBench | Classification | csv |  |
| trec | Classification | [tasksource/trec](https://hf.co/datasets/tasksource/trec) |  |
| vitaminc | Classification | [tals/vitaminc](https://hf.co/datasets/tals/vitaminc) |  |
| hope_edi/english | Classification | csv |  |
| rumoureval_2019/RumourEval2019 | Classification | csv |  |
| ethos/binary | Classification | [SetFit/ethos_binary](https://hf.co/datasets/SetFit/ethos_binary) |  |
| ethos/multilabel | Classification | [tasksource/ethos](https://hf.co/datasets/tasksource/ethos) |  |
| tweet_eval/emoji | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| tweet_eval/emotion | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| tweet_eval/hate | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| tweet_eval/irony | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| tweet_eval/offensive | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| tweet_eval/sentiment | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| tweet_eval/stance_abortion | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | What stance does the tweet take on abortion? |
| tweet_eval/stance_atheism | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | What stance does the tweet take on atheism? |
| tweet_eval/stance_climate | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | What stance does the tweet take on climate change? |
| tweet_eval/stance_feminist | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | What stance does the tweet take on feminism? |
| tweet_eval/stance_hillary | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | What stance does the tweet take on Hillary Clinton? |
| discovery/discovery | Classification | [sileod/discovery](https://hf.co/datasets/sileod/discovery) |  |
| pragmeval/switchboard | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| pragmeval/verifiability | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| pragmeval/mrda | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| pragmeval/emergent | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| pragmeval/gum | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| pragmeval/pdtb | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| pragmeval/persuasiveness-claimtype | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| pragmeval/persuasiveness-premisetype | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| pragmeval/stac | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| pragmeval/sarcasm | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| pragmeval/emobank-arousal | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| pragmeval/emobank-dominance | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| pragmeval/emobank-valence | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| pragmeval/squinky-formality | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| pragmeval/squinky-implicature | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| pragmeval/squinky-informativeness | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| pragmeval/persuasiveness-eloquence | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| pragmeval/persuasiveness-relevance | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| pragmeval/persuasiveness-specificity | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| pragmeval/persuasiveness-strength | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| silicone/oasis | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| silicone/sem | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| silicone/meld_s | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| silicone/meld_e | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| silicone/maptask | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| silicone/dyda_e | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| silicone/dyda_da | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| silicone/iemocap | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| lex_glue/eurlex | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) |  |
| lex_glue/scotus | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) |  |
| lex_glue/ledgar | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) |  |
| lex_glue/unfair_tos | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) | Which kind of unfair term, if any, does this terms-of-service clause contain? |
| lex_glue/case_hold | MultipleChoice | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) |  |
| language-identification | Classification | [papluca/language-identification](https://hf.co/datasets/papluca/language-identification) | What language is this text in? |
| imdb | Classification | [stanfordnlp/imdb](https://hf.co/datasets/stanfordnlp/imdb) |  |
| rotten_tomatoes | Classification | [cornell-movie-review-data/rotten_tomatoes](https://hf.co/datasets/cornell-movie-review-data/rotten_tomatoes) |  |
| ag_news | Classification | [fancyzhx/ag_news](https://hf.co/datasets/fancyzhx/ag_news) |  |
| yelp_review_full/yelp_review_full | Classification | [Yelp/yelp_review_full](https://hf.co/datasets/Yelp/yelp_review_full) |  |
| financial_phrasebank/sentences_allagree | Classification | [ghbacct/financial-phrasebank-all-agree-classification](https://hf.co/datasets/ghbacct/financial-phrasebank-all-agree-classification) |  |
| poem_sentiment | Classification | [google-research-datasets/poem_sentiment](https://hf.co/datasets/google-research-datasets/poem_sentiment) |  |
| emotion | Classification | [dair-ai/emotion](https://hf.co/datasets/dair-ai/emotion) |  |
| dbpedia_14/dbpedia_14 | Classification | [fancyzhx/dbpedia_14](https://hf.co/datasets/fancyzhx/dbpedia_14) |  |
| amazon_polarity/amazon_polarity | Classification | [fancyzhx/amazon_polarity](https://hf.co/datasets/fancyzhx/amazon_polarity) |  |
| app_reviews | Classification | [sealuzh/app_reviews](https://hf.co/datasets/sealuzh/app_reviews) |  |
| hate_speech18 | Classification | [tasksource/hate_speech18](https://hf.co/datasets/tasksource/hate_speech18) |  |
| sms_spam | Classification | [ucirvine/sms_spam](https://hf.co/datasets/ucirvine/sms_spam) |  |
| humicroedit/subtask-1 | Classification | [tasksource/humicroedit](https://hf.co/datasets/tasksource/humicroedit) |  |
| humicroedit/subtask-2 | Classification | [tasksource/humicroedit](https://hf.co/datasets/tasksource/humicroedit) | Which edited headline is funnier? |
| snips_built_in_intents | Classification | [sonos-nlu-benchmark/snips_built_in_intents](https://hf.co/datasets/sonos-nlu-benchmark/snips_built_in_intents) |  |
| hate_speech_offensive | Classification | [tdavidson/hate_speech_offensive](https://hf.co/datasets/tdavidson/hate_speech_offensive) |  |
| yahoo_answers_topics | Classification | [community-datasets/yahoo_answers_topics](https://hf.co/datasets/community-datasets/yahoo_answers_topics) |  |
| stackoverflow-questions | Classification | [pacovaldez/stackoverflow-questions](https://hf.co/datasets/pacovaldez/stackoverflow-questions) |  |
| hyperpartisan_news | Classification | [zapsdcn/hyperpartisan_news](https://hf.co/datasets/zapsdcn/hyperpartisan_news) |  |
| sciie | Classification | [zapsdcn/sciie](https://hf.co/datasets/zapsdcn/sciie) |  |
| citation_intent | Classification | [zapsdcn/citation_intent](https://hf.co/datasets/zapsdcn/citation_intent) |  |
| go_emotions/simplified | Classification | [google-research-datasets/go_emotions](https://hf.co/datasets/google-research-datasets/go_emotions) |  |
| scicite | Classification | [tasksource/scicite](https://hf.co/datasets/tasksource/scicite) |  |
| liar | Classification | [tasksource/liar](https://hf.co/datasets/tasksource/liar) |  |
| lexical_relation_classification/ROOT09 | Classification | json | How is the second word related to the first? |
| lexical_relation_classification/K&H+N | Classification | json | How is the second word related to the first? |
| lexical_relation_classification/BLESS | Classification | json | How is the second word related to the first? |
| lexical_relation_classification/EVALution | Classification | json | How is the second word related to the first? |
| lexical_relation_classification/CogALexV | Classification | json | How is the second word related to the first? |
| linguisticprobing/subj_number | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| linguisticprobing/obj_number | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| linguisticprobing/past_present | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| linguisticprobing/sentence_length | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| linguisticprobing/top_constituents | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| linguisticprobing/tree_depth | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| linguisticprobing/coordination_inversion | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| linguisticprobing/odd_man_out | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| linguisticprobing/bigram_shift | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| crowdflower/airline-sentiment | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) |  |
| crowdflower/corporate-messaging | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) |  |
| crowdflower/economic-news | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) |  |
| crowdflower/political-media-audience | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) |  |
| crowdflower/political-media-bias | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) |  |
| crowdflower/political-media-message | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) |  |
| crowdflower/text_emotion | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) |  |
| crowdflower/sentiment_nuclear_power | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) |  |
| crowdflower/tweet_global_warming | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) |  |
| ethics/commonsense | Classification | csv |  |
| ethics/deontology | Classification | csv |  |
| ethics/justice | Classification | csv |  |
| ethics/virtue | Classification | [hendrycks/ethics](https://hf.co/datasets/hendrycks/ethics) |  |
| emo/emo2019 | Classification | [oneonlee/cleansed_emocontext](https://hf.co/datasets/oneonlee/cleansed_emocontext) |  |
| google_wellformed_query | Classification | [tasksource/google_wellformed_query](https://hf.co/datasets/tasksource/google_wellformed_query) | Is this search query a well-formed question? |
| tweets_hate_speech_detection | Classification | [tweets-hate-speech-detection/tweets_hate_speech_detection](https://hf.co/datasets/tweets-hate-speech-detection/tweets_hate_speech_detection) |  |
| wnut_17/wnut_17 | TokenClassification | [flaitenberger/wnut_17](https://hf.co/datasets/flaitenberger/wnut_17) |  |
| ncbi_disease/ncbi_disease | TokenClassification | [ncbi/ncbi_disease](https://hf.co/datasets/ncbi/ncbi_disease) |  |
| acronym_identification | TokenClassification | [amirveyseh/acronym_identification](https://hf.co/datasets/amirveyseh/acronym_identification) |  |
| jnlpba/jnlpba | TokenClassification | [jnlpba/jnlpba](https://hf.co/datasets/jnlpba/jnlpba) |  |
| ontonotes_english/SpeedOfMagic--ontonotes_english | TokenClassification | [SpeedOfMagic/ontonotes_english](https://hf.co/datasets/SpeedOfMagic/ontonotes_english) |  |
| blog_authorship_corpus/gender | Classification | [tasksource/blog_authorship_corpus](https://hf.co/datasets/tasksource/blog_authorship_corpus) | What is the blogger's gender? |
| blog_authorship_corpus/age | Classification | [tasksource/blog_authorship_corpus](https://hf.co/datasets/tasksource/blog_authorship_corpus) | What is the blogger's age group? |
| blog_authorship_corpus/job | Classification | [tasksource/blog_authorship_corpus](https://hf.co/datasets/tasksource/blog_authorship_corpus) | In which industry does the blogger work? |
| open_question_type | Classification | [Korea-MES/open_question_type](https://hf.co/datasets/Korea-MES/open_question_type) |  |
| health_fact | Classification | [marcov/health_fact_promptsource](https://hf.co/datasets/marcov/health_fact_promptsource) |  |
| commonsense_qa | MultipleChoice | [tau/commonsense_qa](https://hf.co/datasets/tau/commonsense_qa) |  |
| mc_taco | Classification | [marcov/mc_taco_promptsource](https://hf.co/datasets/marcov/mc_taco_promptsource) | Is this answer plausible? |
| ade_corpus_v2/Ade_corpus_v2_classification | Classification | [ade-benchmark-corpus/ade_corpus_v2](https://hf.co/datasets/ade-benchmark-corpus/ade_corpus_v2) |  |
| discosense | MultipleChoice | json |  |
| circa | Classification | [google-research-datasets/circa](https://hf.co/datasets/google-research-datasets/circa) |  |
| code_x_glue_cc_defect_detection | Classification | [google/code_x_glue_cc_defect_detection](https://hf.co/datasets/google/code_x_glue_cc_defect_detection) |  |
| phrase_similarity | Classification | [Deehan1866/processed_phrase_similarity](https://hf.co/datasets/Deehan1866/processed_phrase_similarity) |  |
| scientific-exaggeration-detection | Classification | [copenlu/scientific-exaggeration-detection](https://hf.co/datasets/copenlu/scientific-exaggeration-detection) |  |
| quarel | Classification | [community-datasets/quarel](https://hf.co/datasets/community-datasets/quarel) |  |
| fever-evidence-related | Classification | [mwong/fever-evidence-related](https://hf.co/datasets/mwong/fever-evidence-related) |  |
| numer_sense | Classification | [tasksource/numer_sense](https://hf.co/datasets/tasksource/numer_sense) |  |
| dynasent/dynabench.dynasent.r1.all/r1 | Classification | [tasksource/dynasent](https://hf.co/datasets/tasksource/dynasent) |  |
| dynasent/dynabench.dynasent.r2.all/r2 | Classification | [tasksource/dynasent](https://hf.co/datasets/tasksource/dynasent) |  |
| Sarcasm_News_Headline | Classification | [raquiba/Sarcasm_News_Headline](https://hf.co/datasets/raquiba/Sarcasm_News_Headline) |  |
| sem_eval_2010_task_8 | Classification | [SemEvalWorkshop/sem_eval_2010_task_8](https://hf.co/datasets/SemEvalWorkshop/sem_eval_2010_task_8) |  |
| auditor_review | Classification | [demo-org/auditor_review](https://hf.co/datasets/demo-org/auditor_review) |  |
| medmcqa | MultipleChoice | [openlifescienceai/medmcqa](https://hf.co/datasets/openlifescienceai/medmcqa) |  |
| Dynasent_Disagreement | Classification | [RuyuanWan/Dynasent_Disagreement](https://hf.co/datasets/RuyuanWan/Dynasent_Disagreement) | Would annotators disagree about the sentiment of this text? |
| Politeness_Disagreement | Classification | [RuyuanWan/Politeness_Disagreement](https://hf.co/datasets/RuyuanWan/Politeness_Disagreement) | Would annotators disagree about the politeness of this text? |
| SBIC_Disagreement | Classification | [RuyuanWan/SBIC_Disagreement](https://hf.co/datasets/RuyuanWan/SBIC_Disagreement) | Would annotators disagree about whether this text is offensive? |
| SChem_Disagreement | Classification | [RuyuanWan/SChem_Disagreement](https://hf.co/datasets/RuyuanWan/SChem_Disagreement) | Would annotators disagree about whether this rule of thumb is acceptable? |
| Dilemmas_Disagreement | Classification | [RuyuanWan/Dilemmas_Disagreement](https://hf.co/datasets/RuyuanWan/Dilemmas_Disagreement) | Would annotators disagree about which of these two actions is less ethical? |
| logiqa | MultipleChoice | [fireworks-ai/logiqa](https://hf.co/datasets/fireworks-ai/logiqa) |  |
| wiki_qa | Classification | [microsoft/wiki_qa](https://hf.co/datasets/microsoft/wiki_qa) | Does this sentence answer the question? |
| cycic_classification | Classification | [tasksource/cycic_classification](https://hf.co/datasets/tasksource/cycic_classification) |  |
| cycic_multiplechoice | MultipleChoice | [tasksource/cycic_multiplechoice](https://hf.co/datasets/tasksource/cycic_multiplechoice) |  |
| sts-companion | Classification | [tasksource/sts-companion](https://hf.co/datasets/tasksource/sts-companion) |  |
| commonsense_qa_2.0 | Classification | [tasksource/commonsense_qa_2.0](https://hf.co/datasets/tasksource/commonsense_qa_2.0) |  |
| lingnli | Classification | [tasksource/lingnli](https://hf.co/datasets/tasksource/lingnli) |  |
| monotonicity-entailment | Classification | [tasksource/monotonicity-entailment](https://hf.co/datasets/tasksource/monotonicity-entailment) |  |
| arct | MultipleChoice | [tasksource/arct](https://hf.co/datasets/tasksource/arct) |  |
| scinli | Classification | [tasksource/scinli](https://hf.co/datasets/tasksource/scinli) |  |
| naturallogic | Classification | [tasksource/naturallogic](https://hf.co/datasets/tasksource/naturallogic) |  |
| onestop_qa | MultipleChoice | [malmaud/onestop_qa](https://hf.co/datasets/malmaud/onestop_qa) |  |
| moral_stories/full | MultipleChoice | [LabHC/moral_stories](https://hf.co/datasets/LabHC/moral_stories) |  |
| prost | MultipleChoice | json |  |
| dynahate | Classification | [tasksource/dynahate](https://hf.co/datasets/tasksource/dynahate) |  |
| syntactic-augmentation-nli | Classification | [tasksource/syntactic-augmentation-nli](https://hf.co/datasets/tasksource/syntactic-augmentation-nli) |  |
| autotnli | Classification | [tasksource/autotnli](https://hf.co/datasets/tasksource/autotnli) |  |
| CONDAQA | Classification | [lasha-nlp/CONDAQA](https://hf.co/datasets/lasha-nlp/CONDAQA) |  |
| webgpt_comparisons | MultipleChoice | [heegyu/webgpt_comparisons_ko](https://hf.co/datasets/heegyu/webgpt_comparisons_ko) | Which answer did the human rater prefer? |
| synthetic-instruct-gptj-pairwise | MultipleChoice | [Dahoas/synthetic-instruct-gptj-pairwise](https://hf.co/datasets/Dahoas/synthetic-instruct-gptj-pairwise) | Which response is better? |
| scruples | Classification | [tasksource/scruples](https://hf.co/datasets/tasksource/scruples) | Was the author in the right or in the wrong? |
| wouldyourather | MultipleChoice | [tasksource/wouldyourather](https://hf.co/datasets/tasksource/wouldyourather) | Which would most people rather do? |
| defeasible-nli/atomic | Classification | [tasksource/defeasible-nli](https://hf.co/datasets/tasksource/defeasible-nli) |  |
| defeasible-nli/snli | Classification | [tasksource/defeasible-nli](https://hf.co/datasets/tasksource/defeasible-nli) |  |
| defeasible-nli/social | Classification | [tasksource/defeasible-nli](https://hf.co/datasets/tasksource/defeasible-nli) |  |
| help-nli | Classification | [tasksource/help-nli](https://hf.co/datasets/tasksource/help-nli) |  |
| nli-veridicality-transitivity | Classification | [tasksource/nli-veridicality-transitivity](https://hf.co/datasets/tasksource/nli-veridicality-transitivity) |  |
| lonli | Classification | [tasksource/lonli](https://hf.co/datasets/tasksource/lonli) |  |
| dadc-limit-nli | Classification | [tasksource/dadc-limit-nli](https://hf.co/datasets/tasksource/dadc-limit-nli) |  |
| FLUTE | Classification | [ColumbiaNLP/FLUTE](https://hf.co/datasets/ColumbiaNLP/FLUTE) |  |
| strategy-qa | Classification | [tasksource/strategy-qa](https://hf.co/datasets/tasksource/strategy-qa) |  |
| summarize_from_feedback/comparisons | MultipleChoice | [vwxyzjn/summarize_from_feedback_oai_preprocessing](https://hf.co/datasets/vwxyzjn/summarize_from_feedback_oai_preprocessing) | Which summary did the human rater prefer? |
| folio | Classification | [tasksource/folio](https://hf.co/datasets/tasksource/folio) |  |
| tomi-nli | Classification | [tasksource/tomi-nli](https://hf.co/datasets/tasksource/tomi-nli) |  |
| avicenna | Classification | [tasksource/avicenna](https://hf.co/datasets/tasksource/avicenna) | Do the two premises form a syllogism? |
| SHP | MultipleChoice | [stanfordnlp/SHP](https://hf.co/datasets/stanfordnlp/SHP) | Which reply did readers prefer? |
| MedQA-USMLE-4-options-hf | MultipleChoice | [GBaker/MedQA-USMLE-4-options-hf](https://hf.co/datasets/GBaker/MedQA-USMLE-4-options-hf) |  |
| wikimedqa/medwiki | MultipleChoice | [sileod/wikimedqa](https://hf.co/datasets/sileod/wikimedqa) |  |
| cicero | MultipleChoice | [declare-lab/cicero](https://hf.co/datasets/declare-lab/cicero) |  |
| CREAK | Classification | [amydeng2000/CREAK](https://hf.co/datasets/amydeng2000/CREAK) |  |
| mutual | MultipleChoice | [tasksource/mutual](https://hf.co/datasets/tasksource/mutual) |  |
| puzzte | Classification | [tasksource/puzzte](https://hf.co/datasets/tasksource/puzzte) |  |
| implicatures | MultipleChoice | [tasksource/implicatures](https://hf.co/datasets/tasksource/implicatures) |  |
| race/high | MultipleChoice | [ehovy/race](https://hf.co/datasets/ehovy/race) |  |
| race/middle | MultipleChoice | [ehovy/race](https://hf.co/datasets/ehovy/race) |  |
| race-c | MultipleChoice | [tasksource/race-c](https://hf.co/datasets/tasksource/race-c) |  |
| spartqa-yn | Classification | [tasksource/spartqa-yn](https://hf.co/datasets/tasksource/spartqa-yn) |  |
| spartqa-mchoice | MultipleChoice | [tasksource/spartqa-mchoice](https://hf.co/datasets/tasksource/spartqa-mchoice) |  |
| temporal-nli | Classification | [tasksource/temporal-nli](https://hf.co/datasets/tasksource/temporal-nli) |  |
| riddle_sense | MultipleChoice | [jeggers/riddle_sense](https://hf.co/datasets/jeggers/riddle_sense) |  |
| clcd-english | Classification | [tasksource/clcd-english](https://hf.co/datasets/tasksource/clcd-english) |  |
| twentyquestions | Classification | [tasksource/twentyquestions](https://hf.co/datasets/tasksource/twentyquestions) |  |
| reclor | MultipleChoice | [tasksource/reclor](https://hf.co/datasets/tasksource/reclor) |  |
| counterfactually-augmented-imdb | Classification | [tasksource/counterfactually-augmented-imdb](https://hf.co/datasets/tasksource/counterfactually-augmented-imdb) |  |
| counterfactually-augmented-snli | Classification | [tasksource/counterfactually-augmented-snli](https://hf.co/datasets/tasksource/counterfactually-augmented-snli) |  |
| cnli | Classification | [tasksource/cnli](https://hf.co/datasets/tasksource/cnli) |  |
| boolq-natural-perturbations | Classification | [tasksource/boolq-natural-perturbations](https://hf.co/datasets/tasksource/boolq-natural-perturbations) |  |
| acceptability-prediction | Classification | [tasksource/acceptability-prediction](https://hf.co/datasets/tasksource/acceptability-prediction) | How acceptable is this sentence, from 0 (unacceptable) to 1 (acceptable)? |
| equate | Classification | [tasksource/equate](https://hf.co/datasets/tasksource/equate) |  |
| ScienceQA_text_only | MultipleChoice | [tasksource/ScienceQA_text_only](https://hf.co/datasets/tasksource/ScienceQA_text_only) |  |
| ekar_english | MultipleChoice | [Jiangjie/ekar_english](https://hf.co/datasets/Jiangjie/ekar_english) | Which pair is related in the same way? |
| implicit-hate-stg1 | Classification | [tasksource/implicit-hate-stg1](https://hf.co/datasets/tasksource/implicit-hate-stg1) |  |
| chaos-mnli-ambiguity | Classification | [tasksource/chaos-mnli-ambiguity](https://hf.co/datasets/tasksource/chaos-mnli-ambiguity) | How much would annotators agree on the inference, from 0 (evenly split) to 1 (unanimous)? |
| headline_cause/en_simple | Classification | json |  |
| logiqa-2.0-nli | Classification | [tasksource/logiqa-2.0-nli](https://hf.co/datasets/tasksource/logiqa-2.0-nli) |  |
| oasst2_dense_flat/quality | Classification | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) | How good is the reply, from 0 (low quality) to 1 (high quality)? |
| oasst2_dense_flat/toxicity | Classification | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) | How toxic is the reply, from 0 (not toxic) to 1 (very toxic)? |
| oasst2_dense_flat/helpfulness | Classification | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) | How helpful is the reply, from 0 (unhelpful) to 1 (helpful)? |
| mindgames | Classification | [sileod/mindgames](https://hf.co/datasets/sileod/mindgames) |  |
| universal_dependencies/en_gum/deprel | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/en_ewt/deprel | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/en_lines/deprel | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/en_partut/deprel | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| ambient | Classification | [tasksource/ambient](https://hf.co/datasets/tasksource/ambient) | Is the hypothesis ambiguous? |
| path-naturalness-prediction | MultipleChoice | [tasksource/path-naturalness-prediction](https://hf.co/datasets/tasksource/path-naturalness-prediction) | Which chain of relations is more natural? |
| civil_comments/toxicity | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | Would most raters flag this comment as toxic? |
| civil_comments/severe_toxicity | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | Would most raters flag this comment as severely toxic? |
| civil_comments/obscene | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | Would most raters flag this comment as obscene? |
| civil_comments/threat | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | Would most raters flag this comment as a threat? |
| civil_comments/insult | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | Would most raters flag this comment as insulting? |
| civil_comments/identity_attack | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | Would most raters flag this comment as an identity attack? |
| civil_comments/sexual_explicit | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | Would most raters flag this comment as sexually explicit? |
| cloth | MultipleChoice | [AndyChiang/cloth](https://hf.co/datasets/AndyChiang/cloth) |  |
| dgen | MultipleChoice | [AndyChiang/dgen](https://hf.co/datasets/AndyChiang/dgen) |  |
| I2D2 | Classification | [tasksource/I2D2](https://hf.co/datasets/tasksource/I2D2) |  |
| args_me | Classification | [webis/args_me](https://hf.co/datasets/webis/args_me) |  |
| Touche23-ValueEval | Classification | csv |  |
| starcon | Classification | [tasksource/starcon](https://hf.co/datasets/tasksource/starcon) |  |
| banking77 | Classification | [legacy-datasets/banking77](https://hf.co/datasets/legacy-datasets/banking77) |  |
| it-support-tickets | Classification | [tasksource/it-support-tickets](https://hf.co/datasets/tasksource/it-support-tickets) |  |
| ConTRoL-nli | Classification | [tasksource/ConTRoL-nli](https://hf.co/datasets/tasksource/ConTRoL-nli) |  |
| tracie | Classification | [tasksource/tracie](https://hf.co/datasets/tasksource/tracie) |  |
| sherliic | Classification | [tasksource/sherliic](https://hf.co/datasets/tasksource/sherliic) |  |
| sen-making/1 | MultipleChoice | [tasksource/sen-making](https://hf.co/datasets/tasksource/sen-making) | Which statement makes sense? |
| sen-making/2 | MultipleChoice | [tasksource/sen-making](https://hf.co/datasets/tasksource/sen-making) | Why is this statement implausible? |
| winowhy | Classification | [tasksource/winowhy](https://hf.co/datasets/tasksource/winowhy) | Is this explanation correct? |
| robustLR | Classification | [tasksource/robustLR](https://hf.co/datasets/tasksource/robustLR) |  |
| clutrr | Classification | [tasksource/clutrr](https://hf.co/datasets/tasksource/clutrr) |  |
| logical-fallacy | Classification | [tasksource/logical-fallacy](https://hf.co/datasets/tasksource/logical-fallacy) |  |
| parade | Classification | [tasksource/parade](https://hf.co/datasets/tasksource/parade) |  |
| cladder | Classification | [tasksource/cladder](https://hf.co/datasets/tasksource/cladder) |  |
| subjectivity | Classification | [tasksource/subjectivity](https://hf.co/datasets/tasksource/subjectivity) |  |
| MOH | Classification | [tasksource/MOH](https://hf.co/datasets/tasksource/MOH) |  |
| VUAC | Classification | [tasksource/VUAC](https://hf.co/datasets/tasksource/VUAC) |  |
| TroFi | Classification | parquet |  |
| sharc | Classification | [tasksource/sharc](https://hf.co/datasets/tasksource/sharc) |  |
| conceptrules_v2 | Classification | [tasksource/conceptrules_v2](https://hf.co/datasets/tasksource/conceptrules_v2) | Is the statement true given the context? |
| disrpt/eng.dep.scidtb.rels | Classification | [multilingual-discourse-hub/disrpt](https://hf.co/datasets/multilingual-discourse-hub/disrpt) |  |
| conll2000 | TokenClassification | [eriktks/conll2000](https://hf.co/datasets/eriktks/conll2000) |  |
| few-nerd/supervised | TokenClassification | [DFKI-SLT/few-nerd](https://hf.co/datasets/DFKI-SLT/few-nerd) |  |
| finer-139 | TokenClassification | [nlpaueb/finer-139](https://hf.co/datasets/nlpaueb/finer-139) |  |
| zero-shot-label-nli | Classification | [tasksource/zero-shot-label-nli](https://hf.co/datasets/tasksource/zero-shot-label-nli) |  |
| com2sense | Classification | [tasksource/com2sense](https://hf.co/datasets/tasksource/com2sense) |  |
| scone | Classification | [tasksource/scone](https://hf.co/datasets/tasksource/scone) |  |
| winodict | MultipleChoice | [tasksource/winodict](https://hf.co/datasets/tasksource/winodict) |  |
| fool-me-twice | Classification | [tasksource/fool-me-twice](https://hf.co/datasets/tasksource/fool-me-twice) |  |
| monli | Classification | [tasksource/monli](https://hf.co/datasets/tasksource/monli) |  |
| corr2cause | Classification | [tasksource/corr2cause](https://hf.co/datasets/tasksource/corr2cause) |  |
| lsat_qa/all | MultipleChoice | [lighteval/lsat_qa](https://hf.co/datasets/lighteval/lsat_qa) |  |
| apt | Classification | [tasksource/apt](https://hf.co/datasets/tasksource/apt) |  |
| twitter-financial-news-sentiment | Classification | [zeroshot/twitter-financial-news-sentiment](https://hf.co/datasets/zeroshot/twitter-financial-news-sentiment) |  |
| icl-symbol-tuning-instruct | Classification | [tasksource/icl-symbol-tuning-instruct](https://hf.co/datasets/tasksource/icl-symbol-tuning-instruct) | Is this the right label for the last input? |
| SpaceNLI | Classification | [tasksource/SpaceNLI](https://hf.co/datasets/tasksource/SpaceNLI) |  |
| propsegment/nli | Classification | json |  |
| HatemojiBuild | Classification | [HannahRoseKirk/HatemojiBuild](https://hf.co/datasets/HannahRoseKirk/HatemojiBuild) |  |
| regset | Classification | [tasksource/regset](https://hf.co/datasets/tasksource/regset) | Does the string match the regular expression? |
| esci | Classification | [tasksource/esci](https://hf.co/datasets/tasksource/esci) |  |
| chatbot_arena_conversations | MultipleChoice | [lmsys/chatbot_arena_conversations](https://hf.co/datasets/lmsys/chatbot_arena_conversations) | Which assistant did the user prefer? |
| dnd_style_intents | Classification | [neurae/dnd_style_intents](https://hf.co/datasets/neurae/dnd_style_intents) |  |
| FLD.v2/default | Classification | [hitachi-nlp/FLD.v2](https://hf.co/datasets/hitachi-nlp/FLD.v2) |  |
| FLD.v2/star | Classification | [hitachi-nlp/FLD.v2](https://hf.co/datasets/hitachi-nlp/FLD.v2) |  |
| SDOH-NLI | Classification | [tasksource/SDOH-NLI](https://hf.co/datasets/tasksource/SDOH-NLI) |  |
| scifact_entailment | Classification | [tasksource/scifact_entailment](https://hf.co/datasets/tasksource/scifact_entailment) |  |
| feasibilityQA | Classification | [tasksource/feasibilityQA](https://hf.co/datasets/tasksource/feasibilityQA) |  |
| simple_pair | Classification | [tasksource/simple_pair](https://hf.co/datasets/tasksource/simple_pair) |  |
| AdjectiveScaleProbe-nli | Classification | [tasksource/AdjectiveScaleProbe-nli](https://hf.co/datasets/tasksource/AdjectiveScaleProbe-nli) |  |
| resnli | Classification | [tasksource/resnli](https://hf.co/datasets/tasksource/resnli) |  |
| SpaRTUN | MultipleChoice | [tasksource/SpaRTUN](https://hf.co/datasets/tasksource/SpaRTUN) |  |
| ReSQ | MultipleChoice | [tasksource/ReSQ](https://hf.co/datasets/tasksource/ReSQ) |  |
| semantic_fragments_nli | Classification | [tasksource/semantic_fragments_nli](https://hf.co/datasets/tasksource/semantic_fragments_nli) |  |
| dataset_train_nli | Classification | [MoritzLaurer/dataset_train_nli](https://hf.co/datasets/MoritzLaurer/dataset_train_nli) |  |
| stepgame | Classification | [tasksource/stepgame](https://hf.co/datasets/tasksource/stepgame) |  |
| nlgraph | Classification | [tasksource/nlgraph](https://hf.co/datasets/tasksource/nlgraph) |  |
| oasst2_pairwise_rlhf_reward | MultipleChoice | [tasksource/oasst2_pairwise_rlhf_reward](https://hf.co/datasets/tasksource/oasst2_pairwise_rlhf_reward) | Which reply is better? |
| hh-rlhf/helpful-online | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | Which next assistant reply is more helpful? |
| hh-rlhf/helpful-base | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | Which next assistant reply is more helpful? |
| hh-rlhf/helpful-rejection-sampled | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | Which next assistant reply is more helpful? |
| hh-rlhf/harmless-base | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | Which next assistant reply is more harmless? |
| ruletaker | Classification | [tasksource/ruletaker](https://hf.co/datasets/tasksource/ruletaker) | Does the statement follow from the context? What is not explicitly stated as true is considered false. |
| PARARULE-Plus | Classification | [qbao775/PARARULE-Plus](https://hf.co/datasets/qbao775/PARARULE-Plus) | Is the statement true? What is not explicitly stated as true is considered false. |
| proofwriter | Classification | [tasksource/proofwriter](https://hf.co/datasets/tasksource/proofwriter) |  |
| logical-entailment | Classification | [tasksource/logical-entailment](https://hf.co/datasets/tasksource/logical-entailment) |  |
| nope | Classification | [tasksource/nope](https://hf.co/datasets/tasksource/nope) |  |
| LogicNLI | Classification | [tasksource/LogicNLI](https://hf.co/datasets/tasksource/LogicNLI) |  |
| contract-nli/contractnli_a/seg | Classification | [tasksource/contract-nli](https://hf.co/datasets/tasksource/contract-nli) |  |
| contract-nli/contractnli_b/full | Classification | [tasksource/contract-nli](https://hf.co/datasets/tasksource/contract-nli) |  |
| nli4ct_semeval2024 | Classification | [AshtonIsNotHere/nli4ct_semeval2024](https://hf.co/datasets/AshtonIsNotHere/nli4ct_semeval2024) |  |
| lsat-ar | MultipleChoice | [tasksource/lsat-ar](https://hf.co/datasets/tasksource/lsat-ar) |  |
| lsat-rc | MultipleChoice | [tasksource/lsat-rc](https://hf.co/datasets/tasksource/lsat-rc) |  |
| biosift-nli | Classification | [AshtonIsNotHere/biosift-nli](https://hf.co/datasets/AshtonIsNotHere/biosift-nli) |  |
| brainteasers/SP | MultipleChoice | [tasksource/brainteasers](https://hf.co/datasets/tasksource/brainteasers) |  |
| brainteasers/WP | MultipleChoice | [tasksource/brainteasers](https://hf.co/datasets/tasksource/brainteasers) |  |
| toxigen-data/annotated | Classification | [skg/toxigen-data](https://hf.co/datasets/skg/toxigen-data) |  |
| persuasion | Classification | [Anthropic/persuasion](https://hf.co/datasets/Anthropic/persuasion) |  |
| AmbigNQ-clarifying-question | Classification | [erbacher/AmbigNQ-clarifying-question](https://hf.co/datasets/erbacher/AmbigNQ-clarifying-question) |  |
| SIGA-nli | Classification | [tasksource/SIGA-nli](https://hf.co/datasets/tasksource/SIGA-nli) |  |
| FOL-nli | Classification | [unigram/FOL-nli](https://hf.co/datasets/unigram/FOL-nli) |  |
| goal-step-wikihow/goal | MultipleChoice | [tasksource/goal-step-wikihow](https://hf.co/datasets/tasksource/goal-step-wikihow) |  |
| goal-step-wikihow/step | MultipleChoice | [tasksource/goal-step-wikihow](https://hf.co/datasets/tasksource/goal-step-wikihow) |  |
| goal-step-wikihow/order | MultipleChoice | [tasksource/goal-step-wikihow](https://hf.co/datasets/tasksource/goal-step-wikihow) |  |
| PARADISE | MultipleChoice | [GGLab/PARADISE](https://hf.co/datasets/GGLab/PARADISE) |  |
| doc-nli | Classification | [tasksource/doc-nli](https://hf.co/datasets/tasksource/doc-nli) |  |
| mctest-nli | Classification | [tasksource/mctest-nli](https://hf.co/datasets/tasksource/mctest-nli) |  |
| patent-phrase-similarity | Classification | [tasksource/patent-phrase-similarity](https://hf.co/datasets/tasksource/patent-phrase-similarity) |  |
| natural-language-satisfiability | Classification | [tasksource/natural-language-satisfiability](https://hf.co/datasets/tasksource/natural-language-satisfiability) |  |
| idioms-nli | Classification | [tasksource/idioms-nli](https://hf.co/datasets/tasksource/idioms-nli) |  |
| lifecycle-entailment | Classification | [tasksource/lifecycle-entailment](https://hf.co/datasets/tasksource/lifecycle-entailment) |  |
| toxic-chat/toxicchat0124/toxicity | Classification | [lmsys/toxic-chat](https://hf.co/datasets/lmsys/toxic-chat) | Is this user prompt toxic? |
| toxic-chat/toxicchat0124/jailbreaking | Classification | [lmsys/toxic-chat](https://hf.co/datasets/lmsys/toxic-chat) | Is this user prompt a jailbreak attempt? |
| clinc_oos/plus | Classification | [clinc/clinc_oos](https://hf.co/datasets/clinc/clinc_oos) |  |
| few_rel/default | Classification | [tasksource/few_rel](https://hf.co/datasets/tasksource/few_rel) |  |
| docred | Classification | json |  |
| chemprot/chemprot_full_source | Classification | [bigbio/chemprot](https://hf.co/datasets/bigbio/chemprot) |  |
| PKU-SafeRLHF/helpfulness | MultipleChoice | [PKU-Alignment/PKU-SafeRLHF](https://hf.co/datasets/PKU-Alignment/PKU-SafeRLHF) | Which response is more helpful? |
| PKU-SafeRLHF/safety | MultipleChoice | [PKU-Alignment/PKU-SafeRLHF](https://hf.co/datasets/PKU-Alignment/PKU-SafeRLHF) | Which response is safer? |
| HelpSteer/helpfulness | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | How would you rate the helpfulness of the response? |
| HelpSteer/correctness | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | How would you rate the correctness of the response? |
| HelpSteer/coherence | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | How would you rate the coherence of the response? |
| HelpSteer/complexity | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | How would you rate the complexity of the response? |
| HelpSteer/verbosity | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | How would you rate the verbosity of the response? |
| HelpSteer2/helpfulness | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | How would you rate the helpfulness of the response? |
| HelpSteer2/correctness | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | How would you rate the correctness of the response? |
| HelpSteer2/coherence | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | How would you rate the coherence of the response? |
| HelpSteer2/complexity | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | How would you rate the complexity of the response? |
| HelpSteer2/verbosity | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | How would you rate the verbosity of the response? |
| HelpSteer3/preference | MultipleChoice | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | Which next assistant reply is better? |
| HelpSteer3/principle | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) |  |
| HelpSteer3/edit_quality | MultipleChoice | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | Which edit improves the reply? |
| HelpSteer3/feedback | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | How helpful is the assistant reply? |
| MSciNLI | Classification | [sadat2307/MSciNLI](https://hf.co/datasets/sadat2307/MSciNLI) |  |
| UltraFeedback-paired | MultipleChoice | [pushpdeep/UltraFeedback-paired](https://hf.co/datasets/pushpdeep/UltraFeedback-paired) | Which response is better? |
| prm800k_dpo/solution | MultipleChoice | [tasksource/prm800k_dpo](https://hf.co/datasets/tasksource/prm800k_dpo) | Which solution is correct? |
| prm800k_dpo/step | MultipleChoice | [tasksource/prm800k_dpo](https://hf.co/datasets/tasksource/prm800k_dpo) | Which next step is correct? |
| AES2-essay-scoring | Classification | [tasksource/AES2-essay-scoring](https://hf.co/datasets/tasksource/AES2-essay-scoring) | What holistic score does this student essay deserve? |
| argument-feedback | Classification | [tasksource/argument-feedback](https://hf.co/datasets/tasksource/argument-feedback) | How effective is this element of the student's argument? |
| english-grading/cohesion | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | What cohesion score does this English learner essay deserve? |
| english-grading/syntax | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | What syntax score does this English learner essay deserve? |
| english-grading/vocabulary | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | What vocabulary score does this English learner essay deserve? |
| english-grading/phraseology | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | What phraseology score does this English learner essay deserve? |
| english-grading/grammar | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | What grammar score does this English learner essay deserve? |
| english-grading/conventions | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | What conventions score does this English learner essay deserve? |
| wice | Classification | [tasksource/wice](https://hf.co/datasets/tasksource/wice) |  |
| hover | Classification | [Dzeniks/hover](https://hf.co/datasets/Dzeniks/hover) |  |
| hover-3way/nli | Classification | [Dzeniks/hover-3way](https://hf.co/datasets/Dzeniks/hover-3way) |  |
| tasksource_dpo_pairs | MultipleChoice | [tasksource/tasksource_dpo_pairs](https://hf.co/datasets/tasksource/tasksource_dpo_pairs) | Which response is better? |
| seahorse_summarization_evaluation | Classification | [tasksource/seahorse_summarization_evaluation](https://hf.co/datasets/tasksource/seahorse_summarization_evaluation) |  |
| missing-item-prediction/contrastive | Classification | [sileod/missing-item-prediction](https://hf.co/datasets/sileod/missing-item-prediction) |  |
| jigsaw_toxicity | Classification | [tasksource/jigsaw_toxicity](https://hf.co/datasets/tasksource/jigsaw_toxicity) |  |
| Pol_NLI | Classification | [mlburnham/Pol_NLI](https://hf.co/datasets/mlburnham/Pol_NLI) |  |
| synthetic-retrieval-NLI/position | Classification | [tasksource/synthetic-retrieval-NLI](https://hf.co/datasets/tasksource/synthetic-retrieval-NLI) |  |
| synthetic-retrieval-NLI/binary | Classification | [tasksource/synthetic-retrieval-NLI](https://hf.co/datasets/tasksource/synthetic-retrieval-NLI) |  |
| synthetic-retrieval-NLI/count | Classification | [tasksource/synthetic-retrieval-NLI](https://hf.co/datasets/tasksource/synthetic-retrieval-NLI) |  |
| github-issue-similarity | Classification | [WhereIsAI/github-issue-similarity](https://hf.co/datasets/WhereIsAI/github-issue-similarity) |  |
