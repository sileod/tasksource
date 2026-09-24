504 English tasks. Load one with `load_task(id)`; the annotations are in [tasks.py](src/tasksource/tasks.py), and tasks kept out on purpose are in [parked.py](src/tasksource/parked.py).

| id | type | dataset | config | question | fields |
|---|---|---|---|---|---|
| glue/mnli | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) | mnli |  | sentence1=premise, sentence2=hypothesis, labels=label, splits=[train, None, validation_matched] |
| glue/qnli | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) | qnli |  | sentence1=question, sentence2=sentence, labels=label |
| glue/rte | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) | rte |  | labels=label |
| glue/wnli | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) | wnli |  | labels=label |
| glue/mrpc | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) | mrpc |  | labels=label |
| glue/qqp | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) | qqp |  | sentence1=question1, sentence2=question2, labels=label |
| glue/stsb | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) | stsb | How similar are the two sentences, from 0 (unrelated) to 5 (equivalent)? | labels=label |
| super_glue/boolq | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) | boolq |  | sentence1=question, labels=label |
| super_glue/boolq_passage | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) | boolq |  | sentence1=passage, sentence2=question, labels=label |
| super_glue/cb | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) | cb |  | sentence1=premise, sentence2=hypothesis, labels=label |
| super_glue/multirc | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) | multirc |  | sentence1=fn, sentence2=answer, labels=fn |
| super_glue/wic | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) | wic |  | sentence1=fn, labels=fn |
| super_glue/axg | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) | axg |  | sentence1=premise, sentence2=hypothesis, labels=label, splits=[test, None, None] |
| anli/a1 | Classification | [facebook/anli](https://hf.co/datasets/facebook/anli) |  |  | sentence1=premise, sentence2=hypothesis, labels=label, splits=[train_r1, dev_r1, test_r1] |
| anli/a2 | Classification | [facebook/anli](https://hf.co/datasets/facebook/anli) |  |  | sentence1=premise, sentence2=hypothesis, labels=label, splits=[train_r2, dev_r2, test_r2] |
| anli/a3 | Classification | [facebook/anli](https://hf.co/datasets/facebook/anli) |  |  | sentence1=premise, sentence2=hypothesis, labels=label, splits=[train_r3, dev_r3, test_r3] |
| babi_nli/counting | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) | counting |  | sentence1=premise, sentence2=hypothesis, labels=label |
| babi_nli/indefinite-knowledge | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) | indefinite-knowledge |  | sentence1=premise, sentence2=hypothesis, labels=label |
| babi_nli/lists-sets | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) | lists-sets |  | sentence1=premise, sentence2=hypothesis, labels=label |
| babi_nli/path-finding | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) | path-finding |  | sentence1=premise, sentence2=hypothesis, labels=label |
| babi_nli/positional-reasoning | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) | positional-reasoning |  | sentence1=premise, sentence2=hypothesis, labels=label |
| babi_nli/simple-negation | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) | simple-negation |  | sentence1=premise, sentence2=hypothesis, labels=label |
| babi_nli/size-reasoning | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) | size-reasoning |  | sentence1=premise, sentence2=hypothesis, labels=label |
| babi_nli/conjunction | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) | conjunction |  | sentence1=premise, sentence2=hypothesis, labels=label |
| babi_nli/three-arg-relations | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) | three-arg-relations |  | sentence1=premise, sentence2=hypothesis, labels=label |
| babi_nli/three-supporting-facts | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) | three-supporting-facts |  | sentence1=premise, sentence2=hypothesis, labels=label |
| babi_nli/time-reasoning | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) | time-reasoning |  | sentence1=premise, sentence2=hypothesis, labels=label |
| babi_nli/two-arg-relations | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) | two-arg-relations |  | sentence1=premise, sentence2=hypothesis, labels=label |
| babi_nli/single-supporting-fact | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) | single-supporting-fact |  | sentence1=premise, sentence2=hypothesis, labels=label |
| babi_nli/compound-coreference | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) | compound-coreference |  | sentence1=premise, sentence2=hypothesis, labels=label |
| babi_nli/basic-deduction | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) | basic-deduction |  | sentence1=premise, sentence2=hypothesis, labels=label |
| babi_nli/basic-coreference | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) | basic-coreference |  | sentence1=premise, sentence2=hypothesis, labels=label |
| babi_nli/two-supporting-facts | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) | two-supporting-facts |  | sentence1=premise, sentence2=hypothesis, labels=label |
| babi_nli/basic-induction | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) | basic-induction |  | sentence1=premise, sentence2=hypothesis, labels=label |
| babi_nli/yes-no-questions | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) | yes-no-questions |  | sentence1=premise, sentence2=hypothesis, labels=label |
| sick/label | Classification | [tasksource/sick](https://hf.co/datasets/tasksource/sick) |  |  | sentence1=sentence_A, sentence2=sentence_B, labels=label |
| sick/relatedness | Classification | [tasksource/sick](https://hf.co/datasets/tasksource/sick) |  | How related are the two sentences, from 1 (unrelated) to 5 (very related)? | sentence1=sentence_A, sentence2=sentence_B, labels=relatedness_score |
| snli | Classification | [stanfordnlp/snli](https://hf.co/datasets/stanfordnlp/snli) |  |  | sentence1=premise, sentence2=hypothesis, labels=label, post_process |
| scitail/snli_format | Classification | [allenai/scitail](https://hf.co/datasets/allenai/scitail) | snli_format |  | labels=gold_label |
| hans | Classification | [tasksource/hans](https://hf.co/datasets/tasksource/hans) |  |  |  |
| WANLI | Classification | [alisawuffles/WANLI](https://hf.co/datasets/alisawuffles/WANLI) |  |  | sentence1=premise, sentence2=hypothesis, labels=gold |
| recast/recast_factuality | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) | recast_factuality |  | sentence1=context, sentence2=hypothesis, labels=label |
| recast/recast_verbnet | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) | recast_verbnet |  | sentence1=context, sentence2=hypothesis, labels=label |
| recast/recast_puns | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) | recast_puns |  | sentence1=context, sentence2=hypothesis, labels=label |
| recast/recast_ner | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) | recast_ner |  | sentence1=context, sentence2=hypothesis, labels=label |
| recast/recast_sentiment | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) | recast_sentiment |  | sentence1=context, sentence2=hypothesis, labels=label |
| recast/recast_megaveridicality | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) | recast_megaveridicality |  | sentence1=context, sentence2=hypothesis, labels=label |
| recast/recast_verbcorner | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) | recast_verbcorner |  | sentence1=context, sentence2=hypothesis, labels=label |
| probability_words_nli/usnli | Classification | [sileod/probability_words_nli](https://hf.co/datasets/sileod/probability_words_nli) | usnli |  | sentence1=context, sentence2=hypothesis, labels=label |
| probability_words_nli/reasoning_2hop | Classification | [sileod/probability_words_nli](https://hf.co/datasets/sileod/probability_words_nli) | reasoning_2hop |  | sentence1=context, sentence2=hypothesis, labels=label |
| probability_words_nli/reasoning_1hop | Classification | [sileod/probability_words_nli](https://hf.co/datasets/sileod/probability_words_nli) | reasoning_1hop |  | sentence1=context, sentence2=hypothesis, labels=label |
| nan-nli | Classification | [joey234/nan-nli](https://hf.co/datasets/joey234/nan-nli) |  |  | sentence1=premise, sentence2=hypothesis, labels=label |
| nli_fever | Classification | [pietrolesci/nli_fever](https://hf.co/datasets/pietrolesci/nli_fever) |  |  | sentence1=premise, sentence2=hypothesis, labels=label, splits=[train, dev, None] |
| breaking_nli | Classification | [pietrolesci/breaking_nli](https://hf.co/datasets/pietrolesci/breaking_nli) |  |  | labels=label, splits=[full, None, None], label_values={…} |
| conj_nli | Classification | [pietrolesci/conj_nli](https://hf.co/datasets/pietrolesci/conj_nli) |  |  | sentence1=premise, sentence2=hypothesis, labels=label, splits=[train, dev, None], label_values={…}, post_process |
| fracas | Classification | [pietrolesci/fracas](https://hf.co/datasets/pietrolesci/fracas) |  |  | sentence1=premise, sentence2=hypothesis, labels=label, label_values={…} |
| dialogue_nli | Classification | [pietrolesci/dialogue_nli](https://hf.co/datasets/pietrolesci/dialogue_nli) |  |  | labels=label, label_values={…} |
| mpe | Classification | [pietrolesci/mpe](https://hf.co/datasets/pietrolesci/mpe) |  |  | sentence1=premise, sentence2=hypothesis, labels=label, splits=[train, dev, test], label_values={…} |
| dnc | Classification | [pietrolesci/dnc](https://hf.co/datasets/pietrolesci/dnc) |  |  | sentence1=context, sentence2=hypothesis, labels=label, label_values={…} |
| recast_white/fnplus | Classification | [pietrolesci/recast_white](https://hf.co/datasets/pietrolesci/recast_white) |  |  | sentence1=text, sentence2=hypothesis, labels=label, splits=[fnplus, None, None], label_values={…} |
| recast_white/sprl | Classification | [pietrolesci/recast_white](https://hf.co/datasets/pietrolesci/recast_white) |  |  | sentence1=text, sentence2=hypothesis, labels=label, splits=[sprl, None, None], label_values={…} |
| recast_white/dpr | Classification | [pietrolesci/recast_white](https://hf.co/datasets/pietrolesci/recast_white) |  |  | sentence1=text, sentence2=hypothesis, labels=label, splits=[dpr, None, None], label_values={…} |
| joci | Classification | [pietrolesci/joci](https://hf.co/datasets/pietrolesci/joci) |  |  | sentence1=context, sentence2=hypothesis, labels=fn, splits=[full, None, None], pre_process |
| robust_nli/IS_CS | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |  | sentence1=premise, sentence2=hypothesis, labels=label, splits=[IS_CS, None, None], label_values={…} |
| robust_nli/LI_LI | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |  | sentence1=premise, sentence2=hypothesis, labels=label, splits=[LI_LI, None, None], label_values={…} |
| robust_nli/ST_WO | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |  | sentence1=premise, sentence2=hypothesis, labels=label, splits=[ST_WO, None, None], label_values={…} |
| robust_nli/PI_SP | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |  | sentence1=premise, sentence2=hypothesis, labels=label, splits=[PI_SP, None, None], label_values={…} |
| robust_nli/PI_CD | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |  | sentence1=premise, sentence2=hypothesis, labels=label, splits=[PI_CD, None, None], label_values={…} |
| robust_nli/ST_SE | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |  | sentence1=premise, sentence2=hypothesis, labels=label, splits=[ST_SE, None, None], label_values={…} |
| robust_nli/ST_NE | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |  | sentence1=premise, sentence2=hypothesis, labels=label, splits=[ST_NE, None, None], label_values={…} |
| robust_nli/ST_LM | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |  | sentence1=premise, sentence2=hypothesis, labels=label, splits=[ST_LM, None, None], label_values={…} |
| robust_nli_is_sd | Classification | [pietrolesci/robust_nli_is_sd](https://hf.co/datasets/pietrolesci/robust_nli_is_sd) |  |  | sentence1=premise, sentence2=hypothesis, labels=label, label_values={…} |
| robust_nli_li_ts | Classification | [pietrolesci/robust_nli_li_ts](https://hf.co/datasets/pietrolesci/robust_nli_li_ts) |  |  | sentence1=premise, sentence2=hypothesis, labels=label, label_values={…} |
| gen_debiased_nli/snli_seq_z | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |  | sentence1=premise, sentence2=hypothesis, labels=label, splits=[snli_seq_z, None, None], label_values={…} |
| gen_debiased_nli/snli_z_aug | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |  | sentence1=premise, sentence2=hypothesis, labels=label, splits=[snli_z_aug, None, None], label_values={…} |
| gen_debiased_nli/snli_par_z | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |  | sentence1=premise, sentence2=hypothesis, labels=label, splits=[snli_par_z, None, None], label_values={…} |
| gen_debiased_nli/mnli_par_z | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |  | sentence1=premise, sentence2=hypothesis, labels=label, splits=[mnli_par_z, None, None], label_values={…} |
| gen_debiased_nli/mnli_z_aug | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |  | sentence1=premise, sentence2=hypothesis, labels=label, splits=[mnli_z_aug, None, None], label_values={…} |
| gen_debiased_nli/mnli_seq_z | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |  | sentence1=premise, sentence2=hypothesis, labels=label, splits=[mnli_seq_z, None, None], label_values={…} |
| add_one_rte | Classification | [pietrolesci/add_one_rte](https://hf.co/datasets/pietrolesci/add_one_rte) |  |  | sentence1=premise, sentence2=hypothesis, labels=label, splits=[train, dev, test], label_values={…} |
| imppres/presupposition_question_presupposition/presupposition | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) | presupposition_question_presupposition |  | sentence1=premise, sentence2=hypothesis, labels=gold_label, post_process |
| imppres/presupposition_possessed_definites_uniqueness/presupposition | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) | presupposition_possessed_definites_uniqueness |  | sentence1=premise, sentence2=hypothesis, labels=gold_label, post_process |
| imppres/presupposition_possessed_definites_existence/presupposition | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) | presupposition_possessed_definites_existence |  | sentence1=premise, sentence2=hypothesis, labels=gold_label, post_process |
| imppres/presupposition_only_presupposition/presupposition | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) | presupposition_only_presupposition |  | sentence1=premise, sentence2=hypothesis, labels=gold_label, post_process |
| imppres/presupposition_cleft_uniqueness/presupposition | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) | presupposition_cleft_uniqueness |  | sentence1=premise, sentence2=hypothesis, labels=gold_label, post_process |
| imppres/presupposition_cleft_existence/presupposition | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) | presupposition_cleft_existence |  | sentence1=premise, sentence2=hypothesis, labels=gold_label, post_process |
| imppres/presupposition_change_of_state/presupposition | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) | presupposition_change_of_state |  | sentence1=premise, sentence2=hypothesis, labels=gold_label, post_process |
| imppres/presupposition_both_presupposition/presupposition | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) | presupposition_both_presupposition |  | sentence1=premise, sentence2=hypothesis, labels=gold_label, post_process |
| imppres/presupposition_all_n_presupposition/presupposition | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) | presupposition_all_n_presupposition |  | sentence1=premise, sentence2=hypothesis, labels=gold_label, post_process |
| imppres/implicature_numerals_2_3/prag | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) | implicature_numerals_2_3 |  | sentence1=premise, sentence2=hypothesis, labels=gold_label_prag, post_process |
| imppres/implicature_numerals_10_100/prag | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) | implicature_numerals_10_100 |  | sentence1=premise, sentence2=hypothesis, labels=gold_label_prag, post_process |
| imppres/implicature_modals/prag | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) | implicature_modals |  | sentence1=premise, sentence2=hypothesis, labels=gold_label_prag, post_process |
| imppres/implicature_gradable_verb/prag | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) | implicature_gradable_verb |  | sentence1=premise, sentence2=hypothesis, labels=gold_label_prag, post_process |
| imppres/implicature_gradable_adjective/prag | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) | implicature_gradable_adjective |  | sentence1=premise, sentence2=hypothesis, labels=gold_label_prag, post_process |
| imppres/implicature_connectives/prag | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) | implicature_connectives |  | sentence1=premise, sentence2=hypothesis, labels=gold_label_prag, post_process |
| imppres/implicature_quantifiers/prag | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) | implicature_quantifiers |  | sentence1=premise, sentence2=hypothesis, labels=gold_label_prag, post_process |
| imppres/implicature_modals/log | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) | implicature_modals |  | sentence1=premise, sentence2=hypothesis, labels=gold_label_log, post_process |
| imppres/implicature_gradable_verb/log | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) | implicature_gradable_verb |  | sentence1=premise, sentence2=hypothesis, labels=gold_label_log, post_process |
| imppres/implicature_gradable_adjective/log | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) | implicature_gradable_adjective |  | sentence1=premise, sentence2=hypothesis, labels=gold_label_log, post_process |
| imppres/implicature_quantifiers/log | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) | implicature_quantifiers |  | sentence1=premise, sentence2=hypothesis, labels=gold_label_log, post_process |
| imppres/implicature_numerals_2_3/log | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) | implicature_numerals_2_3 |  | sentence1=premise, sentence2=hypothesis, labels=gold_label_log, post_process |
| imppres/implicature_connectives/log | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) | implicature_connectives |  | sentence1=premise, sentence2=hypothesis, labels=gold_label_log, post_process |
| imppres/implicature_numerals_10_100/log | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) | implicature_numerals_10_100 |  | sentence1=premise, sentence2=hypothesis, labels=gold_label_log, post_process |
| hlgd | Classification | [tasksource/hlgd](https://hf.co/datasets/tasksource/hlgd) |  |  | sentence1=headline_a, sentence2=headline_b, labels=label |
| paws/labeled_final | Classification | [google-research-datasets/paws](https://hf.co/datasets/google-research-datasets/paws) | labeled_final |  | labels=fn |
| paws/labeled_swap | Classification | [google-research-datasets/paws](https://hf.co/datasets/google-research-datasets/paws) | labeled_swap |  | labels=fn, splits=[train, None, None] |
| medical_questions_pairs | Classification | [curaihealth/medical_questions_pairs](https://hf.co/datasets/curaihealth/medical_questions_pairs) |  |  | sentence1=question_1, sentence2=question_2, labels=fn |
| conll2003/pos_tags | TokenClassification | [tomaarsen/conll2003](https://hf.co/datasets/tomaarsen/conll2003) |  |  | labels=pos_tags |
| conll2003/chunk_tags | TokenClassification | [tomaarsen/conll2003](https://hf.co/datasets/tomaarsen/conll2003) |  |  | labels=chunk_tags |
| conll2003/ner_tags | TokenClassification | [tomaarsen/conll2003](https://hf.co/datasets/tomaarsen/conll2003) |  |  | labels=ner_tags |
| fig-qa | MultipleChoice | [nightingal3/fig-qa](https://hf.co/datasets/nightingal3/fig-qa) |  |  | inputs=startphrase, splits=[train, validation, None], choice0=ending1, choice1=ending2 |
| cos_e/v1.0 | MultipleChoice | [Salesforce/cos_e](https://hf.co/datasets/Salesforce/cos_e) | v1.0 |  | inputs=question, labels=fn, choices_list=choices |
| cosmos_qa | MultipleChoice | [Samsoup/cosmos_qa](https://hf.co/datasets/Samsoup/cosmos_qa) |  |  | inputs=fn, labels=label, choice0=answer0, choice1=answer1, choice2=answer2, choice3=answer3 |
| dream | MultipleChoice | [dataset-org/dream](https://hf.co/datasets/dataset-org/dream) |  |  | inputs=fn, labels=fn, choices_list=choice |
| openbookqa | MultipleChoice | [allenai/openbookqa](https://hf.co/datasets/allenai/openbookqa) |  |  | inputs=question_stem, labels=answerKey, choices_list=fn |
| qasc | MultipleChoice | [allenai/qasc](https://hf.co/datasets/allenai/qasc) |  |  | inputs=question, labels=fn, choices_list=fn, splits=[train, validation, None] |
| quartz | MultipleChoice | [allenai/quartz](https://hf.co/datasets/allenai/quartz) |  |  | inputs=question, labels=answerKey, choices_list=fn |
| quail | MultipleChoice | [textmachinelab/quail](https://hf.co/datasets/textmachinelab/quail) |  |  | inputs=fn, labels=correct_answer_id, choices_list=answers |
| head_qa/en | MultipleChoice | [EleutherAI/headqa](https://hf.co/datasets/EleutherAI/headqa) | en |  | inputs=qtext, labels=fn, choices_list=fn |
| sciq | MultipleChoice | [allenai/sciq](https://hf.co/datasets/allenai/sciq) |  |  | inputs=question, labels=fn, choice0=correct_answer, choice1=distractor1, choice2=distractor2, choice3=distractor3 |
| social_i_qa | MultipleChoice | [tasksource/social_i_qa](https://hf.co/datasets/tasksource/social_i_qa) |  |  | inputs=fn, labels=label, choice0=answerA, choice1=answerB, choice2=answerC |
| wiki_hop/original | MultipleChoice | [MoE-UNC/wikihop](https://hf.co/datasets/MoE-UNC/wikihop) | default |  | inputs=fn, labels=fn, choices_list=candidates |
| wiqa | MultipleChoice | [tasksource/wiqa](https://hf.co/datasets/tasksource/wiqa) |  |  | inputs=question_stem, labels=answer_label_as_choice, choices_list=fn |
| piqa | MultipleChoice | [baber/piqa](https://hf.co/datasets/baber/piqa) |  |  | inputs=goal, labels=label, choice0=sol1, choice1=sol2 |
| hellaswag | MultipleChoice | [Rowan/hellaswag](https://hf.co/datasets/Rowan/hellaswag) |  |  | inputs=fn, labels=label, choices_list=fn, splits=[train, validation, None] |
| super_glue/copa | MultipleChoice | [aps/super_glue](https://hf.co/datasets/aps/super_glue) | copa |  | inputs=fn, labels=label, choice0=choice1, choice1=choice2 |
| balanced-copa | MultipleChoice | [pkavumba/balanced-copa](https://hf.co/datasets/pkavumba/balanced-copa) |  |  | inputs=fn, labels=label, choice0=choice1, choice1=choice2 |
| e-CARE | MultipleChoice | [12ml/e-CARE](https://hf.co/datasets/12ml/e-CARE) |  |  | inputs=fn, labels=label, choice0=choice1, choice1=choice2 |
| art | MultipleChoice | [allenai/art](https://hf.co/datasets/allenai/art) |  | What happened in between? | inputs=fn, labels=fn, splits=[train, validation, None], choice0=hypothesis_1, choice1=hypothesis_2 |
| winogrande/winogrande_xl | MultipleChoice | [allenai/winogrande](https://hf.co/datasets/allenai/winogrande) | winogrande_xl |  | inputs=sentence, labels=answer, splits=[train, validation, None], choice0=option1, choice1=option2 |
| codah/codah | MultipleChoice | [jaredfern/codah](https://hf.co/datasets/jaredfern/codah) | codah |  | inputs=question_propmt, labels=correct_answer_idx, choices_list=candidate_answers |
| ai2_arc/ARC-Easy/challenge | MultipleChoice | [allenai/ai2_arc](https://hf.co/datasets/allenai/ai2_arc) | ARC-Easy |  | inputs=question, labels=fn, choices_list=fn |
| ai2_arc/ARC-Challenge/challenge | MultipleChoice | [allenai/ai2_arc](https://hf.co/datasets/allenai/ai2_arc) | ARC-Challenge |  | inputs=question, labels=fn, choices_list=fn |
| definite_pronoun_resolution | MultipleChoice | [community-datasets/definite_pronoun_resolution](https://hf.co/datasets/community-datasets/definite_pronoun_resolution) |  |  | inputs=fn, labels=label, choices_list=candidates, splits=[train, None, test] |
| swag/regular | MultipleChoice | [allenai/swag](https://hf.co/datasets/allenai/swag) | regular |  | inputs=fn, labels=label, choice0=ending0, choice1=ending1, choice2=ending2, choice3=ending3 |
| math_qa | MultipleChoice | [tasksource/math_qa](https://hf.co/datasets/tasksource/math_qa) |  |  | inputs=Problem, labels=fn, choices_list=fn |
| glue/cola | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) | cola |  | sentence1=sentence, labels=label |
| glue/sst2 | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) | sst2 |  | sentence1=sentence, labels=label |
| utilitarianism | Classification | csv |  |  | sentence1=comparison, labels=label, label_values={…}, pre_process |
| amazon_counterfactual/en | Classification | [mteb/amazon_counterfactual](https://hf.co/datasets/mteb/amazon_counterfactual) | en |  | sentence1=text, labels=label_text |
| insincere-questions | Classification | [SetFit/insincere-questions](https://hf.co/datasets/SetFit/insincere-questions) |  |  | sentence1=text, labels=label_text |
| toxic_conversations | Classification | [SetFit/toxic_conversations](https://hf.co/datasets/SetFit/toxic_conversations) |  |  | sentence1=text, labels=label_text |
| TuringBench | Classification | csv |  |  | sentence1=Generation, labels=label, splits=[train, validation, None] |
| trec | Classification | [tasksource/trec](https://hf.co/datasets/tasksource/trec) |  |  | sentence1=text, labels=fine_label |
| vitaminc | Classification | [tals/vitaminc](https://hf.co/datasets/tals/vitaminc) |  |  | sentence1=claim, sentence2=evidence, labels=label |
| hope_edi/english | Classification | csv |  |  | sentence1=text, labels=label, splits=[train, validation, None] |
| rumoureval_2019/RumourEval2019 | Classification | csv |  |  | sentence1=source_text, sentence2=reply_text, labels=label, pre_process |
| ethos/binary | Classification | [SetFit/ethos_binary](https://hf.co/datasets/SetFit/ethos_binary) |  |  | sentence1=text, labels=fn, splits=[train, None, None], pre_process |
| ethos/multilabel | Classification | [tasksource/ethos](https://hf.co/datasets/tasksource/ethos) | multilabel |  | sentence1=comment, sentence2=question, labels=answer, pre_process |
| tweet_eval/emoji | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | emoji |  | sentence1=text, labels=label |
| tweet_eval/emotion | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | emotion |  | sentence1=text, labels=label |
| tweet_eval/hate | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | hate |  | sentence1=text, labels=label |
| tweet_eval/irony | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | irony |  | sentence1=text, labels=label |
| tweet_eval/offensive | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | offensive |  | sentence1=text, labels=label |
| tweet_eval/sentiment | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | sentiment |  | sentence1=text, labels=label |
| tweet_eval/stance_abortion | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | stance_abortion | What stance does the tweet take on abortion? | sentence1=text, labels=label |
| tweet_eval/stance_atheism | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | stance_atheism | What stance does the tweet take on atheism? | sentence1=text, labels=label |
| tweet_eval/stance_climate | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | stance_climate | What stance does the tweet take on climate change? | sentence1=text, labels=label |
| tweet_eval/stance_feminist | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | stance_feminist | What stance does the tweet take on feminism? | sentence1=text, labels=label |
| tweet_eval/stance_hillary | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | stance_hillary | What stance does the tweet take on Hillary Clinton? | sentence1=text, labels=label |
| discovery/discovery | Classification | [sileod/discovery](https://hf.co/datasets/sileod/discovery) | discovery |  | labels=label |
| pragmeval/switchboard | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | switchboard |  | sentence1=sentence, labels=label |
| pragmeval/verifiability | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | verifiability |  | sentence1=sentence, labels=label |
| pragmeval/mrda | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | mrda |  | sentence1=sentence, labels=label |
| pragmeval/emergent | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | emergent |  | labels=label |
| pragmeval/gum | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | gum |  | labels=label |
| pragmeval/pdtb | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | pdtb |  | labels=label |
| pragmeval/persuasiveness-claimtype | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | persuasiveness-claimtype |  | labels=label |
| pragmeval/persuasiveness-premisetype | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | persuasiveness-premisetype |  | labels=label |
| pragmeval/stac | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | stac |  | labels=label |
| pragmeval/sarcasm | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | sarcasm |  | labels=label |
| pragmeval/emobank-arousal | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | emobank-arousal |  | sentence1=sentence, labels=label, label_values={…} |
| pragmeval/emobank-dominance | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | emobank-dominance |  | sentence1=sentence, labels=label, label_values={…} |
| pragmeval/emobank-valence | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | emobank-valence |  | sentence1=sentence, labels=label, label_values={…} |
| pragmeval/squinky-formality | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | squinky-formality |  | sentence1=sentence, labels=label, label_values={…} |
| pragmeval/squinky-implicature | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | squinky-implicature |  | sentence1=sentence, labels=label, label_values={…} |
| pragmeval/squinky-informativeness | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | squinky-informativeness |  | sentence1=sentence, labels=label, label_values={…} |
| pragmeval/persuasiveness-eloquence | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | persuasiveness-eloquence |  | labels=label, label_values={…} |
| pragmeval/persuasiveness-relevance | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | persuasiveness-relevance |  | labels=label, label_values={…} |
| pragmeval/persuasiveness-specificity | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | persuasiveness-specificity |  | labels=label, label_values={…} |
| pragmeval/persuasiveness-strength | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | persuasiveness-strength |  | labels=label, label_values={…} |
| silicone/oasis | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | oasis |  | sentence1=Utterance, labels=Label |
| silicone/sem | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | sem |  | sentence1=Utterance, labels=Label |
| silicone/meld_s | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | meld_s |  | sentence1=Utterance, labels=Label |
| silicone/meld_e | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | meld_e |  | sentence1=Utterance, labels=Label |
| silicone/maptask | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | maptask |  | sentence1=Utterance, labels=Label |
| silicone/dyda_e | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | dyda_e |  | sentence1=Utterance, labels=Label |
| silicone/dyda_da | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | dyda_da |  | sentence1=Utterance, labels=Label |
| silicone/iemocap | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | iemocap |  | sentence1=Utterance, labels=fn, pre_process |
| lex_glue/eurlex | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) | eurlex |  | sentence1=text |
| lex_glue/scotus | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) | scotus |  | sentence1=text, labels=label, label_values={…} |
| lex_glue/ledgar | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) | ledgar |  | sentence1=text, labels=label |
| lex_glue/unfair_tos | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) | unfair_tos | Which kind of unfair term, if any, does this terms-of-service clause contain? | sentence1=text, pre_process |
| lex_glue/case_hold | MultipleChoice | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) | case_hold |  | inputs=context, labels=label, choices_list=endings |
| language-identification | Classification | [papluca/language-identification](https://hf.co/datasets/papluca/language-identification) |  | What language is this text in? | sentence1=text, labels=fn |
| imdb | Classification | [stanfordnlp/imdb](https://hf.co/datasets/stanfordnlp/imdb) |  |  | sentence1=text, labels=label, splits=[train, None, test] |
| rotten_tomatoes | Classification | [cornell-movie-review-data/rotten_tomatoes](https://hf.co/datasets/cornell-movie-review-data/rotten_tomatoes) |  |  | sentence1=text, labels=label |
| ag_news | Classification | [fancyzhx/ag_news](https://hf.co/datasets/fancyzhx/ag_news) |  |  | sentence1=text, labels=label, splits=[train, None, test] |
| yelp_review_full/yelp_review_full | Classification | [Yelp/yelp_review_full](https://hf.co/datasets/Yelp/yelp_review_full) | yelp_review_full |  | sentence1=fn, labels=label, splits=[train, None, test], label_values={…} |
| financial_phrasebank/sentences_allagree | Classification | [ghbacct/financial-phrasebank-all-agree-classification](https://hf.co/datasets/ghbacct/financial-phrasebank-all-agree-classification) |  |  | sentence1=text, labels=label, splits=[train, None, None], pre_process |
| poem_sentiment | Classification | [google-research-datasets/poem_sentiment](https://hf.co/datasets/google-research-datasets/poem_sentiment) |  |  | sentence1=verse_text, labels=label |
| emotion | Classification | [dair-ai/emotion](https://hf.co/datasets/dair-ai/emotion) |  |  | sentence1=text, labels=label |
| dbpedia_14/dbpedia_14 | Classification | [fancyzhx/dbpedia_14](https://hf.co/datasets/fancyzhx/dbpedia_14) | dbpedia_14 |  | sentence1=content, labels=label, splits=[train, None, test] |
| amazon_polarity/amazon_polarity | Classification | [fancyzhx/amazon_polarity](https://hf.co/datasets/fancyzhx/amazon_polarity) | amazon_polarity |  | sentence1=content, labels=label, splits=[train, None, test] |
| app_reviews | Classification | [sealuzh/app_reviews](https://hf.co/datasets/sealuzh/app_reviews) |  |  | sentence1=review, labels=star, splits=[train, None, None], label_values={…} |
| hate_speech18 | Classification | [tasksource/hate_speech18](https://hf.co/datasets/tasksource/hate_speech18) |  |  | sentence1=text, labels=label, splits=[train, None, None], pre_process, post_process |
| sms_spam | Classification | [ucirvine/sms_spam](https://hf.co/datasets/ucirvine/sms_spam) |  |  | sentence1=sms, labels=label, splits=[train, None, None] |
| humicroedit/subtask-1 | Classification | [tasksource/humicroedit](https://hf.co/datasets/tasksource/humicroedit) | subtask-1 |  | sentence1=fn, sentence2=fn, labels=fn, label_values={…} |
| humicroedit/subtask-2 | Classification | [tasksource/humicroedit](https://hf.co/datasets/tasksource/humicroedit) | subtask-2 | Which edited headline is funnier? | sentence1=fn, sentence2=fn, labels=label, label_values={…} |
| snips_built_in_intents | Classification | [sonos-nlu-benchmark/snips_built_in_intents](https://hf.co/datasets/sonos-nlu-benchmark/snips_built_in_intents) |  |  | sentence1=text, labels=label, splits=[train, None, None] |
| hate_speech_offensive | Classification | [tdavidson/hate_speech_offensive](https://hf.co/datasets/tdavidson/hate_speech_offensive) |  |  | sentence1=tweet, labels=class, splits=[train, None, None] |
| yahoo_answers_topics | Classification | [community-datasets/yahoo_answers_topics](https://hf.co/datasets/community-datasets/yahoo_answers_topics) |  |  | sentence1=question_title, sentence2=question_content, labels=topic |
| stackoverflow-questions | Classification | [pacovaldez/stackoverflow-questions](https://hf.co/datasets/pacovaldez/stackoverflow-questions) |  |  | sentence1=title, sentence2=body, labels=label, label_values={…} |
| hyperpartisan_news | Classification | [zapsdcn/hyperpartisan_news](https://hf.co/datasets/zapsdcn/hyperpartisan_news) |  |  | sentence1=text, labels=fn |
| sciie | Classification | [zapsdcn/sciie](https://hf.co/datasets/zapsdcn/sciie) |  |  | sentence1=text, labels=label |
| citation_intent | Classification | [zapsdcn/citation_intent](https://hf.co/datasets/zapsdcn/citation_intent) |  |  | sentence1=text, labels=label |
| go_emotions/simplified | Classification | [google-research-datasets/go_emotions](https://hf.co/datasets/google-research-datasets/go_emotions) | simplified |  | sentence1=text, pre_process |
| scicite | Classification | [tasksource/scicite](https://hf.co/datasets/tasksource/scicite) |  |  | sentence1=string, labels=label |
| liar | Classification | [tasksource/liar](https://hf.co/datasets/tasksource/liar) |  |  | sentence1=statement, labels=label |
| lexical_relation_classification/ROOT09 | Classification | json | ROOT09 | How is the second word related to the first? | sentence1=head, sentence2=tail, labels=fn |
| lexical_relation_classification/K&H+N | Classification | json | K&H+N | How is the second word related to the first? | sentence1=head, sentence2=tail, labels=fn |
| lexical_relation_classification/BLESS | Classification | json | BLESS | How is the second word related to the first? | sentence1=head, sentence2=tail, labels=fn |
| lexical_relation_classification/EVALution | Classification | json | EVALution | How is the second word related to the first? | sentence1=head, sentence2=tail, labels=fn |
| lexical_relation_classification/CogALexV | Classification | json |  | How is the second word related to the first? | sentence1=head, sentence2=tail, labels=relation, pre_process |
| linguisticprobing/subj_number | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | subj_number |  | sentence1=sentence, labels=label, pre_process |
| linguisticprobing/obj_number | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | obj_number |  | sentence1=sentence, labels=label, pre_process |
| linguisticprobing/past_present | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | past_present |  | sentence1=sentence, labels=label, pre_process |
| linguisticprobing/sentence_length | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | sentence_length |  | sentence1=sentence, labels=label, pre_process |
| linguisticprobing/top_constituents | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | top_constituents |  | sentence1=sentence, labels=label, pre_process |
| linguisticprobing/tree_depth | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | tree_depth |  | sentence1=sentence, labels=label, pre_process |
| linguisticprobing/coordination_inversion | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | coordination_inversion |  | sentence1=sentence, labels=label, pre_process |
| linguisticprobing/odd_man_out | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | odd_man_out |  | sentence1=sentence, labels=label, pre_process |
| linguisticprobing/bigram_shift | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | bigram_shift |  | sentence1=sentence, labels=label, pre_process |
| crowdflower/airline-sentiment | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | airline-sentiment |  | sentence1=text, labels=label, splits=[train, None, None] |
| crowdflower/corporate-messaging | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | corporate-messaging |  | sentence1=text, labels=label, splits=[train, None, None] |
| crowdflower/economic-news | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | economic-news |  | sentence1=text, labels=label, splits=[train, None, None] |
| crowdflower/political-media-audience | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | political-media-audience |  | sentence1=text, labels=label, splits=[train, None, None] |
| crowdflower/political-media-bias | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | political-media-bias |  | sentence1=text, labels=label, splits=[train, None, None] |
| crowdflower/political-media-message | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | political-media-message |  | sentence1=text, labels=label, splits=[train, None, None] |
| crowdflower/text_emotion | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | text_emotion |  | sentence1=text, labels=label, splits=[train, None, None] |
| crowdflower/sentiment_nuclear_power | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | sentiment_nuclear_power |  | sentence1=text, labels=label, splits=[train, None, None] |
| crowdflower/tweet_global_warming | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | tweet_global_warming |  | sentence1=text, labels=label, splits=[train, None, None] |
| ethics/commonsense | Classification | csv | commonsense |  | sentence1=input, labels=fn |
| ethics/deontology | Classification | csv | deontology |  | sentence1=scenario, sentence2=excuse, labels=fn |
| ethics/justice | Classification | csv | justice |  | sentence1=scenario, labels=fn |
| ethics/virtue | Classification | [hendrycks/ethics](https://hf.co/datasets/hendrycks/ethics) | default |  | sentence1=fn, sentence2=fn, labels=fn |
| emo/emo2019 | Classification | [oneonlee/cleansed_emocontext](https://hf.co/datasets/oneonlee/cleansed_emocontext) |  |  | sentence1=fn, labels=fn, splits=[train, None, test] |
| google_wellformed_query | Classification | [tasksource/google_wellformed_query](https://hf.co/datasets/tasksource/google_wellformed_query) |  | Is this search query a well-formed question? | sentence1=content, labels=fn, pre_process |
| tweets_hate_speech_detection | Classification | [tweets-hate-speech-detection/tweets_hate_speech_detection](https://hf.co/datasets/tweets-hate-speech-detection/tweets_hate_speech_detection) |  |  | sentence1=tweet, labels=label, splits=[train, None, None] |
| wnut_17/wnut_17 | TokenClassification | [flaitenberger/wnut_17](https://hf.co/datasets/flaitenberger/wnut_17) |  |  | labels=ner_tags |
| ncbi_disease/ncbi_disease | TokenClassification | [ncbi/ncbi_disease](https://hf.co/datasets/ncbi/ncbi_disease) |  |  | labels=ner_tags |
| acronym_identification | TokenClassification | [amirveyseh/acronym_identification](https://hf.co/datasets/amirveyseh/acronym_identification) |  |  |  |
| jnlpba/jnlpba | TokenClassification | [jnlpba/jnlpba](https://hf.co/datasets/jnlpba/jnlpba) |  |  | labels=ner_tags, splits=[train, validation, None] |
| ontonotes_english/SpeedOfMagic--ontonotes_english | TokenClassification | [SpeedOfMagic/ontonotes_english](https://hf.co/datasets/SpeedOfMagic/ontonotes_english) |  |  | labels=ner_tags, pre_process |
| blog_authorship_corpus/gender | Classification | [tasksource/blog_authorship_corpus](https://hf.co/datasets/tasksource/blog_authorship_corpus) |  | What is the blogger's gender? | sentence1=text, labels=gender |
| blog_authorship_corpus/age | Classification | [tasksource/blog_authorship_corpus](https://hf.co/datasets/tasksource/blog_authorship_corpus) |  | What is the blogger's age group? | sentence1=text, labels=fn |
| blog_authorship_corpus/job | Classification | [tasksource/blog_authorship_corpus](https://hf.co/datasets/tasksource/blog_authorship_corpus) |  | In which industry does the blogger work? | sentence1=text, labels=topic, pre_process |
| open_question_type | Classification | [Korea-MES/open_question_type](https://hf.co/datasets/Korea-MES/open_question_type) |  |  | sentence1=question, labels=resolve_type |
| health_fact | Classification | [marcov/health_fact_promptsource](https://hf.co/datasets/marcov/health_fact_promptsource) |  |  | sentence1=claim, labels=label, pre_process |
| commonsense_qa | MultipleChoice | [tau/commonsense_qa](https://hf.co/datasets/tau/commonsense_qa) |  |  | inputs=question, labels=fn, choices_list=fn, splits=[train, validation, None] |
| mc_taco | Classification | [marcov/mc_taco_promptsource](https://hf.co/datasets/marcov/mc_taco_promptsource) |  | Is this answer plausible? | sentence1=fn, sentence2=answer, labels=label, splits=[validation, None, test] |
| ade_corpus_v2/Ade_corpus_v2_classification | Classification | [ade-benchmark-corpus/ade_corpus_v2](https://hf.co/datasets/ade-benchmark-corpus/ade_corpus_v2) | Ade_corpus_v2_classification |  | sentence1=text, labels=label |
| discosense | MultipleChoice | json |  |  | inputs=context, labels=label, choice0=option_0, choice1=option_1, choice2=option_2, choice3=option_3 |
| circa | Classification | [google-research-datasets/circa](https://hf.co/datasets/google-research-datasets/circa) |  |  | sentence1=fn, sentence2=answer-Y, labels=goldstandard2, post_process |
| code_x_glue_cc_defect_detection | Classification | [google/code_x_glue_cc_defect_detection](https://hf.co/datasets/google/code_x_glue_cc_defect_detection) |  |  | sentence1=func, labels=fn |
| phrase_similarity | Classification | [Deehan1866/processed_phrase_similarity](https://hf.co/datasets/Deehan1866/processed_phrase_similarity) |  |  | sentence1=fn, sentence2=fn, labels=fn |
| scientific-exaggeration-detection | Classification | [copenlu/scientific-exaggeration-detection](https://hf.co/datasets/copenlu/scientific-exaggeration-detection) |  |  | sentence1=press_release_conclusion, sentence2=abstract_conclusion, labels=exaggeration_label |
| quarel | Classification | [community-datasets/quarel](https://hf.co/datasets/community-datasets/quarel) |  |  | sentence1=question, labels=fn |
| fever-evidence-related | Classification | [mwong/fever-evidence-related](https://hf.co/datasets/mwong/fever-evidence-related) |  |  | sentence1=claim, sentence2=evidence, labels=fn, splits=[train, valid, test] |
| numer_sense | Classification | [tasksource/numer_sense](https://hf.co/datasets/tasksource/numer_sense) |  |  | sentence1=sentence, labels=target |
| dynasent/dynabench.dynasent.r1.all/r1 | Classification | [tasksource/dynasent](https://hf.co/datasets/tasksource/dynasent) | r1 |  | sentence1=sentence, labels=gold_label, pre_process |
| dynasent/dynabench.dynasent.r2.all/r2 | Classification | [tasksource/dynasent](https://hf.co/datasets/tasksource/dynasent) | r2 |  | sentence1=sentence, labels=gold_label, pre_process |
| Sarcasm_News_Headline | Classification | [raquiba/Sarcasm_News_Headline](https://hf.co/datasets/raquiba/Sarcasm_News_Headline) |  |  | sentence1=headline, labels=fn |
| sem_eval_2010_task_8 | Classification | [SemEvalWorkshop/sem_eval_2010_task_8](https://hf.co/datasets/SemEvalWorkshop/sem_eval_2010_task_8) |  |  | sentence1=sentence, labels=relation |
| auditor_review | Classification | [demo-org/auditor_review](https://hf.co/datasets/demo-org/auditor_review) |  |  | sentence1=sentence, labels=fn |
| medmcqa | MultipleChoice | [openlifescienceai/medmcqa](https://hf.co/datasets/openlifescienceai/medmcqa) |  |  | inputs=question, labels=cop, choice0=opa, choice1=opb, choice2=opc, choice3=opd |
| Dynasent_Disagreement | Classification | [RuyuanWan/Dynasent_Disagreement](https://hf.co/datasets/RuyuanWan/Dynasent_Disagreement) |  | Would annotators disagree about the sentiment of this text? | sentence1=text, labels=fn, pre_process |
| Politeness_Disagreement | Classification | [RuyuanWan/Politeness_Disagreement](https://hf.co/datasets/RuyuanWan/Politeness_Disagreement) |  | Would annotators disagree about the politeness of this text? | sentence1=text, labels=fn, pre_process |
| SBIC_Disagreement | Classification | [RuyuanWan/SBIC_Disagreement](https://hf.co/datasets/RuyuanWan/SBIC_Disagreement) |  | Would annotators disagree about whether this text is offensive? | sentence1=text, labels=fn, pre_process |
| SChem_Disagreement | Classification | [RuyuanWan/SChem_Disagreement](https://hf.co/datasets/RuyuanWan/SChem_Disagreement) |  | Would annotators disagree about whether this rule of thumb is acceptable? | sentence1=text, labels=fn, pre_process |
| Dilemmas_Disagreement | Classification | [RuyuanWan/Dilemmas_Disagreement](https://hf.co/datasets/RuyuanWan/Dilemmas_Disagreement) |  | Would annotators disagree about which of these two actions is less ethical? | sentence1=text, labels=fn, pre_process |
| logiqa | MultipleChoice | [fireworks-ai/logiqa](https://hf.co/datasets/fireworks-ai/logiqa) |  |  | inputs=fn, labels=correct_option, choices_list=options, pre_process |
| wiki_qa | Classification | [microsoft/wiki_qa](https://hf.co/datasets/microsoft/wiki_qa) |  | Does this sentence answer the question? | sentence1=question, sentence2=answer, labels=fn |
| cycic_classification | Classification | [tasksource/cycic_classification](https://hf.co/datasets/tasksource/cycic_classification) |  |  | sentence1=question, labels=fn |
| cycic_multiplechoice | MultipleChoice | [tasksource/cycic_multiplechoice](https://hf.co/datasets/tasksource/cycic_multiplechoice) |  |  | inputs=question, labels=correct_answer, choice0=answer_option0, choice1=answer_option1, choice2=answer_option2, choice3=answer_option3, choice4=answer_option4 |
| sts-companion | Classification | [tasksource/sts-companion](https://hf.co/datasets/tasksource/sts-companion) |  |  | labels=label |
| commonsense_qa_2.0 | Classification | [tasksource/commonsense_qa_2.0](https://hf.co/datasets/tasksource/commonsense_qa_2.0) |  |  | sentence1=question, labels=answer |
| lingnli | Classification | [tasksource/lingnli](https://hf.co/datasets/tasksource/lingnli) |  |  | sentence1=premise, sentence2=hypothesis, labels=label |
| monotonicity-entailment | Classification | [tasksource/monotonicity-entailment](https://hf.co/datasets/tasksource/monotonicity-entailment) |  |  | labels=gold_label |
| arct | MultipleChoice | [tasksource/arct](https://hf.co/datasets/tasksource/arct) |  |  | inputs=fn, labels=correctLabelW0orW1, choice0=warrant0, choice1=warrant1 |
| scinli | Classification | [tasksource/scinli](https://hf.co/datasets/tasksource/scinli) |  |  | labels=label, post_process |
| naturallogic | Classification | [tasksource/naturallogic](https://hf.co/datasets/tasksource/naturallogic) |  |  | sentence1= sent1 , sentence2= sent2 , labels= new_label |
| onestop_qa | MultipleChoice | [malmaud/onestop_qa](https://hf.co/datasets/malmaud/onestop_qa) |  |  | inputs=fn, labels=fn, choices_list=answers |
| moral_stories/full | MultipleChoice | [LabHC/moral_stories](https://hf.co/datasets/LabHC/moral_stories) |  |  | inputs=fn, labels=fn, choice0=moral_action, choice1=immoral_action |
| prost | MultipleChoice | json |  |  | inputs=fn, labels=fn, choice0=A, choice1=B, choice2=C, choice3=D |
| dynahate | Classification | [tasksource/dynahate](https://hf.co/datasets/tasksource/dynahate) |  |  | sentence1=text, labels=label, splits=[train, None, None] |
| syntactic-augmentation-nli | Classification | [tasksource/syntactic-augmentation-nli](https://hf.co/datasets/tasksource/syntactic-augmentation-nli) |  |  | labels=gold_label |
| autotnli | Classification | [tasksource/autotnli](https://hf.co/datasets/tasksource/autotnli) |  |  | sentence1=premises, sentence2=hypothesis, labels=label |
| CONDAQA | Classification | [lasha-nlp/CONDAQA](https://hf.co/datasets/lasha-nlp/CONDAQA) |  |  | labels=label, pre_process |
| webgpt_comparisons | MultipleChoice | [heegyu/webgpt_comparisons_ko](https://hf.co/datasets/heegyu/webgpt_comparisons_ko) |  | Which answer did the human rater prefer? | inputs=fn, labels=fn, pre_process, choice0=answer_0, choice1=answer_1 |
| synthetic-instruct-gptj-pairwise | MultipleChoice | [Dahoas/synthetic-instruct-gptj-pairwise](https://hf.co/datasets/Dahoas/synthetic-instruct-gptj-pairwise) |  | Which response is better? | inputs=prompt, labels=fn, choice0=chosen, choice1=rejected |
| scruples | Classification | [tasksource/scruples](https://hf.co/datasets/tasksource/scruples) |  | Was the author in the right or in the wrong? | sentence1=text, labels=binarized_label |
| wouldyourather | MultipleChoice | [tasksource/wouldyourather](https://hf.co/datasets/tasksource/wouldyourather) |  | Which would most people rather do? | inputs=fn, labels=fn, pre_process, choice0=option_a, choice1=option_b |
| defeasible-nli/atomic | Classification | [tasksource/defeasible-nli](https://hf.co/datasets/tasksource/defeasible-nli) | atomic |  | sentence1=fn, sentence2=Update, labels=UpdateType |
| defeasible-nli/snli | Classification | [tasksource/defeasible-nli](https://hf.co/datasets/tasksource/defeasible-nli) | snli |  | sentence1=fn, sentence2=Update, labels=UpdateType |
| defeasible-nli/social | Classification | [tasksource/defeasible-nli](https://hf.co/datasets/tasksource/defeasible-nli) | social |  | sentence1=Hypothesis, sentence2=Update, labels=UpdateType |
| help-nli | Classification | [tasksource/help-nli](https://hf.co/datasets/tasksource/help-nli) |  |  | sentence1=ori_sentence, sentence2=new_sentence, labels=gold_label |
| nli-veridicality-transitivity | Classification | [tasksource/nli-veridicality-transitivity](https://hf.co/datasets/tasksource/nli-veridicality-transitivity) |  |  | labels=gold_label |
| lonli | Classification | [tasksource/lonli](https://hf.co/datasets/tasksource/lonli) |  |  | sentence1=premise, sentence2=hypothesis, labels=label |
| dadc-limit-nli | Classification | [tasksource/dadc-limit-nli](https://hf.co/datasets/tasksource/dadc-limit-nli) |  |  | labels=label |
| FLUTE | Classification | [ColumbiaNLP/FLUTE](https://hf.co/datasets/ColumbiaNLP/FLUTE) |  |  | sentence1=premise, sentence2=hypothesis, labels=label |
| strategy-qa | Classification | [tasksource/strategy-qa](https://hf.co/datasets/tasksource/strategy-qa) |  |  | sentence1=question, labels=answer, splits=[train, None, None] |
| summarize_from_feedback/comparisons | MultipleChoice | [vwxyzjn/summarize_from_feedback_oai_preprocessing](https://hf.co/datasets/vwxyzjn/summarize_from_feedback_oai_preprocessing) |  | Which summary did the human rater prefer? | inputs=fn, labels=choice, choices_list=fn, pre_process |
| folio | Classification | [tasksource/folio](https://hf.co/datasets/tasksource/folio) |  |  | sentence1=premises, sentence2=conclusion, labels=fn |
| tomi-nli | Classification | [tasksource/tomi-nli](https://hf.co/datasets/tasksource/tomi-nli) |  |  | sentence1=premise, sentence2=hypothesis, labels=label |
| avicenna | Classification | [tasksource/avicenna](https://hf.co/datasets/tasksource/avicenna) |  | Do the two premises form a syllogism? | sentence1=Premise 1, sentence2=Premise 2, labels=Syllogistic relation |
| SHP | MultipleChoice | [stanfordnlp/SHP](https://hf.co/datasets/stanfordnlp/SHP) |  | Which reply did readers prefer? | inputs=fn, labels=fn, pre_process, choice0=human_ref_A, choice1=human_ref_B |
| MedQA-USMLE-4-options-hf | MultipleChoice | [GBaker/MedQA-USMLE-4-options-hf](https://hf.co/datasets/GBaker/MedQA-USMLE-4-options-hf) |  |  | inputs=sent1, labels=label, choice0=ending0, choice1=ending1, choice2=ending2, choice3=ending3 |
| wikimedqa/medwiki | MultipleChoice | [sileod/wikimedqa](https://hf.co/datasets/sileod/wikimedqa) | medwiki |  | inputs=text, labels=label, choice0=option_0, choice1=option_1, choice2=option_2, choice3=option_3, choice4=option_4, choice5=option_5, choice6=option_6, choice7=option_7 |
| cicero | MultipleChoice | [declare-lab/cicero](https://hf.co/datasets/declare-lab/cicero) |  |  | inputs=fn, labels=fn, choices_list=Choices |
| CREAK | Classification | [amydeng2000/CREAK](https://hf.co/datasets/amydeng2000/CREAK) |  |  | sentence1=sentence, labels=label |
| mutual | MultipleChoice | [tasksource/mutual](https://hf.co/datasets/tasksource/mutual) |  |  | inputs=article, labels=fn, choices_list=options, splits=[train, None, None] |
| puzzte | Classification | [tasksource/puzzte](https://hf.co/datasets/tasksource/puzzte) |  |  | sentence1=puzzle_text, sentence2=question, labels=answer, pre_process |
| implicatures | MultipleChoice | [tasksource/implicatures](https://hf.co/datasets/tasksource/implicatures) |  |  | inputs=fn, labels=fn, choice0=correct_implicature, choice1=incorrect_implicature |
| race/high | MultipleChoice | [ehovy/race](https://hf.co/datasets/ehovy/race) | high |  | inputs=fn, labels=fn, choices_list=options |
| race/middle | MultipleChoice | [ehovy/race](https://hf.co/datasets/ehovy/race) | middle |  | inputs=fn, labels=fn, choices_list=options |
| race-c | MultipleChoice | [tasksource/race-c](https://hf.co/datasets/tasksource/race-c) |  |  | inputs=fn, labels=label, choices_list=option |
| spartqa-yn | Classification | [tasksource/spartqa-yn](https://hf.co/datasets/tasksource/spartqa-yn) |  |  | sentence1=story, sentence2=question, labels=fn |
| spartqa-mchoice | MultipleChoice | [tasksource/spartqa-mchoice](https://hf.co/datasets/tasksource/spartqa-mchoice) |  |  | inputs=fn, labels=answer, choices_list=candidate_answers |
| temporal-nli | Classification | [tasksource/temporal-nli](https://hf.co/datasets/tasksource/temporal-nli) |  |  | sentence1=Premise, sentence2=Hypothesis, labels=Label |
| riddle_sense | MultipleChoice | [jeggers/riddle_sense](https://hf.co/datasets/jeggers/riddle_sense) |  |  | inputs=question, labels=fn, choices_list=fn, pre_process |
| clcd-english | Classification | [tasksource/clcd-english](https://hf.co/datasets/tasksource/clcd-english) |  |  | labels=label |
| twentyquestions | Classification | [tasksource/twentyquestions](https://hf.co/datasets/tasksource/twentyquestions) |  |  | sentence1=fn, labels=answer, pre_process |
| reclor | MultipleChoice | [tasksource/reclor](https://hf.co/datasets/tasksource/reclor) |  |  | inputs=fn, labels=label, choices_list=answers, splits=[train, validation, None] |
| counterfactually-augmented-imdb | Classification | [tasksource/counterfactually-augmented-imdb](https://hf.co/datasets/tasksource/counterfactually-augmented-imdb) |  |  | sentence1=Text, labels=Sentiment |
| counterfactually-augmented-snli | Classification | [tasksource/counterfactually-augmented-snli](https://hf.co/datasets/tasksource/counterfactually-augmented-snli) |  |  | labels=gold_label |
| cnli | Classification | [tasksource/cnli](https://hf.co/datasets/tasksource/cnli) |  |  | sentence1=premise, sentence2=hypothesis, labels=label |
| boolq-natural-perturbations | Classification | [tasksource/boolq-natural-perturbations](https://hf.co/datasets/tasksource/boolq-natural-perturbations) |  |  | sentence1=question, labels=hard_label |
| acceptability-prediction | Classification | [tasksource/acceptability-prediction](https://hf.co/datasets/tasksource/acceptability-prediction) |  | How acceptable is this sentence, from 0 (unacceptable) to 1 (acceptable)? | sentence1=text, labels=normalized_score |
| equate | Classification | [tasksource/equate](https://hf.co/datasets/tasksource/equate) |  |  | labels=gold_label |
| ScienceQA_text_only | MultipleChoice | [tasksource/ScienceQA_text_only](https://hf.co/datasets/tasksource/ScienceQA_text_only) |  |  | inputs=question, labels=answer, choices_list=choices |
| ekar_english | MultipleChoice | [Jiangjie/ekar_english](https://hf.co/datasets/Jiangjie/ekar_english) |  | Which pair is related in the same way? | inputs=question, labels=fn, choices_list=fn |
| implicit-hate-stg1 | Classification | [tasksource/implicit-hate-stg1](https://hf.co/datasets/tasksource/implicit-hate-stg1) |  |  | sentence1=post, labels=class |
| chaos-mnli-ambiguity | Classification | [tasksource/chaos-mnli-ambiguity](https://hf.co/datasets/tasksource/chaos-mnli-ambiguity) |  | How much would annotators agree on the inference, from 0 (evenly split) to 1 (unanimous)? | sentence1=premise, sentence2=hypothesis, labels=gini |
| headline_cause/en_simple | Classification | json |  |  | sentence1=left_title, sentence2=right_title, labels=label, label_values={…} |
| logiqa-2.0-nli | Classification | [tasksource/logiqa-2.0-nli](https://hf.co/datasets/tasksource/logiqa-2.0-nli) |  |  | sentence1=premise, sentence2=hypothesis, labels=label |
| oasst2_dense_flat/quality | Classification | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |  | How good is the reply, from 0 (low quality) to 1 (high quality)? | sentence1=parent_text, sentence2=text, labels=quality, pre_process |
| oasst2_dense_flat/toxicity | Classification | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |  | How toxic is the reply, from 0 (not toxic) to 1 (very toxic)? | sentence1=parent_text, sentence2=text, labels=toxicity, pre_process |
| oasst2_dense_flat/helpfulness | Classification | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |  | How helpful is the reply, from 0 (unhelpful) to 1 (helpful)? | sentence1=parent_text, sentence2=text, labels=helpfulness, pre_process |
| mindgames | Classification | [sileod/mindgames](https://hf.co/datasets/sileod/mindgames) |  |  | sentence1=premise, sentence2=hypothesis, labels=label |
| universal_dependencies/en_gum/deprel | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) | en_gum |  | labels=deprel, pre_process |
| universal_dependencies/en_ewt/deprel | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) | en_ewt |  | labels=deprel, pre_process |
| universal_dependencies/en_lines/deprel | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) | en_lines |  | labels=deprel, pre_process |
| universal_dependencies/en_partut/deprel | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) | en_partut |  | labels=deprel, pre_process |
| ambient | Classification | [tasksource/ambient](https://hf.co/datasets/tasksource/ambient) |  | Is the hypothesis ambiguous? | sentence1=premise, sentence2=hypothesis, labels=hypothesis_ambiguous |
| path-naturalness-prediction | MultipleChoice | [tasksource/path-naturalness-prediction](https://hf.co/datasets/tasksource/path-naturalness-prediction) |  | Which chain of relations is more natural? | inputs=fn, labels=label, choice0=choice1, choice1=choice2 |
| civil_comments/toxicity | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |  | Would most raters flag this comment as toxic? | sentence1=text, labels=fn, pre_process |
| civil_comments/severe_toxicity | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |  | Would most raters flag this comment as severely toxic? | sentence1=text, labels=fn, pre_process |
| civil_comments/obscene | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |  | Would most raters flag this comment as obscene? | sentence1=text, labels=fn, pre_process |
| civil_comments/threat | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |  | Would most raters flag this comment as a threat? | sentence1=text, labels=fn, pre_process |
| civil_comments/insult | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |  | Would most raters flag this comment as insulting? | sentence1=text, labels=fn, pre_process |
| civil_comments/identity_attack | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |  | Would most raters flag this comment as an identity attack? | sentence1=text, labels=fn, pre_process |
| civil_comments/sexual_explicit | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |  | Would most raters flag this comment as sexually explicit? | sentence1=text, labels=fn, pre_process |
| cloth | MultipleChoice | [AndyChiang/cloth](https://hf.co/datasets/AndyChiang/cloth) |  |  | inputs=sentence, labels=fn, choices_list=fn |
| dgen | MultipleChoice | [AndyChiang/dgen](https://hf.co/datasets/AndyChiang/dgen) |  |  | inputs=sentence, labels=fn, choices_list=fn |
| I2D2 | Classification | [tasksource/I2D2](https://hf.co/datasets/tasksource/I2D2) |  |  | labels=fn |
| args_me | Classification | [webis/args_me](https://hf.co/datasets/webis/args_me) |  |  | sentence1=argument, sentence2=conclusion, labels=stance |
| Touche23-ValueEval | Classification | csv |  |  | sentence1=Premise, sentence2=Conclusion, labels=Stance |
| starcon | Classification | [tasksource/starcon](https://hf.co/datasets/tasksource/starcon) |  |  | sentence1=argument, sentence2=topic, labels=label |
| banking77 | Classification | [legacy-datasets/banking77](https://hf.co/datasets/legacy-datasets/banking77) |  |  | sentence1=text, labels=label |
| it-support-tickets | Classification | [tasksource/it-support-tickets](https://hf.co/datasets/tasksource/it-support-tickets) |  |  | sentence1=text, labels=label, splits=[train, None, test] |
| ConTRoL-nli | Classification | [tasksource/ConTRoL-nli](https://hf.co/datasets/tasksource/ConTRoL-nli) |  |  | sentence1=premise, sentence2=hypothesis, labels=label |
| tracie | Classification | [tasksource/tracie](https://hf.co/datasets/tasksource/tracie) |  |  | sentence1=premise, sentence2=hypothesis, labels=answer |
| sherliic | Classification | [tasksource/sherliic](https://hf.co/datasets/tasksource/sherliic) |  |  | sentence1=premise, sentence2=hypothesis, labels=label |
| sen-making/1 | MultipleChoice | [tasksource/sen-making](https://hf.co/datasets/tasksource/sen-making) |  | Which statement makes sense? | inputs=fn, labels=false, choice0=sentence0, choice1=sentence1 |
| sen-making/2 | MultipleChoice | [tasksource/sen-making](https://hf.co/datasets/tasksource/sen-making) |  | Why is this statement implausible? | inputs=fn, labels=fn, choice0=A, choice1=B, choice2=C |
| winowhy | Classification | [tasksource/winowhy](https://hf.co/datasets/tasksource/winowhy) |  | Is this explanation correct? | sentence1=sentence, sentence2=fn, labels=fn |
| robustLR | Classification | [tasksource/robustLR](https://hf.co/datasets/tasksource/robustLR) |  |  | sentence1=context, sentence2=statement, labels=label |
| clutrr | Classification | [tasksource/clutrr](https://hf.co/datasets/tasksource/clutrr) |  |  | sentence1=story, sentence2=query, labels=label |
| logical-fallacy | Classification | [tasksource/logical-fallacy](https://hf.co/datasets/tasksource/logical-fallacy) |  |  | sentence1=source_article, labels=logical_fallacies |
| parade | Classification | [tasksource/parade](https://hf.co/datasets/tasksource/parade) |  |  | sentence1=Definition1, sentence2=Definition2, labels=fn |
| cladder | Classification | [tasksource/cladder](https://hf.co/datasets/tasksource/cladder) |  |  | sentence1=given_info, sentence2=question, labels=answer |
| subjectivity | Classification | [tasksource/subjectivity](https://hf.co/datasets/tasksource/subjectivity) |  |  | sentence1=Sentence, labels=fn |
| MOH | Classification | [tasksource/MOH](https://hf.co/datasets/tasksource/MOH) |  |  | sentence1=context, sentence2=expression, labels=label |
| VUAC | Classification | [tasksource/VUAC](https://hf.co/datasets/tasksource/VUAC) |  |  | sentence1=context, sentence2=expression, labels=label |
| TroFi | Classification | parquet |  |  | sentence1=context, sentence2=expression, labels=label, splits=[train, None, test] |
| sharc | Classification | [tasksource/sharc](https://hf.co/datasets/tasksource/sharc) |  |  | sentence1=snippet, sentence2=fn, labels=label |
| conceptrules_v2 | Classification | [tasksource/conceptrules_v2](https://hf.co/datasets/tasksource/conceptrules_v2) |  | Is the statement true given the context? | sentence1=context, sentence2=text, labels=label |
| disrpt/eng.dep.scidtb.rels | Classification | [multilingual-discourse-hub/disrpt](https://hf.co/datasets/multilingual-discourse-hub/disrpt) | eng.dep.scidtb.rels |  | sentence1=unit1_txt, sentence2=unit2_txt, labels=label |
| conll2000 | TokenClassification | [eriktks/conll2000](https://hf.co/datasets/eriktks/conll2000) |  |  | labels=chunk_tags |
| few-nerd/supervised | TokenClassification | [DFKI-SLT/few-nerd](https://hf.co/datasets/DFKI-SLT/few-nerd) | supervised |  | labels=fine_ner_tags |
| finer-139 | TokenClassification | [nlpaueb/finer-139](https://hf.co/datasets/nlpaueb/finer-139) |  |  | labels=ner_tags |
| zero-shot-label-nli | Classification | [tasksource/zero-shot-label-nli](https://hf.co/datasets/tasksource/zero-shot-label-nli) |  |  | sentence1=premise, sentence2=hypothesis |
| com2sense | Classification | [tasksource/com2sense](https://hf.co/datasets/tasksource/com2sense) |  |  | sentence1=sent, labels=label, splits=[train, validation, None] |
| scone | Classification | [tasksource/scone](https://hf.co/datasets/tasksource/scone) |  |  | sentence1=sentence1_edited, sentence2=sentence2_edited, labels=gold_label_edited |
| winodict | MultipleChoice | [tasksource/winodict](https://hf.co/datasets/tasksource/winodict) |  |  | inputs=fn, labels=label, choice0=option1, choice1=option2 |
| fool-me-twice | Classification | [tasksource/fool-me-twice](https://hf.co/datasets/tasksource/fool-me-twice) |  |  | sentence1=fn, sentence2=text, labels=label |
| monli | Classification | [tasksource/monli](https://hf.co/datasets/tasksource/monli) |  |  | labels=gold_label |
| corr2cause | Classification | [tasksource/corr2cause](https://hf.co/datasets/tasksource/corr2cause) |  |  | sentence1=premise, sentence2=hypothesis, labels=relation |
| lsat_qa/all | MultipleChoice | [lighteval/lsat_qa](https://hf.co/datasets/lighteval/lsat_qa) | all |  | inputs=fn, labels=gold_index, choices_list=references |
| apt | Classification | [tasksource/apt](https://hf.co/datasets/tasksource/apt) |  |  | sentence1=text_a, sentence2=text_b, labels=fn |
| twitter-financial-news-sentiment | Classification | [zeroshot/twitter-financial-news-sentiment](https://hf.co/datasets/zeroshot/twitter-financial-news-sentiment) |  |  | sentence1=text, labels=fn |
| icl-symbol-tuning-instruct | Classification | [tasksource/icl-symbol-tuning-instruct](https://hf.co/datasets/tasksource/icl-symbol-tuning-instruct) |  | Is this the right label for the last input? | sentence1=inputs, sentence2=fn, labels=fn, pre_process |
| SpaceNLI | Classification | [tasksource/SpaceNLI](https://hf.co/datasets/tasksource/SpaceNLI) |  |  | sentence1=premises, sentence2=hypothesis, labels=label |
| propsegment/nli | Classification | json |  |  | sentence1=hypothesis, sentence2=premise, labels=fn |
| HatemojiBuild | Classification | [HannahRoseKirk/HatemojiBuild](https://hf.co/datasets/HannahRoseKirk/HatemojiBuild) |  |  | sentence1=text, labels=fn |
| regset | Classification | [tasksource/regset](https://hf.co/datasets/tasksource/regset) |  | Does the string match the regular expression? | sentence1=context, labels=answer |
| esci | Classification | [tasksource/esci](https://hf.co/datasets/tasksource/esci) |  |  | sentence1=query, sentence2=fn, labels=esci_label, pre_process |
| chatbot_arena_conversations | MultipleChoice | [lmsys/chatbot_arena_conversations](https://hf.co/datasets/lmsys/chatbot_arena_conversations) |  | Which assistant did the user prefer? | inputs=prompt, labels=fn, pre_process, choice0=conversation_a, choice1=conversation_b |
| dnd_style_intents | Classification | [neurae/dnd_style_intents](https://hf.co/datasets/neurae/dnd_style_intents) |  |  | sentence1=examples, labels=label_names |
| FLD.v2/default | Classification | [hitachi-nlp/FLD.v2](https://hf.co/datasets/hitachi-nlp/FLD.v2) | default |  | sentence1=context, sentence2=hypothesis, labels=proof_label |
| FLD.v2/star | Classification | [hitachi-nlp/FLD.v2](https://hf.co/datasets/hitachi-nlp/FLD.v2) | star |  | sentence1=context, sentence2=hypothesis, labels=proof_label |
| SDOH-NLI | Classification | [tasksource/SDOH-NLI](https://hf.co/datasets/tasksource/SDOH-NLI) |  |  | sentence1=premise, sentence2=hypothesis, labels=fn |
| scifact_entailment | Classification | [tasksource/scifact_entailment](https://hf.co/datasets/tasksource/scifact_entailment) |  |  | sentence1=fn, sentence2=claim, labels=fn |
| feasibilityQA | Classification | [tasksource/feasibilityQA](https://hf.co/datasets/tasksource/feasibilityQA) |  |  | sentence1=fn, sentence2=hypothesis, labels=binary_classification_label |
| simple_pair | Classification | [tasksource/simple_pair](https://hf.co/datasets/tasksource/simple_pair) |  |  | sentence1=premise, sentence2=hypothesis, labels=label |
| AdjectiveScaleProbe-nli | Classification | [tasksource/AdjectiveScaleProbe-nli](https://hf.co/datasets/tasksource/AdjectiveScaleProbe-nli) |  |  | sentence1=premise, sentence2=hypothesis, labels=label |
| resnli | Classification | [tasksource/resnli](https://hf.co/datasets/tasksource/resnli) |  |  | sentence1=premise, sentence2=hypothesis, labels=label |
| SpaRTUN | MultipleChoice | [tasksource/SpaRTUN](https://hf.co/datasets/tasksource/SpaRTUN) |  |  | inputs=fn, labels=fn, choices_list=candidate_answers, pre_process |
| ReSQ | MultipleChoice | [tasksource/ReSQ](https://hf.co/datasets/tasksource/ReSQ) |  |  | inputs=fn, labels=fn, choices_list=candidate_answers, pre_process |
| semantic_fragments_nli | Classification | [tasksource/semantic_fragments_nli](https://hf.co/datasets/tasksource/semantic_fragments_nli) |  |  | labels=gold_label |
| dataset_train_nli | Classification | [MoritzLaurer/dataset_train_nli](https://hf.co/datasets/MoritzLaurer/dataset_train_nli) |  |  | sentence1=text, sentence2=hypothesis, pre_process |
| stepgame | Classification | [tasksource/stepgame](https://hf.co/datasets/tasksource/stepgame) |  |  | sentence1=story, sentence2=question, labels=label |
| nlgraph | Classification | [tasksource/nlgraph](https://hf.co/datasets/tasksource/nlgraph) |  |  | sentence1=question, labels=fn, pre_process |
| oasst2_pairwise_rlhf_reward | MultipleChoice | [tasksource/oasst2_pairwise_rlhf_reward](https://hf.co/datasets/tasksource/oasst2_pairwise_rlhf_reward) |  | Which reply is better? | inputs=prompt, labels=fn, choice0=chosen, choice1=rejected |
| hh-rlhf/helpful-online | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | helpful-online | Which next assistant reply is more helpful? | inputs=dialogue, labels=fn, pre_process, choice0=chosen_reply, choice1=rejected_reply |
| hh-rlhf/helpful-base | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | helpful-base | Which next assistant reply is more helpful? | inputs=dialogue, labels=fn, pre_process, choice0=chosen_reply, choice1=rejected_reply |
| hh-rlhf/helpful-rejection-sampled | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | helpful-rejection-sampled | Which next assistant reply is more helpful? | inputs=dialogue, labels=fn, pre_process, choice0=chosen_reply, choice1=rejected_reply |
| hh-rlhf/harmless-base | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | harmless-base | Which next assistant reply is more harmless? | inputs=dialogue, labels=fn, pre_process, choice0=chosen_reply, choice1=rejected_reply |
| ruletaker | Classification | [tasksource/ruletaker](https://hf.co/datasets/tasksource/ruletaker) |  | Does the statement follow from the context? What is not explicitly stated as true is considered false. | sentence1=context, sentence2=question, labels=label |
| PARARULE-Plus | Classification | [qbao775/PARARULE-Plus](https://hf.co/datasets/qbao775/PARARULE-Plus) |  | Is the statement true? What is not explicitly stated as true is considered false. | sentence1=context, sentence2=question, labels=fn |
| proofwriter | Classification | [tasksource/proofwriter](https://hf.co/datasets/tasksource/proofwriter) |  |  | sentence1=theory, sentence2=question, labels=answer |
| logical-entailment | Classification | [tasksource/logical-entailment](https://hf.co/datasets/tasksource/logical-entailment) |  |  | sentence1=A, sentence2=B, labels=label |
| nope | Classification | [tasksource/nope](https://hf.co/datasets/tasksource/nope) |  |  | sentence1=premise, sentence2=hypothesis, labels=fn |
| LogicNLI | Classification | [tasksource/LogicNLI](https://hf.co/datasets/tasksource/LogicNLI) |  |  | sentence1=premise, sentence2=hypothesis, labels=label |
| contract-nli/contractnli_a/seg | Classification | [tasksource/contract-nli](https://hf.co/datasets/tasksource/contract-nli) | contractnli_a |  | sentence1=premise, sentence2=hypothesis, labels=label |
| contract-nli/contractnli_b/full | Classification | [tasksource/contract-nli](https://hf.co/datasets/tasksource/contract-nli) | contractnli_b |  | sentence1=premise, sentence2=hypothesis, labels=label |
| nli4ct_semeval2024 | Classification | [AshtonIsNotHere/nli4ct_semeval2024](https://hf.co/datasets/AshtonIsNotHere/nli4ct_semeval2024) |  |  | sentence1=fn, sentence2=Statement, labels=Label, splits=[train, dev, None] |
| lsat-ar | MultipleChoice | [tasksource/lsat-ar](https://hf.co/datasets/tasksource/lsat-ar) |  |  | inputs=fn, labels=label, choices_list=answers |
| lsat-rc | MultipleChoice | [tasksource/lsat-rc](https://hf.co/datasets/tasksource/lsat-rc) |  |  | inputs=fn, labels=label, choices_list=answers |
| biosift-nli | Classification | [AshtonIsNotHere/biosift-nli](https://hf.co/datasets/AshtonIsNotHere/biosift-nli) |  |  | sentence1=Abstract, sentence2=Hypothesis, labels=fn |
| brainteasers/SP | MultipleChoice | [tasksource/brainteasers](https://hf.co/datasets/tasksource/brainteasers) | SP |  | inputs=question, labels=label, choices_list=fn |
| brainteasers/WP | MultipleChoice | [tasksource/brainteasers](https://hf.co/datasets/tasksource/brainteasers) | WP |  | inputs=question, labels=label, choices_list=fn |
| toxigen-data/annotated | Classification | [skg/toxigen-data](https://hf.co/datasets/skg/toxigen-data) | annotated |  | sentence1=text, labels=fn, pre_process |
| persuasion | Classification | [Anthropic/persuasion](https://hf.co/datasets/Anthropic/persuasion) |  |  | sentence1=claim, sentence2=argument, labels=persuasiveness_metric, label_values={…} |
| AmbigNQ-clarifying-question | Classification | [erbacher/AmbigNQ-clarifying-question](https://hf.co/datasets/erbacher/AmbigNQ-clarifying-question) |  |  | sentence1=question, labels=fn |
| SIGA-nli | Classification | [tasksource/SIGA-nli](https://hf.co/datasets/tasksource/SIGA-nli) |  |  | sentence1=premise, sentence2=statement, labels=label |
| FOL-nli | Classification | [unigram/FOL-nli](https://hf.co/datasets/unigram/FOL-nli) |  |  | sentence1=premise, sentence2=hypothesis, labels=label |
| goal-step-wikihow/goal | MultipleChoice | [tasksource/goal-step-wikihow](https://hf.co/datasets/tasksource/goal-step-wikihow) | goal |  | inputs=fn, labels=label, choice0=ending0, choice1=ending1, choice2=ending2, choice3=ending3 |
| goal-step-wikihow/step | MultipleChoice | [tasksource/goal-step-wikihow](https://hf.co/datasets/tasksource/goal-step-wikihow) | step |  | inputs=fn, labels=label, choice0=ending0, choice1=ending1, choice2=ending2, choice3=ending3 |
| goal-step-wikihow/order | MultipleChoice | [tasksource/goal-step-wikihow](https://hf.co/datasets/tasksource/goal-step-wikihow) | order |  | inputs=sent2, labels=label, choice0=ending0, choice1=ending1 |
| PARADISE | MultipleChoice | [GGLab/PARADISE](https://hf.co/datasets/GGLab/PARADISE) |  |  | inputs=sent2, labels=label, choice0=ending0, choice1=ending1, choice2=ending2, choice3=ending3 |
| doc-nli | Classification | [tasksource/doc-nli](https://hf.co/datasets/tasksource/doc-nli) |  |  | sentence1=premise, sentence2=hypothesis, labels=label |
| mctest-nli | Classification | [tasksource/mctest-nli](https://hf.co/datasets/tasksource/mctest-nli) |  |  | sentence1=premise, sentence2=hypothesis, labels=label |
| patent-phrase-similarity | Classification | [tasksource/patent-phrase-similarity](https://hf.co/datasets/tasksource/patent-phrase-similarity) |  |  | sentence1=anchor, sentence2=target, labels=label |
| natural-language-satisfiability | Classification | [tasksource/natural-language-satisfiability](https://hf.co/datasets/tasksource/natural-language-satisfiability) |  |  | sentence1=sentence, labels=label |
| idioms-nli | Classification | [tasksource/idioms-nli](https://hf.co/datasets/tasksource/idioms-nli) |  |  | sentence1=premise, sentence2=hypothesis, labels=label |
| lifecycle-entailment | Classification | [tasksource/lifecycle-entailment](https://hf.co/datasets/tasksource/lifecycle-entailment) |  |  | sentence1=premise, sentence2=hypothesis, labels=label |
| toxic-chat/toxicchat0124/toxicity | Classification | [lmsys/toxic-chat](https://hf.co/datasets/lmsys/toxic-chat) | toxicchat0124 | Is this user prompt toxic? | sentence1=user_input, labels=fn, splits=[train, None, test] |
| toxic-chat/toxicchat0124/jailbreaking | Classification | [lmsys/toxic-chat](https://hf.co/datasets/lmsys/toxic-chat) | toxicchat0124 | Is this user prompt a jailbreak attempt? | sentence1=user_input, labels=fn, splits=[train, None, test] |
| clinc_oos/plus | Classification | [clinc/clinc_oos](https://hf.co/datasets/clinc/clinc_oos) | plus |  | sentence1=text, labels=intent |
| few_rel/default | Classification | [tasksource/few_rel](https://hf.co/datasets/tasksource/few_rel) | default |  | sentence1=text, sentence2=relation, labels=label, splits=[train_wiki, val_wiki, val_nyt], pre_process |
| docred | Classification | json |  |  | sentence1=text, sentence2=entity_pair, labels=relation, splits=[train_annotated, validation, None], pre_process |
| chemprot/chemprot_full_source | Classification | [bigbio/chemprot](https://hf.co/datasets/bigbio/chemprot) | chemprot_full_source |  | sentence1=text, sentence2=entity_pair, labels=relation, pre_process |
| PKU-SafeRLHF/helpfulness | MultipleChoice | [PKU-Alignment/PKU-SafeRLHF](https://hf.co/datasets/PKU-Alignment/PKU-SafeRLHF) |  | Which response is more helpful? | inputs=prompt, labels=better_response_id, choice0=response_0, choice1=response_1 |
| PKU-SafeRLHF/safety | MultipleChoice | [PKU-Alignment/PKU-SafeRLHF](https://hf.co/datasets/PKU-Alignment/PKU-SafeRLHF) |  | Which response is safer? | inputs=prompt, labels=safer_response_id, choice0=response_0, choice1=response_1 |
| HelpSteer/helpfulness | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) |  | How would you rate the helpfulness of the response? | sentence1=prompt, sentence2=response, labels=fn |
| HelpSteer/correctness | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) |  | How would you rate the correctness of the response? | sentence1=prompt, sentence2=response, labels=fn |
| HelpSteer/coherence | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) |  | How would you rate the coherence of the response? | sentence1=prompt, sentence2=response, labels=fn |
| HelpSteer/complexity | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) |  | How would you rate the complexity of the response? | sentence1=prompt, sentence2=response, labels=fn |
| HelpSteer/verbosity | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) |  | How would you rate the verbosity of the response? | sentence1=prompt, sentence2=response, labels=fn |
| HelpSteer2/helpfulness | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) |  | How would you rate the helpfulness of the response? | sentence1=prompt, sentence2=response, labels=fn |
| HelpSteer2/correctness | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) |  | How would you rate the correctness of the response? | sentence1=prompt, sentence2=response, labels=fn |
| HelpSteer2/coherence | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) |  | How would you rate the coherence of the response? | sentence1=prompt, sentence2=response, labels=fn |
| HelpSteer2/complexity | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) |  | How would you rate the complexity of the response? | sentence1=prompt, sentence2=response, labels=fn |
| HelpSteer2/verbosity | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) |  | How would you rate the verbosity of the response? | sentence1=prompt, sentence2=response, labels=fn |
| HelpSteer3/preference | MultipleChoice | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | preference | Which next assistant reply is better? | inputs=fn, labels=fn, pre_process, choice0=response1, choice1=response2 |
| HelpSteer3/principle | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | principle |  | sentence1=fn, sentence2=fn, labels=fulfilment |
| HelpSteer3/edit_quality | MultipleChoice | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | edit_quality | Which edit improves the reply? | inputs=fn, labels=fn, choice0=good_edited_response, choice1=bad_edited_response |
| HelpSteer3/feedback | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | feedback | How helpful is the assistant reply? | sentence1=fn, labels=helpfulness, pre_process |
| MSciNLI | Classification | [sadat2307/MSciNLI](https://hf.co/datasets/sadat2307/MSciNLI) |  |  | labels=label |
| UltraFeedback-paired | MultipleChoice | [pushpdeep/UltraFeedback-paired](https://hf.co/datasets/pushpdeep/UltraFeedback-paired) |  | Which response is better? | inputs=question, labels=fn, choice0=response_j, choice1=response_k |
| prm800k_dpo/solution | MultipleChoice | [tasksource/prm800k_dpo](https://hf.co/datasets/tasksource/prm800k_dpo) | solution | Which solution is correct? | inputs=prompt, labels=fn, splits=[train, None, None], choice0=chosen, choice1=rejected |
| prm800k_dpo/step | MultipleChoice | [tasksource/prm800k_dpo](https://hf.co/datasets/tasksource/prm800k_dpo) | step | Which next step is correct? | inputs=prompt, labels=fn, splits=[train, None, None], choice0=chosen, choice1=rejected |
| AES2-essay-scoring | Classification | [tasksource/AES2-essay-scoring](https://hf.co/datasets/tasksource/AES2-essay-scoring) |  | What holistic score does this student essay deserve? | sentence1=full_text, labels=score, label_values={…} |
| argument-feedback | Classification | [tasksource/argument-feedback](https://hf.co/datasets/tasksource/argument-feedback) |  | How effective is this element of the student's argument? | sentence1=fn, labels=discourse_effectiveness |
| english-grading/cohesion | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) |  | What cohesion score does this English learner essay deserve? | sentence1=full_text, labels=fn |
| english-grading/syntax | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) |  | What syntax score does this English learner essay deserve? | sentence1=full_text, labels=fn |
| english-grading/vocabulary | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) |  | What vocabulary score does this English learner essay deserve? | sentence1=full_text, labels=fn |
| english-grading/phraseology | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) |  | What phraseology score does this English learner essay deserve? | sentence1=full_text, labels=fn |
| english-grading/grammar | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) |  | What grammar score does this English learner essay deserve? | sentence1=full_text, labels=fn |
| english-grading/conventions | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) |  | What conventions score does this English learner essay deserve? | sentence1=full_text, labels=fn |
| wice | Classification | [tasksource/wice](https://hf.co/datasets/tasksource/wice) |  |  | sentence1=fn, sentence2=claim, labels=label |
| hover | Classification | [Dzeniks/hover](https://hf.co/datasets/Dzeniks/hover) |  |  | sentence1=evidence, sentence2=claim, labels=label, label_values={…} |
| hover-3way/nli | Classification | [Dzeniks/hover-3way](https://hf.co/datasets/Dzeniks/hover-3way) |  |  | sentence1=evidence, sentence2=claim, labels=fn |
| tasksource_dpo_pairs | MultipleChoice | [tasksource/tasksource_dpo_pairs](https://hf.co/datasets/tasksource/tasksource_dpo_pairs) |  | Which response is better? | inputs=prompt, labels=fn, choice0=chosen, choice1=rejected |
| seahorse_summarization_evaluation | Classification | [tasksource/seahorse_summarization_evaluation](https://hf.co/datasets/tasksource/seahorse_summarization_evaluation) |  |  | sentence1=article, sentence2=fn, labels=answer |
| missing-item-prediction/contrastive | Classification | [sileod/missing-item-prediction](https://hf.co/datasets/sileod/missing-item-prediction) | contrastive |  | sentence1=fn, labels=fn |
| jigsaw_toxicity | Classification | [tasksource/jigsaw_toxicity](https://hf.co/datasets/tasksource/jigsaw_toxicity) |  |  | sentence1=comment_text, labels=fn |
| Pol_NLI | Classification | [mlburnham/Pol_NLI](https://hf.co/datasets/mlburnham/Pol_NLI) |  |  | sentence1=premise, sentence2=hypothesis, labels=fn |
| synthetic-retrieval-NLI/position | Classification | [tasksource/synthetic-retrieval-NLI](https://hf.co/datasets/tasksource/synthetic-retrieval-NLI) | position |  | sentence1=premise, sentence2=hypothesis, labels=label, pre_process |
| synthetic-retrieval-NLI/binary | Classification | [tasksource/synthetic-retrieval-NLI](https://hf.co/datasets/tasksource/synthetic-retrieval-NLI) | binary |  | sentence1=premise, sentence2=hypothesis, labels=label, pre_process |
| synthetic-retrieval-NLI/count | Classification | [tasksource/synthetic-retrieval-NLI](https://hf.co/datasets/tasksource/synthetic-retrieval-NLI) | count |  | sentence1=premise, sentence2=hypothesis, labels=label, pre_process |
| github-issue-similarity | Classification | [WhereIsAI/github-issue-similarity](https://hf.co/datasets/WhereIsAI/github-issue-similarity) |  |  | sentence1=fn, sentence2=fn, labels=label, label_values={…} |
