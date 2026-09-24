504 English tasks. Load one with `load_task(id)`; the annotations are in [tasks.py](src/tasksource/tasks.py), and tasks kept out on purpose are in [parked.py](src/tasksource/parked.py).

| # | id | type | dataset | question |
|--:|---|---|---|:-:|
| 1 | [glue/mnli](src/tasksource/tasks.py#L28) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 2 | [glue/qnli](src/tasksource/tasks.py#L29) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 3 | [glue/rte](src/tasksource/tasks.py#L30) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 4 | [glue/wnli](src/tasksource/tasks.py#L31) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 5 | [glue/mrpc](src/tasksource/tasks.py#L33) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 6 | [glue/qqp](src/tasksource/tasks.py#L34) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 7 | [glue/stsb](src/tasksource/tasks.py#L35) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) | ✓ |
| 8 | [super_glue/boolq](src/tasksource/tasks.py#L38) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 9 | [super_glue/boolq_passage](src/tasksource/tasks.py#L39) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 10 | [super_glue/cb](src/tasksource/tasks.py#L41) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 11 | [super_glue/multirc](src/tasksource/tasks.py#L42) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 12 | [super_glue/wic](src/tasksource/tasks.py#L47) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 13 | [super_glue/axg](src/tasksource/tasks.py#L52) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 14 | [anli/a1](src/tasksource/tasks.py#L55) | Classification | [facebook/anli](https://hf.co/datasets/facebook/anli) |  |
| 15 | [anli/a2](src/tasksource/tasks.py#L56) | Classification | [facebook/anli](https://hf.co/datasets/facebook/anli) |  |
| 16 | [anli/a3](src/tasksource/tasks.py#L57) | Classification | [facebook/anli](https://hf.co/datasets/facebook/anli) |  |
| 17 | [babi_nli/counting](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 18 | [babi_nli/indefinite-knowledge](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 19 | [babi_nli/lists-sets](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 20 | [babi_nli/path-finding](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 21 | [babi_nli/positional-reasoning](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 22 | [babi_nli/simple-negation](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 23 | [babi_nli/size-reasoning](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 24 | [babi_nli/conjunction](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 25 | [babi_nli/three-arg-relations](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 26 | [babi_nli/three-supporting-facts](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 27 | [babi_nli/time-reasoning](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 28 | [babi_nli/two-arg-relations](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 29 | [babi_nli/single-supporting-fact](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 30 | [babi_nli/compound-coreference](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 31 | [babi_nli/basic-deduction](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 32 | [babi_nli/basic-coreference](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 33 | [babi_nli/two-supporting-facts](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 34 | [babi_nli/basic-induction](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 35 | [babi_nli/yes-no-questions](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 36 | [sick/label](src/tasksource/tasks.py#L66) | Classification | [tasksource/sick](https://hf.co/datasets/tasksource/sick) |  |
| 37 | [sick/relatedness](src/tasksource/tasks.py#L67) | Classification | [tasksource/sick](https://hf.co/datasets/tasksource/sick) | ✓ |
| 38 | [snli](src/tasksource/tasks.py#L122) | Classification | [stanfordnlp/snli](https://hf.co/datasets/stanfordnlp/snli) |  |
| 39 | [scitail/snli_format](src/tasksource/tasks.py#L125) | Classification | [allenai/scitail](https://hf.co/datasets/allenai/scitail) |  |
| 40 | [hans](src/tasksource/tasks.py#L127) | Classification | [tasksource/hans](https://hf.co/datasets/tasksource/hans) |  |
| 41 | [WANLI](src/tasksource/tasks.py#L130) | Classification | [alisawuffles/WANLI](https://hf.co/datasets/alisawuffles/WANLI) |  |
| 42 | [recast/recast_factuality](src/tasksource/tasks.py#L132) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 43 | [recast/recast_verbnet](src/tasksource/tasks.py#L132) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 44 | [recast/recast_puns](src/tasksource/tasks.py#L132) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 45 | [recast/recast_ner](src/tasksource/tasks.py#L132) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 46 | [recast/recast_sentiment](src/tasksource/tasks.py#L132) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 47 | [recast/recast_megaveridicality](src/tasksource/tasks.py#L132) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 48 | [recast/recast_verbcorner](src/tasksource/tasks.py#L132) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 49 | [probability_words_nli/usnli](src/tasksource/tasks.py#L137) | Classification | [sileod/probability_words_nli](https://hf.co/datasets/sileod/probability_words_nli) |  |
| 50 | [probability_words_nli/reasoning_2hop](src/tasksource/tasks.py#L137) | Classification | [sileod/probability_words_nli](https://hf.co/datasets/sileod/probability_words_nli) |  |
| 51 | [probability_words_nli/reasoning_1hop](src/tasksource/tasks.py#L137) | Classification | [sileod/probability_words_nli](https://hf.co/datasets/sileod/probability_words_nli) |  |
| 52 | [nan-nli](src/tasksource/tasks.py#L141) | Classification | [joey234/nan-nli](https://hf.co/datasets/joey234/nan-nli) |  |
| 53 | [nli_fever](src/tasksource/tasks.py#L143) | Classification | [pietrolesci/nli_fever](https://hf.co/datasets/pietrolesci/nli_fever) |  |
| 54 | [breaking_nli](src/tasksource/tasks.py#L146) | Classification | [pietrolesci/breaking_nli](https://hf.co/datasets/pietrolesci/breaking_nli) |  |
| 55 | [conj_nli](src/tasksource/tasks.py#L150) | Classification | [pietrolesci/conj_nli](https://hf.co/datasets/pietrolesci/conj_nli) |  |
| 56 | [fracas](src/tasksource/tasks.py#L154) | Classification | [pietrolesci/fracas](https://hf.co/datasets/pietrolesci/fracas) |  |
| 57 | [dialogue_nli](src/tasksource/tasks.py#L157) | Classification | [pietrolesci/dialogue_nli](https://hf.co/datasets/pietrolesci/dialogue_nli) |  |
| 58 | [mpe](src/tasksource/tasks.py#L160) | Classification | [pietrolesci/mpe](https://hf.co/datasets/pietrolesci/mpe) |  |
| 59 | [dnc](src/tasksource/tasks.py#L164) | Classification | [pietrolesci/dnc](https://hf.co/datasets/pietrolesci/dnc) |  |
| 60 | [recast_white/fnplus](src/tasksource/tasks.py#L168) | Classification | [pietrolesci/recast_white](https://hf.co/datasets/pietrolesci/recast_white) |  |
| 61 | [recast_white/sprl](src/tasksource/tasks.py#L171) | Classification | [pietrolesci/recast_white](https://hf.co/datasets/pietrolesci/recast_white) |  |
| 62 | [recast_white/dpr](src/tasksource/tasks.py#L174) | Classification | [pietrolesci/recast_white](https://hf.co/datasets/pietrolesci/recast_white) |  |
| 63 | [joci](src/tasksource/tasks.py#L178) | Classification | [pietrolesci/joci](https://hf.co/datasets/pietrolesci/joci) |  |
| 64 | [robust_nli/IS_CS](src/tasksource/tasks.py#L184) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 65 | [robust_nli/LI_LI](src/tasksource/tasks.py#L186) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 66 | [robust_nli/ST_WO](src/tasksource/tasks.py#L188) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 67 | [robust_nli/PI_SP](src/tasksource/tasks.py#L190) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 68 | [robust_nli/PI_CD](src/tasksource/tasks.py#L192) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 69 | [robust_nli/ST_SE](src/tasksource/tasks.py#L194) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 70 | [robust_nli/ST_NE](src/tasksource/tasks.py#L196) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 71 | [robust_nli/ST_LM](src/tasksource/tasks.py#L198) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 72 | [robust_nli_is_sd](src/tasksource/tasks.py#L200) | Classification | [pietrolesci/robust_nli_is_sd](https://hf.co/datasets/pietrolesci/robust_nli_is_sd) |  |
| 73 | [robust_nli_li_ts](src/tasksource/tasks.py#L203) | Classification | [pietrolesci/robust_nli_li_ts](https://hf.co/datasets/pietrolesci/robust_nli_li_ts) |  |
| 74 | [gen_debiased_nli/snli_seq_z](src/tasksource/tasks.py#L207) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| 75 | [gen_debiased_nli/snli_z_aug](src/tasksource/tasks.py#L209) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| 76 | [gen_debiased_nli/snli_par_z](src/tasksource/tasks.py#L211) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| 77 | [gen_debiased_nli/mnli_par_z](src/tasksource/tasks.py#L213) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| 78 | [gen_debiased_nli/mnli_z_aug](src/tasksource/tasks.py#L215) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| 79 | [gen_debiased_nli/mnli_seq_z](src/tasksource/tasks.py#L217) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| 80 | [add_one_rte](src/tasksource/tasks.py#L220) | Classification | [pietrolesci/add_one_rte](https://hf.co/datasets/pietrolesci/add_one_rte) |  |
| 81 | [imppres/presupposition_question_presupposition/presupposition](src/tasksource/tasks.py#L230) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| 82 | [imppres/presupposition_possessed_definites_uniqueness/presupposition](src/tasksource/tasks.py#L230) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| 83 | [imppres/presupposition_possessed_definites_existence/presupposition](src/tasksource/tasks.py#L230) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| 84 | [imppres/presupposition_only_presupposition/presupposition](src/tasksource/tasks.py#L230) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| 85 | [imppres/presupposition_cleft_uniqueness/presupposition](src/tasksource/tasks.py#L230) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| 86 | [imppres/presupposition_cleft_existence/presupposition](src/tasksource/tasks.py#L230) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| 87 | [imppres/presupposition_change_of_state/presupposition](src/tasksource/tasks.py#L230) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| 88 | [imppres/presupposition_both_presupposition/presupposition](src/tasksource/tasks.py#L230) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| 89 | [imppres/presupposition_all_n_presupposition/presupposition](src/tasksource/tasks.py#L230) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| 90 | [imppres/implicature_numerals_2_3/prag](src/tasksource/tasks.py#L234) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| 91 | [imppres/implicature_numerals_10_100/prag](src/tasksource/tasks.py#L234) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| 92 | [imppres/implicature_modals/prag](src/tasksource/tasks.py#L234) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| 93 | [imppres/implicature_gradable_verb/prag](src/tasksource/tasks.py#L234) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| 94 | [imppres/implicature_gradable_adjective/prag](src/tasksource/tasks.py#L234) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| 95 | [imppres/implicature_connectives/prag](src/tasksource/tasks.py#L234) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| 96 | [imppres/implicature_quantifiers/prag](src/tasksource/tasks.py#L234) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| 97 | [imppres/implicature_modals/log](src/tasksource/tasks.py#L238) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| 98 | [imppres/implicature_gradable_verb/log](src/tasksource/tasks.py#L238) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| 99 | [imppres/implicature_gradable_adjective/log](src/tasksource/tasks.py#L238) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| 100 | [imppres/implicature_quantifiers/log](src/tasksource/tasks.py#L238) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| 101 | [imppres/implicature_numerals_2_3/log](src/tasksource/tasks.py#L238) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| 102 | [imppres/implicature_connectives/log](src/tasksource/tasks.py#L238) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| 103 | [imppres/implicature_numerals_10_100/log](src/tasksource/tasks.py#L238) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| 104 | [hlgd](src/tasksource/tasks.py#L243) | Classification | [tasksource/hlgd](https://hf.co/datasets/tasksource/hlgd) |  |
| 105 | [paws/labeled_final](src/tasksource/tasks.py#L245) | Classification | [google-research-datasets/paws](https://hf.co/datasets/google-research-datasets/paws) |  |
| 106 | [paws/labeled_swap](src/tasksource/tasks.py#L246) | Classification | [google-research-datasets/paws](https://hf.co/datasets/google-research-datasets/paws) |  |
| 107 | [medical_questions_pairs](src/tasksource/tasks.py#L248) | Classification | [curaihealth/medical_questions_pairs](https://hf.co/datasets/curaihealth/medical_questions_pairs) |  |
| 108 | [conll2003/pos_tags](src/tasksource/tasks.py#L253) | TokenClassification | [tomaarsen/conll2003](https://hf.co/datasets/tomaarsen/conll2003) |  |
| 109 | [conll2003/chunk_tags](src/tasksource/tasks.py#L254) | TokenClassification | [tomaarsen/conll2003](https://hf.co/datasets/tomaarsen/conll2003) |  |
| 110 | [conll2003/ner_tags](src/tasksource/tasks.py#L255) | TokenClassification | [tomaarsen/conll2003](https://hf.co/datasets/tomaarsen/conll2003) |  |
| 111 | [fig-qa](src/tasksource/tasks.py#L261) | MultipleChoice | [nightingal3/fig-qa](https://hf.co/datasets/nightingal3/fig-qa) |  |
| 112 | [cos_e/v1.0](src/tasksource/tasks.py#L270) | MultipleChoice | [Salesforce/cos_e](https://hf.co/datasets/Salesforce/cos_e) |  |
| 113 | [cosmos_qa](src/tasksource/tasks.py#L275) | MultipleChoice | [Samsoup/cosmos_qa](https://hf.co/datasets/Samsoup/cosmos_qa) |  |
| 114 | [dream](src/tasksource/tasks.py#L278) | MultipleChoice | [dataset-org/dream](https://hf.co/datasets/dataset-org/dream) |  |
| 115 | [openbookqa](src/tasksource/tasks.py#L285) | MultipleChoice | [allenai/openbookqa](https://hf.co/datasets/allenai/openbookqa) |  |
| 116 | [qasc](src/tasksource/tasks.py#L291) | MultipleChoice | [allenai/qasc](https://hf.co/datasets/allenai/qasc) |  |
| 117 | [quartz](src/tasksource/tasks.py#L299) | MultipleChoice | [allenai/quartz](https://hf.co/datasets/allenai/quartz) |  |
| 118 | [quail](src/tasksource/tasks.py#L304) | MultipleChoice | [textmachinelab/quail](https://hf.co/datasets/textmachinelab/quail) |  |
| 119 | [head_qa/en](src/tasksource/tasks.py#L310) | MultipleChoice | [EleutherAI/headqa](https://hf.co/datasets/EleutherAI/headqa) |  |
| 120 | [sciq](src/tasksource/tasks.py#L318) | MultipleChoice | [allenai/sciq](https://hf.co/datasets/allenai/sciq) |  |
| 121 | [social_i_qa](src/tasksource/tasks.py#L323) | MultipleChoice | [tasksource/social_i_qa](https://hf.co/datasets/tasksource/social_i_qa) |  |
| 122 | [wiki_hop/original](src/tasksource/tasks.py#L329) | MultipleChoice | [MoE-UNC/wikihop](https://hf.co/datasets/MoE-UNC/wikihop) |  |
| 123 | [wiqa](src/tasksource/tasks.py#L336) | MultipleChoice | [tasksource/wiqa](https://hf.co/datasets/tasksource/wiqa) |  |
| 124 | [piqa](src/tasksource/tasks.py#L341) | MultipleChoice | [baber/piqa](https://hf.co/datasets/baber/piqa) |  |
| 125 | [hellaswag](src/tasksource/tasks.py#L349) | MultipleChoice | [Rowan/hellaswag](https://hf.co/datasets/Rowan/hellaswag) |  |
| 126 | [super_glue/copa](src/tasksource/tasks.py#L358) | MultipleChoice | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 127 | [balanced-copa](src/tasksource/tasks.py#L360) | MultipleChoice | [pkavumba/balanced-copa](https://hf.co/datasets/pkavumba/balanced-copa) |  |
| 128 | [e-CARE](src/tasksource/tasks.py#L363) | MultipleChoice | [12ml/e-CARE](https://hf.co/datasets/12ml/e-CARE) |  |
| 129 | [art](src/tasksource/tasks.py#L366) | MultipleChoice | [allenai/art](https://hf.co/datasets/allenai/art) | ✓ |
| 130 | [winogrande/winogrande_xl](src/tasksource/tasks.py#L374) | MultipleChoice | [allenai/winogrande](https://hf.co/datasets/allenai/winogrande) |  |
| 131 | [codah/codah](src/tasksource/tasks.py#L377) | MultipleChoice | [jaredfern/codah](https://hf.co/datasets/jaredfern/codah) |  |
| 132 | [ai2_arc/ARC-Easy/challenge](src/tasksource/tasks.py#L379) | MultipleChoice | [allenai/ai2_arc](https://hf.co/datasets/allenai/ai2_arc) |  |
| 133 | [ai2_arc/ARC-Challenge/challenge](src/tasksource/tasks.py#L379) | MultipleChoice | [allenai/ai2_arc](https://hf.co/datasets/allenai/ai2_arc) |  |
| 134 | [definite_pronoun_resolution](src/tasksource/tasks.py#L384) | MultipleChoice | [community-datasets/definite_pronoun_resolution](https://hf.co/datasets/community-datasets/definite_pronoun_resolution) |  |
| 135 | [swag/regular](src/tasksource/tasks.py#L390) | MultipleChoice | [allenai/swag](https://hf.co/datasets/allenai/swag) |  |
| 136 | [math_qa](src/tasksource/tasks.py#L396) | MultipleChoice | [tasksource/math_qa](https://hf.co/datasets/tasksource/math_qa) |  |
| 137 | [glue/cola](src/tasksource/tasks.py#L405) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 138 | [glue/sst2](src/tasksource/tasks.py#L406) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 139 | [utilitarianism](src/tasksource/tasks.py#L420) | Classification | csv |  |
| 140 | [amazon_counterfactual/en](src/tasksource/tasks.py#L428) | Classification | [mteb/amazon_counterfactual](https://hf.co/datasets/mteb/amazon_counterfactual) |  |
| 141 | [insincere-questions](src/tasksource/tasks.py#L433) | Classification | [SetFit/insincere-questions](https://hf.co/datasets/SetFit/insincere-questions) |  |
| 142 | [toxic_conversations](src/tasksource/tasks.py#L437) | Classification | [SetFit/toxic_conversations](https://hf.co/datasets/SetFit/toxic_conversations) |  |
| 143 | [TuringBench](src/tasksource/tasks.py#L441) | Classification | csv |  |
| 144 | [trec](src/tasksource/tasks.py#L450) | Classification | [tasksource/trec](https://hf.co/datasets/tasksource/trec) |  |
| 145 | [vitaminc](src/tasksource/tasks.py#L453) | Classification | [tals/vitaminc](https://hf.co/datasets/tals/vitaminc) |  |
| 146 | [hope_edi/english](src/tasksource/tasks.py#L455) | Classification | csv |  |
| 147 | [rumoureval_2019/RumourEval2019](src/tasksource/tasks.py#L469) | Classification | csv |  |
| 148 | [ethos/binary](src/tasksource/tasks.py#L483) | Classification | [SetFit/ethos_binary](https://hf.co/datasets/SetFit/ethos_binary) |  |
| 149 | [ethos/multilabel](src/tasksource/tasks.py#L507) | Classification | [tasksource/ethos](https://hf.co/datasets/tasksource/ethos) |  |
| 150 | [tweet_eval/emoji](src/tasksource/tasks.py#L510) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 151 | [tweet_eval/emotion](src/tasksource/tasks.py#L510) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 152 | [tweet_eval/hate](src/tasksource/tasks.py#L510) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 153 | [tweet_eval/irony](src/tasksource/tasks.py#L510) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 154 | [tweet_eval/offensive](src/tasksource/tasks.py#L510) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 155 | [tweet_eval/sentiment](src/tasksource/tasks.py#L510) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 156 | [tweet_eval/stance_abortion](src/tasksource/tasks.py#L525) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 157 | [tweet_eval/stance_atheism](src/tasksource/tasks.py#L526) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 158 | [tweet_eval/stance_climate](src/tasksource/tasks.py#L527) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 159 | [tweet_eval/stance_feminist](src/tasksource/tasks.py#L528) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 160 | [tweet_eval/stance_hillary](src/tasksource/tasks.py#L529) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 161 | [discovery/discovery](src/tasksource/tasks.py#L532) | Classification | [sileod/discovery](https://hf.co/datasets/sileod/discovery) |  |
| 162 | [pragmeval/switchboard](src/tasksource/tasks.py#L534) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 163 | [pragmeval/verifiability](src/tasksource/tasks.py#L534) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 164 | [pragmeval/mrda](src/tasksource/tasks.py#L534) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 165 | [pragmeval/emergent](src/tasksource/tasks.py#L538) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 166 | [pragmeval/gum](src/tasksource/tasks.py#L538) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 167 | [pragmeval/pdtb](src/tasksource/tasks.py#L538) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 168 | [pragmeval/persuasiveness-claimtype](src/tasksource/tasks.py#L538) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 169 | [pragmeval/persuasiveness-premisetype](src/tasksource/tasks.py#L538) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 170 | [pragmeval/stac](src/tasksource/tasks.py#L538) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 171 | [pragmeval/sarcasm](src/tasksource/tasks.py#L538) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 172 | [pragmeval/emobank-arousal](src/tasksource/tasks.py#L547) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 173 | [pragmeval/emobank-dominance](src/tasksource/tasks.py#L548) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 174 | [pragmeval/emobank-valence](src/tasksource/tasks.py#L549) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 175 | [pragmeval/squinky-formality](src/tasksource/tasks.py#L550) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 176 | [pragmeval/squinky-implicature](src/tasksource/tasks.py#L551) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 177 | [pragmeval/squinky-informativeness](src/tasksource/tasks.py#L552) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 178 | [pragmeval/persuasiveness-eloquence](src/tasksource/tasks.py#L553) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 179 | [pragmeval/persuasiveness-relevance](src/tasksource/tasks.py#L554) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 180 | [pragmeval/persuasiveness-specificity](src/tasksource/tasks.py#L555) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 181 | [pragmeval/persuasiveness-strength](src/tasksource/tasks.py#L556) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 182 | [silicone/oasis](src/tasksource/tasks.py#L558) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| 183 | [silicone/sem](src/tasksource/tasks.py#L558) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| 184 | [silicone/meld_s](src/tasksource/tasks.py#L558) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| 185 | [silicone/meld_e](src/tasksource/tasks.py#L558) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| 186 | [silicone/maptask](src/tasksource/tasks.py#L558) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| 187 | [silicone/dyda_e](src/tasksource/tasks.py#L558) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| 188 | [silicone/dyda_da](src/tasksource/tasks.py#L558) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| 189 | [silicone/iemocap](src/tasksource/tasks.py#L565) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| 190 | [lex_glue/eurlex](src/tasksource/tasks.py#L570) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) |  |
| 191 | [lex_glue/scotus](src/tasksource/tasks.py#L572) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) |  |
| 192 | [lex_glue/ledgar](src/tasksource/tasks.py#L575) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) |  |
| 193 | [lex_glue/unfair_tos](src/tasksource/tasks.py#L577) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) | ✓ |
| 194 | [lex_glue/case_hold](src/tasksource/tasks.py#L580) | MultipleChoice | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) |  |
| 195 | [language-identification](src/tasksource/tasks.py#L588) | Classification | [papluca/language-identification](https://hf.co/datasets/papluca/language-identification) | ✓ |
| 196 | [imdb](src/tasksource/tasks.py#L593) | Classification | [stanfordnlp/imdb](https://hf.co/datasets/stanfordnlp/imdb) |  |
| 197 | [rotten_tomatoes](src/tasksource/tasks.py#L595) | Classification | [cornell-movie-review-data/rotten_tomatoes](https://hf.co/datasets/cornell-movie-review-data/rotten_tomatoes) |  |
| 198 | [ag_news](src/tasksource/tasks.py#L597) | Classification | [fancyzhx/ag_news](https://hf.co/datasets/fancyzhx/ag_news) |  |
| 199 | [yelp_review_full/yelp_review_full](src/tasksource/tasks.py#L599) | Classification | [Yelp/yelp_review_full](https://hf.co/datasets/Yelp/yelp_review_full) |  |
| 200 | [financial_phrasebank/sentences_allagree](src/tasksource/tasks.py#L603) | Classification | [ghbacct/financial-phrasebank-all-agree-classification](https://hf.co/datasets/ghbacct/financial-phrasebank-all-agree-classification) |  |
| 201 | [poem_sentiment](src/tasksource/tasks.py#L608) | Classification | [google-research-datasets/poem_sentiment](https://hf.co/datasets/google-research-datasets/poem_sentiment) |  |
| 202 | [emotion](src/tasksource/tasks.py#L610) | Classification | [dair-ai/emotion](https://hf.co/datasets/dair-ai/emotion) |  |
| 203 | [dbpedia_14/dbpedia_14](src/tasksource/tasks.py#L612) | Classification | [fancyzhx/dbpedia_14](https://hf.co/datasets/fancyzhx/dbpedia_14) |  |
| 204 | [amazon_polarity/amazon_polarity](src/tasksource/tasks.py#L614) | Classification | [fancyzhx/amazon_polarity](https://hf.co/datasets/fancyzhx/amazon_polarity) |  |
| 205 | [app_reviews](src/tasksource/tasks.py#L616) | Classification | [sealuzh/app_reviews](https://hf.co/datasets/sealuzh/app_reviews) |  |
| 206 | [hate_speech18](src/tasksource/tasks.py#L620) | Classification | [tasksource/hate_speech18](https://hf.co/datasets/tasksource/hate_speech18) |  |
| 207 | [sms_spam](src/tasksource/tasks.py#L626) | Classification | [ucirvine/sms_spam](https://hf.co/datasets/ucirvine/sms_spam) |  |
| 208 | [humicroedit/subtask-1](src/tasksource/tasks.py#L629) | Classification | [tasksource/humicroedit](https://hf.co/datasets/tasksource/humicroedit) |  |
| 209 | [humicroedit/subtask-2](src/tasksource/tasks.py#L635) | Classification | [tasksource/humicroedit](https://hf.co/datasets/tasksource/humicroedit) | ✓ |
| 210 | [snips_built_in_intents](src/tasksource/tasks.py#L640) | Classification | [sonos-nlu-benchmark/snips_built_in_intents](https://hf.co/datasets/sonos-nlu-benchmark/snips_built_in_intents) |  |
| 211 | [hate_speech_offensive](src/tasksource/tasks.py#L644) | Classification | [tdavidson/hate_speech_offensive](https://hf.co/datasets/tdavidson/hate_speech_offensive) |  |
| 212 | [yahoo_answers_topics](src/tasksource/tasks.py#L646) | Classification | [community-datasets/yahoo_answers_topics](https://hf.co/datasets/community-datasets/yahoo_answers_topics) |  |
| 213 | [stackoverflow-questions](src/tasksource/tasks.py#L650) | Classification | [pacovaldez/stackoverflow-questions](https://hf.co/datasets/pacovaldez/stackoverflow-questions) |  |
| 214 | [hyperpartisan_news](src/tasksource/tasks.py#L655) | Classification | [zapsdcn/hyperpartisan_news](https://hf.co/datasets/zapsdcn/hyperpartisan_news) |  |
| 215 | [sciie](src/tasksource/tasks.py#L660) | Classification | [zapsdcn/sciie](https://hf.co/datasets/zapsdcn/sciie) |  |
| 216 | [citation_intent](src/tasksource/tasks.py#L661) | Classification | [zapsdcn/citation_intent](https://hf.co/datasets/zapsdcn/citation_intent) |  |
| 217 | [go_emotions/simplified](src/tasksource/tasks.py#L663) | Classification | [google-research-datasets/go_emotions](https://hf.co/datasets/google-research-datasets/go_emotions) |  |
| 218 | [scicite](src/tasksource/tasks.py#L667) | Classification | [tasksource/scicite](https://hf.co/datasets/tasksource/scicite) |  |
| 219 | [liar](src/tasksource/tasks.py#L669) | Classification | [tasksource/liar](https://hf.co/datasets/tasksource/liar) |  |
| 220 | [lexical_relation_classification/ROOT09](src/tasksource/tasks.py#L678) | Classification | json | ✓ |
| 221 | [lexical_relation_classification/K&H+N](src/tasksource/tasks.py#L678) | Classification | json | ✓ |
| 222 | [lexical_relation_classification/BLESS](src/tasksource/tasks.py#L678) | Classification | json | ✓ |
| 223 | [lexical_relation_classification/EVALution](src/tasksource/tasks.py#L678) | Classification | json | ✓ |
| 224 | [lexical_relation_classification/CogALexV](src/tasksource/tasks.py#L704) | Classification | json | ✓ |
| 225 | [linguisticprobing/subj_number](src/tasksource/tasks.py#L722) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 226 | [linguisticprobing/obj_number](src/tasksource/tasks.py#L723) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 227 | [linguisticprobing/past_present](src/tasksource/tasks.py#L724) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 228 | [linguisticprobing/sentence_length](src/tasksource/tasks.py#L725) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 229 | [linguisticprobing/top_constituents](src/tasksource/tasks.py#L726) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 230 | [linguisticprobing/tree_depth](src/tasksource/tasks.py#L728) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 231 | [linguisticprobing/coordination_inversion](src/tasksource/tasks.py#L729) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 232 | [linguisticprobing/odd_man_out](src/tasksource/tasks.py#L731) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 233 | [linguisticprobing/bigram_shift](src/tasksource/tasks.py#L732) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 234 | [crowdflower/airline-sentiment](src/tasksource/tasks.py#L734) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) |  |
| 235 | [crowdflower/corporate-messaging](src/tasksource/tasks.py#L734) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) |  |
| 236 | [crowdflower/economic-news](src/tasksource/tasks.py#L734) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) |  |
| 237 | [crowdflower/political-media-audience](src/tasksource/tasks.py#L734) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) |  |
| 238 | [crowdflower/political-media-bias](src/tasksource/tasks.py#L734) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) |  |
| 239 | [crowdflower/political-media-message](src/tasksource/tasks.py#L734) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) |  |
| 240 | [crowdflower/text_emotion](src/tasksource/tasks.py#L734) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) |  |
| 241 | [crowdflower/sentiment_nuclear_power](src/tasksource/tasks.py#L734) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) |  |
| 242 | [crowdflower/tweet_global_warming](src/tasksource/tasks.py#L734) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) |  |
| 243 | [ethics/commonsense](src/tasksource/tasks.py#L759) | Classification | csv |  |
| 244 | [ethics/deontology](src/tasksource/tasks.py#L767) | Classification | csv |  |
| 245 | [ethics/justice](src/tasksource/tasks.py#L775) | Classification | csv |  |
| 246 | [ethics/virtue](src/tasksource/tasks.py#L783) | Classification | [hendrycks/ethics](https://hf.co/datasets/hendrycks/ethics) |  |
| 247 | [emo/emo2019](src/tasksource/tasks.py#L792) | Classification | [oneonlee/cleansed_emocontext](https://hf.co/datasets/oneonlee/cleansed_emocontext) |  |
| 248 | [google_wellformed_query](src/tasksource/tasks.py#L798) | Classification | [tasksource/google_wellformed_query](https://hf.co/datasets/tasksource/google_wellformed_query) | ✓ |
| 249 | [tweets_hate_speech_detection](src/tasksource/tasks.py#L803) | Classification | [tweets-hate-speech-detection/tweets_hate_speech_detection](https://hf.co/datasets/tweets-hate-speech-detection/tweets_hate_speech_detection) |  |
| 250 | [wnut_17/wnut_17](src/tasksource/tasks.py#L807) | TokenClassification | [flaitenberger/wnut_17](https://hf.co/datasets/flaitenberger/wnut_17) |  |
| 251 | [ncbi_disease/ncbi_disease](src/tasksource/tasks.py#L810) | TokenClassification | [ncbi/ncbi_disease](https://hf.co/datasets/ncbi/ncbi_disease) |  |
| 252 | [acronym_identification](src/tasksource/tasks.py#L813) | TokenClassification | [amirveyseh/acronym_identification](https://hf.co/datasets/amirveyseh/acronym_identification) |  |
| 253 | [jnlpba/jnlpba](src/tasksource/tasks.py#L816) | TokenClassification | [jnlpba/jnlpba](https://hf.co/datasets/jnlpba/jnlpba) |  |
| 254 | [ontonotes_english/SpeedOfMagic--ontonotes_english](src/tasksource/tasks.py#L823) | TokenClassification | [SpeedOfMagic/ontonotes_english](https://hf.co/datasets/SpeedOfMagic/ontonotes_english) |  |
| 255 | [blog_authorship_corpus/gender](src/tasksource/tasks.py#L827) | Classification | [tasksource/blog_authorship_corpus](https://hf.co/datasets/tasksource/blog_authorship_corpus) | ✓ |
| 256 | [blog_authorship_corpus/age](src/tasksource/tasks.py#L829) | Classification | [tasksource/blog_authorship_corpus](https://hf.co/datasets/tasksource/blog_authorship_corpus) | ✓ |
| 257 | [blog_authorship_corpus/job](src/tasksource/tasks.py#L832) | Classification | [tasksource/blog_authorship_corpus](https://hf.co/datasets/tasksource/blog_authorship_corpus) | ✓ |
| 258 | [open_question_type](src/tasksource/tasks.py#L843) | Classification | [Korea-MES/open_question_type](https://hf.co/datasets/Korea-MES/open_question_type) |  |
| 259 | [health_fact](src/tasksource/tasks.py#L845) | Classification | [marcov/health_fact_promptsource](https://hf.co/datasets/marcov/health_fact_promptsource) |  |
| 260 | [commonsense_qa](src/tasksource/tasks.py#L849) | MultipleChoice | [tau/commonsense_qa](https://hf.co/datasets/tau/commonsense_qa) |  |
| 261 | [mc_taco](src/tasksource/tasks.py#L855) | Classification | [marcov/mc_taco_promptsource](https://hf.co/datasets/marcov/mc_taco_promptsource) | ✓ |
| 262 | [ade_corpus_v2/Ade_corpus_v2_classification](src/tasksource/tasks.py#L862) | Classification | [ade-benchmark-corpus/ade_corpus_v2](https://hf.co/datasets/ade-benchmark-corpus/ade_corpus_v2) |  |
| 263 | [discosense](src/tasksource/tasks.py#L864) | MultipleChoice | json |  |
| 264 | [circa](src/tasksource/tasks.py#L871) | Classification | [google-research-datasets/circa](https://hf.co/datasets/google-research-datasets/circa) |  |
| 265 | [code_x_glue_cc_defect_detection](src/tasksource/tasks.py#L876) | Classification | [google/code_x_glue_cc_defect_detection](https://hf.co/datasets/google/code_x_glue_cc_defect_detection) |  |
| 266 | [phrase_similarity](src/tasksource/tasks.py#L880) | Classification | [Deehan1866/processed_phrase_similarity](https://hf.co/datasets/Deehan1866/processed_phrase_similarity) |  |
| 267 | [scientific-exaggeration-detection](src/tasksource/tasks.py#L888) | Classification | [copenlu/scientific-exaggeration-detection](https://hf.co/datasets/copenlu/scientific-exaggeration-detection) |  |
| 268 | [quarel](src/tasksource/tasks.py#L894) | Classification | [community-datasets/quarel](https://hf.co/datasets/community-datasets/quarel) |  |
| 269 | [fever-evidence-related](src/tasksource/tasks.py#L899) | Classification | [mwong/fever-evidence-related](https://hf.co/datasets/mwong/fever-evidence-related) |  |
| 270 | [numer_sense](src/tasksource/tasks.py#L902) | Classification | [tasksource/numer_sense](https://hf.co/datasets/tasksource/numer_sense) |  |
| 271 | [dynasent/dynabench.dynasent.r1.all/r1](src/tasksource/tasks.py#L909) | Classification | [tasksource/dynasent](https://hf.co/datasets/tasksource/dynasent) |  |
| 272 | [dynasent/dynabench.dynasent.r2.all/r2](src/tasksource/tasks.py#L913) | Classification | [tasksource/dynasent](https://hf.co/datasets/tasksource/dynasent) |  |
| 273 | [Sarcasm_News_Headline](src/tasksource/tasks.py#L918) | Classification | [raquiba/Sarcasm_News_Headline](https://hf.co/datasets/raquiba/Sarcasm_News_Headline) |  |
| 274 | [sem_eval_2010_task_8](src/tasksource/tasks.py#L921) | Classification | [SemEvalWorkshop/sem_eval_2010_task_8](https://hf.co/datasets/SemEvalWorkshop/sem_eval_2010_task_8) |  |
| 275 | [auditor_review](src/tasksource/tasks.py#L923) | Classification | [demo-org/auditor_review](https://hf.co/datasets/demo-org/auditor_review) |  |
| 276 | [medmcqa](src/tasksource/tasks.py#L927) | MultipleChoice | [openlifescienceai/medmcqa](https://hf.co/datasets/openlifescienceai/medmcqa) |  |
| 277 | [Dynasent_Disagreement](src/tasksource/tasks.py#L944) | Classification | [RuyuanWan/Dynasent_Disagreement](https://hf.co/datasets/RuyuanWan/Dynasent_Disagreement) | ✓ |
| 278 | [Politeness_Disagreement](src/tasksource/tasks.py#L946) | Classification | [RuyuanWan/Politeness_Disagreement](https://hf.co/datasets/RuyuanWan/Politeness_Disagreement) | ✓ |
| 279 | [SBIC_Disagreement](src/tasksource/tasks.py#L948) | Classification | [RuyuanWan/SBIC_Disagreement](https://hf.co/datasets/RuyuanWan/SBIC_Disagreement) | ✓ |
| 280 | [SChem_Disagreement](src/tasksource/tasks.py#L950) | Classification | [RuyuanWan/SChem_Disagreement](https://hf.co/datasets/RuyuanWan/SChem_Disagreement) | ✓ |
| 281 | [Dilemmas_Disagreement](src/tasksource/tasks.py#L952) | Classification | [RuyuanWan/Dilemmas_Disagreement](https://hf.co/datasets/RuyuanWan/Dilemmas_Disagreement) | ✓ |
| 282 | [logiqa](src/tasksource/tasks.py#L955) | MultipleChoice | [fireworks-ai/logiqa](https://hf.co/datasets/fireworks-ai/logiqa) |  |
| 283 | [wiki_qa](src/tasksource/tasks.py#L964) | Classification | [microsoft/wiki_qa](https://hf.co/datasets/microsoft/wiki_qa) | ✓ |
| 284 | [cycic_classification](src/tasksource/tasks.py#L966) | Classification | [tasksource/cycic_classification](https://hf.co/datasets/tasksource/cycic_classification) |  |
| 285 | [cycic_multiplechoice](src/tasksource/tasks.py#L968) | MultipleChoice | [tasksource/cycic_multiplechoice](https://hf.co/datasets/tasksource/cycic_multiplechoice) |  |
| 286 | [sts-companion](src/tasksource/tasks.py#L972) | Classification | [tasksource/sts-companion](https://hf.co/datasets/tasksource/sts-companion) |  |
| 287 | [commonsense_qa_2.0](src/tasksource/tasks.py#L975) | Classification | [tasksource/commonsense_qa_2.0](https://hf.co/datasets/tasksource/commonsense_qa_2.0) |  |
| 288 | [lingnli](src/tasksource/tasks.py#L978) | Classification | [tasksource/lingnli](https://hf.co/datasets/tasksource/lingnli) |  |
| 289 | [monotonicity-entailment](src/tasksource/tasks.py#L980) | Classification | [tasksource/monotonicity-entailment](https://hf.co/datasets/tasksource/monotonicity-entailment) |  |
| 290 | [arct](src/tasksource/tasks.py#L983) | MultipleChoice | [tasksource/arct](https://hf.co/datasets/tasksource/arct) |  |
| 291 | [scinli](src/tasksource/tasks.py#L986) | Classification | [tasksource/scinli](https://hf.co/datasets/tasksource/scinli) |  |
| 292 | [naturallogic](src/tasksource/tasks.py#L990) | Classification | [tasksource/naturallogic](https://hf.co/datasets/tasksource/naturallogic) |  |
| 293 | [onestop_qa](src/tasksource/tasks.py#L992) | MultipleChoice | [malmaud/onestop_qa](https://hf.co/datasets/malmaud/onestop_qa) |  |
| 294 | [moral_stories/full](src/tasksource/tasks.py#L995) | MultipleChoice | [LabHC/moral_stories](https://hf.co/datasets/LabHC/moral_stories) |  |
| 295 | [prost](src/tasksource/tasks.py#L1003) | MultipleChoice | json |  |
| 296 | [dynahate](src/tasksource/tasks.py#L1008) | Classification | [tasksource/dynahate](https://hf.co/datasets/tasksource/dynahate) |  |
| 297 | [syntactic-augmentation-nli](src/tasksource/tasks.py#L1010) | Classification | [tasksource/syntactic-augmentation-nli](https://hf.co/datasets/tasksource/syntactic-augmentation-nli) |  |
| 298 | [autotnli](src/tasksource/tasks.py#L1012) | Classification | [tasksource/autotnli](https://hf.co/datasets/tasksource/autotnli) |  |
| 299 | [CONDAQA](src/tasksource/tasks.py#L1014) | Classification | [lasha-nlp/CONDAQA](https://hf.co/datasets/lasha-nlp/CONDAQA) |  |
| 300 | [webgpt_comparisons](src/tasksource/tasks.py#L1024) | MultipleChoice | [heegyu/webgpt_comparisons_ko](https://hf.co/datasets/heegyu/webgpt_comparisons_ko) | ✓ |
| 301 | [synthetic-instruct-gptj-pairwise](src/tasksource/tasks.py#L1032) | MultipleChoice | [Dahoas/synthetic-instruct-gptj-pairwise](https://hf.co/datasets/Dahoas/synthetic-instruct-gptj-pairwise) | ✓ |
| 302 | [scruples](src/tasksource/tasks.py#L1035) | Classification | [tasksource/scruples](https://hf.co/datasets/tasksource/scruples) | ✓ |
| 303 | [wouldyourather](src/tasksource/tasks.py#L1037) | MultipleChoice | [tasksource/wouldyourather](https://hf.co/datasets/tasksource/wouldyourather) | ✓ |
| 304 | [defeasible-nli/atomic](src/tasksource/tasks.py#L1045) | Classification | [tasksource/defeasible-nli](https://hf.co/datasets/tasksource/defeasible-nli) |  |
| 305 | [defeasible-nli/snli](src/tasksource/tasks.py#L1045) | Classification | [tasksource/defeasible-nli](https://hf.co/datasets/tasksource/defeasible-nli) |  |
| 306 | [defeasible-nli/social](src/tasksource/tasks.py#L1048) | Classification | [tasksource/defeasible-nli](https://hf.co/datasets/tasksource/defeasible-nli) |  |
| 307 | [help-nli](src/tasksource/tasks.py#L1051) | Classification | [tasksource/help-nli](https://hf.co/datasets/tasksource/help-nli) |  |
| 308 | [nli-veridicality-transitivity](src/tasksource/tasks.py#L1054) | Classification | [tasksource/nli-veridicality-transitivity](https://hf.co/datasets/tasksource/nli-veridicality-transitivity) |  |
| 309 | [lonli](src/tasksource/tasks.py#L1057) | Classification | [tasksource/lonli](https://hf.co/datasets/tasksource/lonli) |  |
| 310 | [dadc-limit-nli](src/tasksource/tasks.py#L1060) | Classification | [tasksource/dadc-limit-nli](https://hf.co/datasets/tasksource/dadc-limit-nli) |  |
| 311 | [FLUTE](src/tasksource/tasks.py#L1063) | Classification | [ColumbiaNLP/FLUTE](https://hf.co/datasets/ColumbiaNLP/FLUTE) |  |
| 312 | [strategy-qa](src/tasksource/tasks.py#L1066) | Classification | [tasksource/strategy-qa](https://hf.co/datasets/tasksource/strategy-qa) |  |
| 313 | [summarize_from_feedback/comparisons](src/tasksource/tasks.py#L1069) | MultipleChoice | [vwxyzjn/summarize_from_feedback_oai_preprocessing](https://hf.co/datasets/vwxyzjn/summarize_from_feedback_oai_preprocessing) | ✓ |
| 314 | [folio](src/tasksource/tasks.py#L1077) | Classification | [tasksource/folio](https://hf.co/datasets/tasksource/folio) |  |
| 315 | [tomi-nli](src/tasksource/tasks.py#L1081) | Classification | [tasksource/tomi-nli](https://hf.co/datasets/tasksource/tomi-nli) |  |
| 316 | [avicenna](src/tasksource/tasks.py#L1084) | Classification | [tasksource/avicenna](https://hf.co/datasets/tasksource/avicenna) | ✓ |
| 317 | [SHP](src/tasksource/tasks.py#L1087) | MultipleChoice | [stanfordnlp/SHP](https://hf.co/datasets/stanfordnlp/SHP) | ✓ |
| 318 | [MedQA-USMLE-4-options-hf](src/tasksource/tasks.py#L1095) | MultipleChoice | [GBaker/MedQA-USMLE-4-options-hf](https://hf.co/datasets/GBaker/MedQA-USMLE-4-options-hf) |  |
| 319 | [wikimedqa/medwiki](src/tasksource/tasks.py#L1098) | MultipleChoice | [sileod/wikimedqa](https://hf.co/datasets/sileod/wikimedqa) |  |
| 320 | [cicero](src/tasksource/tasks.py#L1107) | MultipleChoice | [declare-lab/cicero](https://hf.co/datasets/declare-lab/cicero) |  |
| 321 | [CREAK](src/tasksource/tasks.py#L1111) | Classification | [amydeng2000/CREAK](https://hf.co/datasets/amydeng2000/CREAK) |  |
| 322 | [mutual](src/tasksource/tasks.py#L1114) | MultipleChoice | [tasksource/mutual](https://hf.co/datasets/tasksource/mutual) |  |
| 323 | [puzzte](src/tasksource/tasks.py#L1118) | Classification | [tasksource/puzzte](https://hf.co/datasets/tasksource/puzzte) |  |
| 324 | [implicatures](src/tasksource/tasks.py#L1123) | MultipleChoice | [tasksource/implicatures](https://hf.co/datasets/tasksource/implicatures) |  |
| 325 | [race/high](src/tasksource/tasks.py#L1128) | MultipleChoice | [ehovy/race](https://hf.co/datasets/ehovy/race) |  |
| 326 | [race/middle](src/tasksource/tasks.py#L1128) | MultipleChoice | [ehovy/race](https://hf.co/datasets/ehovy/race) |  |
| 327 | [race-c](src/tasksource/tasks.py#L1132) | MultipleChoice | [tasksource/race-c](https://hf.co/datasets/tasksource/race-c) |  |
| 328 | [spartqa-yn](src/tasksource/tasks.py#L1135) | Classification | [tasksource/spartqa-yn](https://hf.co/datasets/tasksource/spartqa-yn) |  |
| 329 | [spartqa-mchoice](src/tasksource/tasks.py#L1138) | MultipleChoice | [tasksource/spartqa-mchoice](https://hf.co/datasets/tasksource/spartqa-mchoice) |  |
| 330 | [temporal-nli](src/tasksource/tasks.py#L1141) | Classification | [tasksource/temporal-nli](https://hf.co/datasets/tasksource/temporal-nli) |  |
| 331 | [riddle_sense](src/tasksource/tasks.py#L1144) | MultipleChoice | [jeggers/riddle_sense](https://hf.co/datasets/jeggers/riddle_sense) |  |
| 332 | [clcd-english](src/tasksource/tasks.py#L1149) | Classification | [tasksource/clcd-english](https://hf.co/datasets/tasksource/clcd-english) |  |
| 333 | [twentyquestions](src/tasksource/tasks.py#L1161) | Classification | [tasksource/twentyquestions](https://hf.co/datasets/tasksource/twentyquestions) |  |
| 334 | [reclor](src/tasksource/tasks.py#L1166) | MultipleChoice | [tasksource/reclor](https://hf.co/datasets/tasksource/reclor) |  |
| 335 | [counterfactually-augmented-imdb](src/tasksource/tasks.py#L1169) | Classification | [tasksource/counterfactually-augmented-imdb](https://hf.co/datasets/tasksource/counterfactually-augmented-imdb) |  |
| 336 | [counterfactually-augmented-snli](src/tasksource/tasks.py#L1172) | Classification | [tasksource/counterfactually-augmented-snli](https://hf.co/datasets/tasksource/counterfactually-augmented-snli) |  |
| 337 | [cnli](src/tasksource/tasks.py#L1175) | Classification | [tasksource/cnli](https://hf.co/datasets/tasksource/cnli) |  |
| 338 | [boolq-natural-perturbations](src/tasksource/tasks.py#L1178) | Classification | [tasksource/boolq-natural-perturbations](https://hf.co/datasets/tasksource/boolq-natural-perturbations) |  |
| 339 | [acceptability-prediction](src/tasksource/tasks.py#L1182) | Classification | [tasksource/acceptability-prediction](https://hf.co/datasets/tasksource/acceptability-prediction) | ✓ |
| 340 | [equate](src/tasksource/tasks.py#L1186) | Classification | [tasksource/equate](https://hf.co/datasets/tasksource/equate) |  |
| 341 | [ScienceQA_text_only](src/tasksource/tasks.py#L1189) | MultipleChoice | [tasksource/ScienceQA_text_only](https://hf.co/datasets/tasksource/ScienceQA_text_only) |  |
| 342 | [ekar_english](src/tasksource/tasks.py#L1192) | MultipleChoice | [Jiangjie/ekar_english](https://hf.co/datasets/Jiangjie/ekar_english) | ✓ |
| 343 | [implicit-hate-stg1](src/tasksource/tasks.py#L1196) | Classification | [tasksource/implicit-hate-stg1](https://hf.co/datasets/tasksource/implicit-hate-stg1) |  |
| 344 | [chaos-mnli-ambiguity](src/tasksource/tasks.py#L1199) | Classification | [tasksource/chaos-mnli-ambiguity](https://hf.co/datasets/tasksource/chaos-mnli-ambiguity) | ✓ |
| 345 | [headline_cause/en_simple](src/tasksource/tasks.py#L1203) | Classification | json |  |
| 346 | [logiqa-2.0-nli](src/tasksource/tasks.py#L1208) | Classification | [tasksource/logiqa-2.0-nli](https://hf.co/datasets/tasksource/logiqa-2.0-nli) |  |
| 347 | [oasst2_dense_flat/quality](src/tasksource/tasks.py#L1213) | Classification | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) | ✓ |
| 348 | [oasst2_dense_flat/toxicity](src/tasksource/tasks.py#L1215) | Classification | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) | ✓ |
| 349 | [oasst2_dense_flat/helpfulness](src/tasksource/tasks.py#L1217) | Classification | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) | ✓ |
| 350 | [mindgames](src/tasksource/tasks.py#L1220) | Classification | [sileod/mindgames](https://hf.co/datasets/sileod/mindgames) |  |
| 351 | [universal_dependencies/en_gum/deprel](src/tasksource/tasks.py#L1234) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| 352 | [universal_dependencies/en_ewt/deprel](src/tasksource/tasks.py#L1234) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| 353 | [universal_dependencies/en_lines/deprel](src/tasksource/tasks.py#L1234) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| 354 | [universal_dependencies/en_partut/deprel](src/tasksource/tasks.py#L1234) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| 355 | [ambient](src/tasksource/tasks.py#L1240) | Classification | [tasksource/ambient](https://hf.co/datasets/tasksource/ambient) | ✓ |
| 356 | [path-naturalness-prediction](src/tasksource/tasks.py#L1243) | MultipleChoice | [tasksource/path-naturalness-prediction](https://hf.co/datasets/tasksource/path-naturalness-prediction) | ✓ |
| 357 | [civil_comments/toxicity](src/tasksource/tasks.py#L1253) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 358 | [civil_comments/severe_toxicity](src/tasksource/tasks.py#L1254) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 359 | [civil_comments/obscene](src/tasksource/tasks.py#L1255) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 360 | [civil_comments/threat](src/tasksource/tasks.py#L1256) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 361 | [civil_comments/insult](src/tasksource/tasks.py#L1257) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 362 | [civil_comments/identity_attack](src/tasksource/tasks.py#L1258) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 363 | [civil_comments/sexual_explicit](src/tasksource/tasks.py#L1259) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 364 | [cloth](src/tasksource/tasks.py#L1261) | MultipleChoice | [AndyChiang/cloth](https://hf.co/datasets/AndyChiang/cloth) |  |
| 365 | [dgen](src/tasksource/tasks.py#L1262) | MultipleChoice | [AndyChiang/dgen](https://hf.co/datasets/AndyChiang/dgen) |  |
| 366 | [I2D2](src/tasksource/tasks.py#L1264) | Classification | [tasksource/I2D2](https://hf.co/datasets/tasksource/I2D2) |  |
| 367 | [args_me](src/tasksource/tasks.py#L1266) | Classification | [webis/args_me](https://hf.co/datasets/webis/args_me) |  |
| 368 | [Touche23-ValueEval](src/tasksource/tasks.py#L1269) | Classification | csv |  |
| 369 | [starcon](src/tasksource/tasks.py#L1277) | Classification | [tasksource/starcon](https://hf.co/datasets/tasksource/starcon) |  |
| 370 | [banking77](src/tasksource/tasks.py#L1279) | Classification | [legacy-datasets/banking77](https://hf.co/datasets/legacy-datasets/banking77) |  |
| 371 | [it-support-tickets](src/tasksource/tasks.py#L1281) | Classification | [tasksource/it-support-tickets](https://hf.co/datasets/tasksource/it-support-tickets) |  |
| 372 | [ConTRoL-nli](src/tasksource/tasks.py#L1285) | Classification | [tasksource/ConTRoL-nli](https://hf.co/datasets/tasksource/ConTRoL-nli) |  |
| 373 | [tracie](src/tasksource/tasks.py#L1286) | Classification | [tasksource/tracie](https://hf.co/datasets/tasksource/tracie) |  |
| 374 | [sherliic](src/tasksource/tasks.py#L1287) | Classification | [tasksource/sherliic](https://hf.co/datasets/tasksource/sherliic) |  |
| 375 | [sen-making/1](src/tasksource/tasks.py#L1289) | MultipleChoice | [tasksource/sen-making](https://hf.co/datasets/tasksource/sen-making) | ✓ |
| 376 | [sen-making/2](src/tasksource/tasks.py#L1293) | MultipleChoice | [tasksource/sen-making](https://hf.co/datasets/tasksource/sen-making) | ✓ |
| 377 | [winowhy](src/tasksource/tasks.py#L1296) | Classification | [tasksource/winowhy](https://hf.co/datasets/tasksource/winowhy) | ✓ |
| 378 | [robustLR](src/tasksource/tasks.py#L1300) | Classification | [tasksource/robustLR](https://hf.co/datasets/tasksource/robustLR) |  |
| 379 | [clutrr](src/tasksource/tasks.py#L1302) | Classification | [tasksource/clutrr](https://hf.co/datasets/tasksource/clutrr) |  |
| 380 | [logical-fallacy](src/tasksource/tasks.py#L1304) | Classification | [tasksource/logical-fallacy](https://hf.co/datasets/tasksource/logical-fallacy) |  |
| 381 | [parade](src/tasksource/tasks.py#L1306) | Classification | [tasksource/parade](https://hf.co/datasets/tasksource/parade) |  |
| 382 | [cladder](src/tasksource/tasks.py#L1308) | Classification | [tasksource/cladder](https://hf.co/datasets/tasksource/cladder) |  |
| 383 | [subjectivity](src/tasksource/tasks.py#L1310) | Classification | [tasksource/subjectivity](https://hf.co/datasets/tasksource/subjectivity) |  |
| 384 | [MOH](src/tasksource/tasks.py#L1312) | Classification | [tasksource/MOH](https://hf.co/datasets/tasksource/MOH) |  |
| 385 | [VUAC](src/tasksource/tasks.py#L1313) | Classification | [tasksource/VUAC](https://hf.co/datasets/tasksource/VUAC) |  |
| 386 | [TroFi](src/tasksource/tasks.py#L1314) | Classification | parquet |  |
| 387 | [sharc](src/tasksource/tasks.py#L1321) | Classification | [tasksource/sharc](https://hf.co/datasets/tasksource/sharc) |  |
| 388 | [conceptrules_v2](src/tasksource/tasks.py#L1325) | Classification | [tasksource/conceptrules_v2](https://hf.co/datasets/tasksource/conceptrules_v2) | ✓ |
| 389 | [disrpt/eng.dep.scidtb.rels](src/tasksource/tasks.py#L1327) | Classification | [multilingual-discourse-hub/disrpt](https://hf.co/datasets/multilingual-discourse-hub/disrpt) |  |
| 390 | [conll2000](src/tasksource/tasks.py#L1329) | TokenClassification | [eriktks/conll2000](https://hf.co/datasets/eriktks/conll2000) |  |
| 391 | [few-nerd/supervised](src/tasksource/tasks.py#L1332) | TokenClassification | [DFKI-SLT/few-nerd](https://hf.co/datasets/DFKI-SLT/few-nerd) |  |
| 392 | [finer-139](src/tasksource/tasks.py#L1333) | TokenClassification | [nlpaueb/finer-139](https://hf.co/datasets/nlpaueb/finer-139) |  |
| 393 | [zero-shot-label-nli](src/tasksource/tasks.py#L1336) | Classification | [tasksource/zero-shot-label-nli](https://hf.co/datasets/tasksource/zero-shot-label-nli) |  |
| 394 | [com2sense](src/tasksource/tasks.py#L1338) | Classification | [tasksource/com2sense](https://hf.co/datasets/tasksource/com2sense) |  |
| 395 | [scone](src/tasksource/tasks.py#L1340) | Classification | [tasksource/scone](https://hf.co/datasets/tasksource/scone) |  |
| 396 | [winodict](src/tasksource/tasks.py#L1342) | MultipleChoice | [tasksource/winodict](https://hf.co/datasets/tasksource/winodict) |  |
| 397 | [fool-me-twice](src/tasksource/tasks.py#L1344) | Classification | [tasksource/fool-me-twice](https://hf.co/datasets/tasksource/fool-me-twice) |  |
| 398 | [monli](src/tasksource/tasks.py#L1348) | Classification | [tasksource/monli](https://hf.co/datasets/tasksource/monli) |  |
| 399 | [corr2cause](src/tasksource/tasks.py#L1350) | Classification | [tasksource/corr2cause](https://hf.co/datasets/tasksource/corr2cause) |  |
| 400 | [lsat_qa/all](src/tasksource/tasks.py#L1352) | MultipleChoice | [lighteval/lsat_qa](https://hf.co/datasets/lighteval/lsat_qa) |  |
| 401 | [apt](src/tasksource/tasks.py#L1354) | Classification | [tasksource/apt](https://hf.co/datasets/tasksource/apt) |  |
| 402 | [twitter-financial-news-sentiment](src/tasksource/tasks.py#L1357) | Classification | [zeroshot/twitter-financial-news-sentiment](https://hf.co/datasets/zeroshot/twitter-financial-news-sentiment) |  |
| 403 | [icl-symbol-tuning-instruct](src/tasksource/tasks.py#L1364) | Classification | [tasksource/icl-symbol-tuning-instruct](https://hf.co/datasets/tasksource/icl-symbol-tuning-instruct) | ✓ |
| 404 | [SpaceNLI](src/tasksource/tasks.py#L1370) | Classification | [tasksource/SpaceNLI](https://hf.co/datasets/tasksource/SpaceNLI) |  |
| 405 | [propsegment/nli](src/tasksource/tasks.py#L1372) | Classification | json |  |
| 406 | [HatemojiBuild](src/tasksource/tasks.py#L1381) | Classification | [HannahRoseKirk/HatemojiBuild](https://hf.co/datasets/HannahRoseKirk/HatemojiBuild) |  |
| 407 | [regset](src/tasksource/tasks.py#L1384) | Classification | [tasksource/regset](https://hf.co/datasets/tasksource/regset) | ✓ |
| 408 | [esci](src/tasksource/tasks.py#L1390) | Classification | [tasksource/esci](https://hf.co/datasets/tasksource/esci) |  |
| 409 | [chatbot_arena_conversations](src/tasksource/tasks.py#L1409) | MultipleChoice | [lmsys/chatbot_arena_conversations](https://hf.co/datasets/lmsys/chatbot_arena_conversations) | ✓ |
| 410 | [dnd_style_intents](src/tasksource/tasks.py#L1415) | Classification | [neurae/dnd_style_intents](https://hf.co/datasets/neurae/dnd_style_intents) |  |
| 411 | [FLD.v2/default](src/tasksource/tasks.py#L1418) | Classification | [hitachi-nlp/FLD.v2](https://hf.co/datasets/hitachi-nlp/FLD.v2) |  |
| 412 | [FLD.v2/star](src/tasksource/tasks.py#L1421) | Classification | [hitachi-nlp/FLD.v2](https://hf.co/datasets/hitachi-nlp/FLD.v2) |  |
| 413 | [SDOH-NLI](src/tasksource/tasks.py#L1424) | Classification | [tasksource/SDOH-NLI](https://hf.co/datasets/tasksource/SDOH-NLI) |  |
| 414 | [scifact_entailment](src/tasksource/tasks.py#L1427) | Classification | [tasksource/scifact_entailment](https://hf.co/datasets/tasksource/scifact_entailment) |  |
| 415 | [feasibilityQA](src/tasksource/tasks.py#L1431) | Classification | [tasksource/feasibilityQA](https://hf.co/datasets/tasksource/feasibilityQA) |  |
| 416 | [simple_pair](src/tasksource/tasks.py#L1434) | Classification | [tasksource/simple_pair](https://hf.co/datasets/tasksource/simple_pair) |  |
| 417 | [AdjectiveScaleProbe-nli](src/tasksource/tasks.py#L1435) | Classification | [tasksource/AdjectiveScaleProbe-nli](https://hf.co/datasets/tasksource/AdjectiveScaleProbe-nli) |  |
| 418 | [resnli](src/tasksource/tasks.py#L1436) | Classification | [tasksource/resnli](https://hf.co/datasets/tasksource/resnli) |  |
| 419 | [SpaRTUN](src/tasksource/tasks.py#L1438) | MultipleChoice | [tasksource/SpaRTUN](https://hf.co/datasets/tasksource/SpaRTUN) |  |
| 420 | [ReSQ](src/tasksource/tasks.py#L1443) | MultipleChoice | [tasksource/ReSQ](https://hf.co/datasets/tasksource/ReSQ) |  |
| 421 | [semantic_fragments_nli](src/tasksource/tasks.py#L1448) | Classification | [tasksource/semantic_fragments_nli](https://hf.co/datasets/tasksource/semantic_fragments_nli) |  |
| 422 | [dataset_train_nli](src/tasksource/tasks.py#L1451) | Classification | [MoritzLaurer/dataset_train_nli](https://hf.co/datasets/MoritzLaurer/dataset_train_nli) |  |
| 423 | [stepgame](src/tasksource/tasks.py#L1456) | Classification | [tasksource/stepgame](https://hf.co/datasets/tasksource/stepgame) |  |
| 424 | [nlgraph](src/tasksource/tasks.py#L1464) | Classification | [tasksource/nlgraph](https://hf.co/datasets/tasksource/nlgraph) |  |
| 425 | [oasst2_pairwise_rlhf_reward](src/tasksource/tasks.py#L1468) | MultipleChoice | [tasksource/oasst2_pairwise_rlhf_reward](https://hf.co/datasets/tasksource/oasst2_pairwise_rlhf_reward) | ✓ |
| 426 | [hh-rlhf/helpful-online](src/tasksource/tasks.py#L1479) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| 427 | [hh-rlhf/helpful-base](src/tasksource/tasks.py#L1479) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| 428 | [hh-rlhf/helpful-rejection-sampled](src/tasksource/tasks.py#L1479) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| 429 | [hh-rlhf/harmless-base](src/tasksource/tasks.py#L1483) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| 430 | [ruletaker](src/tasksource/tasks.py#L1487) | Classification | [tasksource/ruletaker](https://hf.co/datasets/tasksource/ruletaker) | ✓ |
| 431 | [PARARULE-Plus](src/tasksource/tasks.py#L1491) | Classification | [qbao775/PARARULE-Plus](https://hf.co/datasets/qbao775/PARARULE-Plus) | ✓ |
| 432 | [proofwriter](src/tasksource/tasks.py#L1495) | Classification | [tasksource/proofwriter](https://hf.co/datasets/tasksource/proofwriter) |  |
| 433 | [logical-entailment](src/tasksource/tasks.py#L1498) | Classification | [tasksource/logical-entailment](https://hf.co/datasets/tasksource/logical-entailment) |  |
| 434 | [nope](src/tasksource/tasks.py#L1500) | Classification | [tasksource/nope](https://hf.co/datasets/tasksource/nope) |  |
| 435 | [LogicNLI](src/tasksource/tasks.py#L1504) | Classification | [tasksource/LogicNLI](https://hf.co/datasets/tasksource/LogicNLI) |  |
| 436 | [contract-nli/contractnli_a/seg](src/tasksource/tasks.py#L1506) | Classification | [tasksource/contract-nli](https://hf.co/datasets/tasksource/contract-nli) |  |
| 437 | [contract-nli/contractnli_b/full](src/tasksource/tasks.py#L1508) | Classification | [tasksource/contract-nli](https://hf.co/datasets/tasksource/contract-nli) |  |
| 438 | [nli4ct_semeval2024](src/tasksource/tasks.py#L1510) | Classification | [AshtonIsNotHere/nli4ct_semeval2024](https://hf.co/datasets/AshtonIsNotHere/nli4ct_semeval2024) |  |
| 439 | [lsat-ar](src/tasksource/tasks.py#L1513) | MultipleChoice | [tasksource/lsat-ar](https://hf.co/datasets/tasksource/lsat-ar) |  |
| 440 | [lsat-rc](src/tasksource/tasks.py#L1518) | MultipleChoice | [tasksource/lsat-rc](https://hf.co/datasets/tasksource/lsat-rc) |  |
| 441 | [biosift-nli](src/tasksource/tasks.py#L1523) | Classification | [AshtonIsNotHere/biosift-nli](https://hf.co/datasets/AshtonIsNotHere/biosift-nli) |  |
| 442 | [brainteasers/SP](src/tasksource/tasks.py#L1527) | MultipleChoice | [tasksource/brainteasers](https://hf.co/datasets/tasksource/brainteasers) |  |
| 443 | [brainteasers/WP](src/tasksource/tasks.py#L1527) | MultipleChoice | [tasksource/brainteasers](https://hf.co/datasets/tasksource/brainteasers) |  |
| 444 | [toxigen-data/annotated](src/tasksource/tasks.py#L1533) | Classification | [skg/toxigen-data](https://hf.co/datasets/skg/toxigen-data) |  |
| 445 | [persuasion](src/tasksource/tasks.py#L1544) | Classification | [Anthropic/persuasion](https://hf.co/datasets/Anthropic/persuasion) |  |
| 446 | [AmbigNQ-clarifying-question](src/tasksource/tasks.py#L1550) | Classification | [erbacher/AmbigNQ-clarifying-question](https://hf.co/datasets/erbacher/AmbigNQ-clarifying-question) |  |
| 447 | [SIGA-nli](src/tasksource/tasks.py#L1553) | Classification | [tasksource/SIGA-nli](https://hf.co/datasets/tasksource/SIGA-nli) |  |
| 448 | [FOL-nli](src/tasksource/tasks.py#L1555) | Classification | [unigram/FOL-nli](https://hf.co/datasets/unigram/FOL-nli) |  |
| 449 | [goal-step-wikihow/goal](src/tasksource/tasks.py#L1557) | MultipleChoice | [tasksource/goal-step-wikihow](https://hf.co/datasets/tasksource/goal-step-wikihow) |  |
| 450 | [goal-step-wikihow/step](src/tasksource/tasks.py#L1560) | MultipleChoice | [tasksource/goal-step-wikihow](https://hf.co/datasets/tasksource/goal-step-wikihow) |  |
| 451 | [goal-step-wikihow/order](src/tasksource/tasks.py#L1563) | MultipleChoice | [tasksource/goal-step-wikihow](https://hf.co/datasets/tasksource/goal-step-wikihow) |  |
| 452 | [PARADISE](src/tasksource/tasks.py#L1566) | MultipleChoice | [GGLab/PARADISE](https://hf.co/datasets/GGLab/PARADISE) |  |
| 453 | [doc-nli](src/tasksource/tasks.py#L1569) | Classification | [tasksource/doc-nli](https://hf.co/datasets/tasksource/doc-nli) |  |
| 454 | [mctest-nli](src/tasksource/tasks.py#L1571) | Classification | [tasksource/mctest-nli](https://hf.co/datasets/tasksource/mctest-nli) |  |
| 455 | [patent-phrase-similarity](src/tasksource/tasks.py#L1573) | Classification | [tasksource/patent-phrase-similarity](https://hf.co/datasets/tasksource/patent-phrase-similarity) |  |
| 456 | [natural-language-satisfiability](src/tasksource/tasks.py#L1575) | Classification | [tasksource/natural-language-satisfiability](https://hf.co/datasets/tasksource/natural-language-satisfiability) |  |
| 457 | [idioms-nli](src/tasksource/tasks.py#L1577) | Classification | [tasksource/idioms-nli](https://hf.co/datasets/tasksource/idioms-nli) |  |
| 458 | [lifecycle-entailment](src/tasksource/tasks.py#L1579) | Classification | [tasksource/lifecycle-entailment](https://hf.co/datasets/tasksource/lifecycle-entailment) |  |
| 459 | [toxic-chat/toxicchat0124/toxicity](src/tasksource/tasks.py#L1584) | Classification | [lmsys/toxic-chat](https://hf.co/datasets/lmsys/toxic-chat) | ✓ |
| 460 | [toxic-chat/toxicchat0124/jailbreaking](src/tasksource/tasks.py#L1589) | Classification | [lmsys/toxic-chat](https://hf.co/datasets/lmsys/toxic-chat) | ✓ |
| 461 | [clinc_oos/plus](src/tasksource/tasks.py#L1595) | Classification | [clinc/clinc_oos](https://hf.co/datasets/clinc/clinc_oos) |  |
| 462 | [few_rel/default](src/tasksource/tasks.py#L1632) | Classification | [tasksource/few_rel](https://hf.co/datasets/tasksource/few_rel) |  |
| 463 | [docred](src/tasksource/tasks.py#L1678) | Classification | json |  |
| 464 | [chemprot/chemprot_full_source](src/tasksource/tasks.py#L1706) | Classification | [bigbio/chemprot](https://hf.co/datasets/bigbio/chemprot) |  |
| 465 | [PKU-SafeRLHF/helpfulness](src/tasksource/tasks.py#L1711) | MultipleChoice | [PKU-Alignment/PKU-SafeRLHF](https://hf.co/datasets/PKU-Alignment/PKU-SafeRLHF) | ✓ |
| 466 | [PKU-SafeRLHF/safety](src/tasksource/tasks.py#L1716) | MultipleChoice | [PKU-Alignment/PKU-SafeRLHF](https://hf.co/datasets/PKU-Alignment/PKU-SafeRLHF) | ✓ |
| 467 | [HelpSteer/helpfulness](src/tasksource/tasks.py#L1730) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 468 | [HelpSteer/correctness](src/tasksource/tasks.py#L1731) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 469 | [HelpSteer/coherence](src/tasksource/tasks.py#L1732) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 470 | [HelpSteer/complexity](src/tasksource/tasks.py#L1733) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 471 | [HelpSteer/verbosity](src/tasksource/tasks.py#L1734) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 472 | [HelpSteer2/helpfulness](src/tasksource/tasks.py#L1736) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 473 | [HelpSteer2/correctness](src/tasksource/tasks.py#L1737) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 474 | [HelpSteer2/coherence](src/tasksource/tasks.py#L1738) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 475 | [HelpSteer2/complexity](src/tasksource/tasks.py#L1739) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 476 | [HelpSteer2/verbosity](src/tasksource/tasks.py#L1740) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 477 | [HelpSteer3/preference](src/tasksource/tasks.py#L1745) | MultipleChoice | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | ✓ |
| 478 | [HelpSteer3/principle](src/tasksource/tasks.py#L1750) | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) |  |
| 479 | [HelpSteer3/edit_quality](src/tasksource/tasks.py#L1755) | MultipleChoice | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | ✓ |
| 480 | [HelpSteer3/feedback](src/tasksource/tasks.py#L1778) | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | ✓ |
| 481 | [MSciNLI](src/tasksource/tasks.py#L1783) | Classification | [sadat2307/MSciNLI](https://hf.co/datasets/sadat2307/MSciNLI) |  |
| 482 | [UltraFeedback-paired](src/tasksource/tasks.py#L1786) | MultipleChoice | [pushpdeep/UltraFeedback-paired](https://hf.co/datasets/pushpdeep/UltraFeedback-paired) | ✓ |
| 483 | [prm800k_dpo/solution](src/tasksource/tasks.py#L1790) | MultipleChoice | [tasksource/prm800k_dpo](https://hf.co/datasets/tasksource/prm800k_dpo) | ✓ |
| 484 | [prm800k_dpo/step](src/tasksource/tasks.py#L1793) | MultipleChoice | [tasksource/prm800k_dpo](https://hf.co/datasets/tasksource/prm800k_dpo) | ✓ |
| 485 | [AES2-essay-scoring](src/tasksource/tasks.py#L1797) | Classification | [tasksource/AES2-essay-scoring](https://hf.co/datasets/tasksource/AES2-essay-scoring) | ✓ |
| 486 | [argument-feedback](src/tasksource/tasks.py#L1801) | Classification | [tasksource/argument-feedback](https://hf.co/datasets/tasksource/argument-feedback) | ✓ |
| 487 | [english-grading/cohesion](src/tasksource/tasks.py#L1808) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 488 | [english-grading/syntax](src/tasksource/tasks.py#L1809) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 489 | [english-grading/vocabulary](src/tasksource/tasks.py#L1810) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 490 | [english-grading/phraseology](src/tasksource/tasks.py#L1811) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 491 | [english-grading/grammar](src/tasksource/tasks.py#L1812) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 492 | [english-grading/conventions](src/tasksource/tasks.py#L1813) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 493 | [wice](src/tasksource/tasks.py#L1815) | Classification | [tasksource/wice](https://hf.co/datasets/tasksource/wice) |  |
| 494 | [hover](src/tasksource/tasks.py#L1818) | Classification | [Dzeniks/hover](https://hf.co/datasets/Dzeniks/hover) |  |
| 495 | [hover-3way/nli](src/tasksource/tasks.py#L1822) | Classification | [Dzeniks/hover-3way](https://hf.co/datasets/Dzeniks/hover-3way) |  |
| 496 | [tasksource_dpo_pairs](src/tasksource/tasks.py#L1825) | MultipleChoice | [tasksource/tasksource_dpo_pairs](https://hf.co/datasets/tasksource/tasksource_dpo_pairs) | ✓ |
| 497 | [seahorse_summarization_evaluation](src/tasksource/tasks.py#L1828) | Classification | [tasksource/seahorse_summarization_evaluation](https://hf.co/datasets/tasksource/seahorse_summarization_evaluation) |  |
| 498 | [missing-item-prediction/contrastive](src/tasksource/tasks.py#L1831) | Classification | [sileod/missing-item-prediction](https://hf.co/datasets/sileod/missing-item-prediction) |  |
| 499 | [jigsaw_toxicity](src/tasksource/tasks.py#L1835) | Classification | [tasksource/jigsaw_toxicity](https://hf.co/datasets/tasksource/jigsaw_toxicity) |  |
| 500 | [Pol_NLI](src/tasksource/tasks.py#L1838) | Classification | [mlburnham/Pol_NLI](https://hf.co/datasets/mlburnham/Pol_NLI) |  |
| 501 | [synthetic-retrieval-NLI/position](src/tasksource/tasks.py#L1841) | Classification | [tasksource/synthetic-retrieval-NLI](https://hf.co/datasets/tasksource/synthetic-retrieval-NLI) |  |
| 502 | [synthetic-retrieval-NLI/binary](src/tasksource/tasks.py#L1841) | Classification | [tasksource/synthetic-retrieval-NLI](https://hf.co/datasets/tasksource/synthetic-retrieval-NLI) |  |
| 503 | [synthetic-retrieval-NLI/count](src/tasksource/tasks.py#L1841) | Classification | [tasksource/synthetic-retrieval-NLI](https://hf.co/datasets/tasksource/synthetic-retrieval-NLI) |  |
| 504 | [github-issue-similarity](src/tasksource/tasks.py#L1850) | Classification | [WhereIsAI/github-issue-similarity](https://hf.co/datasets/WhereIsAI/github-issue-similarity) |  |
