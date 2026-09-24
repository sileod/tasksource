504 English tasks. Load one with `load_task(id)`; the annotations are in [tasks.py](src/tasksource/tasks.py), and tasks kept out on purpose are in [parked.py](src/tasksource/parked.py).

| id | type | dataset | question |
|---|---|---|:-:|
| [glue/mnli](src/tasksource/tasks.py#L28) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| [glue/qnli](src/tasksource/tasks.py#L29) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| [glue/rte](src/tasksource/tasks.py#L30) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| [glue/wnli](src/tasksource/tasks.py#L31) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| [glue/mrpc](src/tasksource/tasks.py#L33) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| [glue/qqp](src/tasksource/tasks.py#L34) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| [glue/stsb](src/tasksource/tasks.py#L35) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) | ✓ |
| [super_glue/boolq](src/tasksource/tasks.py#L38) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| [super_glue/boolq_passage](src/tasksource/tasks.py#L39) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| [super_glue/cb](src/tasksource/tasks.py#L41) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| [super_glue/multirc](src/tasksource/tasks.py#L42) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| [super_glue/wic](src/tasksource/tasks.py#L47) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| [super_glue/axg](src/tasksource/tasks.py#L52) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| [anli/a1](src/tasksource/tasks.py#L55) | Classification | [facebook/anli](https://hf.co/datasets/facebook/anli) |  |
| [anli/a2](src/tasksource/tasks.py#L56) | Classification | [facebook/anli](https://hf.co/datasets/facebook/anli) |  |
| [anli/a3](src/tasksource/tasks.py#L57) | Classification | [facebook/anli](https://hf.co/datasets/facebook/anli) |  |
| [babi_nli/counting](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| [babi_nli/indefinite-knowledge](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| [babi_nli/lists-sets](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| [babi_nli/path-finding](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| [babi_nli/positional-reasoning](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| [babi_nli/simple-negation](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| [babi_nli/size-reasoning](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| [babi_nli/conjunction](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| [babi_nli/three-arg-relations](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| [babi_nli/three-supporting-facts](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| [babi_nli/time-reasoning](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| [babi_nli/two-arg-relations](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| [babi_nli/single-supporting-fact](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| [babi_nli/compound-coreference](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| [babi_nli/basic-deduction](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| [babi_nli/basic-coreference](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| [babi_nli/two-supporting-facts](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| [babi_nli/basic-induction](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| [babi_nli/yes-no-questions](src/tasksource/tasks.py#L60) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| [sick/label](src/tasksource/tasks.py#L66) | Classification | [tasksource/sick](https://hf.co/datasets/tasksource/sick) |  |
| [sick/relatedness](src/tasksource/tasks.py#L67) | Classification | [tasksource/sick](https://hf.co/datasets/tasksource/sick) | ✓ |
| [snli](src/tasksource/tasks.py#L122) | Classification | [stanfordnlp/snli](https://hf.co/datasets/stanfordnlp/snli) |  |
| [scitail/snli_format](src/tasksource/tasks.py#L125) | Classification | [allenai/scitail](https://hf.co/datasets/allenai/scitail) |  |
| [hans](src/tasksource/tasks.py#L127) | Classification | [tasksource/hans](https://hf.co/datasets/tasksource/hans) |  |
| [WANLI](src/tasksource/tasks.py#L130) | Classification | [alisawuffles/WANLI](https://hf.co/datasets/alisawuffles/WANLI) |  |
| [recast/recast_factuality](src/tasksource/tasks.py#L132) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| [recast/recast_verbnet](src/tasksource/tasks.py#L132) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| [recast/recast_puns](src/tasksource/tasks.py#L132) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| [recast/recast_ner](src/tasksource/tasks.py#L132) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| [recast/recast_sentiment](src/tasksource/tasks.py#L132) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| [recast/recast_megaveridicality](src/tasksource/tasks.py#L132) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| [recast/recast_verbcorner](src/tasksource/tasks.py#L132) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| [probability_words_nli/usnli](src/tasksource/tasks.py#L137) | Classification | [sileod/probability_words_nli](https://hf.co/datasets/sileod/probability_words_nli) |  |
| [probability_words_nli/reasoning_2hop](src/tasksource/tasks.py#L137) | Classification | [sileod/probability_words_nli](https://hf.co/datasets/sileod/probability_words_nli) |  |
| [probability_words_nli/reasoning_1hop](src/tasksource/tasks.py#L137) | Classification | [sileod/probability_words_nli](https://hf.co/datasets/sileod/probability_words_nli) |  |
| [nan-nli](src/tasksource/tasks.py#L141) | Classification | [joey234/nan-nli](https://hf.co/datasets/joey234/nan-nli) |  |
| [nli_fever](src/tasksource/tasks.py#L143) | Classification | [pietrolesci/nli_fever](https://hf.co/datasets/pietrolesci/nli_fever) |  |
| [breaking_nli](src/tasksource/tasks.py#L146) | Classification | [pietrolesci/breaking_nli](https://hf.co/datasets/pietrolesci/breaking_nli) |  |
| [conj_nli](src/tasksource/tasks.py#L150) | Classification | [pietrolesci/conj_nli](https://hf.co/datasets/pietrolesci/conj_nli) |  |
| [fracas](src/tasksource/tasks.py#L154) | Classification | [pietrolesci/fracas](https://hf.co/datasets/pietrolesci/fracas) |  |
| [dialogue_nli](src/tasksource/tasks.py#L157) | Classification | [pietrolesci/dialogue_nli](https://hf.co/datasets/pietrolesci/dialogue_nli) |  |
| [mpe](src/tasksource/tasks.py#L160) | Classification | [pietrolesci/mpe](https://hf.co/datasets/pietrolesci/mpe) |  |
| [dnc](src/tasksource/tasks.py#L164) | Classification | [pietrolesci/dnc](https://hf.co/datasets/pietrolesci/dnc) |  |
| [recast_white/fnplus](src/tasksource/tasks.py#L168) | Classification | [pietrolesci/recast_white](https://hf.co/datasets/pietrolesci/recast_white) |  |
| [recast_white/sprl](src/tasksource/tasks.py#L171) | Classification | [pietrolesci/recast_white](https://hf.co/datasets/pietrolesci/recast_white) |  |
| [recast_white/dpr](src/tasksource/tasks.py#L174) | Classification | [pietrolesci/recast_white](https://hf.co/datasets/pietrolesci/recast_white) |  |
| [joci](src/tasksource/tasks.py#L178) | Classification | [pietrolesci/joci](https://hf.co/datasets/pietrolesci/joci) |  |
| [robust_nli/IS_CS](src/tasksource/tasks.py#L184) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| [robust_nli/LI_LI](src/tasksource/tasks.py#L186) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| [robust_nli/ST_WO](src/tasksource/tasks.py#L188) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| [robust_nli/PI_SP](src/tasksource/tasks.py#L190) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| [robust_nli/PI_CD](src/tasksource/tasks.py#L192) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| [robust_nli/ST_SE](src/tasksource/tasks.py#L194) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| [robust_nli/ST_NE](src/tasksource/tasks.py#L196) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| [robust_nli/ST_LM](src/tasksource/tasks.py#L198) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| [robust_nli_is_sd](src/tasksource/tasks.py#L200) | Classification | [pietrolesci/robust_nli_is_sd](https://hf.co/datasets/pietrolesci/robust_nli_is_sd) |  |
| [robust_nli_li_ts](src/tasksource/tasks.py#L203) | Classification | [pietrolesci/robust_nli_li_ts](https://hf.co/datasets/pietrolesci/robust_nli_li_ts) |  |
| [gen_debiased_nli/snli_seq_z](src/tasksource/tasks.py#L207) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| [gen_debiased_nli/snli_z_aug](src/tasksource/tasks.py#L209) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| [gen_debiased_nli/snli_par_z](src/tasksource/tasks.py#L211) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| [gen_debiased_nli/mnli_par_z](src/tasksource/tasks.py#L213) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| [gen_debiased_nli/mnli_z_aug](src/tasksource/tasks.py#L215) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| [gen_debiased_nli/mnli_seq_z](src/tasksource/tasks.py#L217) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| [add_one_rte](src/tasksource/tasks.py#L220) | Classification | [pietrolesci/add_one_rte](https://hf.co/datasets/pietrolesci/add_one_rte) |  |
| [imppres/presupposition_question_presupposition/presupposition](src/tasksource/tasks.py#L230) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| [imppres/presupposition_possessed_definites_uniqueness/presupposition](src/tasksource/tasks.py#L230) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| [imppres/presupposition_possessed_definites_existence/presupposition](src/tasksource/tasks.py#L230) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| [imppres/presupposition_only_presupposition/presupposition](src/tasksource/tasks.py#L230) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| [imppres/presupposition_cleft_uniqueness/presupposition](src/tasksource/tasks.py#L230) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| [imppres/presupposition_cleft_existence/presupposition](src/tasksource/tasks.py#L230) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| [imppres/presupposition_change_of_state/presupposition](src/tasksource/tasks.py#L230) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| [imppres/presupposition_both_presupposition/presupposition](src/tasksource/tasks.py#L230) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| [imppres/presupposition_all_n_presupposition/presupposition](src/tasksource/tasks.py#L230) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| [imppres/implicature_numerals_2_3/prag](src/tasksource/tasks.py#L234) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| [imppres/implicature_numerals_10_100/prag](src/tasksource/tasks.py#L234) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| [imppres/implicature_modals/prag](src/tasksource/tasks.py#L234) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| [imppres/implicature_gradable_verb/prag](src/tasksource/tasks.py#L234) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| [imppres/implicature_gradable_adjective/prag](src/tasksource/tasks.py#L234) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| [imppres/implicature_connectives/prag](src/tasksource/tasks.py#L234) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| [imppres/implicature_quantifiers/prag](src/tasksource/tasks.py#L234) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| [imppres/implicature_modals/log](src/tasksource/tasks.py#L238) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| [imppres/implicature_gradable_verb/log](src/tasksource/tasks.py#L238) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| [imppres/implicature_gradable_adjective/log](src/tasksource/tasks.py#L238) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| [imppres/implicature_quantifiers/log](src/tasksource/tasks.py#L238) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| [imppres/implicature_numerals_2_3/log](src/tasksource/tasks.py#L238) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| [imppres/implicature_connectives/log](src/tasksource/tasks.py#L238) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| [imppres/implicature_numerals_10_100/log](src/tasksource/tasks.py#L238) | Classification | [tasksource/imppres](https://hf.co/datasets/tasksource/imppres) |  |
| [hlgd](src/tasksource/tasks.py#L243) | Classification | [tasksource/hlgd](https://hf.co/datasets/tasksource/hlgd) |  |
| [paws/labeled_final](src/tasksource/tasks.py#L245) | Classification | [google-research-datasets/paws](https://hf.co/datasets/google-research-datasets/paws) |  |
| [paws/labeled_swap](src/tasksource/tasks.py#L246) | Classification | [google-research-datasets/paws](https://hf.co/datasets/google-research-datasets/paws) |  |
| [medical_questions_pairs](src/tasksource/tasks.py#L248) | Classification | [curaihealth/medical_questions_pairs](https://hf.co/datasets/curaihealth/medical_questions_pairs) |  |
| [conll2003/pos_tags](src/tasksource/tasks.py#L253) | TokenClassification | [tomaarsen/conll2003](https://hf.co/datasets/tomaarsen/conll2003) |  |
| [conll2003/chunk_tags](src/tasksource/tasks.py#L254) | TokenClassification | [tomaarsen/conll2003](https://hf.co/datasets/tomaarsen/conll2003) |  |
| [conll2003/ner_tags](src/tasksource/tasks.py#L255) | TokenClassification | [tomaarsen/conll2003](https://hf.co/datasets/tomaarsen/conll2003) |  |
| [fig-qa](src/tasksource/tasks.py#L261) | MultipleChoice | [nightingal3/fig-qa](https://hf.co/datasets/nightingal3/fig-qa) |  |
| [cos_e/v1.0](src/tasksource/tasks.py#L270) | MultipleChoice | [Salesforce/cos_e](https://hf.co/datasets/Salesforce/cos_e) |  |
| [cosmos_qa](src/tasksource/tasks.py#L275) | MultipleChoice | [Samsoup/cosmos_qa](https://hf.co/datasets/Samsoup/cosmos_qa) |  |
| [dream](src/tasksource/tasks.py#L278) | MultipleChoice | [dataset-org/dream](https://hf.co/datasets/dataset-org/dream) |  |
| [openbookqa](src/tasksource/tasks.py#L285) | MultipleChoice | [allenai/openbookqa](https://hf.co/datasets/allenai/openbookqa) |  |
| [qasc](src/tasksource/tasks.py#L291) | MultipleChoice | [allenai/qasc](https://hf.co/datasets/allenai/qasc) |  |
| [quartz](src/tasksource/tasks.py#L299) | MultipleChoice | [allenai/quartz](https://hf.co/datasets/allenai/quartz) |  |
| [quail](src/tasksource/tasks.py#L304) | MultipleChoice | [textmachinelab/quail](https://hf.co/datasets/textmachinelab/quail) |  |
| [head_qa/en](src/tasksource/tasks.py#L310) | MultipleChoice | [EleutherAI/headqa](https://hf.co/datasets/EleutherAI/headqa) |  |
| [sciq](src/tasksource/tasks.py#L318) | MultipleChoice | [allenai/sciq](https://hf.co/datasets/allenai/sciq) |  |
| [social_i_qa](src/tasksource/tasks.py#L323) | MultipleChoice | [tasksource/social_i_qa](https://hf.co/datasets/tasksource/social_i_qa) |  |
| [wiki_hop/original](src/tasksource/tasks.py#L329) | MultipleChoice | [MoE-UNC/wikihop](https://hf.co/datasets/MoE-UNC/wikihop) |  |
| [wiqa](src/tasksource/tasks.py#L336) | MultipleChoice | [tasksource/wiqa](https://hf.co/datasets/tasksource/wiqa) |  |
| [piqa](src/tasksource/tasks.py#L341) | MultipleChoice | [baber/piqa](https://hf.co/datasets/baber/piqa) |  |
| [hellaswag](src/tasksource/tasks.py#L349) | MultipleChoice | [Rowan/hellaswag](https://hf.co/datasets/Rowan/hellaswag) |  |
| [super_glue/copa](src/tasksource/tasks.py#L358) | MultipleChoice | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| [balanced-copa](src/tasksource/tasks.py#L360) | MultipleChoice | [pkavumba/balanced-copa](https://hf.co/datasets/pkavumba/balanced-copa) |  |
| [e-CARE](src/tasksource/tasks.py#L363) | MultipleChoice | [12ml/e-CARE](https://hf.co/datasets/12ml/e-CARE) |  |
| [art](src/tasksource/tasks.py#L366) | MultipleChoice | [allenai/art](https://hf.co/datasets/allenai/art) | ✓ |
| [winogrande/winogrande_xl](src/tasksource/tasks.py#L374) | MultipleChoice | [allenai/winogrande](https://hf.co/datasets/allenai/winogrande) |  |
| [codah/codah](src/tasksource/tasks.py#L377) | MultipleChoice | [jaredfern/codah](https://hf.co/datasets/jaredfern/codah) |  |
| [ai2_arc/ARC-Easy/challenge](src/tasksource/tasks.py#L379) | MultipleChoice | [allenai/ai2_arc](https://hf.co/datasets/allenai/ai2_arc) |  |
| [ai2_arc/ARC-Challenge/challenge](src/tasksource/tasks.py#L379) | MultipleChoice | [allenai/ai2_arc](https://hf.co/datasets/allenai/ai2_arc) |  |
| [definite_pronoun_resolution](src/tasksource/tasks.py#L384) | MultipleChoice | [community-datasets/definite_pronoun_resolution](https://hf.co/datasets/community-datasets/definite_pronoun_resolution) |  |
| [swag/regular](src/tasksource/tasks.py#L390) | MultipleChoice | [allenai/swag](https://hf.co/datasets/allenai/swag) |  |
| [math_qa](src/tasksource/tasks.py#L396) | MultipleChoice | [tasksource/math_qa](https://hf.co/datasets/tasksource/math_qa) |  |
| [glue/cola](src/tasksource/tasks.py#L405) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| [glue/sst2](src/tasksource/tasks.py#L406) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| [utilitarianism](src/tasksource/tasks.py#L420) | Classification | csv |  |
| [amazon_counterfactual/en](src/tasksource/tasks.py#L428) | Classification | [mteb/amazon_counterfactual](https://hf.co/datasets/mteb/amazon_counterfactual) |  |
| [insincere-questions](src/tasksource/tasks.py#L433) | Classification | [SetFit/insincere-questions](https://hf.co/datasets/SetFit/insincere-questions) |  |
| [toxic_conversations](src/tasksource/tasks.py#L437) | Classification | [SetFit/toxic_conversations](https://hf.co/datasets/SetFit/toxic_conversations) |  |
| [TuringBench](src/tasksource/tasks.py#L441) | Classification | csv |  |
| [trec](src/tasksource/tasks.py#L450) | Classification | [tasksource/trec](https://hf.co/datasets/tasksource/trec) |  |
| [vitaminc](src/tasksource/tasks.py#L453) | Classification | [tals/vitaminc](https://hf.co/datasets/tals/vitaminc) |  |
| [hope_edi/english](src/tasksource/tasks.py#L455) | Classification | csv |  |
| [rumoureval_2019/RumourEval2019](src/tasksource/tasks.py#L469) | Classification | csv |  |
| [ethos/binary](src/tasksource/tasks.py#L483) | Classification | [SetFit/ethos_binary](https://hf.co/datasets/SetFit/ethos_binary) |  |
| [ethos/multilabel](src/tasksource/tasks.py#L507) | Classification | [tasksource/ethos](https://hf.co/datasets/tasksource/ethos) |  |
| [tweet_eval/emoji](src/tasksource/tasks.py#L510) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| [tweet_eval/emotion](src/tasksource/tasks.py#L510) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| [tweet_eval/hate](src/tasksource/tasks.py#L510) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| [tweet_eval/irony](src/tasksource/tasks.py#L510) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| [tweet_eval/offensive](src/tasksource/tasks.py#L510) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| [tweet_eval/sentiment](src/tasksource/tasks.py#L510) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| [tweet_eval/stance_abortion](src/tasksource/tasks.py#L525) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| [tweet_eval/stance_atheism](src/tasksource/tasks.py#L526) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| [tweet_eval/stance_climate](src/tasksource/tasks.py#L527) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| [tweet_eval/stance_feminist](src/tasksource/tasks.py#L528) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| [tweet_eval/stance_hillary](src/tasksource/tasks.py#L529) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| [discovery/discovery](src/tasksource/tasks.py#L532) | Classification | [sileod/discovery](https://hf.co/datasets/sileod/discovery) |  |
| [pragmeval/switchboard](src/tasksource/tasks.py#L534) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| [pragmeval/verifiability](src/tasksource/tasks.py#L534) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| [pragmeval/mrda](src/tasksource/tasks.py#L534) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| [pragmeval/emergent](src/tasksource/tasks.py#L538) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| [pragmeval/gum](src/tasksource/tasks.py#L538) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| [pragmeval/pdtb](src/tasksource/tasks.py#L538) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| [pragmeval/persuasiveness-claimtype](src/tasksource/tasks.py#L538) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| [pragmeval/persuasiveness-premisetype](src/tasksource/tasks.py#L538) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| [pragmeval/stac](src/tasksource/tasks.py#L538) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| [pragmeval/sarcasm](src/tasksource/tasks.py#L538) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| [pragmeval/emobank-arousal](src/tasksource/tasks.py#L547) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| [pragmeval/emobank-dominance](src/tasksource/tasks.py#L548) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| [pragmeval/emobank-valence](src/tasksource/tasks.py#L549) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| [pragmeval/squinky-formality](src/tasksource/tasks.py#L550) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| [pragmeval/squinky-implicature](src/tasksource/tasks.py#L551) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| [pragmeval/squinky-informativeness](src/tasksource/tasks.py#L552) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| [pragmeval/persuasiveness-eloquence](src/tasksource/tasks.py#L553) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| [pragmeval/persuasiveness-relevance](src/tasksource/tasks.py#L554) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| [pragmeval/persuasiveness-specificity](src/tasksource/tasks.py#L555) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| [pragmeval/persuasiveness-strength](src/tasksource/tasks.py#L556) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| [silicone/oasis](src/tasksource/tasks.py#L558) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| [silicone/sem](src/tasksource/tasks.py#L558) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| [silicone/meld_s](src/tasksource/tasks.py#L558) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| [silicone/meld_e](src/tasksource/tasks.py#L558) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| [silicone/maptask](src/tasksource/tasks.py#L558) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| [silicone/dyda_e](src/tasksource/tasks.py#L558) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| [silicone/dyda_da](src/tasksource/tasks.py#L558) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| [silicone/iemocap](src/tasksource/tasks.py#L565) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| [lex_glue/eurlex](src/tasksource/tasks.py#L570) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) |  |
| [lex_glue/scotus](src/tasksource/tasks.py#L572) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) |  |
| [lex_glue/ledgar](src/tasksource/tasks.py#L575) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) |  |
| [lex_glue/unfair_tos](src/tasksource/tasks.py#L577) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) | ✓ |
| [lex_glue/case_hold](src/tasksource/tasks.py#L580) | MultipleChoice | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) |  |
| [language-identification](src/tasksource/tasks.py#L588) | Classification | [papluca/language-identification](https://hf.co/datasets/papluca/language-identification) | ✓ |
| [imdb](src/tasksource/tasks.py#L593) | Classification | [stanfordnlp/imdb](https://hf.co/datasets/stanfordnlp/imdb) |  |
| [rotten_tomatoes](src/tasksource/tasks.py#L595) | Classification | [cornell-movie-review-data/rotten_tomatoes](https://hf.co/datasets/cornell-movie-review-data/rotten_tomatoes) |  |
| [ag_news](src/tasksource/tasks.py#L597) | Classification | [fancyzhx/ag_news](https://hf.co/datasets/fancyzhx/ag_news) |  |
| [yelp_review_full/yelp_review_full](src/tasksource/tasks.py#L599) | Classification | [Yelp/yelp_review_full](https://hf.co/datasets/Yelp/yelp_review_full) |  |
| [financial_phrasebank/sentences_allagree](src/tasksource/tasks.py#L603) | Classification | [ghbacct/financial-phrasebank-all-agree-classification](https://hf.co/datasets/ghbacct/financial-phrasebank-all-agree-classification) |  |
| [poem_sentiment](src/tasksource/tasks.py#L608) | Classification | [google-research-datasets/poem_sentiment](https://hf.co/datasets/google-research-datasets/poem_sentiment) |  |
| [emotion](src/tasksource/tasks.py#L610) | Classification | [dair-ai/emotion](https://hf.co/datasets/dair-ai/emotion) |  |
| [dbpedia_14/dbpedia_14](src/tasksource/tasks.py#L612) | Classification | [fancyzhx/dbpedia_14](https://hf.co/datasets/fancyzhx/dbpedia_14) |  |
| [amazon_polarity/amazon_polarity](src/tasksource/tasks.py#L614) | Classification | [fancyzhx/amazon_polarity](https://hf.co/datasets/fancyzhx/amazon_polarity) |  |
| [app_reviews](src/tasksource/tasks.py#L616) | Classification | [sealuzh/app_reviews](https://hf.co/datasets/sealuzh/app_reviews) |  |
| [hate_speech18](src/tasksource/tasks.py#L620) | Classification | [tasksource/hate_speech18](https://hf.co/datasets/tasksource/hate_speech18) |  |
| [sms_spam](src/tasksource/tasks.py#L626) | Classification | [ucirvine/sms_spam](https://hf.co/datasets/ucirvine/sms_spam) |  |
| [humicroedit/subtask-1](src/tasksource/tasks.py#L629) | Classification | [tasksource/humicroedit](https://hf.co/datasets/tasksource/humicroedit) |  |
| [humicroedit/subtask-2](src/tasksource/tasks.py#L635) | Classification | [tasksource/humicroedit](https://hf.co/datasets/tasksource/humicroedit) | ✓ |
| [snips_built_in_intents](src/tasksource/tasks.py#L640) | Classification | [sonos-nlu-benchmark/snips_built_in_intents](https://hf.co/datasets/sonos-nlu-benchmark/snips_built_in_intents) |  |
| [hate_speech_offensive](src/tasksource/tasks.py#L644) | Classification | [tdavidson/hate_speech_offensive](https://hf.co/datasets/tdavidson/hate_speech_offensive) |  |
| [yahoo_answers_topics](src/tasksource/tasks.py#L646) | Classification | [community-datasets/yahoo_answers_topics](https://hf.co/datasets/community-datasets/yahoo_answers_topics) |  |
| [stackoverflow-questions](src/tasksource/tasks.py#L650) | Classification | [pacovaldez/stackoverflow-questions](https://hf.co/datasets/pacovaldez/stackoverflow-questions) |  |
| [hyperpartisan_news](src/tasksource/tasks.py#L655) | Classification | [zapsdcn/hyperpartisan_news](https://hf.co/datasets/zapsdcn/hyperpartisan_news) |  |
| [sciie](src/tasksource/tasks.py#L660) | Classification | [zapsdcn/sciie](https://hf.co/datasets/zapsdcn/sciie) |  |
| [citation_intent](src/tasksource/tasks.py#L661) | Classification | [zapsdcn/citation_intent](https://hf.co/datasets/zapsdcn/citation_intent) |  |
| [go_emotions/simplified](src/tasksource/tasks.py#L663) | Classification | [google-research-datasets/go_emotions](https://hf.co/datasets/google-research-datasets/go_emotions) |  |
| [scicite](src/tasksource/tasks.py#L667) | Classification | [tasksource/scicite](https://hf.co/datasets/tasksource/scicite) |  |
| [liar](src/tasksource/tasks.py#L669) | Classification | [tasksource/liar](https://hf.co/datasets/tasksource/liar) |  |
| [lexical_relation_classification/ROOT09](src/tasksource/tasks.py#L678) | Classification | json | ✓ |
| [lexical_relation_classification/K&H+N](src/tasksource/tasks.py#L678) | Classification | json | ✓ |
| [lexical_relation_classification/BLESS](src/tasksource/tasks.py#L678) | Classification | json | ✓ |
| [lexical_relation_classification/EVALution](src/tasksource/tasks.py#L678) | Classification | json | ✓ |
| [lexical_relation_classification/CogALexV](src/tasksource/tasks.py#L704) | Classification | json | ✓ |
| [linguisticprobing/subj_number](src/tasksource/tasks.py#L722) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| [linguisticprobing/obj_number](src/tasksource/tasks.py#L723) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| [linguisticprobing/past_present](src/tasksource/tasks.py#L724) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| [linguisticprobing/sentence_length](src/tasksource/tasks.py#L725) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| [linguisticprobing/top_constituents](src/tasksource/tasks.py#L726) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| [linguisticprobing/tree_depth](src/tasksource/tasks.py#L728) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| [linguisticprobing/coordination_inversion](src/tasksource/tasks.py#L729) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| [linguisticprobing/odd_man_out](src/tasksource/tasks.py#L731) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| [linguisticprobing/bigram_shift](src/tasksource/tasks.py#L732) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| [crowdflower/airline-sentiment](src/tasksource/tasks.py#L734) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) |  |
| [crowdflower/corporate-messaging](src/tasksource/tasks.py#L734) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) |  |
| [crowdflower/economic-news](src/tasksource/tasks.py#L734) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) |  |
| [crowdflower/political-media-audience](src/tasksource/tasks.py#L734) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) |  |
| [crowdflower/political-media-bias](src/tasksource/tasks.py#L734) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) |  |
| [crowdflower/political-media-message](src/tasksource/tasks.py#L734) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) |  |
| [crowdflower/text_emotion](src/tasksource/tasks.py#L734) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) |  |
| [crowdflower/sentiment_nuclear_power](src/tasksource/tasks.py#L734) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) |  |
| [crowdflower/tweet_global_warming](src/tasksource/tasks.py#L734) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) |  |
| [ethics/commonsense](src/tasksource/tasks.py#L759) | Classification | csv |  |
| [ethics/deontology](src/tasksource/tasks.py#L767) | Classification | csv |  |
| [ethics/justice](src/tasksource/tasks.py#L775) | Classification | csv |  |
| [ethics/virtue](src/tasksource/tasks.py#L783) | Classification | [hendrycks/ethics](https://hf.co/datasets/hendrycks/ethics) |  |
| [emo/emo2019](src/tasksource/tasks.py#L792) | Classification | [oneonlee/cleansed_emocontext](https://hf.co/datasets/oneonlee/cleansed_emocontext) |  |
| [google_wellformed_query](src/tasksource/tasks.py#L798) | Classification | [tasksource/google_wellformed_query](https://hf.co/datasets/tasksource/google_wellformed_query) | ✓ |
| [tweets_hate_speech_detection](src/tasksource/tasks.py#L803) | Classification | [tweets-hate-speech-detection/tweets_hate_speech_detection](https://hf.co/datasets/tweets-hate-speech-detection/tweets_hate_speech_detection) |  |
| [wnut_17/wnut_17](src/tasksource/tasks.py#L807) | TokenClassification | [flaitenberger/wnut_17](https://hf.co/datasets/flaitenberger/wnut_17) |  |
| [ncbi_disease/ncbi_disease](src/tasksource/tasks.py#L810) | TokenClassification | [ncbi/ncbi_disease](https://hf.co/datasets/ncbi/ncbi_disease) |  |
| [acronym_identification](src/tasksource/tasks.py#L813) | TokenClassification | [amirveyseh/acronym_identification](https://hf.co/datasets/amirveyseh/acronym_identification) |  |
| [jnlpba/jnlpba](src/tasksource/tasks.py#L816) | TokenClassification | [jnlpba/jnlpba](https://hf.co/datasets/jnlpba/jnlpba) |  |
| [ontonotes_english/SpeedOfMagic--ontonotes_english](src/tasksource/tasks.py#L823) | TokenClassification | [SpeedOfMagic/ontonotes_english](https://hf.co/datasets/SpeedOfMagic/ontonotes_english) |  |
| [blog_authorship_corpus/gender](src/tasksource/tasks.py#L827) | Classification | [tasksource/blog_authorship_corpus](https://hf.co/datasets/tasksource/blog_authorship_corpus) | ✓ |
| [blog_authorship_corpus/age](src/tasksource/tasks.py#L829) | Classification | [tasksource/blog_authorship_corpus](https://hf.co/datasets/tasksource/blog_authorship_corpus) | ✓ |
| [blog_authorship_corpus/job](src/tasksource/tasks.py#L832) | Classification | [tasksource/blog_authorship_corpus](https://hf.co/datasets/tasksource/blog_authorship_corpus) | ✓ |
| [open_question_type](src/tasksource/tasks.py#L843) | Classification | [Korea-MES/open_question_type](https://hf.co/datasets/Korea-MES/open_question_type) |  |
| [health_fact](src/tasksource/tasks.py#L845) | Classification | [marcov/health_fact_promptsource](https://hf.co/datasets/marcov/health_fact_promptsource) |  |
| [commonsense_qa](src/tasksource/tasks.py#L849) | MultipleChoice | [tau/commonsense_qa](https://hf.co/datasets/tau/commonsense_qa) |  |
| [mc_taco](src/tasksource/tasks.py#L855) | Classification | [marcov/mc_taco_promptsource](https://hf.co/datasets/marcov/mc_taco_promptsource) | ✓ |
| [ade_corpus_v2/Ade_corpus_v2_classification](src/tasksource/tasks.py#L862) | Classification | [ade-benchmark-corpus/ade_corpus_v2](https://hf.co/datasets/ade-benchmark-corpus/ade_corpus_v2) |  |
| [discosense](src/tasksource/tasks.py#L864) | MultipleChoice | json |  |
| [circa](src/tasksource/tasks.py#L871) | Classification | [google-research-datasets/circa](https://hf.co/datasets/google-research-datasets/circa) |  |
| [code_x_glue_cc_defect_detection](src/tasksource/tasks.py#L876) | Classification | [google/code_x_glue_cc_defect_detection](https://hf.co/datasets/google/code_x_glue_cc_defect_detection) |  |
| [phrase_similarity](src/tasksource/tasks.py#L880) | Classification | [Deehan1866/processed_phrase_similarity](https://hf.co/datasets/Deehan1866/processed_phrase_similarity) |  |
| [scientific-exaggeration-detection](src/tasksource/tasks.py#L888) | Classification | [copenlu/scientific-exaggeration-detection](https://hf.co/datasets/copenlu/scientific-exaggeration-detection) |  |
| [quarel](src/tasksource/tasks.py#L894) | Classification | [community-datasets/quarel](https://hf.co/datasets/community-datasets/quarel) |  |
| [fever-evidence-related](src/tasksource/tasks.py#L899) | Classification | [mwong/fever-evidence-related](https://hf.co/datasets/mwong/fever-evidence-related) |  |
| [numer_sense](src/tasksource/tasks.py#L902) | Classification | [tasksource/numer_sense](https://hf.co/datasets/tasksource/numer_sense) |  |
| [dynasent/dynabench.dynasent.r1.all/r1](src/tasksource/tasks.py#L909) | Classification | [tasksource/dynasent](https://hf.co/datasets/tasksource/dynasent) |  |
| [dynasent/dynabench.dynasent.r2.all/r2](src/tasksource/tasks.py#L913) | Classification | [tasksource/dynasent](https://hf.co/datasets/tasksource/dynasent) |  |
| [Sarcasm_News_Headline](src/tasksource/tasks.py#L918) | Classification | [raquiba/Sarcasm_News_Headline](https://hf.co/datasets/raquiba/Sarcasm_News_Headline) |  |
| [sem_eval_2010_task_8](src/tasksource/tasks.py#L921) | Classification | [SemEvalWorkshop/sem_eval_2010_task_8](https://hf.co/datasets/SemEvalWorkshop/sem_eval_2010_task_8) |  |
| [auditor_review](src/tasksource/tasks.py#L923) | Classification | [demo-org/auditor_review](https://hf.co/datasets/demo-org/auditor_review) |  |
| [medmcqa](src/tasksource/tasks.py#L927) | MultipleChoice | [openlifescienceai/medmcqa](https://hf.co/datasets/openlifescienceai/medmcqa) |  |
| [Dynasent_Disagreement](src/tasksource/tasks.py#L944) | Classification | [RuyuanWan/Dynasent_Disagreement](https://hf.co/datasets/RuyuanWan/Dynasent_Disagreement) | ✓ |
| [Politeness_Disagreement](src/tasksource/tasks.py#L946) | Classification | [RuyuanWan/Politeness_Disagreement](https://hf.co/datasets/RuyuanWan/Politeness_Disagreement) | ✓ |
| [SBIC_Disagreement](src/tasksource/tasks.py#L948) | Classification | [RuyuanWan/SBIC_Disagreement](https://hf.co/datasets/RuyuanWan/SBIC_Disagreement) | ✓ |
| [SChem_Disagreement](src/tasksource/tasks.py#L950) | Classification | [RuyuanWan/SChem_Disagreement](https://hf.co/datasets/RuyuanWan/SChem_Disagreement) | ✓ |
| [Dilemmas_Disagreement](src/tasksource/tasks.py#L952) | Classification | [RuyuanWan/Dilemmas_Disagreement](https://hf.co/datasets/RuyuanWan/Dilemmas_Disagreement) | ✓ |
| [logiqa](src/tasksource/tasks.py#L955) | MultipleChoice | [fireworks-ai/logiqa](https://hf.co/datasets/fireworks-ai/logiqa) |  |
| [wiki_qa](src/tasksource/tasks.py#L964) | Classification | [microsoft/wiki_qa](https://hf.co/datasets/microsoft/wiki_qa) | ✓ |
| [cycic_classification](src/tasksource/tasks.py#L966) | Classification | [tasksource/cycic_classification](https://hf.co/datasets/tasksource/cycic_classification) |  |
| [cycic_multiplechoice](src/tasksource/tasks.py#L968) | MultipleChoice | [tasksource/cycic_multiplechoice](https://hf.co/datasets/tasksource/cycic_multiplechoice) |  |
| [sts-companion](src/tasksource/tasks.py#L972) | Classification | [tasksource/sts-companion](https://hf.co/datasets/tasksource/sts-companion) |  |
| [commonsense_qa_2.0](src/tasksource/tasks.py#L975) | Classification | [tasksource/commonsense_qa_2.0](https://hf.co/datasets/tasksource/commonsense_qa_2.0) |  |
| [lingnli](src/tasksource/tasks.py#L978) | Classification | [tasksource/lingnli](https://hf.co/datasets/tasksource/lingnli) |  |
| [monotonicity-entailment](src/tasksource/tasks.py#L980) | Classification | [tasksource/monotonicity-entailment](https://hf.co/datasets/tasksource/monotonicity-entailment) |  |
| [arct](src/tasksource/tasks.py#L983) | MultipleChoice | [tasksource/arct](https://hf.co/datasets/tasksource/arct) |  |
| [scinli](src/tasksource/tasks.py#L986) | Classification | [tasksource/scinli](https://hf.co/datasets/tasksource/scinli) |  |
| [naturallogic](src/tasksource/tasks.py#L990) | Classification | [tasksource/naturallogic](https://hf.co/datasets/tasksource/naturallogic) |  |
| [onestop_qa](src/tasksource/tasks.py#L992) | MultipleChoice | [malmaud/onestop_qa](https://hf.co/datasets/malmaud/onestop_qa) |  |
| [moral_stories/full](src/tasksource/tasks.py#L995) | MultipleChoice | [LabHC/moral_stories](https://hf.co/datasets/LabHC/moral_stories) |  |
| [prost](src/tasksource/tasks.py#L1003) | MultipleChoice | json |  |
| [dynahate](src/tasksource/tasks.py#L1008) | Classification | [tasksource/dynahate](https://hf.co/datasets/tasksource/dynahate) |  |
| [syntactic-augmentation-nli](src/tasksource/tasks.py#L1010) | Classification | [tasksource/syntactic-augmentation-nli](https://hf.co/datasets/tasksource/syntactic-augmentation-nli) |  |
| [autotnli](src/tasksource/tasks.py#L1012) | Classification | [tasksource/autotnli](https://hf.co/datasets/tasksource/autotnli) |  |
| [CONDAQA](src/tasksource/tasks.py#L1014) | Classification | [lasha-nlp/CONDAQA](https://hf.co/datasets/lasha-nlp/CONDAQA) |  |
| [webgpt_comparisons](src/tasksource/tasks.py#L1024) | MultipleChoice | [heegyu/webgpt_comparisons_ko](https://hf.co/datasets/heegyu/webgpt_comparisons_ko) | ✓ |
| [synthetic-instruct-gptj-pairwise](src/tasksource/tasks.py#L1032) | MultipleChoice | [Dahoas/synthetic-instruct-gptj-pairwise](https://hf.co/datasets/Dahoas/synthetic-instruct-gptj-pairwise) | ✓ |
| [scruples](src/tasksource/tasks.py#L1035) | Classification | [tasksource/scruples](https://hf.co/datasets/tasksource/scruples) | ✓ |
| [wouldyourather](src/tasksource/tasks.py#L1037) | MultipleChoice | [tasksource/wouldyourather](https://hf.co/datasets/tasksource/wouldyourather) | ✓ |
| [defeasible-nli/atomic](src/tasksource/tasks.py#L1045) | Classification | [tasksource/defeasible-nli](https://hf.co/datasets/tasksource/defeasible-nli) |  |
| [defeasible-nli/snli](src/tasksource/tasks.py#L1045) | Classification | [tasksource/defeasible-nli](https://hf.co/datasets/tasksource/defeasible-nli) |  |
| [defeasible-nli/social](src/tasksource/tasks.py#L1048) | Classification | [tasksource/defeasible-nli](https://hf.co/datasets/tasksource/defeasible-nli) |  |
| [help-nli](src/tasksource/tasks.py#L1051) | Classification | [tasksource/help-nli](https://hf.co/datasets/tasksource/help-nli) |  |
| [nli-veridicality-transitivity](src/tasksource/tasks.py#L1054) | Classification | [tasksource/nli-veridicality-transitivity](https://hf.co/datasets/tasksource/nli-veridicality-transitivity) |  |
| [lonli](src/tasksource/tasks.py#L1057) | Classification | [tasksource/lonli](https://hf.co/datasets/tasksource/lonli) |  |
| [dadc-limit-nli](src/tasksource/tasks.py#L1060) | Classification | [tasksource/dadc-limit-nli](https://hf.co/datasets/tasksource/dadc-limit-nli) |  |
| [FLUTE](src/tasksource/tasks.py#L1063) | Classification | [ColumbiaNLP/FLUTE](https://hf.co/datasets/ColumbiaNLP/FLUTE) |  |
| [strategy-qa](src/tasksource/tasks.py#L1066) | Classification | [tasksource/strategy-qa](https://hf.co/datasets/tasksource/strategy-qa) |  |
| [summarize_from_feedback/comparisons](src/tasksource/tasks.py#L1069) | MultipleChoice | [vwxyzjn/summarize_from_feedback_oai_preprocessing](https://hf.co/datasets/vwxyzjn/summarize_from_feedback_oai_preprocessing) | ✓ |
| [folio](src/tasksource/tasks.py#L1077) | Classification | [tasksource/folio](https://hf.co/datasets/tasksource/folio) |  |
| [tomi-nli](src/tasksource/tasks.py#L1081) | Classification | [tasksource/tomi-nli](https://hf.co/datasets/tasksource/tomi-nli) |  |
| [avicenna](src/tasksource/tasks.py#L1084) | Classification | [tasksource/avicenna](https://hf.co/datasets/tasksource/avicenna) | ✓ |
| [SHP](src/tasksource/tasks.py#L1087) | MultipleChoice | [stanfordnlp/SHP](https://hf.co/datasets/stanfordnlp/SHP) | ✓ |
| [MedQA-USMLE-4-options-hf](src/tasksource/tasks.py#L1095) | MultipleChoice | [GBaker/MedQA-USMLE-4-options-hf](https://hf.co/datasets/GBaker/MedQA-USMLE-4-options-hf) |  |
| [wikimedqa/medwiki](src/tasksource/tasks.py#L1098) | MultipleChoice | [sileod/wikimedqa](https://hf.co/datasets/sileod/wikimedqa) |  |
| [cicero](src/tasksource/tasks.py#L1107) | MultipleChoice | [declare-lab/cicero](https://hf.co/datasets/declare-lab/cicero) |  |
| [CREAK](src/tasksource/tasks.py#L1111) | Classification | [amydeng2000/CREAK](https://hf.co/datasets/amydeng2000/CREAK) |  |
| [mutual](src/tasksource/tasks.py#L1114) | MultipleChoice | [tasksource/mutual](https://hf.co/datasets/tasksource/mutual) |  |
| [puzzte](src/tasksource/tasks.py#L1118) | Classification | [tasksource/puzzte](https://hf.co/datasets/tasksource/puzzte) |  |
| [implicatures](src/tasksource/tasks.py#L1123) | MultipleChoice | [tasksource/implicatures](https://hf.co/datasets/tasksource/implicatures) |  |
| [race/high](src/tasksource/tasks.py#L1128) | MultipleChoice | [ehovy/race](https://hf.co/datasets/ehovy/race) |  |
| [race/middle](src/tasksource/tasks.py#L1128) | MultipleChoice | [ehovy/race](https://hf.co/datasets/ehovy/race) |  |
| [race-c](src/tasksource/tasks.py#L1132) | MultipleChoice | [tasksource/race-c](https://hf.co/datasets/tasksource/race-c) |  |
| [spartqa-yn](src/tasksource/tasks.py#L1135) | Classification | [tasksource/spartqa-yn](https://hf.co/datasets/tasksource/spartqa-yn) |  |
| [spartqa-mchoice](src/tasksource/tasks.py#L1138) | MultipleChoice | [tasksource/spartqa-mchoice](https://hf.co/datasets/tasksource/spartqa-mchoice) |  |
| [temporal-nli](src/tasksource/tasks.py#L1141) | Classification | [tasksource/temporal-nli](https://hf.co/datasets/tasksource/temporal-nli) |  |
| [riddle_sense](src/tasksource/tasks.py#L1144) | MultipleChoice | [jeggers/riddle_sense](https://hf.co/datasets/jeggers/riddle_sense) |  |
| [clcd-english](src/tasksource/tasks.py#L1149) | Classification | [tasksource/clcd-english](https://hf.co/datasets/tasksource/clcd-english) |  |
| [twentyquestions](src/tasksource/tasks.py#L1161) | Classification | [tasksource/twentyquestions](https://hf.co/datasets/tasksource/twentyquestions) |  |
| [reclor](src/tasksource/tasks.py#L1166) | MultipleChoice | [tasksource/reclor](https://hf.co/datasets/tasksource/reclor) |  |
| [counterfactually-augmented-imdb](src/tasksource/tasks.py#L1169) | Classification | [tasksource/counterfactually-augmented-imdb](https://hf.co/datasets/tasksource/counterfactually-augmented-imdb) |  |
| [counterfactually-augmented-snli](src/tasksource/tasks.py#L1172) | Classification | [tasksource/counterfactually-augmented-snli](https://hf.co/datasets/tasksource/counterfactually-augmented-snli) |  |
| [cnli](src/tasksource/tasks.py#L1175) | Classification | [tasksource/cnli](https://hf.co/datasets/tasksource/cnli) |  |
| [boolq-natural-perturbations](src/tasksource/tasks.py#L1178) | Classification | [tasksource/boolq-natural-perturbations](https://hf.co/datasets/tasksource/boolq-natural-perturbations) |  |
| [acceptability-prediction](src/tasksource/tasks.py#L1182) | Classification | [tasksource/acceptability-prediction](https://hf.co/datasets/tasksource/acceptability-prediction) | ✓ |
| [equate](src/tasksource/tasks.py#L1186) | Classification | [tasksource/equate](https://hf.co/datasets/tasksource/equate) |  |
| [ScienceQA_text_only](src/tasksource/tasks.py#L1189) | MultipleChoice | [tasksource/ScienceQA_text_only](https://hf.co/datasets/tasksource/ScienceQA_text_only) |  |
| [ekar_english](src/tasksource/tasks.py#L1192) | MultipleChoice | [Jiangjie/ekar_english](https://hf.co/datasets/Jiangjie/ekar_english) | ✓ |
| [implicit-hate-stg1](src/tasksource/tasks.py#L1196) | Classification | [tasksource/implicit-hate-stg1](https://hf.co/datasets/tasksource/implicit-hate-stg1) |  |
| [chaos-mnli-ambiguity](src/tasksource/tasks.py#L1199) | Classification | [tasksource/chaos-mnli-ambiguity](https://hf.co/datasets/tasksource/chaos-mnli-ambiguity) | ✓ |
| [headline_cause/en_simple](src/tasksource/tasks.py#L1203) | Classification | json |  |
| [logiqa-2.0-nli](src/tasksource/tasks.py#L1208) | Classification | [tasksource/logiqa-2.0-nli](https://hf.co/datasets/tasksource/logiqa-2.0-nli) |  |
| [oasst2_dense_flat/quality](src/tasksource/tasks.py#L1213) | Classification | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) | ✓ |
| [oasst2_dense_flat/toxicity](src/tasksource/tasks.py#L1215) | Classification | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) | ✓ |
| [oasst2_dense_flat/helpfulness](src/tasksource/tasks.py#L1217) | Classification | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) | ✓ |
| [mindgames](src/tasksource/tasks.py#L1220) | Classification | [sileod/mindgames](https://hf.co/datasets/sileod/mindgames) |  |
| [universal_dependencies/en_gum/deprel](src/tasksource/tasks.py#L1234) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| [universal_dependencies/en_ewt/deprel](src/tasksource/tasks.py#L1234) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| [universal_dependencies/en_lines/deprel](src/tasksource/tasks.py#L1234) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| [universal_dependencies/en_partut/deprel](src/tasksource/tasks.py#L1234) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| [ambient](src/tasksource/tasks.py#L1240) | Classification | [tasksource/ambient](https://hf.co/datasets/tasksource/ambient) | ✓ |
| [path-naturalness-prediction](src/tasksource/tasks.py#L1243) | MultipleChoice | [tasksource/path-naturalness-prediction](https://hf.co/datasets/tasksource/path-naturalness-prediction) | ✓ |
| [civil_comments/toxicity](src/tasksource/tasks.py#L1253) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| [civil_comments/severe_toxicity](src/tasksource/tasks.py#L1254) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| [civil_comments/obscene](src/tasksource/tasks.py#L1255) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| [civil_comments/threat](src/tasksource/tasks.py#L1256) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| [civil_comments/insult](src/tasksource/tasks.py#L1257) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| [civil_comments/identity_attack](src/tasksource/tasks.py#L1258) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| [civil_comments/sexual_explicit](src/tasksource/tasks.py#L1259) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| [cloth](src/tasksource/tasks.py#L1261) | MultipleChoice | [AndyChiang/cloth](https://hf.co/datasets/AndyChiang/cloth) |  |
| [dgen](src/tasksource/tasks.py#L1262) | MultipleChoice | [AndyChiang/dgen](https://hf.co/datasets/AndyChiang/dgen) |  |
| [I2D2](src/tasksource/tasks.py#L1264) | Classification | [tasksource/I2D2](https://hf.co/datasets/tasksource/I2D2) |  |
| [args_me](src/tasksource/tasks.py#L1266) | Classification | [webis/args_me](https://hf.co/datasets/webis/args_me) |  |
| [Touche23-ValueEval](src/tasksource/tasks.py#L1269) | Classification | csv |  |
| [starcon](src/tasksource/tasks.py#L1277) | Classification | [tasksource/starcon](https://hf.co/datasets/tasksource/starcon) |  |
| [banking77](src/tasksource/tasks.py#L1279) | Classification | [legacy-datasets/banking77](https://hf.co/datasets/legacy-datasets/banking77) |  |
| [it-support-tickets](src/tasksource/tasks.py#L1281) | Classification | [tasksource/it-support-tickets](https://hf.co/datasets/tasksource/it-support-tickets) |  |
| [ConTRoL-nli](src/tasksource/tasks.py#L1285) | Classification | [tasksource/ConTRoL-nli](https://hf.co/datasets/tasksource/ConTRoL-nli) |  |
| [tracie](src/tasksource/tasks.py#L1286) | Classification | [tasksource/tracie](https://hf.co/datasets/tasksource/tracie) |  |
| [sherliic](src/tasksource/tasks.py#L1287) | Classification | [tasksource/sherliic](https://hf.co/datasets/tasksource/sherliic) |  |
| [sen-making/1](src/tasksource/tasks.py#L1289) | MultipleChoice | [tasksource/sen-making](https://hf.co/datasets/tasksource/sen-making) | ✓ |
| [sen-making/2](src/tasksource/tasks.py#L1293) | MultipleChoice | [tasksource/sen-making](https://hf.co/datasets/tasksource/sen-making) | ✓ |
| [winowhy](src/tasksource/tasks.py#L1296) | Classification | [tasksource/winowhy](https://hf.co/datasets/tasksource/winowhy) | ✓ |
| [robustLR](src/tasksource/tasks.py#L1300) | Classification | [tasksource/robustLR](https://hf.co/datasets/tasksource/robustLR) |  |
| [clutrr](src/tasksource/tasks.py#L1302) | Classification | [tasksource/clutrr](https://hf.co/datasets/tasksource/clutrr) |  |
| [logical-fallacy](src/tasksource/tasks.py#L1304) | Classification | [tasksource/logical-fallacy](https://hf.co/datasets/tasksource/logical-fallacy) |  |
| [parade](src/tasksource/tasks.py#L1306) | Classification | [tasksource/parade](https://hf.co/datasets/tasksource/parade) |  |
| [cladder](src/tasksource/tasks.py#L1308) | Classification | [tasksource/cladder](https://hf.co/datasets/tasksource/cladder) |  |
| [subjectivity](src/tasksource/tasks.py#L1310) | Classification | [tasksource/subjectivity](https://hf.co/datasets/tasksource/subjectivity) |  |
| [MOH](src/tasksource/tasks.py#L1312) | Classification | [tasksource/MOH](https://hf.co/datasets/tasksource/MOH) |  |
| [VUAC](src/tasksource/tasks.py#L1313) | Classification | [tasksource/VUAC](https://hf.co/datasets/tasksource/VUAC) |  |
| [TroFi](src/tasksource/tasks.py#L1314) | Classification | parquet |  |
| [sharc](src/tasksource/tasks.py#L1321) | Classification | [tasksource/sharc](https://hf.co/datasets/tasksource/sharc) |  |
| [conceptrules_v2](src/tasksource/tasks.py#L1325) | Classification | [tasksource/conceptrules_v2](https://hf.co/datasets/tasksource/conceptrules_v2) | ✓ |
| [disrpt/eng.dep.scidtb.rels](src/tasksource/tasks.py#L1327) | Classification | [multilingual-discourse-hub/disrpt](https://hf.co/datasets/multilingual-discourse-hub/disrpt) |  |
| [conll2000](src/tasksource/tasks.py#L1329) | TokenClassification | [eriktks/conll2000](https://hf.co/datasets/eriktks/conll2000) |  |
| [few-nerd/supervised](src/tasksource/tasks.py#L1332) | TokenClassification | [DFKI-SLT/few-nerd](https://hf.co/datasets/DFKI-SLT/few-nerd) |  |
| [finer-139](src/tasksource/tasks.py#L1333) | TokenClassification | [nlpaueb/finer-139](https://hf.co/datasets/nlpaueb/finer-139) |  |
| [zero-shot-label-nli](src/tasksource/tasks.py#L1336) | Classification | [tasksource/zero-shot-label-nli](https://hf.co/datasets/tasksource/zero-shot-label-nli) |  |
| [com2sense](src/tasksource/tasks.py#L1338) | Classification | [tasksource/com2sense](https://hf.co/datasets/tasksource/com2sense) |  |
| [scone](src/tasksource/tasks.py#L1340) | Classification | [tasksource/scone](https://hf.co/datasets/tasksource/scone) |  |
| [winodict](src/tasksource/tasks.py#L1342) | MultipleChoice | [tasksource/winodict](https://hf.co/datasets/tasksource/winodict) |  |
| [fool-me-twice](src/tasksource/tasks.py#L1344) | Classification | [tasksource/fool-me-twice](https://hf.co/datasets/tasksource/fool-me-twice) |  |
| [monli](src/tasksource/tasks.py#L1348) | Classification | [tasksource/monli](https://hf.co/datasets/tasksource/monli) |  |
| [corr2cause](src/tasksource/tasks.py#L1350) | Classification | [tasksource/corr2cause](https://hf.co/datasets/tasksource/corr2cause) |  |
| [lsat_qa/all](src/tasksource/tasks.py#L1352) | MultipleChoice | [lighteval/lsat_qa](https://hf.co/datasets/lighteval/lsat_qa) |  |
| [apt](src/tasksource/tasks.py#L1354) | Classification | [tasksource/apt](https://hf.co/datasets/tasksource/apt) |  |
| [twitter-financial-news-sentiment](src/tasksource/tasks.py#L1357) | Classification | [zeroshot/twitter-financial-news-sentiment](https://hf.co/datasets/zeroshot/twitter-financial-news-sentiment) |  |
| [icl-symbol-tuning-instruct](src/tasksource/tasks.py#L1364) | Classification | [tasksource/icl-symbol-tuning-instruct](https://hf.co/datasets/tasksource/icl-symbol-tuning-instruct) | ✓ |
| [SpaceNLI](src/tasksource/tasks.py#L1370) | Classification | [tasksource/SpaceNLI](https://hf.co/datasets/tasksource/SpaceNLI) |  |
| [propsegment/nli](src/tasksource/tasks.py#L1372) | Classification | json |  |
| [HatemojiBuild](src/tasksource/tasks.py#L1381) | Classification | [HannahRoseKirk/HatemojiBuild](https://hf.co/datasets/HannahRoseKirk/HatemojiBuild) |  |
| [regset](src/tasksource/tasks.py#L1384) | Classification | [tasksource/regset](https://hf.co/datasets/tasksource/regset) | ✓ |
| [esci](src/tasksource/tasks.py#L1390) | Classification | [tasksource/esci](https://hf.co/datasets/tasksource/esci) |  |
| [chatbot_arena_conversations](src/tasksource/tasks.py#L1409) | MultipleChoice | [lmsys/chatbot_arena_conversations](https://hf.co/datasets/lmsys/chatbot_arena_conversations) | ✓ |
| [dnd_style_intents](src/tasksource/tasks.py#L1415) | Classification | [neurae/dnd_style_intents](https://hf.co/datasets/neurae/dnd_style_intents) |  |
| [FLD.v2/default](src/tasksource/tasks.py#L1418) | Classification | [hitachi-nlp/FLD.v2](https://hf.co/datasets/hitachi-nlp/FLD.v2) |  |
| [FLD.v2/star](src/tasksource/tasks.py#L1421) | Classification | [hitachi-nlp/FLD.v2](https://hf.co/datasets/hitachi-nlp/FLD.v2) |  |
| [SDOH-NLI](src/tasksource/tasks.py#L1424) | Classification | [tasksource/SDOH-NLI](https://hf.co/datasets/tasksource/SDOH-NLI) |  |
| [scifact_entailment](src/tasksource/tasks.py#L1427) | Classification | [tasksource/scifact_entailment](https://hf.co/datasets/tasksource/scifact_entailment) |  |
| [feasibilityQA](src/tasksource/tasks.py#L1431) | Classification | [tasksource/feasibilityQA](https://hf.co/datasets/tasksource/feasibilityQA) |  |
| [simple_pair](src/tasksource/tasks.py#L1434) | Classification | [tasksource/simple_pair](https://hf.co/datasets/tasksource/simple_pair) |  |
| [AdjectiveScaleProbe-nli](src/tasksource/tasks.py#L1435) | Classification | [tasksource/AdjectiveScaleProbe-nli](https://hf.co/datasets/tasksource/AdjectiveScaleProbe-nli) |  |
| [resnli](src/tasksource/tasks.py#L1436) | Classification | [tasksource/resnli](https://hf.co/datasets/tasksource/resnli) |  |
| [SpaRTUN](src/tasksource/tasks.py#L1438) | MultipleChoice | [tasksource/SpaRTUN](https://hf.co/datasets/tasksource/SpaRTUN) |  |
| [ReSQ](src/tasksource/tasks.py#L1443) | MultipleChoice | [tasksource/ReSQ](https://hf.co/datasets/tasksource/ReSQ) |  |
| [semantic_fragments_nli](src/tasksource/tasks.py#L1448) | Classification | [tasksource/semantic_fragments_nli](https://hf.co/datasets/tasksource/semantic_fragments_nli) |  |
| [dataset_train_nli](src/tasksource/tasks.py#L1451) | Classification | [MoritzLaurer/dataset_train_nli](https://hf.co/datasets/MoritzLaurer/dataset_train_nli) |  |
| [stepgame](src/tasksource/tasks.py#L1456) | Classification | [tasksource/stepgame](https://hf.co/datasets/tasksource/stepgame) |  |
| [nlgraph](src/tasksource/tasks.py#L1464) | Classification | [tasksource/nlgraph](https://hf.co/datasets/tasksource/nlgraph) |  |
| [oasst2_pairwise_rlhf_reward](src/tasksource/tasks.py#L1468) | MultipleChoice | [tasksource/oasst2_pairwise_rlhf_reward](https://hf.co/datasets/tasksource/oasst2_pairwise_rlhf_reward) | ✓ |
| [hh-rlhf/helpful-online](src/tasksource/tasks.py#L1479) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| [hh-rlhf/helpful-base](src/tasksource/tasks.py#L1479) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| [hh-rlhf/helpful-rejection-sampled](src/tasksource/tasks.py#L1479) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| [hh-rlhf/harmless-base](src/tasksource/tasks.py#L1483) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| [ruletaker](src/tasksource/tasks.py#L1487) | Classification | [tasksource/ruletaker](https://hf.co/datasets/tasksource/ruletaker) | ✓ |
| [PARARULE-Plus](src/tasksource/tasks.py#L1491) | Classification | [qbao775/PARARULE-Plus](https://hf.co/datasets/qbao775/PARARULE-Plus) | ✓ |
| [proofwriter](src/tasksource/tasks.py#L1495) | Classification | [tasksource/proofwriter](https://hf.co/datasets/tasksource/proofwriter) |  |
| [logical-entailment](src/tasksource/tasks.py#L1498) | Classification | [tasksource/logical-entailment](https://hf.co/datasets/tasksource/logical-entailment) |  |
| [nope](src/tasksource/tasks.py#L1500) | Classification | [tasksource/nope](https://hf.co/datasets/tasksource/nope) |  |
| [LogicNLI](src/tasksource/tasks.py#L1504) | Classification | [tasksource/LogicNLI](https://hf.co/datasets/tasksource/LogicNLI) |  |
| [contract-nli/contractnli_a/seg](src/tasksource/tasks.py#L1506) | Classification | [tasksource/contract-nli](https://hf.co/datasets/tasksource/contract-nli) |  |
| [contract-nli/contractnli_b/full](src/tasksource/tasks.py#L1508) | Classification | [tasksource/contract-nli](https://hf.co/datasets/tasksource/contract-nli) |  |
| [nli4ct_semeval2024](src/tasksource/tasks.py#L1510) | Classification | [AshtonIsNotHere/nli4ct_semeval2024](https://hf.co/datasets/AshtonIsNotHere/nli4ct_semeval2024) |  |
| [lsat-ar](src/tasksource/tasks.py#L1513) | MultipleChoice | [tasksource/lsat-ar](https://hf.co/datasets/tasksource/lsat-ar) |  |
| [lsat-rc](src/tasksource/tasks.py#L1518) | MultipleChoice | [tasksource/lsat-rc](https://hf.co/datasets/tasksource/lsat-rc) |  |
| [biosift-nli](src/tasksource/tasks.py#L1523) | Classification | [AshtonIsNotHere/biosift-nli](https://hf.co/datasets/AshtonIsNotHere/biosift-nli) |  |
| [brainteasers/SP](src/tasksource/tasks.py#L1527) | MultipleChoice | [tasksource/brainteasers](https://hf.co/datasets/tasksource/brainteasers) |  |
| [brainteasers/WP](src/tasksource/tasks.py#L1527) | MultipleChoice | [tasksource/brainteasers](https://hf.co/datasets/tasksource/brainteasers) |  |
| [toxigen-data/annotated](src/tasksource/tasks.py#L1533) | Classification | [skg/toxigen-data](https://hf.co/datasets/skg/toxigen-data) |  |
| [persuasion](src/tasksource/tasks.py#L1544) | Classification | [Anthropic/persuasion](https://hf.co/datasets/Anthropic/persuasion) |  |
| [AmbigNQ-clarifying-question](src/tasksource/tasks.py#L1550) | Classification | [erbacher/AmbigNQ-clarifying-question](https://hf.co/datasets/erbacher/AmbigNQ-clarifying-question) |  |
| [SIGA-nli](src/tasksource/tasks.py#L1553) | Classification | [tasksource/SIGA-nli](https://hf.co/datasets/tasksource/SIGA-nli) |  |
| [FOL-nli](src/tasksource/tasks.py#L1555) | Classification | [unigram/FOL-nli](https://hf.co/datasets/unigram/FOL-nli) |  |
| [goal-step-wikihow/goal](src/tasksource/tasks.py#L1557) | MultipleChoice | [tasksource/goal-step-wikihow](https://hf.co/datasets/tasksource/goal-step-wikihow) |  |
| [goal-step-wikihow/step](src/tasksource/tasks.py#L1560) | MultipleChoice | [tasksource/goal-step-wikihow](https://hf.co/datasets/tasksource/goal-step-wikihow) |  |
| [goal-step-wikihow/order](src/tasksource/tasks.py#L1563) | MultipleChoice | [tasksource/goal-step-wikihow](https://hf.co/datasets/tasksource/goal-step-wikihow) |  |
| [PARADISE](src/tasksource/tasks.py#L1566) | MultipleChoice | [GGLab/PARADISE](https://hf.co/datasets/GGLab/PARADISE) |  |
| [doc-nli](src/tasksource/tasks.py#L1569) | Classification | [tasksource/doc-nli](https://hf.co/datasets/tasksource/doc-nli) |  |
| [mctest-nli](src/tasksource/tasks.py#L1571) | Classification | [tasksource/mctest-nli](https://hf.co/datasets/tasksource/mctest-nli) |  |
| [patent-phrase-similarity](src/tasksource/tasks.py#L1573) | Classification | [tasksource/patent-phrase-similarity](https://hf.co/datasets/tasksource/patent-phrase-similarity) |  |
| [natural-language-satisfiability](src/tasksource/tasks.py#L1575) | Classification | [tasksource/natural-language-satisfiability](https://hf.co/datasets/tasksource/natural-language-satisfiability) |  |
| [idioms-nli](src/tasksource/tasks.py#L1577) | Classification | [tasksource/idioms-nli](https://hf.co/datasets/tasksource/idioms-nli) |  |
| [lifecycle-entailment](src/tasksource/tasks.py#L1579) | Classification | [tasksource/lifecycle-entailment](https://hf.co/datasets/tasksource/lifecycle-entailment) |  |
| [toxic-chat/toxicchat0124/toxicity](src/tasksource/tasks.py#L1584) | Classification | [lmsys/toxic-chat](https://hf.co/datasets/lmsys/toxic-chat) | ✓ |
| [toxic-chat/toxicchat0124/jailbreaking](src/tasksource/tasks.py#L1589) | Classification | [lmsys/toxic-chat](https://hf.co/datasets/lmsys/toxic-chat) | ✓ |
| [clinc_oos/plus](src/tasksource/tasks.py#L1595) | Classification | [clinc/clinc_oos](https://hf.co/datasets/clinc/clinc_oos) |  |
| [few_rel/default](src/tasksource/tasks.py#L1632) | Classification | [tasksource/few_rel](https://hf.co/datasets/tasksource/few_rel) |  |
| [docred](src/tasksource/tasks.py#L1678) | Classification | json |  |
| [chemprot/chemprot_full_source](src/tasksource/tasks.py#L1706) | Classification | [bigbio/chemprot](https://hf.co/datasets/bigbio/chemprot) |  |
| [PKU-SafeRLHF/helpfulness](src/tasksource/tasks.py#L1711) | MultipleChoice | [PKU-Alignment/PKU-SafeRLHF](https://hf.co/datasets/PKU-Alignment/PKU-SafeRLHF) | ✓ |
| [PKU-SafeRLHF/safety](src/tasksource/tasks.py#L1716) | MultipleChoice | [PKU-Alignment/PKU-SafeRLHF](https://hf.co/datasets/PKU-Alignment/PKU-SafeRLHF) | ✓ |
| [HelpSteer/helpfulness](src/tasksource/tasks.py#L1730) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| [HelpSteer/correctness](src/tasksource/tasks.py#L1731) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| [HelpSteer/coherence](src/tasksource/tasks.py#L1732) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| [HelpSteer/complexity](src/tasksource/tasks.py#L1733) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| [HelpSteer/verbosity](src/tasksource/tasks.py#L1734) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| [HelpSteer2/helpfulness](src/tasksource/tasks.py#L1736) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| [HelpSteer2/correctness](src/tasksource/tasks.py#L1737) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| [HelpSteer2/coherence](src/tasksource/tasks.py#L1738) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| [HelpSteer2/complexity](src/tasksource/tasks.py#L1739) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| [HelpSteer2/verbosity](src/tasksource/tasks.py#L1740) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| [HelpSteer3/preference](src/tasksource/tasks.py#L1745) | MultipleChoice | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | ✓ |
| [HelpSteer3/principle](src/tasksource/tasks.py#L1750) | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) |  |
| [HelpSteer3/edit_quality](src/tasksource/tasks.py#L1755) | MultipleChoice | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | ✓ |
| [HelpSteer3/feedback](src/tasksource/tasks.py#L1778) | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | ✓ |
| [MSciNLI](src/tasksource/tasks.py#L1783) | Classification | [sadat2307/MSciNLI](https://hf.co/datasets/sadat2307/MSciNLI) |  |
| [UltraFeedback-paired](src/tasksource/tasks.py#L1786) | MultipleChoice | [pushpdeep/UltraFeedback-paired](https://hf.co/datasets/pushpdeep/UltraFeedback-paired) | ✓ |
| [prm800k_dpo/solution](src/tasksource/tasks.py#L1790) | MultipleChoice | [tasksource/prm800k_dpo](https://hf.co/datasets/tasksource/prm800k_dpo) | ✓ |
| [prm800k_dpo/step](src/tasksource/tasks.py#L1793) | MultipleChoice | [tasksource/prm800k_dpo](https://hf.co/datasets/tasksource/prm800k_dpo) | ✓ |
| [AES2-essay-scoring](src/tasksource/tasks.py#L1797) | Classification | [tasksource/AES2-essay-scoring](https://hf.co/datasets/tasksource/AES2-essay-scoring) | ✓ |
| [argument-feedback](src/tasksource/tasks.py#L1801) | Classification | [tasksource/argument-feedback](https://hf.co/datasets/tasksource/argument-feedback) | ✓ |
| [english-grading/cohesion](src/tasksource/tasks.py#L1808) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| [english-grading/syntax](src/tasksource/tasks.py#L1809) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| [english-grading/vocabulary](src/tasksource/tasks.py#L1810) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| [english-grading/phraseology](src/tasksource/tasks.py#L1811) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| [english-grading/grammar](src/tasksource/tasks.py#L1812) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| [english-grading/conventions](src/tasksource/tasks.py#L1813) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| [wice](src/tasksource/tasks.py#L1815) | Classification | [tasksource/wice](https://hf.co/datasets/tasksource/wice) |  |
| [hover](src/tasksource/tasks.py#L1818) | Classification | [Dzeniks/hover](https://hf.co/datasets/Dzeniks/hover) |  |
| [hover-3way/nli](src/tasksource/tasks.py#L1822) | Classification | [Dzeniks/hover-3way](https://hf.co/datasets/Dzeniks/hover-3way) |  |
| [tasksource_dpo_pairs](src/tasksource/tasks.py#L1825) | MultipleChoice | [tasksource/tasksource_dpo_pairs](https://hf.co/datasets/tasksource/tasksource_dpo_pairs) | ✓ |
| [seahorse_summarization_evaluation](src/tasksource/tasks.py#L1828) | Classification | [tasksource/seahorse_summarization_evaluation](https://hf.co/datasets/tasksource/seahorse_summarization_evaluation) |  |
| [missing-item-prediction/contrastive](src/tasksource/tasks.py#L1831) | Classification | [sileod/missing-item-prediction](https://hf.co/datasets/sileod/missing-item-prediction) |  |
| [jigsaw_toxicity](src/tasksource/tasks.py#L1835) | Classification | [tasksource/jigsaw_toxicity](https://hf.co/datasets/tasksource/jigsaw_toxicity) |  |
| [Pol_NLI](src/tasksource/tasks.py#L1838) | Classification | [mlburnham/Pol_NLI](https://hf.co/datasets/mlburnham/Pol_NLI) |  |
| [synthetic-retrieval-NLI/position](src/tasksource/tasks.py#L1841) | Classification | [tasksource/synthetic-retrieval-NLI](https://hf.co/datasets/tasksource/synthetic-retrieval-NLI) |  |
| [synthetic-retrieval-NLI/binary](src/tasksource/tasks.py#L1841) | Classification | [tasksource/synthetic-retrieval-NLI](https://hf.co/datasets/tasksource/synthetic-retrieval-NLI) |  |
| [synthetic-retrieval-NLI/count](src/tasksource/tasks.py#L1841) | Classification | [tasksource/synthetic-retrieval-NLI](https://hf.co/datasets/tasksource/synthetic-retrieval-NLI) |  |
| [github-issue-similarity](src/tasksource/tasks.py#L1850) | Classification | [WhereIsAI/github-issue-similarity](https://hf.co/datasets/WhereIsAI/github-issue-similarity) |  |
