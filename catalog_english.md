502 English tasks. Load one with `load_task(id)`; the annotations are in [tasks.py](src/tasksource/tasks.py). Evaluation benchmarks are in [eval_only.py](src/tasksource/eval_only.py); other excluded annotations are in [parked.py](src/tasksource/parked.py).

| # | id | type | dataset | question |
|--:|---|---|---|:-:|
| 1 | [glue/mnli](src/tasksource/tasks.py#L32) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 2 | [glue/qnli](src/tasksource/tasks.py#L33) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 3 | [glue/rte](src/tasksource/tasks.py#L34) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 4 | [glue/wnli](src/tasksource/tasks.py#L35) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 5 | [glue/mrpc](src/tasksource/tasks.py#L37) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 6 | [glue/qqp](src/tasksource/tasks.py#L38) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 7 | [glue/stsb](src/tasksource/tasks.py#L44) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) | ✓ |
| 8 | [super_glue/boolq](src/tasksource/tasks.py#L49) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 9 | [super_glue/boolq_passage](src/tasksource/tasks.py#L50) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 10 | [super_glue/cb](src/tasksource/tasks.py#L52) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 11 | [super_glue/multirc](src/tasksource/tasks.py#L53) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 12 | [super_glue/wic](src/tasksource/tasks.py#L58) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 13 | [super_glue/axg](src/tasksource/tasks.py#L63) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 14 | [anli/a1](src/tasksource/tasks.py#L66) | Classification | [facebook/anli](https://hf.co/datasets/facebook/anli) |  |
| 15 | [anli/a2](src/tasksource/tasks.py#L67) | Classification | [facebook/anli](https://hf.co/datasets/facebook/anli) |  |
| 16 | [anli/a3](src/tasksource/tasks.py#L68) | Classification | [facebook/anli](https://hf.co/datasets/facebook/anli) |  |
| 17 | [babi_nli/basic-coreference](src/tasksource/tasks.py#L71) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 18 | [babi_nli/counting](src/tasksource/tasks.py#L71) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 19 | [babi_nli/conjunction](src/tasksource/tasks.py#L71) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 20 | [babi_nli/compound-coreference](src/tasksource/tasks.py#L71) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 21 | [babi_nli/basic-induction](src/tasksource/tasks.py#L71) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 22 | [babi_nli/basic-deduction](src/tasksource/tasks.py#L71) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 23 | [babi_nli/path-finding](src/tasksource/tasks.py#L71) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 24 | [babi_nli/positional-reasoning](src/tasksource/tasks.py#L71) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 25 | [babi_nli/simple-negation](src/tasksource/tasks.py#L71) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 26 | [babi_nli/indefinite-knowledge](src/tasksource/tasks.py#L71) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 27 | [babi_nli/single-supporting-fact](src/tasksource/tasks.py#L71) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 28 | [babi_nli/size-reasoning](src/tasksource/tasks.py#L71) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 29 | [babi_nli/three-arg-relations](src/tasksource/tasks.py#L71) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 30 | [babi_nli/three-supporting-facts](src/tasksource/tasks.py#L71) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 31 | [babi_nli/time-reasoning](src/tasksource/tasks.py#L71) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 32 | [babi_nli/two-arg-relations](src/tasksource/tasks.py#L71) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 33 | [babi_nli/two-supporting-facts](src/tasksource/tasks.py#L71) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 34 | [babi_nli/yes-no-questions](src/tasksource/tasks.py#L71) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 35 | [babi_nli/lists-sets](src/tasksource/tasks.py#L71) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 36 | [sick/label](src/tasksource/tasks.py#L76) | Classification | [tasksource/sick](https://hf.co/datasets/tasksource/sick) |  |
| 37 | [sick/relatedness](src/tasksource/tasks.py#L78) | Classification | [tasksource/sick](https://hf.co/datasets/tasksource/sick) | ✓ |
| 38 | [snli](src/tasksource/tasks.py#L136) | Classification | [stanfordnlp/snli](https://hf.co/datasets/stanfordnlp/snli) |  |
| 39 | [scitail/snli_format](src/tasksource/tasks.py#L139) | Classification | [allenai/scitail](https://hf.co/datasets/allenai/scitail) |  |
| 40 | [hans](src/tasksource/tasks.py#L141) | Classification | [tasksource/hans](https://hf.co/datasets/tasksource/hans) |  |
| 41 | [WANLI](src/tasksource/tasks.py#L144) | Classification | [alisawuffles/WANLI](https://hf.co/datasets/alisawuffles/WANLI) |  |
| 42 | [recast/recast_megaveridicality](src/tasksource/tasks.py#L146) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 43 | [recast/recast_sentiment](src/tasksource/tasks.py#L146) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 44 | [recast/recast_ner](src/tasksource/tasks.py#L146) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 45 | [recast/recast_verbcorner](src/tasksource/tasks.py#L146) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 46 | [recast/recast_verbnet](src/tasksource/tasks.py#L146) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 47 | [recast/recast_factuality](src/tasksource/tasks.py#L146) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 48 | [recast/recast_puns](src/tasksource/tasks.py#L146) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 49 | [probability_words_nli/reasoning_1hop](src/tasksource/tasks.py#L151) | Classification | [sileod/probability_words_nli](https://hf.co/datasets/sileod/probability_words_nli) |  |
| 50 | [probability_words_nli/reasoning_2hop](src/tasksource/tasks.py#L151) | Classification | [sileod/probability_words_nli](https://hf.co/datasets/sileod/probability_words_nli) |  |
| 51 | [probability_words_nli/usnli](src/tasksource/tasks.py#L151) | Classification | [sileod/probability_words_nli](https://hf.co/datasets/sileod/probability_words_nli) |  |
| 52 | [nan-nli](src/tasksource/tasks.py#L155) | Classification | [joey234/nan-nli](https://hf.co/datasets/joey234/nan-nli) |  |
| 53 | [nli_fever](src/tasksource/tasks.py#L157) | Classification | [pietrolesci/nli_fever](https://hf.co/datasets/pietrolesci/nli_fever) |  |
| 54 | [breaking_nli](src/tasksource/tasks.py#L160) | Classification | [pietrolesci/breaking_nli](https://hf.co/datasets/pietrolesci/breaking_nli) |  |
| 55 | [conj_nli](src/tasksource/tasks.py#L164) | Classification | [pietrolesci/conj_nli](https://hf.co/datasets/pietrolesci/conj_nli) |  |
| 56 | [fracas](src/tasksource/tasks.py#L168) | Classification | [pietrolesci/fracas](https://hf.co/datasets/pietrolesci/fracas) |  |
| 57 | [dialogue_nli](src/tasksource/tasks.py#L171) | Classification | [pietrolesci/dialogue_nli](https://hf.co/datasets/pietrolesci/dialogue_nli) |  |
| 58 | [mpe](src/tasksource/tasks.py#L174) | Classification | [pietrolesci/mpe](https://hf.co/datasets/pietrolesci/mpe) |  |
| 59 | [dnc](src/tasksource/tasks.py#L178) | Classification | [pietrolesci/dnc](https://hf.co/datasets/pietrolesci/dnc) |  |
| 60 | [recast_white/fnplus](src/tasksource/tasks.py#L182) | Classification | [pietrolesci/recast_white](https://hf.co/datasets/pietrolesci/recast_white) |  |
| 61 | [recast_white/sprl](src/tasksource/tasks.py#L185) | Classification | [pietrolesci/recast_white](https://hf.co/datasets/pietrolesci/recast_white) |  |
| 62 | [recast_white/dpr](src/tasksource/tasks.py#L188) | Classification | [pietrolesci/recast_white](https://hf.co/datasets/pietrolesci/recast_white) |  |
| 63 | [joci](src/tasksource/tasks.py#L192) | Classification | [pietrolesci/joci](https://hf.co/datasets/pietrolesci/joci) | ✓ |
| 64 | [robust_nli/IS_CS](src/tasksource/tasks.py#L199) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 65 | [robust_nli/LI_LI](src/tasksource/tasks.py#L201) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 66 | [robust_nli/ST_WO](src/tasksource/tasks.py#L203) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 67 | [robust_nli/PI_SP](src/tasksource/tasks.py#L205) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 68 | [robust_nli/PI_CD](src/tasksource/tasks.py#L207) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 69 | [robust_nli/ST_SE](src/tasksource/tasks.py#L209) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 70 | [robust_nli/ST_NE](src/tasksource/tasks.py#L211) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 71 | [robust_nli/ST_LM](src/tasksource/tasks.py#L213) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 72 | [robust_nli_is_sd](src/tasksource/tasks.py#L215) | Classification | [pietrolesci/robust_nli_is_sd](https://hf.co/datasets/pietrolesci/robust_nli_is_sd) |  |
| 73 | [robust_nli_li_ts](src/tasksource/tasks.py#L218) | Classification | [pietrolesci/robust_nli_li_ts](https://hf.co/datasets/pietrolesci/robust_nli_li_ts) |  |
| 74 | [gen_debiased_nli/snli_seq_z](src/tasksource/tasks.py#L222) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| 75 | [gen_debiased_nli/snli_z_aug](src/tasksource/tasks.py#L224) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| 76 | [gen_debiased_nli/snli_par_z](src/tasksource/tasks.py#L226) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| 77 | [gen_debiased_nli/mnli_par_z](src/tasksource/tasks.py#L228) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| 78 | [gen_debiased_nli/mnli_z_aug](src/tasksource/tasks.py#L230) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| 79 | [gen_debiased_nli/mnli_seq_z](src/tasksource/tasks.py#L232) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| 80 | [add_one_rte](src/tasksource/tasks.py#L235) | Classification | [pietrolesci/add_one_rte](https://hf.co/datasets/pietrolesci/add_one_rte) |  |
| 81 | [hlgd](src/tasksource/tasks.py#L239) | Classification | [tasksource/hlgd](https://hf.co/datasets/tasksource/hlgd) |  |
| 82 | [paws/labeled_final](src/tasksource/tasks.py#L241) | Classification | [google-research-datasets/paws](https://hf.co/datasets/google-research-datasets/paws) |  |
| 83 | [paws/labeled_swap](src/tasksource/tasks.py#L242) | Classification | [google-research-datasets/paws](https://hf.co/datasets/google-research-datasets/paws) |  |
| 84 | [medical_questions_pairs](src/tasksource/tasks.py#L244) | Classification | [curaihealth/medical_questions_pairs](https://hf.co/datasets/curaihealth/medical_questions_pairs) |  |
| 85 | [conll2003/pos_tags](src/tasksource/tasks.py#L249) | TokenClassification | [tomaarsen/conll2003](https://hf.co/datasets/tomaarsen/conll2003) |  |
| 86 | [conll2003/chunk_tags](src/tasksource/tasks.py#L250) | TokenClassification | [tomaarsen/conll2003](https://hf.co/datasets/tomaarsen/conll2003) |  |
| 87 | [conll2003/ner_tags](src/tasksource/tasks.py#L251) | TokenClassification | [tomaarsen/conll2003](https://hf.co/datasets/tomaarsen/conll2003) |  |
| 88 | [fig-qa](src/tasksource/tasks.py#L257) | MultipleChoice | [nightingal3/fig-qa](https://hf.co/datasets/nightingal3/fig-qa) |  |
| 89 | [cos_e/v1.0](src/tasksource/tasks.py#L266) | MultipleChoice | [Salesforce/cos_e](https://hf.co/datasets/Salesforce/cos_e) |  |
| 90 | [cosmos_qa](src/tasksource/tasks.py#L271) | MultipleChoice | [Samsoup/cosmos_qa](https://hf.co/datasets/Samsoup/cosmos_qa) |  |
| 91 | [dream](src/tasksource/tasks.py#L274) | MultipleChoice | [dataset-org/dream](https://hf.co/datasets/dataset-org/dream) |  |
| 92 | [openbookqa](src/tasksource/tasks.py#L281) | MultipleChoice | [allenai/openbookqa](https://hf.co/datasets/allenai/openbookqa) |  |
| 93 | [qasc](src/tasksource/tasks.py#L287) | MultipleChoice | [allenai/qasc](https://hf.co/datasets/allenai/qasc) |  |
| 94 | [quartz](src/tasksource/tasks.py#L295) | MultipleChoice | [allenai/quartz](https://hf.co/datasets/allenai/quartz) |  |
| 95 | [quail](src/tasksource/tasks.py#L300) | MultipleChoice | [textmachinelab/quail](https://hf.co/datasets/textmachinelab/quail) |  |
| 96 | [head_qa/en](src/tasksource/tasks.py#L306) | MultipleChoice | [EleutherAI/headqa](https://hf.co/datasets/EleutherAI/headqa) |  |
| 97 | [sciq](src/tasksource/tasks.py#L314) | MultipleChoice | [allenai/sciq](https://hf.co/datasets/allenai/sciq) |  |
| 98 | [social_i_qa](src/tasksource/tasks.py#L319) | MultipleChoice | [tasksource/social_i_qa](https://hf.co/datasets/tasksource/social_i_qa) |  |
| 99 | [wiki_hop/original](src/tasksource/tasks.py#L325) | MultipleChoice | [MoE-UNC/wikihop](https://hf.co/datasets/MoE-UNC/wikihop) |  |
| 100 | [wiqa](src/tasksource/tasks.py#L332) | MultipleChoice | [tasksource/wiqa](https://hf.co/datasets/tasksource/wiqa) |  |
| 101 | [piqa](src/tasksource/tasks.py#L337) | MultipleChoice | [baber/piqa](https://hf.co/datasets/baber/piqa) |  |
| 102 | [hellaswag](src/tasksource/tasks.py#L345) | MultipleChoice | [Rowan/hellaswag](https://hf.co/datasets/Rowan/hellaswag) |  |
| 103 | [super_glue/copa](src/tasksource/tasks.py#L354) | MultipleChoice | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 104 | [balanced-copa](src/tasksource/tasks.py#L356) | MultipleChoice | [pkavumba/balanced-copa](https://hf.co/datasets/pkavumba/balanced-copa) |  |
| 105 | [e-CARE](src/tasksource/tasks.py#L359) | MultipleChoice | [12ml/e-CARE](https://hf.co/datasets/12ml/e-CARE) |  |
| 106 | [art](src/tasksource/tasks.py#L362) | MultipleChoice | [allenai/art](https://hf.co/datasets/allenai/art) | ✓ |
| 107 | [winogrande/winogrande_xl](src/tasksource/tasks.py#L370) | MultipleChoice | [allenai/winogrande](https://hf.co/datasets/allenai/winogrande) |  |
| 108 | [codah/codah](src/tasksource/tasks.py#L373) | MultipleChoice | [jaredfern/codah](https://hf.co/datasets/jaredfern/codah) |  |
| 109 | [ai2_arc/ARC-Easy/challenge](src/tasksource/tasks.py#L375) | MultipleChoice | [allenai/ai2_arc](https://hf.co/datasets/allenai/ai2_arc) |  |
| 110 | [ai2_arc/ARC-Challenge/challenge](src/tasksource/tasks.py#L375) | MultipleChoice | [allenai/ai2_arc](https://hf.co/datasets/allenai/ai2_arc) |  |
| 111 | [definite_pronoun_resolution](src/tasksource/tasks.py#L380) | MultipleChoice | [community-datasets/definite_pronoun_resolution](https://hf.co/datasets/community-datasets/definite_pronoun_resolution) |  |
| 112 | [swag/regular](src/tasksource/tasks.py#L386) | MultipleChoice | [allenai/swag](https://hf.co/datasets/allenai/swag) |  |
| 113 | [math_qa](src/tasksource/tasks.py#L392) | MultipleChoice | [tasksource/math_qa](https://hf.co/datasets/tasksource/math_qa) |  |
| 114 | [glue/cola](src/tasksource/tasks.py#L401) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 115 | [glue/sst2](src/tasksource/tasks.py#L402) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 116 | [utilitarianism](src/tasksource/tasks.py#L416) | Classification | csv |  |
| 117 | [amazon_counterfactual/en](src/tasksource/tasks.py#L424) | Classification | [mteb/amazon_counterfactual](https://hf.co/datasets/mteb/amazon_counterfactual) |  |
| 118 | [insincere-questions](src/tasksource/tasks.py#L429) | Classification | [SetFit/insincere-questions](https://hf.co/datasets/SetFit/insincere-questions) |  |
| 119 | [toxic_conversations](src/tasksource/tasks.py#L433) | Classification | [SetFit/toxic_conversations](https://hf.co/datasets/SetFit/toxic_conversations) |  |
| 120 | [TuringBench](src/tasksource/tasks.py#L437) | Classification | csv | ✓ |
| 121 | [trec](src/tasksource/tasks.py#L447) | Classification | [tasksource/trec](https://hf.co/datasets/tasksource/trec) |  |
| 122 | [vitaminc](src/tasksource/tasks.py#L450) | Classification | [tals/vitaminc](https://hf.co/datasets/tals/vitaminc) |  |
| 123 | [hope_edi/english](src/tasksource/tasks.py#L452) | Classification | csv |  |
| 124 | [rumoureval_2019/RumourEval2019](src/tasksource/tasks.py#L466) | Classification | csv |  |
| 125 | [ethos/binary](src/tasksource/tasks.py#L480) | Classification | [SetFit/ethos_binary](https://hf.co/datasets/SetFit/ethos_binary) |  |
| 126 | [ethos/multilabel](src/tasksource/tasks.py#L504) | Classification | [tasksource/ethos](https://hf.co/datasets/tasksource/ethos) |  |
| 127 | [tweet_eval/offensive](src/tasksource/tasks.py#L507) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 128 | [tweet_eval/irony](src/tasksource/tasks.py#L507) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 129 | [tweet_eval/hate](src/tasksource/tasks.py#L507) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 130 | [tweet_eval/sentiment](src/tasksource/tasks.py#L507) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 131 | [tweet_eval/emotion](src/tasksource/tasks.py#L507) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 132 | [tweet_eval/emoji](src/tasksource/tasks.py#L507) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 133 | [tweet_eval/stance_abortion](src/tasksource/tasks.py#L522) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 134 | [tweet_eval/stance_atheism](src/tasksource/tasks.py#L523) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 135 | [tweet_eval/stance_climate](src/tasksource/tasks.py#L524) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 136 | [tweet_eval/stance_feminist](src/tasksource/tasks.py#L525) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 137 | [tweet_eval/stance_hillary](src/tasksource/tasks.py#L526) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 138 | [discovery/discovery](src/tasksource/tasks.py#L529) | Classification | [sileod/discovery](https://hf.co/datasets/sileod/discovery) |  |
| 139 | [pragmeval/verifiability](src/tasksource/tasks.py#L531) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 140 | [pragmeval/mrda](src/tasksource/tasks.py#L531) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 141 | [pragmeval/switchboard](src/tasksource/tasks.py#L531) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 142 | [pragmeval/persuasiveness-claimtype](src/tasksource/tasks.py#L535) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 143 | [pragmeval/pdtb](src/tasksource/tasks.py#L535) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 144 | [pragmeval/gum](src/tasksource/tasks.py#L535) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 145 | [pragmeval/persuasiveness-premisetype](src/tasksource/tasks.py#L535) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 146 | [pragmeval/emergent](src/tasksource/tasks.py#L535) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 147 | [pragmeval/sarcasm](src/tasksource/tasks.py#L535) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 148 | [pragmeval/stac](src/tasksource/tasks.py#L535) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 149 | [pragmeval/emobank-arousal](src/tasksource/tasks.py#L544) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 150 | [pragmeval/emobank-dominance](src/tasksource/tasks.py#L545) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 151 | [pragmeval/emobank-valence](src/tasksource/tasks.py#L546) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 152 | [pragmeval/squinky-formality](src/tasksource/tasks.py#L547) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 153 | [pragmeval/squinky-implicature](src/tasksource/tasks.py#L548) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 154 | [pragmeval/squinky-informativeness](src/tasksource/tasks.py#L549) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 155 | [pragmeval/persuasiveness-eloquence](src/tasksource/tasks.py#L550) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 156 | [pragmeval/persuasiveness-relevance](src/tasksource/tasks.py#L551) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 157 | [pragmeval/persuasiveness-specificity](src/tasksource/tasks.py#L552) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 158 | [pragmeval/persuasiveness-strength](src/tasksource/tasks.py#L553) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 159 | [silicone/dyda_da](src/tasksource/tasks.py#L555) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| 160 | [silicone/meld_s](src/tasksource/tasks.py#L555) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| 161 | [silicone/dyda_e](src/tasksource/tasks.py#L555) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| 162 | [silicone/sem](src/tasksource/tasks.py#L555) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| 163 | [silicone/maptask](src/tasksource/tasks.py#L555) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| 164 | [silicone/meld_e](src/tasksource/tasks.py#L555) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| 165 | [silicone/oasis](src/tasksource/tasks.py#L555) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| 166 | [silicone/iemocap](src/tasksource/tasks.py#L562) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| 167 | [lex_glue/eurlex](src/tasksource/tasks.py#L567) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) |  |
| 168 | [lex_glue/scotus](src/tasksource/tasks.py#L569) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) |  |
| 169 | [lex_glue/ledgar](src/tasksource/tasks.py#L572) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) |  |
| 170 | [lex_glue/unfair_tos](src/tasksource/tasks.py#L574) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) | ✓ |
| 171 | [lex_glue/case_hold](src/tasksource/tasks.py#L577) | MultipleChoice | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) |  |
| 172 | [language-identification](src/tasksource/tasks.py#L585) | Classification | [papluca/language-identification](https://hf.co/datasets/papluca/language-identification) | ✓ |
| 173 | [imdb](src/tasksource/tasks.py#L590) | Classification | [stanfordnlp/imdb](https://hf.co/datasets/stanfordnlp/imdb) |  |
| 174 | [rotten_tomatoes](src/tasksource/tasks.py#L592) | Classification | [cornell-movie-review-data/rotten_tomatoes](https://hf.co/datasets/cornell-movie-review-data/rotten_tomatoes) |  |
| 175 | [ag_news](src/tasksource/tasks.py#L594) | Classification | [fancyzhx/ag_news](https://hf.co/datasets/fancyzhx/ag_news) |  |
| 176 | [yelp_review_full/yelp_review_full](src/tasksource/tasks.py#L596) | Classification | [Yelp/yelp_review_full](https://hf.co/datasets/Yelp/yelp_review_full) | ✓ |
| 177 | [financial_phrasebank/sentences_allagree](src/tasksource/tasks.py#L601) | Classification | [ghbacct/financial-phrasebank-all-agree-classification](https://hf.co/datasets/ghbacct/financial-phrasebank-all-agree-classification) |  |
| 178 | [poem_sentiment](src/tasksource/tasks.py#L606) | Classification | [google-research-datasets/poem_sentiment](https://hf.co/datasets/google-research-datasets/poem_sentiment) |  |
| 179 | [emotion](src/tasksource/tasks.py#L608) | Classification | [dair-ai/emotion](https://hf.co/datasets/dair-ai/emotion) |  |
| 180 | [dbpedia_14/dbpedia_14](src/tasksource/tasks.py#L610) | Classification | [fancyzhx/dbpedia_14](https://hf.co/datasets/fancyzhx/dbpedia_14) |  |
| 181 | [amazon_polarity/amazon_polarity](src/tasksource/tasks.py#L612) | Classification | [fancyzhx/amazon_polarity](https://hf.co/datasets/fancyzhx/amazon_polarity) |  |
| 182 | [app_reviews](src/tasksource/tasks.py#L614) | Classification | [sealuzh/app_reviews](https://hf.co/datasets/sealuzh/app_reviews) | ✓ |
| 183 | [hate_speech18](src/tasksource/tasks.py#L619) | Classification | [tasksource/hate_speech18](https://hf.co/datasets/tasksource/hate_speech18) |  |
| 184 | [sms_spam](src/tasksource/tasks.py#L625) | Classification | [ucirvine/sms_spam](https://hf.co/datasets/ucirvine/sms_spam) |  |
| 185 | [humicroedit/subtask-1](src/tasksource/tasks.py#L628) | Classification | [tasksource/humicroedit](https://hf.co/datasets/tasksource/humicroedit) | ✓ |
| 186 | [humicroedit/subtask-2](src/tasksource/tasks.py#L634) | Classification | [tasksource/humicroedit](https://hf.co/datasets/tasksource/humicroedit) | ✓ |
| 187 | [snips_built_in_intents](src/tasksource/tasks.py#L639) | Classification | [sonos-nlu-benchmark/snips_built_in_intents](https://hf.co/datasets/sonos-nlu-benchmark/snips_built_in_intents) |  |
| 188 | [hate_speech_offensive](src/tasksource/tasks.py#L643) | Classification | [tdavidson/hate_speech_offensive](https://hf.co/datasets/tdavidson/hate_speech_offensive) |  |
| 189 | [yahoo_answers_topics](src/tasksource/tasks.py#L645) | Classification | [community-datasets/yahoo_answers_topics](https://hf.co/datasets/community-datasets/yahoo_answers_topics) |  |
| 190 | [stackoverflow-questions](src/tasksource/tasks.py#L649) | Classification | [pacovaldez/stackoverflow-questions](https://hf.co/datasets/pacovaldez/stackoverflow-questions) | ✓ |
| 191 | [hyperpartisan_news](src/tasksource/tasks.py#L655) | Classification | [zapsdcn/hyperpartisan_news](https://hf.co/datasets/zapsdcn/hyperpartisan_news) |  |
| 192 | [sciie](src/tasksource/tasks.py#L660) | Classification | [zapsdcn/sciie](https://hf.co/datasets/zapsdcn/sciie) |  |
| 193 | [citation_intent](src/tasksource/tasks.py#L661) | Classification | [zapsdcn/citation_intent](https://hf.co/datasets/zapsdcn/citation_intent) |  |
| 194 | [go_emotions/simplified](src/tasksource/tasks.py#L663) | Classification | [google-research-datasets/go_emotions](https://hf.co/datasets/google-research-datasets/go_emotions) |  |
| 195 | [scicite](src/tasksource/tasks.py#L667) | Classification | [tasksource/scicite](https://hf.co/datasets/tasksource/scicite) |  |
| 196 | [liar](src/tasksource/tasks.py#L669) | Classification | [tasksource/liar](https://hf.co/datasets/tasksource/liar) | ✓ |
| 197 | [lexical_relation_classification/BLESS](src/tasksource/tasks.py#L680) | Classification | json | ✓ |
| 198 | [lexical_relation_classification/EVALution](src/tasksource/tasks.py#L680) | Classification | json | ✓ |
| 199 | [lexical_relation_classification/K&H+N](src/tasksource/tasks.py#L680) | Classification | json | ✓ |
| 200 | [lexical_relation_classification/ROOT09](src/tasksource/tasks.py#L680) | Classification | json | ✓ |
| 201 | [lexical_relation_classification/CogALexV](src/tasksource/tasks.py#L706) | Classification | json | ✓ |
| 202 | [linguisticprobing/subj_number](src/tasksource/tasks.py#L724) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 203 | [linguisticprobing/obj_number](src/tasksource/tasks.py#L725) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 204 | [linguisticprobing/past_present](src/tasksource/tasks.py#L726) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 205 | [linguisticprobing/sentence_length](src/tasksource/tasks.py#L727) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 206 | [linguisticprobing/top_constituents](src/tasksource/tasks.py#L728) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 207 | [linguisticprobing/tree_depth](src/tasksource/tasks.py#L730) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 208 | [linguisticprobing/coordination_inversion](src/tasksource/tasks.py#L731) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 209 | [linguisticprobing/odd_man_out](src/tasksource/tasks.py#L733) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 210 | [linguisticprobing/bigram_shift](src/tasksource/tasks.py#L734) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 211 | [crowdflower/airline-sentiment](src/tasksource/tasks.py#L736) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 212 | [crowdflower/political-media-bias](src/tasksource/tasks.py#L736) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 213 | [crowdflower/political-media-message](src/tasksource/tasks.py#L736) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 214 | [crowdflower/text_emotion](src/tasksource/tasks.py#L736) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 215 | [crowdflower/political-media-audience](src/tasksource/tasks.py#L736) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 216 | [crowdflower/sentiment_nuclear_power](src/tasksource/tasks.py#L736) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 217 | [crowdflower/corporate-messaging](src/tasksource/tasks.py#L736) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 218 | [crowdflower/economic-news](src/tasksource/tasks.py#L736) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 219 | [crowdflower/tweet_global_warming](src/tasksource/tasks.py#L736) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 220 | [ethics/commonsense](src/tasksource/tasks.py#L772) | Classification | csv |  |
| 221 | [ethics/deontology](src/tasksource/tasks.py#L780) | Classification | csv |  |
| 222 | [ethics/justice](src/tasksource/tasks.py#L788) | Classification | csv |  |
| 223 | [ethics/virtue](src/tasksource/tasks.py#L796) | Classification | [hendrycks/ethics](https://hf.co/datasets/hendrycks/ethics) |  |
| 224 | [emo/emo2019](src/tasksource/tasks.py#L805) | Classification | [oneonlee/cleansed_emocontext](https://hf.co/datasets/oneonlee/cleansed_emocontext) |  |
| 225 | [google_wellformed_query](src/tasksource/tasks.py#L811) | Classification | [tasksource/google_wellformed_query](https://hf.co/datasets/tasksource/google_wellformed_query) | ✓ |
| 226 | [tweets_hate_speech_detection](src/tasksource/tasks.py#L817) | Classification | [tweets-hate-speech-detection/tweets_hate_speech_detection](https://hf.co/datasets/tweets-hate-speech-detection/tweets_hate_speech_detection) |  |
| 227 | [wnut_17/wnut_17](src/tasksource/tasks.py#L821) | TokenClassification | [flaitenberger/wnut_17](https://hf.co/datasets/flaitenberger/wnut_17) |  |
| 228 | [ncbi_disease/ncbi_disease](src/tasksource/tasks.py#L824) | TokenClassification | [ncbi/ncbi_disease](https://hf.co/datasets/ncbi/ncbi_disease) |  |
| 229 | [acronym_identification](src/tasksource/tasks.py#L827) | TokenClassification | [amirveyseh/acronym_identification](https://hf.co/datasets/amirveyseh/acronym_identification) |  |
| 230 | [jnlpba/jnlpba](src/tasksource/tasks.py#L830) | TokenClassification | [jnlpba/jnlpba](https://hf.co/datasets/jnlpba/jnlpba) |  |
| 231 | [ontonotes_english/SpeedOfMagic--ontonotes_english](src/tasksource/tasks.py#L837) | TokenClassification | [SpeedOfMagic/ontonotes_english](https://hf.co/datasets/SpeedOfMagic/ontonotes_english) |  |
| 232 | [blog_authorship_corpus/gender](src/tasksource/tasks.py#L841) | Classification | [tasksource/blog_authorship_corpus](https://hf.co/datasets/tasksource/blog_authorship_corpus) | ✓ |
| 233 | [blog_authorship_corpus/age](src/tasksource/tasks.py#L843) | Classification | [tasksource/blog_authorship_corpus](https://hf.co/datasets/tasksource/blog_authorship_corpus) | ✓ |
| 234 | [blog_authorship_corpus/job](src/tasksource/tasks.py#L846) | Classification | [tasksource/blog_authorship_corpus](https://hf.co/datasets/tasksource/blog_authorship_corpus) | ✓ |
| 235 | [open_question_type](src/tasksource/tasks.py#L857) | Classification | [Korea-MES/open_question_type](https://hf.co/datasets/Korea-MES/open_question_type) |  |
| 236 | [health_fact](src/tasksource/tasks.py#L859) | Classification | [marcov/health_fact_promptsource](https://hf.co/datasets/marcov/health_fact_promptsource) |  |
| 237 | [commonsense_qa](src/tasksource/tasks.py#L863) | MultipleChoice | [tau/commonsense_qa](https://hf.co/datasets/tau/commonsense_qa) |  |
| 238 | [mc_taco](src/tasksource/tasks.py#L869) | Classification | [marcov/mc_taco_promptsource](https://hf.co/datasets/marcov/mc_taco_promptsource) | ✓ |
| 239 | [ade_corpus_v2/Ade_corpus_v2_classification](src/tasksource/tasks.py#L876) | Classification | [ade-benchmark-corpus/ade_corpus_v2](https://hf.co/datasets/ade-benchmark-corpus/ade_corpus_v2) |  |
| 240 | [discosense](src/tasksource/tasks.py#L878) | MultipleChoice | json |  |
| 241 | [circa](src/tasksource/tasks.py#L885) | Classification | [google-research-datasets/circa](https://hf.co/datasets/google-research-datasets/circa) |  |
| 242 | [code_x_glue_cc_defect_detection](src/tasksource/tasks.py#L890) | Classification | [google/code_x_glue_cc_defect_detection](https://hf.co/datasets/google/code_x_glue_cc_defect_detection) |  |
| 243 | [phrase_similarity](src/tasksource/tasks.py#L894) | Classification | [Deehan1866/processed_phrase_similarity](https://hf.co/datasets/Deehan1866/processed_phrase_similarity) |  |
| 244 | [scientific-exaggeration-detection](src/tasksource/tasks.py#L902) | Classification | [copenlu/scientific-exaggeration-detection](https://hf.co/datasets/copenlu/scientific-exaggeration-detection) |  |
| 245 | [quarel](src/tasksource/tasks.py#L908) | Classification | [community-datasets/quarel](https://hf.co/datasets/community-datasets/quarel) |  |
| 246 | [fever-evidence-related](src/tasksource/tasks.py#L913) | Classification | [mwong/fever-evidence-related](https://hf.co/datasets/mwong/fever-evidence-related) |  |
| 247 | [numer_sense](src/tasksource/tasks.py#L916) | Classification | [tasksource/numer_sense](https://hf.co/datasets/tasksource/numer_sense) |  |
| 248 | [dynasent/dynabench.dynasent.r1.all/r1](src/tasksource/tasks.py#L923) | Classification | [tasksource/dynasent](https://hf.co/datasets/tasksource/dynasent) |  |
| 249 | [dynasent/dynabench.dynasent.r2.all/r2](src/tasksource/tasks.py#L927) | Classification | [tasksource/dynasent](https://hf.co/datasets/tasksource/dynasent) |  |
| 250 | [Sarcasm_News_Headline](src/tasksource/tasks.py#L932) | Classification | [raquiba/Sarcasm_News_Headline](https://hf.co/datasets/raquiba/Sarcasm_News_Headline) |  |
| 251 | [sem_eval_2010_task_8](src/tasksource/tasks.py#L935) | Classification | [SemEvalWorkshop/sem_eval_2010_task_8](https://hf.co/datasets/SemEvalWorkshop/sem_eval_2010_task_8) |  |
| 252 | [auditor_review](src/tasksource/tasks.py#L937) | Classification | [demo-org/auditor_review](https://hf.co/datasets/demo-org/auditor_review) |  |
| 253 | [medmcqa](src/tasksource/tasks.py#L941) | MultipleChoice | [openlifescienceai/medmcqa](https://hf.co/datasets/openlifescienceai/medmcqa) |  |
| 254 | [Dynasent_Disagreement](src/tasksource/tasks.py#L958) | Classification | [RuyuanWan/Dynasent_Disagreement](https://hf.co/datasets/RuyuanWan/Dynasent_Disagreement) | ✓ |
| 255 | [Politeness_Disagreement](src/tasksource/tasks.py#L960) | Classification | [RuyuanWan/Politeness_Disagreement](https://hf.co/datasets/RuyuanWan/Politeness_Disagreement) | ✓ |
| 256 | [SBIC_Disagreement](src/tasksource/tasks.py#L962) | Classification | [RuyuanWan/SBIC_Disagreement](https://hf.co/datasets/RuyuanWan/SBIC_Disagreement) | ✓ |
| 257 | [SChem_Disagreement](src/tasksource/tasks.py#L964) | Classification | [RuyuanWan/SChem_Disagreement](https://hf.co/datasets/RuyuanWan/SChem_Disagreement) | ✓ |
| 258 | [Dilemmas_Disagreement](src/tasksource/tasks.py#L966) | Classification | [RuyuanWan/Dilemmas_Disagreement](https://hf.co/datasets/RuyuanWan/Dilemmas_Disagreement) | ✓ |
| 259 | [logiqa](src/tasksource/tasks.py#L969) | MultipleChoice | [fireworks-ai/logiqa](https://hf.co/datasets/fireworks-ai/logiqa) |  |
| 260 | [wiki_qa](src/tasksource/tasks.py#L978) | Classification | [microsoft/wiki_qa](https://hf.co/datasets/microsoft/wiki_qa) | ✓ |
| 261 | [cycic_classification](src/tasksource/tasks.py#L980) | Classification | [tasksource/cycic_classification](https://hf.co/datasets/tasksource/cycic_classification) |  |
| 262 | [cycic_multiplechoice](src/tasksource/tasks.py#L982) | MultipleChoice | [tasksource/cycic_multiplechoice](https://hf.co/datasets/tasksource/cycic_multiplechoice) |  |
| 263 | [sts-companion](src/tasksource/tasks.py#L986) | Classification | [tasksource/sts-companion](https://hf.co/datasets/tasksource/sts-companion) |  |
| 264 | [commonsense_qa_2.0](src/tasksource/tasks.py#L989) | Classification | [tasksource/commonsense_qa_2.0](https://hf.co/datasets/tasksource/commonsense_qa_2.0) |  |
| 265 | [lingnli](src/tasksource/tasks.py#L992) | Classification | [tasksource/lingnli](https://hf.co/datasets/tasksource/lingnli) |  |
| 266 | [monotonicity-entailment](src/tasksource/tasks.py#L994) | Classification | [tasksource/monotonicity-entailment](https://hf.co/datasets/tasksource/monotonicity-entailment) |  |
| 267 | [arct](src/tasksource/tasks.py#L997) | MultipleChoice | [tasksource/arct](https://hf.co/datasets/tasksource/arct) |  |
| 268 | [scinli](src/tasksource/tasks.py#L1000) | Classification | [tasksource/scinli](https://hf.co/datasets/tasksource/scinli) |  |
| 269 | [naturallogic](src/tasksource/tasks.py#L1004) | Classification | [tasksource/naturallogic](https://hf.co/datasets/tasksource/naturallogic) |  |
| 270 | [onestop_qa](src/tasksource/tasks.py#L1006) | MultipleChoice | [malmaud/onestop_qa](https://hf.co/datasets/malmaud/onestop_qa) |  |
| 271 | [moral_stories/full](src/tasksource/tasks.py#L1009) | MultipleChoice | [LabHC/moral_stories](https://hf.co/datasets/LabHC/moral_stories) |  |
| 272 | [prost](src/tasksource/tasks.py#L1017) | MultipleChoice | json |  |
| 273 | [dynahate](src/tasksource/tasks.py#L1022) | Classification | [tasksource/dynahate](https://hf.co/datasets/tasksource/dynahate) |  |
| 274 | [syntactic-augmentation-nli](src/tasksource/tasks.py#L1024) | Classification | [tasksource/syntactic-augmentation-nli](https://hf.co/datasets/tasksource/syntactic-augmentation-nli) |  |
| 275 | [autotnli](src/tasksource/tasks.py#L1026) | Classification | [tasksource/autotnli](https://hf.co/datasets/tasksource/autotnli) |  |
| 276 | [CONDAQA](src/tasksource/tasks.py#L1028) | Classification | [lasha-nlp/CONDAQA](https://hf.co/datasets/lasha-nlp/CONDAQA) |  |
| 277 | [webgpt_comparisons](src/tasksource/tasks.py#L1038) | MultipleChoice | [heegyu/webgpt_comparisons_ko](https://hf.co/datasets/heegyu/webgpt_comparisons_ko) | ✓ |
| 278 | [synthetic-instruct-gptj-pairwise](src/tasksource/tasks.py#L1046) | MultipleChoice | [Dahoas/synthetic-instruct-gptj-pairwise](https://hf.co/datasets/Dahoas/synthetic-instruct-gptj-pairwise) | ✓ |
| 279 | [scruples](src/tasksource/tasks.py#L1049) | Classification | [tasksource/scruples](https://hf.co/datasets/tasksource/scruples) | ✓ |
| 280 | [wouldyourather](src/tasksource/tasks.py#L1052) | MultipleChoice | [tasksource/wouldyourather](https://hf.co/datasets/tasksource/wouldyourather) | ✓ |
| 281 | [defeasible-nli/snli](src/tasksource/tasks.py#L1060) | Classification | [tasksource/defeasible-nli](https://hf.co/datasets/tasksource/defeasible-nli) |  |
| 282 | [defeasible-nli/atomic](src/tasksource/tasks.py#L1060) | Classification | [tasksource/defeasible-nli](https://hf.co/datasets/tasksource/defeasible-nli) |  |
| 283 | [defeasible-nli/social](src/tasksource/tasks.py#L1063) | Classification | [tasksource/defeasible-nli](https://hf.co/datasets/tasksource/defeasible-nli) |  |
| 284 | [help-nli](src/tasksource/tasks.py#L1066) | Classification | [tasksource/help-nli](https://hf.co/datasets/tasksource/help-nli) |  |
| 285 | [nli-veridicality-transitivity](src/tasksource/tasks.py#L1069) | Classification | [tasksource/nli-veridicality-transitivity](https://hf.co/datasets/tasksource/nli-veridicality-transitivity) |  |
| 286 | [lonli](src/tasksource/tasks.py#L1072) | Classification | [tasksource/lonli](https://hf.co/datasets/tasksource/lonli) |  |
| 287 | [dadc-limit-nli](src/tasksource/tasks.py#L1075) | Classification | [tasksource/dadc-limit-nli](https://hf.co/datasets/tasksource/dadc-limit-nli) |  |
| 288 | [FLUTE](src/tasksource/tasks.py#L1078) | Classification | [ColumbiaNLP/FLUTE](https://hf.co/datasets/ColumbiaNLP/FLUTE) |  |
| 289 | [strategy-qa](src/tasksource/tasks.py#L1081) | Classification | [tasksource/strategy-qa](https://hf.co/datasets/tasksource/strategy-qa) |  |
| 290 | [summarize_from_feedback/comparisons](src/tasksource/tasks.py#L1084) | MultipleChoice | [vwxyzjn/summarize_from_feedback_oai_preprocessing](https://hf.co/datasets/vwxyzjn/summarize_from_feedback_oai_preprocessing) | ✓ |
| 291 | [folio](src/tasksource/tasks.py#L1092) | Classification | [tasksource/folio](https://hf.co/datasets/tasksource/folio) |  |
| 292 | [tomi-nli](src/tasksource/tasks.py#L1096) | Classification | [tasksource/tomi-nli](https://hf.co/datasets/tasksource/tomi-nli) |  |
| 293 | [avicenna](src/tasksource/tasks.py#L1099) | Classification | [tasksource/avicenna](https://hf.co/datasets/tasksource/avicenna) | ✓ |
| 294 | [SHP](src/tasksource/tasks.py#L1102) | MultipleChoice | [stanfordnlp/SHP](https://hf.co/datasets/stanfordnlp/SHP) | ✓ |
| 295 | [MedQA-USMLE-4-options-hf](src/tasksource/tasks.py#L1110) | MultipleChoice | [GBaker/MedQA-USMLE-4-options-hf](https://hf.co/datasets/GBaker/MedQA-USMLE-4-options-hf) |  |
| 296 | [wikimedqa/medwiki](src/tasksource/tasks.py#L1113) | MultipleChoice | [sileod/wikimedqa](https://hf.co/datasets/sileod/wikimedqa) |  |
| 297 | [cicero](src/tasksource/tasks.py#L1122) | MultipleChoice | [declare-lab/cicero](https://hf.co/datasets/declare-lab/cicero) |  |
| 298 | [CREAK](src/tasksource/tasks.py#L1126) | Classification | [amydeng2000/CREAK](https://hf.co/datasets/amydeng2000/CREAK) |  |
| 299 | [mutual](src/tasksource/tasks.py#L1129) | MultipleChoice | [tasksource/mutual](https://hf.co/datasets/tasksource/mutual) |  |
| 300 | [puzzte](src/tasksource/tasks.py#L1133) | Classification | [tasksource/puzzte](https://hf.co/datasets/tasksource/puzzte) |  |
| 301 | [implicatures](src/tasksource/tasks.py#L1138) | MultipleChoice | [tasksource/implicatures](https://hf.co/datasets/tasksource/implicatures) |  |
| 302 | [race/middle](src/tasksource/tasks.py#L1143) | MultipleChoice | [ehovy/race](https://hf.co/datasets/ehovy/race) |  |
| 303 | [race/high](src/tasksource/tasks.py#L1143) | MultipleChoice | [ehovy/race](https://hf.co/datasets/ehovy/race) |  |
| 304 | [race-c](src/tasksource/tasks.py#L1147) | MultipleChoice | [tasksource/race-c](https://hf.co/datasets/tasksource/race-c) |  |
| 305 | [spartqa-yn](src/tasksource/tasks.py#L1150) | Classification | [tasksource/spartqa-yn](https://hf.co/datasets/tasksource/spartqa-yn) |  |
| 306 | [spartqa-mchoice](src/tasksource/tasks.py#L1153) | MultipleChoice | [tasksource/spartqa-mchoice](https://hf.co/datasets/tasksource/spartqa-mchoice) |  |
| 307 | [temporal-nli](src/tasksource/tasks.py#L1156) | Classification | [tasksource/temporal-nli](https://hf.co/datasets/tasksource/temporal-nli) |  |
| 308 | [riddle_sense](src/tasksource/tasks.py#L1159) | MultipleChoice | [jeggers/riddle_sense](https://hf.co/datasets/jeggers/riddle_sense) |  |
| 309 | [clcd-english](src/tasksource/tasks.py#L1164) | Classification | [tasksource/clcd-english](https://hf.co/datasets/tasksource/clcd-english) |  |
| 310 | [twentyquestions](src/tasksource/tasks.py#L1176) | Classification | [tasksource/twentyquestions](https://hf.co/datasets/tasksource/twentyquestions) |  |
| 311 | [reclor](src/tasksource/tasks.py#L1181) | MultipleChoice | [tasksource/reclor](https://hf.co/datasets/tasksource/reclor) |  |
| 312 | [counterfactually-augmented-imdb](src/tasksource/tasks.py#L1184) | Classification | [tasksource/counterfactually-augmented-imdb](https://hf.co/datasets/tasksource/counterfactually-augmented-imdb) |  |
| 313 | [counterfactually-augmented-snli](src/tasksource/tasks.py#L1187) | Classification | [tasksource/counterfactually-augmented-snli](https://hf.co/datasets/tasksource/counterfactually-augmented-snli) |  |
| 314 | [cnli](src/tasksource/tasks.py#L1190) | Classification | [tasksource/cnli](https://hf.co/datasets/tasksource/cnli) |  |
| 315 | [boolq-natural-perturbations](src/tasksource/tasks.py#L1193) | Classification | [tasksource/boolq-natural-perturbations](https://hf.co/datasets/tasksource/boolq-natural-perturbations) |  |
| 316 | [acceptability-prediction](src/tasksource/tasks.py#L1198) | Classification | [tasksource/acceptability-prediction](https://hf.co/datasets/tasksource/acceptability-prediction) | ✓ |
| 317 | [equate](src/tasksource/tasks.py#L1224) | Classification | [tasksource/equate](https://hf.co/datasets/tasksource/equate) |  |
| 318 | [ScienceQA_text_only](src/tasksource/tasks.py#L1227) | MultipleChoice | [tasksource/ScienceQA_text_only](https://hf.co/datasets/tasksource/ScienceQA_text_only) |  |
| 319 | [ekar_english](src/tasksource/tasks.py#L1230) | MultipleChoice | [Jiangjie/ekar_english](https://hf.co/datasets/Jiangjie/ekar_english) | ✓ |
| 320 | [implicit-hate-stg1](src/tasksource/tasks.py#L1234) | Classification | [tasksource/implicit-hate-stg1](https://hf.co/datasets/tasksource/implicit-hate-stg1) |  |
| 321 | [chaos-mnli-ambiguity](src/tasksource/tasks.py#L1237) | Classification | [tasksource/chaos-mnli-ambiguity](https://hf.co/datasets/tasksource/chaos-mnli-ambiguity) | ✓ |
| 322 | [headline_cause/en_simple](src/tasksource/tasks.py#L1241) | Classification | json |  |
| 323 | [logiqa-2.0-nli](src/tasksource/tasks.py#L1246) | Classification | [tasksource/logiqa-2.0-nli](https://hf.co/datasets/tasksource/logiqa-2.0-nli) |  |
| 324 | [oasst2_dense_flat/quality](src/tasksource/tasks.py#L1251) | Classification | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) | ✓ |
| 325 | [oasst2_dense_flat/toxicity](src/tasksource/tasks.py#L1253) | Classification | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) | ✓ |
| 326 | [oasst2_dense_flat/helpfulness](src/tasksource/tasks.py#L1255) | Classification | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) | ✓ |
| 327 | [mindgames](src/tasksource/tasks.py#L1258) | Classification | [sileod/mindgames](https://hf.co/datasets/sileod/mindgames) |  |
| 328 | [universal_dependencies/en_partut/deprel](src/tasksource/tasks.py#L1272) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| 329 | [universal_dependencies/en_lines/deprel](src/tasksource/tasks.py#L1272) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| 330 | [universal_dependencies/en_gum/deprel](src/tasksource/tasks.py#L1272) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| 331 | [universal_dependencies/en_ewt/deprel](src/tasksource/tasks.py#L1272) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| 332 | [ambient](src/tasksource/tasks.py#L1278) | Classification | [tasksource/ambient](https://hf.co/datasets/tasksource/ambient) | ✓ |
| 333 | [path-naturalness-prediction](src/tasksource/tasks.py#L1281) | MultipleChoice | [tasksource/path-naturalness-prediction](https://hf.co/datasets/tasksource/path-naturalness-prediction) | ✓ |
| 334 | [civil_comments/toxicity](src/tasksource/tasks.py#L1291) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 335 | [civil_comments/severe_toxicity](src/tasksource/tasks.py#L1292) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 336 | [civil_comments/obscene](src/tasksource/tasks.py#L1293) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 337 | [civil_comments/threat](src/tasksource/tasks.py#L1294) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 338 | [civil_comments/insult](src/tasksource/tasks.py#L1295) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 339 | [civil_comments/identity_attack](src/tasksource/tasks.py#L1296) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 340 | [civil_comments/sexual_explicit](src/tasksource/tasks.py#L1297) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 341 | [cloth](src/tasksource/tasks.py#L1299) | MultipleChoice | [AndyChiang/cloth](https://hf.co/datasets/AndyChiang/cloth) |  |
| 342 | [dgen](src/tasksource/tasks.py#L1300) | MultipleChoice | [AndyChiang/dgen](https://hf.co/datasets/AndyChiang/dgen) |  |
| 343 | [I2D2](src/tasksource/tasks.py#L1302) | Classification | [tasksource/I2D2](https://hf.co/datasets/tasksource/I2D2) | ✓ |
| 344 | [args_me](src/tasksource/tasks.py#L1307) | Classification | [webis/args_me](https://hf.co/datasets/webis/args_me) | ✓ |
| 345 | [Touche23-ValueEval](src/tasksource/tasks.py#L1312) | Classification | csv | ✓ |
| 346 | [starcon](src/tasksource/tasks.py#L1321) | Classification | [tasksource/starcon](https://hf.co/datasets/tasksource/starcon) | ✓ |
| 347 | [banking77](src/tasksource/tasks.py#L1326) | Classification | [legacy-datasets/banking77](https://hf.co/datasets/legacy-datasets/banking77) |  |
| 348 | [it-support-tickets](src/tasksource/tasks.py#L1328) | Classification | [tasksource/it-support-tickets](https://hf.co/datasets/tasksource/it-support-tickets) |  |
| 349 | [ConTRoL-nli](src/tasksource/tasks.py#L1332) | Classification | [tasksource/ConTRoL-nli](https://hf.co/datasets/tasksource/ConTRoL-nli) |  |
| 350 | [tracie](src/tasksource/tasks.py#L1333) | Classification | [tasksource/tracie](https://hf.co/datasets/tasksource/tracie) |  |
| 351 | [sherliic](src/tasksource/tasks.py#L1334) | Classification | [tasksource/sherliic](https://hf.co/datasets/tasksource/sherliic) |  |
| 352 | [sen-making/1](src/tasksource/tasks.py#L1336) | MultipleChoice | [tasksource/sen-making](https://hf.co/datasets/tasksource/sen-making) | ✓ |
| 353 | [sen-making/2](src/tasksource/tasks.py#L1340) | MultipleChoice | [tasksource/sen-making](https://hf.co/datasets/tasksource/sen-making) | ✓ |
| 354 | [winowhy](src/tasksource/tasks.py#L1343) | Classification | [tasksource/winowhy](https://hf.co/datasets/tasksource/winowhy) | ✓ |
| 355 | [robustLR](src/tasksource/tasks.py#L1347) | Classification | [tasksource/robustLR](https://hf.co/datasets/tasksource/robustLR) |  |
| 356 | [clutrr](src/tasksource/tasks.py#L1349) | Classification | [tasksource/clutrr](https://hf.co/datasets/tasksource/clutrr) |  |
| 357 | [logical-fallacy](src/tasksource/tasks.py#L1351) | Classification | [tasksource/logical-fallacy](https://hf.co/datasets/tasksource/logical-fallacy) |  |
| 358 | [parade](src/tasksource/tasks.py#L1353) | Classification | [tasksource/parade](https://hf.co/datasets/tasksource/parade) |  |
| 359 | [cladder](src/tasksource/tasks.py#L1355) | Classification | [tasksource/cladder](https://hf.co/datasets/tasksource/cladder) |  |
| 360 | [subjectivity](src/tasksource/tasks.py#L1357) | Classification | [tasksource/subjectivity](https://hf.co/datasets/tasksource/subjectivity) | ✓ |
| 361 | [MOH](src/tasksource/tasks.py#L1363) | Classification | [tasksource/MOH](https://hf.co/datasets/tasksource/MOH) | ✓ |
| 362 | [VUAC](src/tasksource/tasks.py#L1364) | Classification | [tasksource/VUAC](https://hf.co/datasets/tasksource/VUAC) | ✓ |
| 363 | [TroFi](src/tasksource/tasks.py#L1365) | Classification | parquet | ✓ |
| 364 | [sharc](src/tasksource/tasks.py#L1373) | Classification | [tasksource/sharc](https://hf.co/datasets/tasksource/sharc) |  |
| 365 | [conceptrules_v2](src/tasksource/tasks.py#L1377) | Classification | [tasksource/conceptrules_v2](https://hf.co/datasets/tasksource/conceptrules_v2) | ✓ |
| 366 | [disrpt/eng.dep.scidtb.rels](src/tasksource/tasks.py#L1379) | Classification | [multilingual-discourse-hub/disrpt](https://hf.co/datasets/multilingual-discourse-hub/disrpt) |  |
| 367 | [conll2000](src/tasksource/tasks.py#L1381) | TokenClassification | [eriktks/conll2000](https://hf.co/datasets/eriktks/conll2000) |  |
| 368 | [few-nerd/supervised](src/tasksource/tasks.py#L1384) | TokenClassification | [DFKI-SLT/few-nerd](https://hf.co/datasets/DFKI-SLT/few-nerd) |  |
| 369 | [finer-139](src/tasksource/tasks.py#L1385) | TokenClassification | [nlpaueb/finer-139](https://hf.co/datasets/nlpaueb/finer-139) |  |
| 370 | [zero-shot-label-nli](src/tasksource/tasks.py#L1388) | Classification | [tasksource/zero-shot-label-nli](https://hf.co/datasets/tasksource/zero-shot-label-nli) |  |
| 371 | [com2sense](src/tasksource/tasks.py#L1390) | Classification | [tasksource/com2sense](https://hf.co/datasets/tasksource/com2sense) |  |
| 372 | [scone](src/tasksource/tasks.py#L1392) | Classification | [tasksource/scone](https://hf.co/datasets/tasksource/scone) |  |
| 373 | [winodict](src/tasksource/tasks.py#L1394) | MultipleChoice | [tasksource/winodict](https://hf.co/datasets/tasksource/winodict) |  |
| 374 | [fool-me-twice](src/tasksource/tasks.py#L1396) | Classification | [tasksource/fool-me-twice](https://hf.co/datasets/tasksource/fool-me-twice) |  |
| 375 | [monli](src/tasksource/tasks.py#L1400) | Classification | [tasksource/monli](https://hf.co/datasets/tasksource/monli) |  |
| 376 | [corr2cause](src/tasksource/tasks.py#L1402) | Classification | [tasksource/corr2cause](https://hf.co/datasets/tasksource/corr2cause) |  |
| 377 | [lsat_qa/all](src/tasksource/tasks.py#L1404) | MultipleChoice | [lighteval/lsat_qa](https://hf.co/datasets/lighteval/lsat_qa) |  |
| 378 | [apt](src/tasksource/tasks.py#L1406) | Classification | [tasksource/apt](https://hf.co/datasets/tasksource/apt) |  |
| 379 | [twitter-financial-news-sentiment](src/tasksource/tasks.py#L1409) | Classification | [zeroshot/twitter-financial-news-sentiment](https://hf.co/datasets/zeroshot/twitter-financial-news-sentiment) |  |
| 380 | [icl-symbol-tuning-instruct](src/tasksource/tasks.py#L1416) | Classification | [tasksource/icl-symbol-tuning-instruct](https://hf.co/datasets/tasksource/icl-symbol-tuning-instruct) | ✓ |
| 381 | [SpaceNLI](src/tasksource/tasks.py#L1422) | Classification | [tasksource/SpaceNLI](https://hf.co/datasets/tasksource/SpaceNLI) |  |
| 382 | [propsegment/nli](src/tasksource/tasks.py#L1424) | Classification | json |  |
| 383 | [HatemojiBuild](src/tasksource/tasks.py#L1433) | Classification | [HannahRoseKirk/HatemojiBuild](https://hf.co/datasets/HannahRoseKirk/HatemojiBuild) |  |
| 384 | [regset](src/tasksource/tasks.py#L1436) | Classification | [tasksource/regset](https://hf.co/datasets/tasksource/regset) | ✓ |
| 385 | [esci](src/tasksource/tasks.py#L1442) | Classification | [tasksource/esci](https://hf.co/datasets/tasksource/esci) |  |
| 386 | [chatbot_arena_conversations](src/tasksource/tasks.py#L1461) | MultipleChoice | [lmsys/chatbot_arena_conversations](https://hf.co/datasets/lmsys/chatbot_arena_conversations) | ✓ |
| 387 | [dnd_style_intents](src/tasksource/tasks.py#L1467) | Classification | [neurae/dnd_style_intents](https://hf.co/datasets/neurae/dnd_style_intents) |  |
| 388 | [FLD.v2/default](src/tasksource/tasks.py#L1470) | Classification | [hitachi-nlp/FLD.v2](https://hf.co/datasets/hitachi-nlp/FLD.v2) |  |
| 389 | [FLD.v2/star](src/tasksource/tasks.py#L1473) | Classification | [hitachi-nlp/FLD.v2](https://hf.co/datasets/hitachi-nlp/FLD.v2) |  |
| 390 | [SDOH-NLI](src/tasksource/tasks.py#L1476) | Classification | [tasksource/SDOH-NLI](https://hf.co/datasets/tasksource/SDOH-NLI) |  |
| 391 | [scifact_entailment](src/tasksource/tasks.py#L1479) | Classification | [tasksource/scifact_entailment](https://hf.co/datasets/tasksource/scifact_entailment) |  |
| 392 | [feasibilityQA](src/tasksource/tasks.py#L1483) | Classification | [tasksource/feasibilityQA](https://hf.co/datasets/tasksource/feasibilityQA) |  |
| 393 | [simple_pair](src/tasksource/tasks.py#L1486) | Classification | [tasksource/simple_pair](https://hf.co/datasets/tasksource/simple_pair) |  |
| 394 | [AdjectiveScaleProbe-nli](src/tasksource/tasks.py#L1487) | Classification | [tasksource/AdjectiveScaleProbe-nli](https://hf.co/datasets/tasksource/AdjectiveScaleProbe-nli) |  |
| 395 | [resnli](src/tasksource/tasks.py#L1488) | Classification | [tasksource/resnli](https://hf.co/datasets/tasksource/resnli) |  |
| 396 | [SpaRTUN](src/tasksource/tasks.py#L1490) | MultipleChoice | [tasksource/SpaRTUN](https://hf.co/datasets/tasksource/SpaRTUN) |  |
| 397 | [ReSQ](src/tasksource/tasks.py#L1495) | MultipleChoice | [tasksource/ReSQ](https://hf.co/datasets/tasksource/ReSQ) |  |
| 398 | [semantic_fragments_nli](src/tasksource/tasks.py#L1500) | Classification | [tasksource/semantic_fragments_nli](https://hf.co/datasets/tasksource/semantic_fragments_nli) |  |
| 399 | [dataset_train_nli](src/tasksource/tasks.py#L1503) | Classification | [MoritzLaurer/dataset_train_nli](https://hf.co/datasets/MoritzLaurer/dataset_train_nli) |  |
| 400 | [stepgame](src/tasksource/tasks.py#L1508) | Classification | [tasksource/stepgame](https://hf.co/datasets/tasksource/stepgame) |  |
| 401 | [nlgraph](src/tasksource/tasks.py#L1516) | Classification | [tasksource/nlgraph](https://hf.co/datasets/tasksource/nlgraph) |  |
| 402 | [oasst2_pairwise_rlhf_reward](src/tasksource/tasks.py#L1520) | MultipleChoice | [tasksource/oasst2_pairwise_rlhf_reward](https://hf.co/datasets/tasksource/oasst2_pairwise_rlhf_reward) | ✓ |
| 403 | [hh-rlhf/helpful-rejection-sampled](src/tasksource/tasks.py#L1531) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| 404 | [hh-rlhf/helpful-online](src/tasksource/tasks.py#L1531) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| 405 | [hh-rlhf/helpful-base](src/tasksource/tasks.py#L1531) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| 406 | [hh-rlhf/harmless-base](src/tasksource/tasks.py#L1535) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| 407 | [ruletaker](src/tasksource/tasks.py#L1539) | Classification | [tasksource/ruletaker](https://hf.co/datasets/tasksource/ruletaker) | ✓ |
| 408 | [PARARULE-Plus](src/tasksource/tasks.py#L1543) | Classification | [qbao775/PARARULE-Plus](https://hf.co/datasets/qbao775/PARARULE-Plus) | ✓ |
| 409 | [proofwriter](src/tasksource/tasks.py#L1547) | Classification | [tasksource/proofwriter](https://hf.co/datasets/tasksource/proofwriter) |  |
| 410 | [logical-entailment](src/tasksource/tasks.py#L1550) | Classification | [tasksource/logical-entailment](https://hf.co/datasets/tasksource/logical-entailment) |  |
| 411 | [nope](src/tasksource/tasks.py#L1552) | Classification | [tasksource/nope](https://hf.co/datasets/tasksource/nope) |  |
| 412 | [LogicNLI](src/tasksource/tasks.py#L1556) | Classification | [tasksource/LogicNLI](https://hf.co/datasets/tasksource/LogicNLI) |  |
| 413 | [contract-nli/contractnli_a/seg](src/tasksource/tasks.py#L1558) | Classification | [tasksource/contract-nli](https://hf.co/datasets/tasksource/contract-nli) |  |
| 414 | [contract-nli/contractnli_b/full](src/tasksource/tasks.py#L1560) | Classification | [tasksource/contract-nli](https://hf.co/datasets/tasksource/contract-nli) |  |
| 415 | [nli4ct_semeval2024](src/tasksource/tasks.py#L1562) | Classification | [AshtonIsNotHere/nli4ct_semeval2024](https://hf.co/datasets/AshtonIsNotHere/nli4ct_semeval2024) |  |
| 416 | [lsat-ar](src/tasksource/tasks.py#L1565) | MultipleChoice | [tasksource/lsat-ar](https://hf.co/datasets/tasksource/lsat-ar) |  |
| 417 | [lsat-rc](src/tasksource/tasks.py#L1570) | MultipleChoice | [tasksource/lsat-rc](https://hf.co/datasets/tasksource/lsat-rc) |  |
| 418 | [biosift-nli](src/tasksource/tasks.py#L1575) | Classification | [AshtonIsNotHere/biosift-nli](https://hf.co/datasets/AshtonIsNotHere/biosift-nli) |  |
| 419 | [brainteasers/WP](src/tasksource/tasks.py#L1579) | MultipleChoice | [tasksource/brainteasers](https://hf.co/datasets/tasksource/brainteasers) |  |
| 420 | [brainteasers/SP](src/tasksource/tasks.py#L1579) | MultipleChoice | [tasksource/brainteasers](https://hf.co/datasets/tasksource/brainteasers) |  |
| 421 | [toxigen-data/annotated](src/tasksource/tasks.py#L1585) | Classification | [skg/toxigen-data](https://hf.co/datasets/skg/toxigen-data) |  |
| 422 | [persuasion](src/tasksource/tasks.py#L1596) | Classification | [Anthropic/persuasion](https://hf.co/datasets/Anthropic/persuasion) |  |
| 423 | [AmbigNQ-clarifying-question](src/tasksource/tasks.py#L1602) | Classification | [erbacher/AmbigNQ-clarifying-question](https://hf.co/datasets/erbacher/AmbigNQ-clarifying-question) | ✓ |
| 424 | [SIGA-nli](src/tasksource/tasks.py#L1607) | Classification | [tasksource/SIGA-nli](https://hf.co/datasets/tasksource/SIGA-nli) |  |
| 425 | [FOL-nli](src/tasksource/tasks.py#L1609) | Classification | [unigram/FOL-nli](https://hf.co/datasets/unigram/FOL-nli) |  |
| 426 | [goal-step-wikihow/goal](src/tasksource/tasks.py#L1611) | MultipleChoice | [tasksource/goal-step-wikihow](https://hf.co/datasets/tasksource/goal-step-wikihow) |  |
| 427 | [goal-step-wikihow/step](src/tasksource/tasks.py#L1614) | MultipleChoice | [tasksource/goal-step-wikihow](https://hf.co/datasets/tasksource/goal-step-wikihow) |  |
| 428 | [goal-step-wikihow/order](src/tasksource/tasks.py#L1617) | MultipleChoice | [tasksource/goal-step-wikihow](https://hf.co/datasets/tasksource/goal-step-wikihow) |  |
| 429 | [PARADISE](src/tasksource/tasks.py#L1620) | MultipleChoice | [GGLab/PARADISE](https://hf.co/datasets/GGLab/PARADISE) |  |
| 430 | [doc-nli](src/tasksource/tasks.py#L1623) | Classification | [tasksource/doc-nli](https://hf.co/datasets/tasksource/doc-nli) |  |
| 431 | [mctest-nli](src/tasksource/tasks.py#L1625) | Classification | [tasksource/mctest-nli](https://hf.co/datasets/tasksource/mctest-nli) |  |
| 432 | [patent-phrase-similarity](src/tasksource/tasks.py#L1627) | Classification | [tasksource/patent-phrase-similarity](https://hf.co/datasets/tasksource/patent-phrase-similarity) |  |
| 433 | [natural-language-satisfiability](src/tasksource/tasks.py#L1629) | Classification | [tasksource/natural-language-satisfiability](https://hf.co/datasets/tasksource/natural-language-satisfiability) |  |
| 434 | [idioms-nli](src/tasksource/tasks.py#L1631) | Classification | [tasksource/idioms-nli](https://hf.co/datasets/tasksource/idioms-nli) |  |
| 435 | [lifecycle-entailment](src/tasksource/tasks.py#L1633) | Classification | [tasksource/lifecycle-entailment](https://hf.co/datasets/tasksource/lifecycle-entailment) |  |
| 436 | [safe-guard-prompt-injection](src/tasksource/tasks.py#L1640) | Classification | [xTRam1/safe-guard-prompt-injection](https://hf.co/datasets/xTRam1/safe-guard-prompt-injection) | ✓ |
| 437 | [prompt-injections](src/tasksource/tasks.py#L1646) | Classification | [deepset/prompt-injections](https://hf.co/datasets/deepset/prompt-injections) | ✓ |
| 438 | [prompt-injection-dataset](src/tasksource/tasks.py#L1652) | Classification | [S-Labs/prompt-injection-dataset](https://hf.co/datasets/S-Labs/prompt-injection-dataset) | ✓ |
| 439 | [Prompt-injection-dataset/full](src/tasksource/tasks.py#L1658) | Classification | [neuralchemy/Prompt-injection-dataset](https://hf.co/datasets/neuralchemy/Prompt-injection-dataset) | ✓ |
| 440 | [PromptShield](src/tasksource/tasks.py#L1664) | Classification | [hendzh/PromptShield](https://hf.co/datasets/hendzh/PromptShield) | ✓ |
| 441 | [shell-safety-v2](src/tasksource/tasks.py#L1670) | Classification | [tomngdev/shell-safety-v2](https://hf.co/datasets/tomngdev/shell-safety-v2) | ✓ |
| 442 | [agent_action_safety](src/tasksource/tasks.py#L1675) | Classification | json | ✓ |
| 443 | [ShellRisk-Bench](src/tasksource/tasks.py#L1690) | Classification | [kontext-security/ShellRisk-Bench](https://hf.co/datasets/kontext-security/ShellRisk-Bench) | ✓ |
| 444 | [wildguardmix-cleaned/prompt_harm](src/tasksource/tasks.py#L1715) | Classification | [bogdanminko/wildguardmix-cleaned](https://hf.co/datasets/bogdanminko/wildguardmix-cleaned) | ✓ |
| 445 | [wildguardmix-cleaned/response_harm](src/tasksource/tasks.py#L1721) | Classification | [bogdanminko/wildguardmix-cleaned](https://hf.co/datasets/bogdanminko/wildguardmix-cleaned) | ✓ |
| 446 | [wildguardmix-cleaned/response_refusal](src/tasksource/tasks.py#L1726) | Classification | [bogdanminko/wildguardmix-cleaned](https://hf.co/datasets/bogdanminko/wildguardmix-cleaned) | ✓ |
| 447 | [BeaverTails](src/tasksource/tasks.py#L1742) | Classification | [PKU-Alignment/BeaverTails](https://hf.co/datasets/PKU-Alignment/BeaverTails) | ✓ |
| 448 | [privacy-200k-Mistral-Large-3](src/tasksource/tasks.py#L1758) | Classification | [gabrielloiseau/privacy-200k-Mistral-Large-3](https://hf.co/datasets/gabrielloiseau/privacy-200k-Mistral-Large-3) | ✓ |
| 449 | [toxic-chat/toxicchat0124/toxicity](src/tasksource/tasks.py#L1767) | Classification | [lmsys/toxic-chat](https://hf.co/datasets/lmsys/toxic-chat) | ✓ |
| 450 | [toxic-chat/toxicchat0124/jailbreaking](src/tasksource/tasks.py#L1772) | Classification | [lmsys/toxic-chat](https://hf.co/datasets/lmsys/toxic-chat) | ✓ |
| 451 | [clinc_oos/plus](src/tasksource/tasks.py#L1778) | Classification | [clinc/clinc_oos](https://hf.co/datasets/clinc/clinc_oos) |  |
| 452 | [IntentGrasp/all](src/tasksource/tasks.py#L1796) | MultipleChoice | [yuweiyin/IntentGrasp](https://hf.co/datasets/yuweiyin/IntentGrasp) |  |
| 453 | [few_rel/default](src/tasksource/tasks.py#L1834) | Classification | [tasksource/few_rel](https://hf.co/datasets/tasksource/few_rel) |  |
| 454 | [docred](src/tasksource/tasks.py#L1880) | Classification | json |  |
| 455 | [chemprot/chemprot_full_source](src/tasksource/tasks.py#L1908) | Classification | [bigbio/chemprot](https://hf.co/datasets/bigbio/chemprot) |  |
| 456 | [PKU-SafeRLHF/helpfulness](src/tasksource/tasks.py#L1913) | MultipleChoice | [PKU-Alignment/PKU-SafeRLHF](https://hf.co/datasets/PKU-Alignment/PKU-SafeRLHF) | ✓ |
| 457 | [PKU-SafeRLHF/safety](src/tasksource/tasks.py#L1918) | MultipleChoice | [PKU-Alignment/PKU-SafeRLHF](https://hf.co/datasets/PKU-Alignment/PKU-SafeRLHF) | ✓ |
| 458 | [HelpSteer/helpfulness](src/tasksource/tasks.py#L1940) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 459 | [HelpSteer/correctness](src/tasksource/tasks.py#L1941) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 460 | [HelpSteer/coherence](src/tasksource/tasks.py#L1942) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 461 | [HelpSteer/complexity](src/tasksource/tasks.py#L1943) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 462 | [HelpSteer/verbosity](src/tasksource/tasks.py#L1944) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 463 | [HelpSteer2/helpfulness](src/tasksource/tasks.py#L1946) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 464 | [HelpSteer2/correctness](src/tasksource/tasks.py#L1947) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 465 | [HelpSteer2/coherence](src/tasksource/tasks.py#L1948) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 466 | [HelpSteer2/complexity](src/tasksource/tasks.py#L1949) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 467 | [HelpSteer2/verbosity](src/tasksource/tasks.py#L1950) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 468 | [HelpSteer3/preference](src/tasksource/tasks.py#L1955) | MultipleChoice | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | ✓ |
| 469 | [HelpSteer3/preference_strength](src/tasksource/tasks.py#L1968) | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | ✓ |
| 470 | [HelpSteer3/principle](src/tasksource/tasks.py#L1980) | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) |  |
| 471 | [HelpSteer3/edit_quality](src/tasksource/tasks.py#L1985) | MultipleChoice | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | ✓ |
| 472 | [HelpSteer3/feedback](src/tasksource/tasks.py#L2007) | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | ✓ |
| 473 | [MSciNLI](src/tasksource/tasks.py#L2013) | Classification | [sadat2307/MSciNLI](https://hf.co/datasets/sadat2307/MSciNLI) |  |
| 474 | [UltraFeedback-paired](src/tasksource/tasks.py#L2016) | MultipleChoice | [pushpdeep/UltraFeedback-paired](https://hf.co/datasets/pushpdeep/UltraFeedback-paired) | ✓ |
| 475 | [prm800k_dpo/solution](src/tasksource/tasks.py#L2020) | MultipleChoice | [tasksource/prm800k_dpo](https://hf.co/datasets/tasksource/prm800k_dpo) | ✓ |
| 476 | [prm800k_dpo/step](src/tasksource/tasks.py#L2023) | MultipleChoice | [tasksource/prm800k_dpo](https://hf.co/datasets/tasksource/prm800k_dpo) | ✓ |
| 477 | [AES2-essay-scoring](src/tasksource/tasks.py#L2027) | Classification | [tasksource/AES2-essay-scoring](https://hf.co/datasets/tasksource/AES2-essay-scoring) | ✓ |
| 478 | [argument-feedback](src/tasksource/tasks.py#L2031) | Classification | [tasksource/argument-feedback](https://hf.co/datasets/tasksource/argument-feedback) | ✓ |
| 479 | [english-grading/cohesion](src/tasksource/tasks.py#L2038) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 480 | [english-grading/syntax](src/tasksource/tasks.py#L2039) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 481 | [english-grading/vocabulary](src/tasksource/tasks.py#L2040) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 482 | [english-grading/phraseology](src/tasksource/tasks.py#L2041) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 483 | [english-grading/grammar](src/tasksource/tasks.py#L2042) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 484 | [english-grading/conventions](src/tasksource/tasks.py#L2043) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 485 | [wice](src/tasksource/tasks.py#L2045) | Classification | [tasksource/wice](https://hf.co/datasets/tasksource/wice) |  |
| 486 | [hover](src/tasksource/tasks.py#L2048) | Classification | [Dzeniks/hover](https://hf.co/datasets/Dzeniks/hover) |  |
| 487 | [hover-3way/nli](src/tasksource/tasks.py#L2052) | Classification | [Dzeniks/hover-3way](https://hf.co/datasets/Dzeniks/hover-3way) |  |
| 488 | [tasksource_dpo_pairs](src/tasksource/tasks.py#L2055) | MultipleChoice | [tasksource/tasksource_dpo_pairs](https://hf.co/datasets/tasksource/tasksource_dpo_pairs) | ✓ |
| 489 | [seahorse_summarization_evaluation](src/tasksource/tasks.py#L2058) | Classification | [tasksource/seahorse_summarization_evaluation](https://hf.co/datasets/tasksource/seahorse_summarization_evaluation) |  |
| 490 | [missing-item-prediction/contrastive](src/tasksource/tasks.py#L2061) | Classification | [sileod/missing-item-prediction](https://hf.co/datasets/sileod/missing-item-prediction) |  |
| 491 | [jigsaw_toxicity](src/tasksource/tasks.py#L2065) | Classification | [tasksource/jigsaw_toxicity](https://hf.co/datasets/tasksource/jigsaw_toxicity) |  |
| 492 | [Pol_NLI](src/tasksource/tasks.py#L2068) | Classification | [mlburnham/Pol_NLI](https://hf.co/datasets/mlburnham/Pol_NLI) |  |
| 493 | [synthetic-retrieval-NLI/count](src/tasksource/tasks.py#L2071) | Classification | [tasksource/synthetic-retrieval-NLI](https://hf.co/datasets/tasksource/synthetic-retrieval-NLI) |  |
| 494 | [synthetic-retrieval-NLI/position](src/tasksource/tasks.py#L2071) | Classification | [tasksource/synthetic-retrieval-NLI](https://hf.co/datasets/tasksource/synthetic-retrieval-NLI) |  |
| 495 | [synthetic-retrieval-NLI/binary](src/tasksource/tasks.py#L2071) | Classification | [tasksource/synthetic-retrieval-NLI](https://hf.co/datasets/tasksource/synthetic-retrieval-NLI) |  |
| 496 | [github-issue-similarity](src/tasksource/tasks.py#L2080) | Classification | [WhereIsAI/github-issue-similarity](https://hf.co/datasets/WhereIsAI/github-issue-similarity) |  |

| 497 | [webinstruct/mc](src/tasksource/tasks.py#L2379) | MultipleChoice | [tasksource/webinstruct](https://hf.co/datasets/tasksource/webinstruct) |  |
| 498 | [webinstruct/binary](src/tasksource/tasks.py#L2381) | Classification | [tasksource/webinstruct](https://hf.co/datasets/tasksource/webinstruct) |  |
| 499 | [weblinx/action](src/tasksource/tasks.py#L2467) | Classification | [McGill-NLP/WebLINX](https://hf.co/datasets/McGill-NLP/WebLINX) | ✓ |
| 500 | [weblinx/dom-element](src/tasksource/tasks.py#L2470) | MultipleChoice | [McGill-NLP/WebLINX](https://hf.co/datasets/McGill-NLP/WebLINX) | ✓ |
| 501 | [websrc/yesno](src/tasksource/tasks.py#L2476) | Classification | [tasksource/websrc](https://hf.co/datasets/tasksource/websrc) |  |
| 502 | [websrc/element](src/tasksource/tasks.py#L2478) | MultipleChoice | [tasksource/websrc](https://hf.co/datasets/tasksource/websrc) | ✓ |


## Soft labels

Annotations whose label is a distribution (annotator votes, rater shares, survey counts), loaded with `load_task(id, soft=True)`. Those with a hard view are also listed above, by their majority label; the others have soft labels only. `votes` are shares of annotators, `mean` a mean rating; annotators is the typical count per item (vote shares from fewer than five are coarse, and the Jev build leaves them out).

| id | kind | aggregation | annotators | default view | dataset |
|---|---|---|--:|---|---|
| [glue/stsb](src/tasksource/tasks.py#L44) | score | mean |  | regression | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |
| [sick/relatedness](src/tasksource/tasks.py#L78) | score | mean | 10 | regression | [tasksource/sick](https://hf.co/datasets/tasksource/sick) |
| [google_wellformed_query](src/tasksource/tasks.py#L811) | noul | votes | 5 | Classification | [tasksource/google_wellformed_query](https://hf.co/datasets/tasksource/google_wellformed_query) |
| [sts-companion](src/tasksource/tasks.py#L986) | score | mean |  | regression | [tasksource/sts-companion](https://hf.co/datasets/tasksource/sts-companion) |
| [wouldyourather](src/tasksource/tasks.py#L1052) | choice | votes | 1000 | MultipleChoice | [tasksource/wouldyourather](https://hf.co/datasets/tasksource/wouldyourather) |
| [acceptability-prediction/rating_votes](src/tasksource/tasks.py#L1213) | score | votes | 15 |  | [tasksource/acceptability-prediction](https://hf.co/datasets/tasksource/acceptability-prediction) |
| [acceptability-prediction/binary_votes](src/tasksource/tasks.py#L1218) | noul | votes | 15 |  | [tasksource/acceptability-prediction](https://hf.co/datasets/tasksource/acceptability-prediction) |
| [HelpSteer3/individual_preferences](src/tasksource/tasks.py#L1974) | score | votes | 3 |  | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) |
| [HelpSteer3/feedback](src/tasksource/tasks.py#L2007) | score | votes | 3 | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) |
| [proto_qa/proto_qa](src/tasksource/tasks.py#L2090) | choice | votes | 100 |  | [community-datasets/proto_qa](https://hf.co/datasets/community-datasets/proto_qa) |
| [UNLI](src/tasksource/tasks.py#L2104) | noul | mean | 2 |  | [Zhengping/UNLI](https://hf.co/datasets/Zhengping/UNLI) |
| [chaos-mnli-ambiguity/votes](src/tasksource/tasks.py#L2108) | choice | votes | 100 |  | [tasksource/chaos-mnli-ambiguity](https://hf.co/datasets/tasksource/chaos-mnli-ambiguity) |
| [hate_speech_offensive/votes](src/tasksource/tasks.py#L2112) | choice | votes | 3 |  | [tdavidson/hate_speech_offensive](https://hf.co/datasets/tdavidson/hate_speech_offensive) |
| [scruples/verdict_votes](src/tasksource/tasks.py#L2120) | choice | votes | 8 |  | [tasksource/scruples](https://hf.co/datasets/tasksource/scruples) |
| [BeaverTails/unsafe_votes](src/tasksource/tasks.py#L2134) | noul | votes | 3 |  | [PKU-Alignment/BeaverTails](https://hf.co/datasets/PKU-Alignment/BeaverTails) |
| [dynasent/r1_votes](src/tasksource/tasks.py#L2145) | choice | votes | 5 |  | [dynabench/dynasent](https://hf.co/datasets/dynabench/dynasent) |
| [dynasent/r2_votes](src/tasksource/tasks.py#L2149) | choice | votes | 5 |  | [dynabench/dynasent](https://hf.co/datasets/dynabench/dynasent) |
| [hatexplain/votes](src/tasksource/tasks.py#L2154) | choice | votes | 3 |  | [Hate-speech-CNERG/hatexplain](https://hf.co/datasets/Hate-speech-CNERG/hatexplain) |
| [measuring-hate-speech/sentiment](src/tasksource/tasks.py#L2169) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/respect](src/tasksource/tasks.py#L2171) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/insult](src/tasksource/tasks.py#L2173) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/humiliate](src/tasksource/tasks.py#L2174) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/status](src/tasksource/tasks.py#L2175) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/dehumanize](src/tasksource/tasks.py#L2177) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/violence](src/tasksource/tasks.py#L2179) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/genocide](src/tasksource/tasks.py#L2180) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/attack_defend](src/tasksource/tasks.py#L2182) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/hatespeech](src/tasksource/tasks.py#L2184) | choice | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [wikipedia-detox/attack](src/tasksource/tasks.py#L2311) | choice | votes | 10 |  | [tasksource/wikipedia-detox-votes](https://hf.co/datasets/tasksource/wikipedia-detox-votes) |
| [wikipedia-detox/aggression](src/tasksource/tasks.py#L2313) | choice | votes | 10 |  | [tasksource/wikipedia-detox-votes](https://hf.co/datasets/tasksource/wikipedia-detox-votes) |
| [wikipedia-detox/aggression_score](src/tasksource/tasks.py#L2315) | score | votes | 10 |  | [tasksource/wikipedia-detox-votes](https://hf.co/datasets/tasksource/wikipedia-detox-votes) |
| [wikipedia-detox/toxicity](src/tasksource/tasks.py#L2319) | choice | votes | 10 |  | [tasksource/wikipedia-detox-votes](https://hf.co/datasets/tasksource/wikipedia-detox-votes) |
| [wikipedia-detox/toxicity_score](src/tasksource/tasks.py#L2321) | score | votes | 10 |  | [tasksource/wikipedia-detox-votes](https://hf.co/datasets/tasksource/wikipedia-detox-votes) |
| [lewidi/md_agreement](src/tasksource/tasks.py#L2199) | noul | votes | 5 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/hs_brexit](src/tasksource/tasks.py#L2200) | noul | votes | 6 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/armis](src/tasksource/tasks.py#L2202) | noul | votes | 3 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/conv_abuse](src/tasksource/tasks.py#L2204) | noul | votes | 3 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/mp](src/tasksource/tasks.py#L2206) | noul | votes | 5 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/csc](src/tasksource/tasks.py#L2207) | score | votes | 4 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/varierrnli/entailment](src/tasksource/tasks.py#L2215) | noul | votes | 4 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/varierrnli/neutral](src/tasksource/tasks.py#L2216) | noul | votes | 4 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/varierrnli/contradiction](src/tasksource/tasks.py#L2217) | noul | votes | 4 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [civil_comments/toxicity_share](src/tasksource/tasks.py#L2225) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/severe_toxicity_share](src/tasksource/tasks.py#L2226) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/obscene_share](src/tasksource/tasks.py#L2227) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/threat_share](src/tasksource/tasks.py#L2228) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/insult_share](src/tasksource/tasks.py#L2229) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/identity_attack_share](src/tasksource/tasks.py#L2230) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/sexual_explicit_share](src/tasksource/tasks.py#L2231) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [oasst2/quality](src/tasksource/tasks.py#L2248) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/helpfulness](src/tasksource/tasks.py#L2249) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/creativity](src/tasksource/tasks.py#L2250) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/humor](src/tasksource/tasks.py#L2251) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/toxicity](src/tasksource/tasks.py#L2252) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/violence](src/tasksource/tasks.py#L2253) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/spam](src/tasksource/tasks.py#L2254) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/fails_task](src/tasksource/tasks.py#L2255) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/not_appropriate](src/tasksource/tasks.py#L2256) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/hate_speech](src/tasksource/tasks.py#L2257) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/sexual_content](src/tasksource/tasks.py#L2258) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/pii](src/tasksource/tasks.py#L2259) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/lang_mismatch](src/tasksource/tasks.py#L2260) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
