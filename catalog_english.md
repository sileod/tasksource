545 English tasks. Load one with `load_task(id)`; the annotations are in [tasks.py](src/tasksource/tasks.py). Evaluation benchmarks are in [eval_only.py](src/tasksource/eval_only.py); other excluded annotations are in [parked.py](src/tasksource/parked.py).

| # | id | type | dataset | question |
|--:|---|---|---|:-:|
| 1 | [glue/mnli](src/tasksource/tasks.py#L33) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 2 | [glue/qnli](src/tasksource/tasks.py#L34) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) | ✓ |
| 3 | [glue/rte](src/tasksource/tasks.py#L36) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 4 | [glue/wnli](src/tasksource/tasks.py#L37) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 5 | [glue/mrpc](src/tasksource/tasks.py#L39) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 6 | [glue/qqp](src/tasksource/tasks.py#L40) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 7 | [glue/stsb](src/tasksource/tasks.py#L46) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) | ✓ |
| 8 | [super_glue/boolq](src/tasksource/tasks.py#L51) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 9 | [super_glue/boolq_passage](src/tasksource/tasks.py#L52) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 10 | [super_glue/cb](src/tasksource/tasks.py#L54) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 11 | [super_glue/multirc](src/tasksource/tasks.py#L55) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 12 | [super_glue/wic](src/tasksource/tasks.py#L60) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 13 | [super_glue/axg](src/tasksource/tasks.py#L65) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 14 | [anli/a1](src/tasksource/tasks.py#L68) | Classification | [facebook/anli](https://hf.co/datasets/facebook/anli) |  |
| 15 | [anli/a2](src/tasksource/tasks.py#L69) | Classification | [facebook/anli](https://hf.co/datasets/facebook/anli) |  |
| 16 | [anli/a3](src/tasksource/tasks.py#L70) | Classification | [facebook/anli](https://hf.co/datasets/facebook/anli) |  |
| 17 | [babi_nli/two-arg-relations](src/tasksource/tasks.py#L73) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 18 | [babi_nli/two-supporting-facts](src/tasksource/tasks.py#L73) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 19 | [babi_nli/time-reasoning](src/tasksource/tasks.py#L73) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 20 | [babi_nli/three-supporting-facts](src/tasksource/tasks.py#L73) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 21 | [babi_nli/three-arg-relations](src/tasksource/tasks.py#L73) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 22 | [babi_nli/size-reasoning](src/tasksource/tasks.py#L73) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 23 | [babi_nli/single-supporting-fact](src/tasksource/tasks.py#L73) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 24 | [babi_nli/simple-negation](src/tasksource/tasks.py#L73) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 25 | [babi_nli/positional-reasoning](src/tasksource/tasks.py#L73) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 26 | [babi_nli/yes-no-questions](src/tasksource/tasks.py#L73) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 27 | [babi_nli/lists-sets](src/tasksource/tasks.py#L73) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 28 | [babi_nli/indefinite-knowledge](src/tasksource/tasks.py#L73) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 29 | [babi_nli/counting](src/tasksource/tasks.py#L73) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 30 | [babi_nli/conjunction](src/tasksource/tasks.py#L73) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 31 | [babi_nli/compound-coreference](src/tasksource/tasks.py#L73) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 32 | [babi_nli/path-finding](src/tasksource/tasks.py#L73) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 33 | [babi_nli/basic-coreference](src/tasksource/tasks.py#L73) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 34 | [babi_nli/basic-deduction](src/tasksource/tasks.py#L73) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 35 | [babi_nli/basic-induction](src/tasksource/tasks.py#L78) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) | ✓ |
| 36 | [sick/label](src/tasksource/tasks.py#L83) | Classification | [tasksource/sick](https://hf.co/datasets/tasksource/sick) |  |
| 37 | [sick/relatedness](src/tasksource/tasks.py#L85) | Classification | [tasksource/sick](https://hf.co/datasets/tasksource/sick) | ✓ |
| 38 | [snli](src/tasksource/tasks.py#L143) | Classification | [stanfordnlp/snli](https://hf.co/datasets/stanfordnlp/snli) |  |
| 39 | [scitail/snli_format](src/tasksource/tasks.py#L146) | Classification | [allenai/scitail](https://hf.co/datasets/allenai/scitail) |  |
| 40 | [hans](src/tasksource/tasks.py#L148) | Classification | [tasksource/hans](https://hf.co/datasets/tasksource/hans) |  |
| 41 | [WANLI](src/tasksource/tasks.py#L151) | Classification | [alisawuffles/WANLI](https://hf.co/datasets/alisawuffles/WANLI) |  |
| 42 | [recast/recast_puns](src/tasksource/tasks.py#L153) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 43 | [recast/recast_factuality](src/tasksource/tasks.py#L153) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 44 | [recast/recast_verbnet](src/tasksource/tasks.py#L153) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 45 | [recast/recast_sentiment](src/tasksource/tasks.py#L153) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 46 | [recast/recast_ner](src/tasksource/tasks.py#L153) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 47 | [recast/recast_verbcorner](src/tasksource/tasks.py#L153) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 48 | [recast/recast_megaveridicality](src/tasksource/tasks.py#L153) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 49 | [probability_words_nli/reasoning_2hop](src/tasksource/tasks.py#L158) | Classification | [sileod/probability_words_nli](https://hf.co/datasets/sileod/probability_words_nli) | ✓ |
| 50 | [probability_words_nli/reasoning_1hop](src/tasksource/tasks.py#L158) | Classification | [sileod/probability_words_nli](https://hf.co/datasets/sileod/probability_words_nli) | ✓ |
| 51 | [probability_words_nli/usnli](src/tasksource/tasks.py#L158) | Classification | [sileod/probability_words_nli](https://hf.co/datasets/sileod/probability_words_nli) | ✓ |
| 52 | [nan-nli](src/tasksource/tasks.py#L163) | Classification | [joey234/nan-nli](https://hf.co/datasets/joey234/nan-nli) |  |
| 53 | [nli_fever](src/tasksource/tasks.py#L166) | Classification | [pietrolesci/nli_fever](https://hf.co/datasets/pietrolesci/nli_fever) |  |
| 54 | [breaking_nli](src/tasksource/tasks.py#L169) | Classification | [pietrolesci/breaking_nli](https://hf.co/datasets/pietrolesci/breaking_nli) |  |
| 55 | [conj_nli](src/tasksource/tasks.py#L173) | Classification | [pietrolesci/conj_nli](https://hf.co/datasets/pietrolesci/conj_nli) |  |
| 56 | [fracas](src/tasksource/tasks.py#L177) | Classification | [pietrolesci/fracas](https://hf.co/datasets/pietrolesci/fracas) |  |
| 57 | [dialogue_nli](src/tasksource/tasks.py#L180) | Classification | [pietrolesci/dialogue_nli](https://hf.co/datasets/pietrolesci/dialogue_nli) |  |
| 58 | [mpe](src/tasksource/tasks.py#L183) | Classification | [pietrolesci/mpe](https://hf.co/datasets/pietrolesci/mpe) |  |
| 59 | [dnc](src/tasksource/tasks.py#L187) | Classification | [pietrolesci/dnc](https://hf.co/datasets/pietrolesci/dnc) |  |
| 60 | [recast_white/fnplus](src/tasksource/tasks.py#L191) | Classification | [pietrolesci/recast_white](https://hf.co/datasets/pietrolesci/recast_white) |  |
| 61 | [recast_white/sprl](src/tasksource/tasks.py#L194) | Classification | [pietrolesci/recast_white](https://hf.co/datasets/pietrolesci/recast_white) |  |
| 62 | [recast_white/dpr](src/tasksource/tasks.py#L197) | Classification | [pietrolesci/recast_white](https://hf.co/datasets/pietrolesci/recast_white) |  |
| 63 | [joci](src/tasksource/tasks.py#L201) | Classification | [pietrolesci/joci](https://hf.co/datasets/pietrolesci/joci) | ✓ |
| 64 | [robust_nli/IS_CS](src/tasksource/tasks.py#L208) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 65 | [robust_nli/LI_LI](src/tasksource/tasks.py#L210) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 66 | [robust_nli/ST_WO](src/tasksource/tasks.py#L212) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 67 | [robust_nli/PI_SP](src/tasksource/tasks.py#L214) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 68 | [robust_nli/PI_CD](src/tasksource/tasks.py#L216) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 69 | [robust_nli/ST_SE](src/tasksource/tasks.py#L218) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 70 | [robust_nli/ST_NE](src/tasksource/tasks.py#L220) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 71 | [robust_nli/ST_LM](src/tasksource/tasks.py#L222) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 72 | [robust_nli_is_sd](src/tasksource/tasks.py#L224) | Classification | [pietrolesci/robust_nli_is_sd](https://hf.co/datasets/pietrolesci/robust_nli_is_sd) |  |
| 73 | [robust_nli_li_ts](src/tasksource/tasks.py#L227) | Classification | [pietrolesci/robust_nli_li_ts](https://hf.co/datasets/pietrolesci/robust_nli_li_ts) |  |
| 74 | [gen_debiased_nli/snli_seq_z](src/tasksource/tasks.py#L231) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| 75 | [gen_debiased_nli/snli_z_aug](src/tasksource/tasks.py#L233) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| 76 | [gen_debiased_nli/snli_par_z](src/tasksource/tasks.py#L235) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| 77 | [gen_debiased_nli/mnli_par_z](src/tasksource/tasks.py#L237) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| 78 | [gen_debiased_nli/mnli_z_aug](src/tasksource/tasks.py#L239) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| 79 | [gen_debiased_nli/mnli_seq_z](src/tasksource/tasks.py#L241) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| 80 | [add_one_rte](src/tasksource/tasks.py#L244) | Classification | [pietrolesci/add_one_rte](https://hf.co/datasets/pietrolesci/add_one_rte) |  |
| 81 | [hlgd](src/tasksource/tasks.py#L248) | Classification | [tasksource/hlgd](https://hf.co/datasets/tasksource/hlgd) | ✓ |
| 82 | [paws/labeled_final](src/tasksource/tasks.py#L252) | Classification | [google-research-datasets/paws](https://hf.co/datasets/google-research-datasets/paws) |  |
| 83 | [paws/labeled_swap](src/tasksource/tasks.py#L253) | Classification | [google-research-datasets/paws](https://hf.co/datasets/google-research-datasets/paws) |  |
| 84 | [medical_questions_pairs](src/tasksource/tasks.py#L255) | Classification | [curaihealth/medical_questions_pairs](https://hf.co/datasets/curaihealth/medical_questions_pairs) | ✓ |
| 85 | [conll2003/pos_tags](src/tasksource/tasks.py#L261) | TokenClassification | [tomaarsen/conll2003](https://hf.co/datasets/tomaarsen/conll2003) |  |
| 86 | [conll2003/chunk_tags](src/tasksource/tasks.py#L262) | TokenClassification | [tomaarsen/conll2003](https://hf.co/datasets/tomaarsen/conll2003) |  |
| 87 | [conll2003/ner_tags](src/tasksource/tasks.py#L263) | TokenClassification | [tomaarsen/conll2003](https://hf.co/datasets/tomaarsen/conll2003) |  |
| 88 | [fig-qa](src/tasksource/tasks.py#L269) | MultipleChoice | [nightingal3/fig-qa](https://hf.co/datasets/nightingal3/fig-qa) |  |
| 89 | [cos_e/v1.0](src/tasksource/tasks.py#L278) | MultipleChoice | [Salesforce/cos_e](https://hf.co/datasets/Salesforce/cos_e) |  |
| 90 | [cosmos_qa](src/tasksource/tasks.py#L283) | MultipleChoice | [Samsoup/cosmos_qa](https://hf.co/datasets/Samsoup/cosmos_qa) |  |
| 91 | [dream](src/tasksource/tasks.py#L286) | MultipleChoice | [dataset-org/dream](https://hf.co/datasets/dataset-org/dream) |  |
| 92 | [openbookqa](src/tasksource/tasks.py#L293) | MultipleChoice | [allenai/openbookqa](https://hf.co/datasets/allenai/openbookqa) |  |
| 93 | [qasc](src/tasksource/tasks.py#L299) | MultipleChoice | [allenai/qasc](https://hf.co/datasets/allenai/qasc) |  |
| 94 | [quartz](src/tasksource/tasks.py#L307) | MultipleChoice | [allenai/quartz](https://hf.co/datasets/allenai/quartz) |  |
| 95 | [quail](src/tasksource/tasks.py#L312) | MultipleChoice | [textmachinelab/quail](https://hf.co/datasets/textmachinelab/quail) |  |
| 96 | [head_qa/en](src/tasksource/tasks.py#L318) | MultipleChoice | [EleutherAI/headqa](https://hf.co/datasets/EleutherAI/headqa) |  |
| 97 | [sciq](src/tasksource/tasks.py#L326) | MultipleChoice | [allenai/sciq](https://hf.co/datasets/allenai/sciq) |  |
| 98 | [social_i_qa](src/tasksource/tasks.py#L331) | MultipleChoice | [tasksource/social_i_qa](https://hf.co/datasets/tasksource/social_i_qa) |  |
| 99 | [wiki_hop/original](src/tasksource/tasks.py#L337) | MultipleChoice | [MoE-UNC/wikihop](https://hf.co/datasets/MoE-UNC/wikihop) |  |
| 100 | [wiqa](src/tasksource/tasks.py#L350) | MultipleChoice | parquet |  |
| 101 | [piqa](src/tasksource/tasks.py#L357) | MultipleChoice | [baber/piqa](https://hf.co/datasets/baber/piqa) |  |
| 102 | [hellaswag](src/tasksource/tasks.py#L365) | MultipleChoice | [Rowan/hellaswag](https://hf.co/datasets/Rowan/hellaswag) |  |
| 103 | [super_glue/copa](src/tasksource/tasks.py#L374) | MultipleChoice | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 104 | [balanced-copa](src/tasksource/tasks.py#L376) | MultipleChoice | [pkavumba/balanced-copa](https://hf.co/datasets/pkavumba/balanced-copa) |  |
| 105 | [e-CARE](src/tasksource/tasks.py#L379) | MultipleChoice | [12ml/e-CARE](https://hf.co/datasets/12ml/e-CARE) |  |
| 106 | [art](src/tasksource/tasks.py#L382) | MultipleChoice | [allenai/art](https://hf.co/datasets/allenai/art) | ✓ |
| 107 | [winogrande/winogrande_xl](src/tasksource/tasks.py#L390) | MultipleChoice | [allenai/winogrande](https://hf.co/datasets/allenai/winogrande) |  |
| 108 | [codah/codah](src/tasksource/tasks.py#L393) | MultipleChoice | [jaredfern/codah](https://hf.co/datasets/jaredfern/codah) |  |
| 109 | [ai2_arc/ARC-Easy/challenge](src/tasksource/tasks.py#L395) | MultipleChoice | [allenai/ai2_arc](https://hf.co/datasets/allenai/ai2_arc) |  |
| 110 | [ai2_arc/ARC-Challenge/challenge](src/tasksource/tasks.py#L395) | MultipleChoice | [allenai/ai2_arc](https://hf.co/datasets/allenai/ai2_arc) |  |
| 111 | [definite_pronoun_resolution](src/tasksource/tasks.py#L400) | MultipleChoice | [community-datasets/definite_pronoun_resolution](https://hf.co/datasets/community-datasets/definite_pronoun_resolution) |  |
| 112 | [swag/regular](src/tasksource/tasks.py#L406) | MultipleChoice | [allenai/swag](https://hf.co/datasets/allenai/swag) |  |
| 113 | [math_qa](src/tasksource/tasks.py#L412) | MultipleChoice | [tasksource/math_qa](https://hf.co/datasets/tasksource/math_qa) |  |
| 114 | [glue/cola](src/tasksource/tasks.py#L421) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 115 | [glue/sst2](src/tasksource/tasks.py#L422) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 116 | [utilitarianism](src/tasksource/tasks.py#L436) | Classification | csv |  |
| 117 | [amazon_counterfactual/en](src/tasksource/tasks.py#L444) | Classification | [mteb/amazon_counterfactual](https://hf.co/datasets/mteb/amazon_counterfactual) |  |
| 118 | [insincere-questions](src/tasksource/tasks.py#L449) | Classification | [SetFit/insincere-questions](https://hf.co/datasets/SetFit/insincere-questions) |  |
| 119 | [toxic_conversations](src/tasksource/tasks.py#L453) | Classification | [SetFit/toxic_conversations](https://hf.co/datasets/SetFit/toxic_conversations) |  |
| 120 | [TuringBench](src/tasksource/tasks.py#L457) | Classification | csv | ✓ |
| 121 | [trec](src/tasksource/tasks.py#L467) | Classification | [tasksource/trec](https://hf.co/datasets/tasksource/trec) |  |
| 122 | [vitaminc](src/tasksource/tasks.py#L470) | Classification | [tals/vitaminc](https://hf.co/datasets/tals/vitaminc) |  |
| 123 | [hope_edi/english](src/tasksource/tasks.py#L472) | Classification | csv |  |
| 124 | [rumoureval_2019/RumourEval2019](src/tasksource/tasks.py#L486) | Classification | csv |  |
| 125 | [ethos/binary](src/tasksource/tasks.py#L500) | Classification | [SetFit/ethos_binary](https://hf.co/datasets/SetFit/ethos_binary) |  |
| 126 | [ethos/multilabel](src/tasksource/tasks.py#L524) | Classification | [tasksource/ethos](https://hf.co/datasets/tasksource/ethos) |  |
| 127 | [tweet_eval/emoji](src/tasksource/tasks.py#L527) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 128 | [tweet_eval/sentiment](src/tasksource/tasks.py#L527) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 129 | [tweet_eval/irony](src/tasksource/tasks.py#L527) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 130 | [tweet_eval/offensive](src/tasksource/tasks.py#L527) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 131 | [tweet_eval/emotion](src/tasksource/tasks.py#L527) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 132 | [tweet_eval/hate](src/tasksource/tasks.py#L527) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 133 | [tweet_eval/stance_abortion](src/tasksource/tasks.py#L542) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 134 | [tweet_eval/stance_atheism](src/tasksource/tasks.py#L543) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 135 | [tweet_eval/stance_climate](src/tasksource/tasks.py#L544) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 136 | [tweet_eval/stance_feminist](src/tasksource/tasks.py#L545) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 137 | [tweet_eval/stance_hillary](src/tasksource/tasks.py#L546) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 138 | [discovery/discovery](src/tasksource/tasks.py#L549) | Classification | [sileod/discovery](https://hf.co/datasets/sileod/discovery) | ✓ |
| 139 | [pragmeval/switchboard](src/tasksource/tasks.py#L552) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 140 | [pragmeval/verifiability](src/tasksource/tasks.py#L552) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 141 | [pragmeval/mrda](src/tasksource/tasks.py#L552) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 142 | [pragmeval/gum](src/tasksource/tasks.py#L556) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 143 | [pragmeval/persuasiveness-premisetype](src/tasksource/tasks.py#L556) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 144 | [pragmeval/persuasiveness-claimtype](src/tasksource/tasks.py#L556) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 145 | [pragmeval/pdtb](src/tasksource/tasks.py#L556) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 146 | [pragmeval/emergent](src/tasksource/tasks.py#L556) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 147 | [pragmeval/stac](src/tasksource/tasks.py#L556) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 148 | [pragmeval/sarcasm](src/tasksource/tasks.py#L556) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 149 | [pragmeval/emobank-arousal](src/tasksource/tasks.py#L569) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 150 | [pragmeval/emobank-dominance](src/tasksource/tasks.py#L570) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 151 | [pragmeval/emobank-valence](src/tasksource/tasks.py#L571) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 152 | [pragmeval/squinky-formality](src/tasksource/tasks.py#L572) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 153 | [pragmeval/squinky-implicature](src/tasksource/tasks.py#L573) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 154 | [pragmeval/squinky-informativeness](src/tasksource/tasks.py#L574) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 155 | [pragmeval/persuasiveness-eloquence](src/tasksource/tasks.py#L575) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 156 | [pragmeval/persuasiveness-relevance](src/tasksource/tasks.py#L576) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 157 | [pragmeval/persuasiveness-specificity](src/tasksource/tasks.py#L577) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 158 | [pragmeval/persuasiveness-strength](src/tasksource/tasks.py#L578) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 159 | [silicone/sem](src/tasksource/tasks.py#L580) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| 160 | [silicone/dyda_da](src/tasksource/tasks.py#L580) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| 161 | [silicone/dyda_e](src/tasksource/tasks.py#L580) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| 162 | [silicone/maptask](src/tasksource/tasks.py#L580) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| 163 | [silicone/meld_e](src/tasksource/tasks.py#L580) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| 164 | [silicone/meld_s](src/tasksource/tasks.py#L580) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| 165 | [silicone/oasis](src/tasksource/tasks.py#L580) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| 166 | [silicone/iemocap](src/tasksource/tasks.py#L587) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| 167 | [lex_glue/eurlex](src/tasksource/tasks.py#L592) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) |  |
| 168 | [lex_glue/scotus](src/tasksource/tasks.py#L594) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) |  |
| 169 | [lex_glue/ledgar](src/tasksource/tasks.py#L597) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) |  |
| 170 | [lex_glue/unfair_tos](src/tasksource/tasks.py#L599) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) | ✓ |
| 171 | [lex_glue/case_hold](src/tasksource/tasks.py#L602) | MultipleChoice | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) |  |
| 172 | [language-identification](src/tasksource/tasks.py#L610) | Classification | [papluca/language-identification](https://hf.co/datasets/papluca/language-identification) | ✓ |
| 173 | [imdb](src/tasksource/tasks.py#L615) | Classification | [stanfordnlp/imdb](https://hf.co/datasets/stanfordnlp/imdb) |  |
| 174 | [rotten_tomatoes](src/tasksource/tasks.py#L617) | Classification | [cornell-movie-review-data/rotten_tomatoes](https://hf.co/datasets/cornell-movie-review-data/rotten_tomatoes) |  |
| 175 | [ag_news](src/tasksource/tasks.py#L619) | Classification | [fancyzhx/ag_news](https://hf.co/datasets/fancyzhx/ag_news) |  |
| 176 | [yelp_review_full/yelp_review_full](src/tasksource/tasks.py#L621) | Classification | [Yelp/yelp_review_full](https://hf.co/datasets/Yelp/yelp_review_full) | ✓ |
| 177 | [financial_phrasebank/sentences_allagree](src/tasksource/tasks.py#L626) | Classification | [ghbacct/financial-phrasebank-all-agree-classification](https://hf.co/datasets/ghbacct/financial-phrasebank-all-agree-classification) |  |
| 178 | [poem_sentiment](src/tasksource/tasks.py#L631) | Classification | [google-research-datasets/poem_sentiment](https://hf.co/datasets/google-research-datasets/poem_sentiment) |  |
| 179 | [emotion](src/tasksource/tasks.py#L633) | Classification | [dair-ai/emotion](https://hf.co/datasets/dair-ai/emotion) |  |
| 180 | [dbpedia_14/dbpedia_14](src/tasksource/tasks.py#L635) | Classification | [fancyzhx/dbpedia_14](https://hf.co/datasets/fancyzhx/dbpedia_14) |  |
| 181 | [amazon_polarity/amazon_polarity](src/tasksource/tasks.py#L637) | Classification | [fancyzhx/amazon_polarity](https://hf.co/datasets/fancyzhx/amazon_polarity) |  |
| 182 | [app_reviews](src/tasksource/tasks.py#L639) | Classification | [sealuzh/app_reviews](https://hf.co/datasets/sealuzh/app_reviews) | ✓ |
| 183 | [hate_speech18](src/tasksource/tasks.py#L644) | Classification | [tasksource/hate_speech18](https://hf.co/datasets/tasksource/hate_speech18) |  |
| 184 | [sms_spam](src/tasksource/tasks.py#L650) | Classification | [ucirvine/sms_spam](https://hf.co/datasets/ucirvine/sms_spam) |  |
| 185 | [humicroedit/subtask-1](src/tasksource/tasks.py#L653) | Classification | [tasksource/humicroedit](https://hf.co/datasets/tasksource/humicroedit) | ✓ |
| 186 | [humicroedit/subtask-2](src/tasksource/tasks.py#L659) | Classification | [tasksource/humicroedit](https://hf.co/datasets/tasksource/humicroedit) | ✓ |
| 187 | [snips_built_in_intents](src/tasksource/tasks.py#L664) | Classification | [sonos-nlu-benchmark/snips_built_in_intents](https://hf.co/datasets/sonos-nlu-benchmark/snips_built_in_intents) |  |
| 188 | [hate_speech_offensive](src/tasksource/tasks.py#L668) | Classification | [tdavidson/hate_speech_offensive](https://hf.co/datasets/tdavidson/hate_speech_offensive) |  |
| 189 | [yahoo_answers_topics](src/tasksource/tasks.py#L670) | Classification | [community-datasets/yahoo_answers_topics](https://hf.co/datasets/community-datasets/yahoo_answers_topics) |  |
| 190 | [stackoverflow-questions](src/tasksource/tasks.py#L674) | Classification | [pacovaldez/stackoverflow-questions](https://hf.co/datasets/pacovaldez/stackoverflow-questions) | ✓ |
| 191 | [hyperpartisan_news](src/tasksource/tasks.py#L680) | Classification | [zapsdcn/hyperpartisan_news](https://hf.co/datasets/zapsdcn/hyperpartisan_news) |  |
| 192 | [sciie](src/tasksource/tasks.py#L685) | Classification | [zapsdcn/sciie](https://hf.co/datasets/zapsdcn/sciie) |  |
| 193 | [citation_intent](src/tasksource/tasks.py#L686) | Classification | [zapsdcn/citation_intent](https://hf.co/datasets/zapsdcn/citation_intent) |  |
| 194 | [go_emotions/simplified](src/tasksource/tasks.py#L688) | Classification | [google-research-datasets/go_emotions](https://hf.co/datasets/google-research-datasets/go_emotions) |  |
| 195 | [scicite](src/tasksource/tasks.py#L692) | Classification | [tasksource/scicite](https://hf.co/datasets/tasksource/scicite) |  |
| 196 | [liar](src/tasksource/tasks.py#L694) | Classification | [tasksource/liar](https://hf.co/datasets/tasksource/liar) | ✓ |
| 197 | [lexical_relation_classification/ROOT09](src/tasksource/tasks.py#L705) | Classification | json | ✓ |
| 198 | [lexical_relation_classification/K&H+N](src/tasksource/tasks.py#L705) | Classification | json | ✓ |
| 199 | [lexical_relation_classification/EVALution](src/tasksource/tasks.py#L705) | Classification | json | ✓ |
| 200 | [lexical_relation_classification/BLESS](src/tasksource/tasks.py#L705) | Classification | json | ✓ |
| 201 | [lexical_relation_classification/CogALexV](src/tasksource/tasks.py#L731) | Classification | json | ✓ |
| 202 | [linguisticprobing/subj_number](src/tasksource/tasks.py#L758) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 203 | [linguisticprobing/obj_number](src/tasksource/tasks.py#L759) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 204 | [linguisticprobing/past_present](src/tasksource/tasks.py#L760) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 205 | [linguisticprobing/sentence_length](src/tasksource/tasks.py#L761) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | ✓ |
| 206 | [linguisticprobing/top_constituents](src/tasksource/tasks.py#L762) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | ✓ |
| 207 | [linguisticprobing/tree_depth](src/tasksource/tasks.py#L764) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | ✓ |
| 208 | [linguisticprobing/coordination_inversion](src/tasksource/tasks.py#L765) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | ✓ |
| 209 | [linguisticprobing/odd_man_out](src/tasksource/tasks.py#L767) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | ✓ |
| 210 | [linguisticprobing/bigram_shift](src/tasksource/tasks.py#L768) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | ✓ |
| 211 | [crowdflower/political-media-bias](src/tasksource/tasks.py#L770) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 212 | [crowdflower/political-media-audience](src/tasksource/tasks.py#L770) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 213 | [crowdflower/political-media-message](src/tasksource/tasks.py#L770) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 214 | [crowdflower/economic-news](src/tasksource/tasks.py#L770) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 215 | [crowdflower/text_emotion](src/tasksource/tasks.py#L770) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 216 | [crowdflower/airline-sentiment](src/tasksource/tasks.py#L770) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 217 | [crowdflower/corporate-messaging](src/tasksource/tasks.py#L770) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 218 | [crowdflower/tweet_global_warming](src/tasksource/tasks.py#L770) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 219 | [crowdflower/sentiment_nuclear_power](src/tasksource/tasks.py#L770) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 220 | [ethics/commonsense](src/tasksource/tasks.py#L806) | Classification | csv |  |
| 221 | [ethics/deontology](src/tasksource/tasks.py#L814) | Classification | csv | ✓ |
| 222 | [ethics/justice](src/tasksource/tasks.py#L823) | Classification | csv |  |
| 223 | [ethics/virtue](src/tasksource/tasks.py#L831) | Classification | [hendrycks/ethics](https://hf.co/datasets/hendrycks/ethics) |  |
| 224 | [emo/emo2019](src/tasksource/tasks.py#L840) | Classification | [oneonlee/cleansed_emocontext](https://hf.co/datasets/oneonlee/cleansed_emocontext) |  |
| 225 | [google_wellformed_query](src/tasksource/tasks.py#L846) | Classification | [tasksource/google_wellformed_query](https://hf.co/datasets/tasksource/google_wellformed_query) | ✓ |
| 226 | [tweets_hate_speech_detection](src/tasksource/tasks.py#L852) | Classification | [tweets-hate-speech-detection/tweets_hate_speech_detection](https://hf.co/datasets/tweets-hate-speech-detection/tweets_hate_speech_detection) |  |
| 227 | [wnut_17/wnut_17](src/tasksource/tasks.py#L856) | TokenClassification | [flaitenberger/wnut_17](https://hf.co/datasets/flaitenberger/wnut_17) |  |
| 228 | [ncbi_disease/ncbi_disease](src/tasksource/tasks.py#L859) | TokenClassification | [ncbi/ncbi_disease](https://hf.co/datasets/ncbi/ncbi_disease) |  |
| 229 | [acronym_identification](src/tasksource/tasks.py#L862) | TokenClassification | [amirveyseh/acronym_identification](https://hf.co/datasets/amirveyseh/acronym_identification) |  |
| 230 | [jnlpba/jnlpba](src/tasksource/tasks.py#L865) | TokenClassification | [jnlpba/jnlpba](https://hf.co/datasets/jnlpba/jnlpba) |  |
| 231 | [ontonotes_english/SpeedOfMagic--ontonotes_english](src/tasksource/tasks.py#L872) | TokenClassification | [SpeedOfMagic/ontonotes_english](https://hf.co/datasets/SpeedOfMagic/ontonotes_english) |  |
| 232 | [blog_authorship_corpus/gender](src/tasksource/tasks.py#L876) | Classification | [tasksource/blog_authorship_corpus](https://hf.co/datasets/tasksource/blog_authorship_corpus) | ✓ |
| 233 | [blog_authorship_corpus/age](src/tasksource/tasks.py#L878) | Classification | [tasksource/blog_authorship_corpus](https://hf.co/datasets/tasksource/blog_authorship_corpus) | ✓ |
| 234 | [blog_authorship_corpus/job](src/tasksource/tasks.py#L881) | Classification | [tasksource/blog_authorship_corpus](https://hf.co/datasets/tasksource/blog_authorship_corpus) | ✓ |
| 235 | [open_question_type](src/tasksource/tasks.py#L892) | Classification | [Korea-MES/open_question_type](https://hf.co/datasets/Korea-MES/open_question_type) |  |
| 236 | [health_fact](src/tasksource/tasks.py#L894) | Classification | [marcov/health_fact_promptsource](https://hf.co/datasets/marcov/health_fact_promptsource) |  |
| 237 | [commonsense_qa](src/tasksource/tasks.py#L898) | MultipleChoice | [tau/commonsense_qa](https://hf.co/datasets/tau/commonsense_qa) |  |
| 238 | [mc_taco](src/tasksource/tasks.py#L904) | Classification | [marcov/mc_taco_promptsource](https://hf.co/datasets/marcov/mc_taco_promptsource) | ✓ |
| 239 | [ade_corpus_v2/Ade_corpus_v2_classification](src/tasksource/tasks.py#L911) | Classification | [ade-benchmark-corpus/ade_corpus_v2](https://hf.co/datasets/ade-benchmark-corpus/ade_corpus_v2) | ✓ |
| 240 | [discosense](src/tasksource/tasks.py#L914) | MultipleChoice | json |  |
| 241 | [circa](src/tasksource/tasks.py#L921) | Classification | [google-research-datasets/circa](https://hf.co/datasets/google-research-datasets/circa) |  |
| 242 | [code_x_glue_cc_defect_detection](src/tasksource/tasks.py#L926) | Classification | [google/code_x_glue_cc_defect_detection](https://hf.co/datasets/google/code_x_glue_cc_defect_detection) | ✓ |
| 243 | [phrase_similarity](src/tasksource/tasks.py#L931) | Classification | [Deehan1866/processed_phrase_similarity](https://hf.co/datasets/Deehan1866/processed_phrase_similarity) |  |
| 244 | [scientific-exaggeration-detection](src/tasksource/tasks.py#L939) | Classification | [copenlu/scientific-exaggeration-detection](https://hf.co/datasets/copenlu/scientific-exaggeration-detection) |  |
| 245 | [quarel](src/tasksource/tasks.py#L953) | MultipleChoice | [community-datasets/quarel](https://hf.co/datasets/community-datasets/quarel) | ✓ |
| 246 | [fever-evidence-related](src/tasksource/tasks.py#L960) | Classification | [mwong/fever-evidence-related](https://hf.co/datasets/mwong/fever-evidence-related) |  |
| 247 | [numer_sense](src/tasksource/tasks.py#L963) | Classification | [tasksource/numer_sense](https://hf.co/datasets/tasksource/numer_sense) |  |
| 248 | [dynasent/dynabench.dynasent.r1.all/r1](src/tasksource/tasks.py#L970) | Classification | [tasksource/dynasent](https://hf.co/datasets/tasksource/dynasent) |  |
| 249 | [dynasent/dynabench.dynasent.r2.all/r2](src/tasksource/tasks.py#L974) | Classification | [tasksource/dynasent](https://hf.co/datasets/tasksource/dynasent) |  |
| 250 | [Sarcasm_News_Headline](src/tasksource/tasks.py#L979) | Classification | [raquiba/Sarcasm_News_Headline](https://hf.co/datasets/raquiba/Sarcasm_News_Headline) |  |
| 251 | [sem_eval_2010_task_8](src/tasksource/tasks.py#L982) | Classification | [SemEvalWorkshop/sem_eval_2010_task_8](https://hf.co/datasets/SemEvalWorkshop/sem_eval_2010_task_8) |  |
| 252 | [auditor_review](src/tasksource/tasks.py#L984) | Classification | [demo-org/auditor_review](https://hf.co/datasets/demo-org/auditor_review) |  |
| 253 | [medmcqa](src/tasksource/tasks.py#L988) | MultipleChoice | [openlifescienceai/medmcqa](https://hf.co/datasets/openlifescienceai/medmcqa) |  |
| 254 | [Dynasent_Disagreement](src/tasksource/tasks.py#L1005) | Classification | [RuyuanWan/Dynasent_Disagreement](https://hf.co/datasets/RuyuanWan/Dynasent_Disagreement) | ✓ |
| 255 | [Politeness_Disagreement](src/tasksource/tasks.py#L1007) | Classification | [RuyuanWan/Politeness_Disagreement](https://hf.co/datasets/RuyuanWan/Politeness_Disagreement) | ✓ |
| 256 | [SBIC_Disagreement](src/tasksource/tasks.py#L1009) | Classification | [RuyuanWan/SBIC_Disagreement](https://hf.co/datasets/RuyuanWan/SBIC_Disagreement) | ✓ |
| 257 | [SChem_Disagreement](src/tasksource/tasks.py#L1011) | Classification | [RuyuanWan/SChem_Disagreement](https://hf.co/datasets/RuyuanWan/SChem_Disagreement) | ✓ |
| 258 | [Dilemmas_Disagreement](src/tasksource/tasks.py#L1013) | Classification | [RuyuanWan/Dilemmas_Disagreement](https://hf.co/datasets/RuyuanWan/Dilemmas_Disagreement) | ✓ |
| 259 | [logiqa](src/tasksource/tasks.py#L1016) | MultipleChoice | [fireworks-ai/logiqa](https://hf.co/datasets/fireworks-ai/logiqa) |  |
| 260 | [wiki_qa](src/tasksource/tasks.py#L1025) | Classification | [microsoft/wiki_qa](https://hf.co/datasets/microsoft/wiki_qa) | ✓ |
| 261 | [cycic_classification](src/tasksource/tasks.py#L1027) | Classification | [tasksource/cycic_classification](https://hf.co/datasets/tasksource/cycic_classification) |  |
| 262 | [cycic_multiplechoice](src/tasksource/tasks.py#L1029) | MultipleChoice | [tasksource/cycic_multiplechoice](https://hf.co/datasets/tasksource/cycic_multiplechoice) |  |
| 263 | [sts-companion](src/tasksource/tasks.py#L1033) | Classification | [tasksource/sts-companion](https://hf.co/datasets/tasksource/sts-companion) |  |
| 264 | [commonsense_qa_2.0](src/tasksource/tasks.py#L1036) | Classification | [tasksource/commonsense_qa_2.0](https://hf.co/datasets/tasksource/commonsense_qa_2.0) |  |
| 265 | [lingnli](src/tasksource/tasks.py#L1039) | Classification | [tasksource/lingnli](https://hf.co/datasets/tasksource/lingnli) |  |
| 266 | [monotonicity-entailment](src/tasksource/tasks.py#L1041) | Classification | [tasksource/monotonicity-entailment](https://hf.co/datasets/tasksource/monotonicity-entailment) |  |
| 267 | [arct](src/tasksource/tasks.py#L1044) | MultipleChoice | [tasksource/arct](https://hf.co/datasets/tasksource/arct) |  |
| 268 | [scinli](src/tasksource/tasks.py#L1047) | Classification | [tasksource/scinli](https://hf.co/datasets/tasksource/scinli) |  |
| 269 | [naturallogic](src/tasksource/tasks.py#L1051) | Classification | [tasksource/naturallogic](https://hf.co/datasets/tasksource/naturallogic) |  |
| 270 | [onestop_qa](src/tasksource/tasks.py#L1053) | MultipleChoice | [malmaud/onestop_qa](https://hf.co/datasets/malmaud/onestop_qa) |  |
| 271 | [moral_stories/full](src/tasksource/tasks.py#L1056) | MultipleChoice | [LabHC/moral_stories](https://hf.co/datasets/LabHC/moral_stories) |  |
| 272 | [prost](src/tasksource/tasks.py#L1064) | MultipleChoice | json |  |
| 273 | [dynahate](src/tasksource/tasks.py#L1069) | Classification | [tasksource/dynahate](https://hf.co/datasets/tasksource/dynahate) |  |
| 274 | [syntactic-augmentation-nli](src/tasksource/tasks.py#L1071) | Classification | [tasksource/syntactic-augmentation-nli](https://hf.co/datasets/tasksource/syntactic-augmentation-nli) |  |
| 275 | [autotnli](src/tasksource/tasks.py#L1073) | Classification | [tasksource/autotnli](https://hf.co/datasets/tasksource/autotnli) |  |
| 276 | [CONDAQA](src/tasksource/tasks.py#L1075) | Classification | [lasha-nlp/CONDAQA](https://hf.co/datasets/lasha-nlp/CONDAQA) |  |
| 277 | [webgpt_comparisons](src/tasksource/tasks.py#L1085) | MultipleChoice | [heegyu/webgpt_comparisons_ko](https://hf.co/datasets/heegyu/webgpt_comparisons_ko) | ✓ |
| 278 | [synthetic-instruct-gptj-pairwise](src/tasksource/tasks.py#L1093) | MultipleChoice | [Dahoas/synthetic-instruct-gptj-pairwise](https://hf.co/datasets/Dahoas/synthetic-instruct-gptj-pairwise) | ✓ |
| 279 | [scruples](src/tasksource/tasks.py#L1096) | Classification | [tasksource/scruples](https://hf.co/datasets/tasksource/scruples) | ✓ |
| 280 | [wouldyourather](src/tasksource/tasks.py#L1099) | MultipleChoice | [tasksource/wouldyourather](https://hf.co/datasets/tasksource/wouldyourather) | ✓ |
| 281 | [defeasible-nli/snli](src/tasksource/tasks.py#L1107) | Classification | [tasksource/defeasible-nli](https://hf.co/datasets/tasksource/defeasible-nli) |  |
| 282 | [defeasible-nli/atomic](src/tasksource/tasks.py#L1107) | Classification | [tasksource/defeasible-nli](https://hf.co/datasets/tasksource/defeasible-nli) |  |
| 283 | [defeasible-nli/social](src/tasksource/tasks.py#L1110) | Classification | [tasksource/defeasible-nli](https://hf.co/datasets/tasksource/defeasible-nli) |  |
| 284 | [help-nli](src/tasksource/tasks.py#L1113) | Classification | [tasksource/help-nli](https://hf.co/datasets/tasksource/help-nli) |  |
| 285 | [nli-veridicality-transitivity](src/tasksource/tasks.py#L1116) | Classification | [tasksource/nli-veridicality-transitivity](https://hf.co/datasets/tasksource/nli-veridicality-transitivity) |  |
| 286 | [lonli](src/tasksource/tasks.py#L1119) | Classification | [tasksource/lonli](https://hf.co/datasets/tasksource/lonli) |  |
| 287 | [dadc-limit-nli](src/tasksource/tasks.py#L1122) | Classification | [tasksource/dadc-limit-nli](https://hf.co/datasets/tasksource/dadc-limit-nli) |  |
| 288 | [FLUTE](src/tasksource/tasks.py#L1125) | Classification | [ColumbiaNLP/FLUTE](https://hf.co/datasets/ColumbiaNLP/FLUTE) |  |
| 289 | [strategy-qa](src/tasksource/tasks.py#L1128) | Classification | [tasksource/strategy-qa](https://hf.co/datasets/tasksource/strategy-qa) |  |
| 290 | [summarize_from_feedback/comparisons](src/tasksource/tasks.py#L1131) | MultipleChoice | [vwxyzjn/summarize_from_feedback_oai_preprocessing](https://hf.co/datasets/vwxyzjn/summarize_from_feedback_oai_preprocessing) | ✓ |
| 291 | [folio](src/tasksource/tasks.py#L1139) | Classification | [tasksource/folio](https://hf.co/datasets/tasksource/folio) |  |
| 292 | [tomi-nli](src/tasksource/tasks.py#L1166) | Classification | [tasksource/tomi-nli](https://hf.co/datasets/tasksource/tomi-nli) |  |
| 293 | [avicenna](src/tasksource/tasks.py#L1169) | Classification | [tasksource/avicenna](https://hf.co/datasets/tasksource/avicenna) | ✓ |
| 294 | [SHP](src/tasksource/tasks.py#L1172) | MultipleChoice | [stanfordnlp/SHP](https://hf.co/datasets/stanfordnlp/SHP) | ✓ |
| 295 | [MedQA-USMLE-4-options-hf](src/tasksource/tasks.py#L1180) | MultipleChoice | [GBaker/MedQA-USMLE-4-options-hf](https://hf.co/datasets/GBaker/MedQA-USMLE-4-options-hf) |  |
| 296 | [wikimedqa/medwiki](src/tasksource/tasks.py#L1183) | MultipleChoice | [sileod/wikimedqa](https://hf.co/datasets/sileod/wikimedqa) |  |
| 297 | [cicero](src/tasksource/tasks.py#L1192) | MultipleChoice | [declare-lab/cicero](https://hf.co/datasets/declare-lab/cicero) |  |
| 298 | [CREAK](src/tasksource/tasks.py#L1196) | Classification | [amydeng2000/CREAK](https://hf.co/datasets/amydeng2000/CREAK) |  |
| 299 | [mutual](src/tasksource/tasks.py#L1199) | MultipleChoice | [tasksource/mutual](https://hf.co/datasets/tasksource/mutual) |  |
| 300 | [puzzte](src/tasksource/tasks.py#L1203) | Classification | [tasksource/puzzte](https://hf.co/datasets/tasksource/puzzte) |  |
| 301 | [implicatures](src/tasksource/tasks.py#L1208) | MultipleChoice | [tasksource/implicatures](https://hf.co/datasets/tasksource/implicatures) |  |
| 302 | [race/middle](src/tasksource/tasks.py#L1213) | MultipleChoice | [ehovy/race](https://hf.co/datasets/ehovy/race) |  |
| 303 | [race/high](src/tasksource/tasks.py#L1213) | MultipleChoice | [ehovy/race](https://hf.co/datasets/ehovy/race) |  |
| 304 | [race-c](src/tasksource/tasks.py#L1217) | MultipleChoice | [tasksource/race-c](https://hf.co/datasets/tasksource/race-c) |  |
| 305 | [spartqa-yn](src/tasksource/tasks.py#L1220) | Classification | [tasksource/spartqa-yn](https://hf.co/datasets/tasksource/spartqa-yn) |  |
| 306 | [spartqa-mchoice](src/tasksource/tasks.py#L1223) | MultipleChoice | [tasksource/spartqa-mchoice](https://hf.co/datasets/tasksource/spartqa-mchoice) |  |
| 307 | [temporal-nli](src/tasksource/tasks.py#L1226) | Classification | [tasksource/temporal-nli](https://hf.co/datasets/tasksource/temporal-nli) |  |
| 308 | [riddle_sense](src/tasksource/tasks.py#L1229) | MultipleChoice | [jeggers/riddle_sense](https://hf.co/datasets/jeggers/riddle_sense) |  |
| 309 | [clcd-english](src/tasksource/tasks.py#L1234) | Classification | [tasksource/clcd-english](https://hf.co/datasets/tasksource/clcd-english) |  |
| 310 | [twentyquestions](src/tasksource/tasks.py#L1246) | Classification | [tasksource/twentyquestions](https://hf.co/datasets/tasksource/twentyquestions) |  |
| 311 | [reclor](src/tasksource/tasks.py#L1251) | MultipleChoice | [tasksource/reclor](https://hf.co/datasets/tasksource/reclor) |  |
| 312 | [counterfactually-augmented-imdb](src/tasksource/tasks.py#L1254) | Classification | [tasksource/counterfactually-augmented-imdb](https://hf.co/datasets/tasksource/counterfactually-augmented-imdb) |  |
| 313 | [counterfactually-augmented-snli](src/tasksource/tasks.py#L1257) | Classification | [tasksource/counterfactually-augmented-snli](https://hf.co/datasets/tasksource/counterfactually-augmented-snli) |  |
| 314 | [cnli](src/tasksource/tasks.py#L1260) | Classification | [tasksource/cnli](https://hf.co/datasets/tasksource/cnli) |  |
| 315 | [boolq-natural-perturbations](src/tasksource/tasks.py#L1263) | Classification | [tasksource/boolq-natural-perturbations](https://hf.co/datasets/tasksource/boolq-natural-perturbations) |  |
| 316 | [acceptability-prediction](src/tasksource/tasks.py#L1268) | Classification | [tasksource/acceptability-prediction](https://hf.co/datasets/tasksource/acceptability-prediction) | ✓ |
| 317 | [equate](src/tasksource/tasks.py#L1294) | Classification | [tasksource/equate](https://hf.co/datasets/tasksource/equate) |  |
| 318 | [ScienceQA_text_only](src/tasksource/tasks.py#L1297) | MultipleChoice | [tasksource/ScienceQA_text_only](https://hf.co/datasets/tasksource/ScienceQA_text_only) |  |
| 319 | [ekar_english](src/tasksource/tasks.py#L1300) | MultipleChoice | [Jiangjie/ekar_english](https://hf.co/datasets/Jiangjie/ekar_english) | ✓ |
| 320 | [implicit-hate-stg1](src/tasksource/tasks.py#L1304) | Classification | [tasksource/implicit-hate-stg1](https://hf.co/datasets/tasksource/implicit-hate-stg1) | ✓ |
| 321 | [chaos-mnli-ambiguity](src/tasksource/tasks.py#L1308) | Classification | [tasksource/chaos-mnli-ambiguity](https://hf.co/datasets/tasksource/chaos-mnli-ambiguity) | ✓ |
| 322 | [headline_cause/en_simple](src/tasksource/tasks.py#L1312) | Classification | json |  |
| 323 | [logiqa-2.0-nli](src/tasksource/tasks.py#L1317) | Classification | [tasksource/logiqa-2.0-nli](https://hf.co/datasets/tasksource/logiqa-2.0-nli) |  |
| 324 | [oasst2_dense_flat/quality](src/tasksource/tasks.py#L1322) | Classification | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) | ✓ |
| 325 | [oasst2_dense_flat/toxicity](src/tasksource/tasks.py#L1324) | Classification | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) | ✓ |
| 326 | [oasst2_dense_flat/helpfulness](src/tasksource/tasks.py#L1326) | Classification | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) | ✓ |
| 327 | [mindgames](src/tasksource/tasks.py#L1329) | Classification | [sileod/mindgames](https://hf.co/datasets/sileod/mindgames) |  |
| 328 | [universal_dependencies/en_gum/deprel](src/tasksource/tasks.py#L1343) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| 329 | [universal_dependencies/en_ewt/deprel](src/tasksource/tasks.py#L1343) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| 330 | [universal_dependencies/en_lines/deprel](src/tasksource/tasks.py#L1343) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| 331 | [universal_dependencies/en_partut/deprel](src/tasksource/tasks.py#L1343) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| 332 | [ambient](src/tasksource/tasks.py#L1349) | Classification | [tasksource/ambient](https://hf.co/datasets/tasksource/ambient) | ✓ |
| 333 | [path-naturalness-prediction](src/tasksource/tasks.py#L1354) | MultipleChoice | [tasksource/path-naturalness-prediction](https://hf.co/datasets/tasksource/path-naturalness-prediction) | ✓ |
| 334 | [civil_comments/toxicity](src/tasksource/tasks.py#L1364) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 335 | [civil_comments/severe_toxicity](src/tasksource/tasks.py#L1365) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 336 | [civil_comments/obscene](src/tasksource/tasks.py#L1366) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 337 | [civil_comments/threat](src/tasksource/tasks.py#L1367) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 338 | [civil_comments/insult](src/tasksource/tasks.py#L1368) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 339 | [civil_comments/identity_attack](src/tasksource/tasks.py#L1369) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 340 | [civil_comments/sexual_explicit](src/tasksource/tasks.py#L1370) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 341 | [cloth](src/tasksource/tasks.py#L1372) | MultipleChoice | [AndyChiang/cloth](https://hf.co/datasets/AndyChiang/cloth) |  |
| 342 | [dgen](src/tasksource/tasks.py#L1373) | MultipleChoice | [AndyChiang/dgen](https://hf.co/datasets/AndyChiang/dgen) |  |
| 343 | [I2D2](src/tasksource/tasks.py#L1375) | Classification | [tasksource/I2D2](https://hf.co/datasets/tasksource/I2D2) | ✓ |
| 344 | [args_me](src/tasksource/tasks.py#L1380) | Classification | [webis/args_me](https://hf.co/datasets/webis/args_me) | ✓ |
| 345 | [Touche23-ValueEval](src/tasksource/tasks.py#L1385) | Classification | csv | ✓ |
| 346 | [starcon](src/tasksource/tasks.py#L1394) | Classification | [tasksource/starcon](https://hf.co/datasets/tasksource/starcon) | ✓ |
| 347 | [banking77](src/tasksource/tasks.py#L1399) | Classification | [legacy-datasets/banking77](https://hf.co/datasets/legacy-datasets/banking77) |  |
| 348 | [it-support-tickets](src/tasksource/tasks.py#L1401) | Classification | [tasksource/it-support-tickets](https://hf.co/datasets/tasksource/it-support-tickets) |  |
| 349 | [ConTRoL-nli](src/tasksource/tasks.py#L1405) | Classification | [tasksource/ConTRoL-nli](https://hf.co/datasets/tasksource/ConTRoL-nli) |  |
| 350 | [tracie](src/tasksource/tasks.py#L1407) | Classification | [tasksource/tracie](https://hf.co/datasets/tasksource/tracie) | ✓ |
| 351 | [sherliic](src/tasksource/tasks.py#L1409) | Classification | [tasksource/sherliic](https://hf.co/datasets/tasksource/sherliic) |  |
| 352 | [sen-making/1](src/tasksource/tasks.py#L1411) | MultipleChoice | [tasksource/sen-making](https://hf.co/datasets/tasksource/sen-making) | ✓ |
| 353 | [sen-making/2](src/tasksource/tasks.py#L1415) | MultipleChoice | [tasksource/sen-making](https://hf.co/datasets/tasksource/sen-making) | ✓ |
| 354 | [winowhy](src/tasksource/tasks.py#L1418) | Classification | [tasksource/winowhy](https://hf.co/datasets/tasksource/winowhy) | ✓ |
| 355 | [robustLR](src/tasksource/tasks.py#L1422) | Classification | [tasksource/robustLR](https://hf.co/datasets/tasksource/robustLR) |  |
| 356 | [clutrr](src/tasksource/tasks.py#L1424) | Classification | [tasksource/clutrr](https://hf.co/datasets/tasksource/clutrr) |  |
| 357 | [logical-fallacy](src/tasksource/tasks.py#L1426) | Classification | [tasksource/logical-fallacy](https://hf.co/datasets/tasksource/logical-fallacy) |  |
| 358 | [parade](src/tasksource/tasks.py#L1428) | Classification | [tasksource/parade](https://hf.co/datasets/tasksource/parade) |  |
| 359 | [subjectivity](src/tasksource/tasks.py#L1431) | Classification | [tasksource/subjectivity](https://hf.co/datasets/tasksource/subjectivity) | ✓ |
| 360 | [MOH](src/tasksource/tasks.py#L1437) | Classification | [tasksource/MOH](https://hf.co/datasets/tasksource/MOH) | ✓ |
| 361 | [VUAC](src/tasksource/tasks.py#L1438) | Classification | [tasksource/VUAC](https://hf.co/datasets/tasksource/VUAC) | ✓ |
| 362 | [TroFi](src/tasksource/tasks.py#L1439) | Classification | parquet | ✓ |
| 363 | [sharc](src/tasksource/tasks.py#L1447) | Classification | [tasksource/sharc](https://hf.co/datasets/tasksource/sharc) |  |
| 364 | [conceptrules_v2](src/tasksource/tasks.py#L1451) | Classification | [tasksource/conceptrules_v2](https://hf.co/datasets/tasksource/conceptrules_v2) | ✓ |
| 365 | [disrpt/eng.dep.scidtb.rels](src/tasksource/tasks.py#L1453) | Classification | [multilingual-discourse-hub/disrpt](https://hf.co/datasets/multilingual-discourse-hub/disrpt) | ✓ |
| 366 | [conll2000](src/tasksource/tasks.py#L1456) | TokenClassification | [eriktks/conll2000](https://hf.co/datasets/eriktks/conll2000) |  |
| 367 | [few-nerd/supervised](src/tasksource/tasks.py#L1459) | TokenClassification | [DFKI-SLT/few-nerd](https://hf.co/datasets/DFKI-SLT/few-nerd) |  |
| 368 | [finer-139](src/tasksource/tasks.py#L1460) | TokenClassification | [nlpaueb/finer-139](https://hf.co/datasets/nlpaueb/finer-139) |  |
| 369 | [zero-shot-label-nli](src/tasksource/tasks.py#L1463) | Classification | [tasksource/zero-shot-label-nli](https://hf.co/datasets/tasksource/zero-shot-label-nli) |  |
| 370 | [com2sense](src/tasksource/tasks.py#L1465) | Classification | [tasksource/com2sense](https://hf.co/datasets/tasksource/com2sense) |  |
| 371 | [scone](src/tasksource/tasks.py#L1467) | Classification | [tasksource/scone](https://hf.co/datasets/tasksource/scone) |  |
| 372 | [winodict](src/tasksource/tasks.py#L1469) | MultipleChoice | [tasksource/winodict](https://hf.co/datasets/tasksource/winodict) |  |
| 373 | [fool-me-twice](src/tasksource/tasks.py#L1471) | Classification | [tasksource/fool-me-twice](https://hf.co/datasets/tasksource/fool-me-twice) |  |
| 374 | [monli](src/tasksource/tasks.py#L1475) | Classification | [tasksource/monli](https://hf.co/datasets/tasksource/monli) |  |
| 375 | [corr2cause](src/tasksource/tasks.py#L1477) | Classification | [tasksource/corr2cause](https://hf.co/datasets/tasksource/corr2cause) |  |
| 376 | [lsat_qa/all](src/tasksource/tasks.py#L1479) | MultipleChoice | [lighteval/lsat_qa](https://hf.co/datasets/lighteval/lsat_qa) |  |
| 377 | [apt](src/tasksource/tasks.py#L1481) | Classification | [tasksource/apt](https://hf.co/datasets/tasksource/apt) |  |
| 378 | [twitter-financial-news-sentiment](src/tasksource/tasks.py#L1484) | Classification | [zeroshot/twitter-financial-news-sentiment](https://hf.co/datasets/zeroshot/twitter-financial-news-sentiment) |  |
| 379 | [icl-symbol-tuning-instruct](src/tasksource/tasks.py#L1491) | Classification | [tasksource/icl-symbol-tuning-instruct](https://hf.co/datasets/tasksource/icl-symbol-tuning-instruct) | ✓ |
| 380 | [SpaceNLI](src/tasksource/tasks.py#L1497) | Classification | [tasksource/SpaceNLI](https://hf.co/datasets/tasksource/SpaceNLI) |  |
| 381 | [propsegment/nli](src/tasksource/tasks.py#L1499) | Classification | json |  |
| 382 | [HatemojiBuild](src/tasksource/tasks.py#L1508) | Classification | [HannahRoseKirk/HatemojiBuild](https://hf.co/datasets/HannahRoseKirk/HatemojiBuild) |  |
| 383 | [regset](src/tasksource/tasks.py#L1511) | Classification | [tasksource/regset](https://hf.co/datasets/tasksource/regset) | ✓ |
| 384 | [esci](src/tasksource/tasks.py#L1517) | Classification | [tasksource/esci](https://hf.co/datasets/tasksource/esci) |  |
| 385 | [chatbot_arena_conversations](src/tasksource/tasks.py#L1536) | MultipleChoice | [lmsys/chatbot_arena_conversations](https://hf.co/datasets/lmsys/chatbot_arena_conversations) | ✓ |
| 386 | [dnd_style_intents](src/tasksource/tasks.py#L1542) | Classification | [neurae/dnd_style_intents](https://hf.co/datasets/neurae/dnd_style_intents) |  |
| 387 | [FLD.v2/default](src/tasksource/tasks.py#L1545) | Classification | [hitachi-nlp/FLD.v2](https://hf.co/datasets/hitachi-nlp/FLD.v2) | ✓ |
| 388 | [FLD.v2/star](src/tasksource/tasks.py#L1549) | Classification | [hitachi-nlp/FLD.v2](https://hf.co/datasets/hitachi-nlp/FLD.v2) | ✓ |
| 389 | [SDOH-NLI](src/tasksource/tasks.py#L1553) | Classification | [tasksource/SDOH-NLI](https://hf.co/datasets/tasksource/SDOH-NLI) |  |
| 390 | [scifact_entailment](src/tasksource/tasks.py#L1556) | Classification | [tasksource/scifact_entailment](https://hf.co/datasets/tasksource/scifact_entailment) |  |
| 391 | [feasibilityQA](src/tasksource/tasks.py#L1560) | Classification | [tasksource/feasibilityQA](https://hf.co/datasets/tasksource/feasibilityQA) |  |
| 392 | [simple_pair](src/tasksource/tasks.py#L1563) | Classification | [tasksource/simple_pair](https://hf.co/datasets/tasksource/simple_pair) |  |
| 393 | [AdjectiveScaleProbe-nli](src/tasksource/tasks.py#L1564) | Classification | [tasksource/AdjectiveScaleProbe-nli](https://hf.co/datasets/tasksource/AdjectiveScaleProbe-nli) |  |
| 394 | [resnli](src/tasksource/tasks.py#L1565) | Classification | [tasksource/resnli](https://hf.co/datasets/tasksource/resnli) |  |
| 395 | [SpaRTUN](src/tasksource/tasks.py#L1567) | MultipleChoice | [tasksource/SpaRTUN](https://hf.co/datasets/tasksource/SpaRTUN) |  |
| 396 | [ReSQ](src/tasksource/tasks.py#L1572) | MultipleChoice | [tasksource/ReSQ](https://hf.co/datasets/tasksource/ReSQ) |  |
| 397 | [semantic_fragments_nli](src/tasksource/tasks.py#L1577) | Classification | [tasksource/semantic_fragments_nli](https://hf.co/datasets/tasksource/semantic_fragments_nli) |  |
| 398 | [dataset_train_nli](src/tasksource/tasks.py#L1580) | Classification | [MoritzLaurer/dataset_train_nli](https://hf.co/datasets/MoritzLaurer/dataset_train_nli) |  |
| 399 | [stepgame](src/tasksource/tasks.py#L1585) | Classification | [tasksource/stepgame](https://hf.co/datasets/tasksource/stepgame) |  |
| 400 | [nlgraph](src/tasksource/tasks.py#L1593) | Classification | [tasksource/nlgraph](https://hf.co/datasets/tasksource/nlgraph) |  |
| 401 | [oasst2_pairwise_rlhf_reward](src/tasksource/tasks.py#L1597) | MultipleChoice | [tasksource/oasst2_pairwise_rlhf_reward](https://hf.co/datasets/tasksource/oasst2_pairwise_rlhf_reward) | ✓ |
| 402 | [hh-rlhf/helpful-base](src/tasksource/tasks.py#L1608) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| 403 | [hh-rlhf/helpful-online](src/tasksource/tasks.py#L1608) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| 404 | [hh-rlhf/helpful-rejection-sampled](src/tasksource/tasks.py#L1608) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| 405 | [hh-rlhf/harmless-base](src/tasksource/tasks.py#L1612) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| 406 | [ruletaker](src/tasksource/tasks.py#L1616) | Classification | [tasksource/ruletaker](https://hf.co/datasets/tasksource/ruletaker) | ✓ |
| 407 | [PARARULE-Plus](src/tasksource/tasks.py#L1620) | Classification | [qbao775/PARARULE-Plus](https://hf.co/datasets/qbao775/PARARULE-Plus) | ✓ |
| 408 | [proofwriter](src/tasksource/tasks.py#L1624) | Classification | [tasksource/proofwriter](https://hf.co/datasets/tasksource/proofwriter) |  |
| 409 | [logical-entailment](src/tasksource/tasks.py#L1627) | Classification | [tasksource/logical-entailment](https://hf.co/datasets/tasksource/logical-entailment) | ✓ |
| 410 | [nope](src/tasksource/tasks.py#L1630) | Classification | [tasksource/nope](https://hf.co/datasets/tasksource/nope) |  |
| 411 | [LogicNLI](src/tasksource/tasks.py#L1634) | Classification | [tasksource/LogicNLI](https://hf.co/datasets/tasksource/LogicNLI) | ✓ |
| 412 | [contract-nli/contractnli_a/seg](src/tasksource/tasks.py#L1637) | Classification | [tasksource/contract-nli](https://hf.co/datasets/tasksource/contract-nli) |  |
| 413 | [contract-nli/contractnli_b/full](src/tasksource/tasks.py#L1639) | Classification | [tasksource/contract-nli](https://hf.co/datasets/tasksource/contract-nli) |  |
| 414 | [nli4ct_semeval2024](src/tasksource/tasks.py#L1641) | Classification | [AshtonIsNotHere/nli4ct_semeval2024](https://hf.co/datasets/AshtonIsNotHere/nli4ct_semeval2024) |  |
| 415 | [lsat-ar](src/tasksource/tasks.py#L1644) | MultipleChoice | [tasksource/lsat-ar](https://hf.co/datasets/tasksource/lsat-ar) |  |
| 416 | [lsat-rc](src/tasksource/tasks.py#L1649) | MultipleChoice | [tasksource/lsat-rc](https://hf.co/datasets/tasksource/lsat-rc) |  |
| 417 | [biosift-nli](src/tasksource/tasks.py#L1654) | Classification | [AshtonIsNotHere/biosift-nli](https://hf.co/datasets/AshtonIsNotHere/biosift-nli) |  |
| 418 | [brainteasers/SP](src/tasksource/tasks.py#L1658) | MultipleChoice | [tasksource/brainteasers](https://hf.co/datasets/tasksource/brainteasers) |  |
| 419 | [brainteasers/WP](src/tasksource/tasks.py#L1658) | MultipleChoice | [tasksource/brainteasers](https://hf.co/datasets/tasksource/brainteasers) |  |
| 420 | [toxigen-data/annotated](src/tasksource/tasks.py#L1664) | Classification | [skg/toxigen-data](https://hf.co/datasets/skg/toxigen-data) |  |
| 421 | [persuasion](src/tasksource/tasks.py#L1675) | Classification | [Anthropic/persuasion](https://hf.co/datasets/Anthropic/persuasion) |  |
| 422 | [AmbigNQ-clarifying-question](src/tasksource/tasks.py#L1681) | Classification | [erbacher/AmbigNQ-clarifying-question](https://hf.co/datasets/erbacher/AmbigNQ-clarifying-question) | ✓ |
| 423 | [SIGA-nli](src/tasksource/tasks.py#L1686) | Classification | [tasksource/SIGA-nli](https://hf.co/datasets/tasksource/SIGA-nli) |  |
| 424 | [FOL-nli](src/tasksource/tasks.py#L1688) | Classification | [unigram/FOL-nli](https://hf.co/datasets/unigram/FOL-nli) |  |
| 425 | [goal-step-wikihow/goal](src/tasksource/tasks.py#L1690) | MultipleChoice | [tasksource/goal-step-wikihow](https://hf.co/datasets/tasksource/goal-step-wikihow) |  |
| 426 | [goal-step-wikihow/step](src/tasksource/tasks.py#L1693) | MultipleChoice | [tasksource/goal-step-wikihow](https://hf.co/datasets/tasksource/goal-step-wikihow) |  |
| 427 | [goal-step-wikihow/order](src/tasksource/tasks.py#L1696) | MultipleChoice | [tasksource/goal-step-wikihow](https://hf.co/datasets/tasksource/goal-step-wikihow) | ✓ |
| 428 | [PARADISE](src/tasksource/tasks.py#L1700) | MultipleChoice | [GGLab/PARADISE](https://hf.co/datasets/GGLab/PARADISE) |  |
| 429 | [doc-nli](src/tasksource/tasks.py#L1703) | Classification | [tasksource/doc-nli](https://hf.co/datasets/tasksource/doc-nli) |  |
| 430 | [mctest-nli](src/tasksource/tasks.py#L1705) | Classification | [tasksource/mctest-nli](https://hf.co/datasets/tasksource/mctest-nli) |  |
| 431 | [patent-phrase-similarity](src/tasksource/tasks.py#L1707) | Classification | [tasksource/patent-phrase-similarity](https://hf.co/datasets/tasksource/patent-phrase-similarity) | ✓ |
| 432 | [natural-language-satisfiability](src/tasksource/tasks.py#L1710) | Classification | [tasksource/natural-language-satisfiability](https://hf.co/datasets/tasksource/natural-language-satisfiability) |  |
| 433 | [idioms-nli](src/tasksource/tasks.py#L1712) | Classification | [tasksource/idioms-nli](https://hf.co/datasets/tasksource/idioms-nli) |  |
| 434 | [lifecycle-entailment](src/tasksource/tasks.py#L1714) | Classification | [tasksource/lifecycle-entailment](https://hf.co/datasets/tasksource/lifecycle-entailment) |  |
| 435 | [safe-guard-prompt-injection](src/tasksource/tasks.py#L1721) | Classification | [xTRam1/safe-guard-prompt-injection](https://hf.co/datasets/xTRam1/safe-guard-prompt-injection) | ✓ |
| 436 | [prompt-injections](src/tasksource/tasks.py#L1727) | Classification | [deepset/prompt-injections](https://hf.co/datasets/deepset/prompt-injections) | ✓ |
| 437 | [prompt-injection-dataset](src/tasksource/tasks.py#L1733) | Classification | [S-Labs/prompt-injection-dataset](https://hf.co/datasets/S-Labs/prompt-injection-dataset) | ✓ |
| 438 | [Prompt-injection-dataset/full](src/tasksource/tasks.py#L1739) | Classification | [neuralchemy/Prompt-injection-dataset](https://hf.co/datasets/neuralchemy/Prompt-injection-dataset) | ✓ |
| 439 | [PromptShield](src/tasksource/tasks.py#L1745) | Classification | [hendzh/PromptShield](https://hf.co/datasets/hendzh/PromptShield) | ✓ |
| 440 | [shell-safety-v2](src/tasksource/tasks.py#L1751) | Classification | [tomngdev/shell-safety-v2](https://hf.co/datasets/tomngdev/shell-safety-v2) | ✓ |
| 441 | [agent_action_safety](src/tasksource/tasks.py#L1756) | Classification | json | ✓ |
| 442 | [ShellRisk-Bench](src/tasksource/tasks.py#L1771) | Classification | [kontext-security/ShellRisk-Bench](https://hf.co/datasets/kontext-security/ShellRisk-Bench) | ✓ |
| 443 | [wildguardmix-cleaned/prompt_harm](src/tasksource/tasks.py#L1796) | Classification | [bogdanminko/wildguardmix-cleaned](https://hf.co/datasets/bogdanminko/wildguardmix-cleaned) | ✓ |
| 444 | [wildguardmix-cleaned/response_harm](src/tasksource/tasks.py#L1802) | Classification | [bogdanminko/wildguardmix-cleaned](https://hf.co/datasets/bogdanminko/wildguardmix-cleaned) | ✓ |
| 445 | [wildguardmix-cleaned/response_refusal](src/tasksource/tasks.py#L1807) | Classification | [bogdanminko/wildguardmix-cleaned](https://hf.co/datasets/bogdanminko/wildguardmix-cleaned) | ✓ |
| 446 | [BeaverTails](src/tasksource/tasks.py#L1823) | Classification | [PKU-Alignment/BeaverTails](https://hf.co/datasets/PKU-Alignment/BeaverTails) | ✓ |
| 447 | [privacy-200k-Mistral-Large-3](src/tasksource/tasks.py#L1839) | Classification | [gabrielloiseau/privacy-200k-Mistral-Large-3](https://hf.co/datasets/gabrielloiseau/privacy-200k-Mistral-Large-3) | ✓ |
| 448 | [toxic-chat/toxicchat0124/toxicity](src/tasksource/tasks.py#L1848) | Classification | [lmsys/toxic-chat](https://hf.co/datasets/lmsys/toxic-chat) | ✓ |
| 449 | [toxic-chat/toxicchat0124/jailbreaking](src/tasksource/tasks.py#L1853) | Classification | [lmsys/toxic-chat](https://hf.co/datasets/lmsys/toxic-chat) | ✓ |
| 450 | [clinc_oos/plus](src/tasksource/tasks.py#L1859) | Classification | [clinc/clinc_oos](https://hf.co/datasets/clinc/clinc_oos) |  |
| 451 | [IntentGrasp/all](src/tasksource/tasks.py#L1877) | MultipleChoice | [yuweiyin/IntentGrasp](https://hf.co/datasets/yuweiyin/IntentGrasp) |  |
| 452 | [few_rel/default](src/tasksource/tasks.py#L1915) | Classification | [tasksource/few_rel](https://hf.co/datasets/tasksource/few_rel) |  |
| 453 | [docred](src/tasksource/tasks.py#L1993) | Classification | json |  |
| 454 | [chemprot/chemprot_full_source](src/tasksource/tasks.py#L2028) | Classification | [bigbio/chemprot](https://hf.co/datasets/bigbio/chemprot) | ✓ |
| 455 | [PKU-SafeRLHF/helpfulness](src/tasksource/tasks.py#L2034) | MultipleChoice | [PKU-Alignment/PKU-SafeRLHF](https://hf.co/datasets/PKU-Alignment/PKU-SafeRLHF) | ✓ |
| 456 | [PKU-SafeRLHF/safety](src/tasksource/tasks.py#L2039) | MultipleChoice | [PKU-Alignment/PKU-SafeRLHF](https://hf.co/datasets/PKU-Alignment/PKU-SafeRLHF) | ✓ |
| 457 | [HelpSteer/helpfulness](src/tasksource/tasks.py#L2061) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 458 | [HelpSteer/correctness](src/tasksource/tasks.py#L2062) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 459 | [HelpSteer/coherence](src/tasksource/tasks.py#L2063) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 460 | [HelpSteer/complexity](src/tasksource/tasks.py#L2064) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 461 | [HelpSteer/verbosity](src/tasksource/tasks.py#L2065) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 462 | [HelpSteer2/helpfulness](src/tasksource/tasks.py#L2067) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 463 | [HelpSteer2/correctness](src/tasksource/tasks.py#L2068) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 464 | [HelpSteer2/coherence](src/tasksource/tasks.py#L2069) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 465 | [HelpSteer2/complexity](src/tasksource/tasks.py#L2070) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 466 | [HelpSteer2/verbosity](src/tasksource/tasks.py#L2071) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 467 | [HelpSteer3/preference](src/tasksource/tasks.py#L2076) | MultipleChoice | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | ✓ |
| 468 | [HelpSteer3/preference_strength](src/tasksource/tasks.py#L2089) | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | ✓ |
| 469 | [HelpSteer3/principle](src/tasksource/tasks.py#L2101) | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) |  |
| 470 | [HelpSteer3/edit_quality](src/tasksource/tasks.py#L2106) | MultipleChoice | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | ✓ |
| 471 | [HelpSteer3/feedback](src/tasksource/tasks.py#L2128) | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | ✓ |
| 472 | [MSciNLI](src/tasksource/tasks.py#L2134) | Classification | [sadat2307/MSciNLI](https://hf.co/datasets/sadat2307/MSciNLI) |  |
| 473 | [UltraFeedback-paired](src/tasksource/tasks.py#L2137) | MultipleChoice | [pushpdeep/UltraFeedback-paired](https://hf.co/datasets/pushpdeep/UltraFeedback-paired) | ✓ |
| 474 | [prm800k_dpo/solution](src/tasksource/tasks.py#L2141) | MultipleChoice | [tasksource/prm800k_dpo](https://hf.co/datasets/tasksource/prm800k_dpo) | ✓ |
| 475 | [prm800k_dpo/step](src/tasksource/tasks.py#L2144) | MultipleChoice | [tasksource/prm800k_dpo](https://hf.co/datasets/tasksource/prm800k_dpo) | ✓ |
| 476 | [AES2-essay-scoring](src/tasksource/tasks.py#L2148) | Classification | [tasksource/AES2-essay-scoring](https://hf.co/datasets/tasksource/AES2-essay-scoring) | ✓ |
| 477 | [argument-feedback](src/tasksource/tasks.py#L2152) | Classification | [tasksource/argument-feedback](https://hf.co/datasets/tasksource/argument-feedback) | ✓ |
| 478 | [english-grading/cohesion](src/tasksource/tasks.py#L2159) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 479 | [english-grading/syntax](src/tasksource/tasks.py#L2160) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 480 | [english-grading/vocabulary](src/tasksource/tasks.py#L2161) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 481 | [english-grading/phraseology](src/tasksource/tasks.py#L2162) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 482 | [english-grading/grammar](src/tasksource/tasks.py#L2163) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 483 | [english-grading/conventions](src/tasksource/tasks.py#L2164) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 484 | [wice](src/tasksource/tasks.py#L2166) | Classification | [tasksource/wice](https://hf.co/datasets/tasksource/wice) |  |
| 485 | [hover](src/tasksource/tasks.py#L2169) | Classification | [Dzeniks/hover](https://hf.co/datasets/Dzeniks/hover) |  |
| 486 | [hover-3way/nli](src/tasksource/tasks.py#L2173) | Classification | [Dzeniks/hover-3way](https://hf.co/datasets/Dzeniks/hover-3way) |  |
| 487 | [tasksource_dpo_pairs](src/tasksource/tasks.py#L2176) | MultipleChoice | [tasksource/tasksource_dpo_pairs](https://hf.co/datasets/tasksource/tasksource_dpo_pairs) | ✓ |
| 488 | [seahorse_summarization_evaluation](src/tasksource/tasks.py#L2179) | Classification | [tasksource/seahorse_summarization_evaluation](https://hf.co/datasets/tasksource/seahorse_summarization_evaluation) |  |
| 489 | [missing-item-prediction/contrastive](src/tasksource/tasks.py#L2191) | Classification | [sileod/missing-item-prediction](https://hf.co/datasets/sileod/missing-item-prediction) | ✓ |
| 490 | [jigsaw_toxicity](src/tasksource/tasks.py#L2197) | Classification | [tasksource/jigsaw_toxicity](https://hf.co/datasets/tasksource/jigsaw_toxicity) |  |
| 491 | [Pol_NLI](src/tasksource/tasks.py#L2200) | Classification | [mlburnham/Pol_NLI](https://hf.co/datasets/mlburnham/Pol_NLI) |  |
| 492 | [synthetic-retrieval-NLI/binary](src/tasksource/tasks.py#L2203) | Classification | [tasksource/synthetic-retrieval-NLI](https://hf.co/datasets/tasksource/synthetic-retrieval-NLI) |  |
| 493 | [synthetic-retrieval-NLI/position](src/tasksource/tasks.py#L2203) | Classification | [tasksource/synthetic-retrieval-NLI](https://hf.co/datasets/tasksource/synthetic-retrieval-NLI) |  |
| 494 | [synthetic-retrieval-NLI/count](src/tasksource/tasks.py#L2203) | Classification | [tasksource/synthetic-retrieval-NLI](https://hf.co/datasets/tasksource/synthetic-retrieval-NLI) |  |
| 495 | [github-issue-similarity](src/tasksource/tasks.py#L2212) | Classification | [WhereIsAI/github-issue-similarity](https://hf.co/datasets/WhereIsAI/github-issue-similarity) |  |
| 496 | [webinstruct/mc](src/tasksource/tasks.py#L2417) | MultipleChoice | [tasksource/webinstruct](https://hf.co/datasets/tasksource/webinstruct) |  |
| 497 | [webinstruct/binary](src/tasksource/tasks.py#L2419) | Classification | [tasksource/webinstruct](https://hf.co/datasets/tasksource/webinstruct) |  |
| 498 | [weblinx/action](src/tasksource/tasks.py#L2483) | Classification | [McGill-NLP/WebLINX](https://hf.co/datasets/McGill-NLP/WebLINX) | ✓ |
| 499 | [weblinx/dom-element](src/tasksource/tasks.py#L2486) | MultipleChoice | [McGill-NLP/WebLINX](https://hf.co/datasets/McGill-NLP/WebLINX) | ✓ |
| 500 | [websrc/yesno](src/tasksource/tasks.py#L2492) | Classification | [tasksource/websrc](https://hf.co/datasets/tasksource/websrc) |  |
| 501 | [websrc/element](src/tasksource/tasks.py#L2494) | MultipleChoice | [tasksource/websrc](https://hf.co/datasets/tasksource/websrc) | ✓ |
| 502 | [mind2web/action](src/tasksource/tasks.py#L2501) | Classification | [tasksource/mind2web-dom](https://hf.co/datasets/tasksource/mind2web-dom) | ✓ |
| 503 | [mind2web/dom-element](src/tasksource/tasks.py#L2504) | MultipleChoice | [tasksource/mind2web-dom](https://hf.co/datasets/tasksource/mind2web-dom) | ✓ |
| 504 | [sentfin](src/tasksource/tasks.py#L2509) | Classification | [tasksource/sentfin](https://hf.co/datasets/tasksource/sentfin) | ✓ |
| 505 | [wands](src/tasksource/tasks.py#L2513) | Classification | [tasksource/wands](https://hf.co/datasets/tasksource/wands) | ✓ |
| 506 | [scirepeval/search](src/tasksource/tasks.py#L2517) | Classification | [tasksource/scirepeval-search](https://hf.co/datasets/tasksource/scirepeval-search) | ✓ |
| 507 | [maud/consideration](src/tasksource/tasks.py#L2568) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 508 | [maud/bringdown-timing](src/tasksource/tasks.py#L2572) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 509 | [maud/ordinary-course-consent](src/tasksource/tasks.py#L2576) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 510 | [maud/ordinary-course-efforts](src/tasksource/tasks.py#L2580) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 511 | [maud/antitrust-efforts](src/tasksource/tasks.py#L2584) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 512 | [maud/specific-performance](src/tasksource/tasks.py#L2588) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 513 | [ddi/relation](src/tasksource/tasks.py#L2593) | Classification | [OpenMed/DDI-Corpus-Processed](https://hf.co/datasets/OpenMed/DDI-Corpus-Processed) | ✓ |
| 514 | [patent-classification/abstract](src/tasksource/tasks.py#L2600) | Classification | [ccdv/patent-classification](https://hf.co/datasets/ccdv/patent-classification) | ✓ |
| 515 | [tabfact](src/tasksource/tasks.py#L2607) | Classification | [wenhu/tab_fact](https://hf.co/datasets/wenhu/tab_fact) | ✓ |
| 516 | [infotabs](src/tasksource/tasks.py#L2615) | Classification | [table-benchmark/infotabs](https://hf.co/datasets/table-benchmark/infotabs) | ✓ |
| 517 | [squad-v2/answerability](src/tasksource/tasks.py#L2627) | Classification | [rajpurkar/squad_v2](https://hf.co/datasets/rajpurkar/squad_v2) | ✓ |
| 518 | [cuad/answerability](src/tasksource/tasks.py#L2632) | Classification | [theatticusproject/cuad-qa](https://hf.co/datasets/theatticusproject/cuad-qa) | ✓ |
| 519 | [hallmarks-of-cancer/yesno](src/tasksource/tasks.py#L2651) | Classification | [bigbio/hallmarks_of_cancer](https://hf.co/datasets/bigbio/hallmarks_of_cancer) | ✓ |
| 520 | [semeval2014-absa/restaurants](src/tasksource/tasks.py#L2673) | Classification | [alexcadillon/SemEval2014Task4](https://hf.co/datasets/alexcadillon/SemEval2014Task4) | ✓ |
| 521 | [semeval2014-absa/laptops](src/tasksource/tasks.py#L2673) | Classification | [alexcadillon/SemEval2014Task4](https://hf.co/datasets/alexcadillon/SemEval2014Task4) | ✓ |
| 522 | [record/candidate](src/tasksource/tasks.py#L2695) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) | ✓ |
| 523 | [multidogo/finance/intent](src/tasksource/tasks.py#L2718) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 524 | [multidogo/fastfood/intent](src/tasksource/tasks.py#L2718) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 525 | [multidogo/software/intent](src/tasksource/tasks.py#L2718) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 526 | [multidogo/airline/intent](src/tasksource/tasks.py#L2718) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 527 | [multidogo/insurance/intent](src/tasksource/tasks.py#L2718) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 528 | [multidogo/media/intent](src/tasksource/tasks.py#L2718) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 529 | [wikisql/aggregation](src/tasksource/tasks.py#L2746) | Classification | [Salesforce/wikisql](https://hf.co/datasets/Salesforce/wikisql) | ✓ |
| 530 | [wikisql/column](src/tasksource/tasks.py#L2751) | MultipleChoice | [Salesforce/wikisql](https://hf.co/datasets/Salesforce/wikisql) | ✓ |
| 531 | [wikisql/operator](src/tasksource/tasks.py#L2756) | Classification | [Salesforce/wikisql](https://hf.co/datasets/Salesforce/wikisql) | ✓ |
| 532 | [acord](src/tasksource/tasks.py#L2761) | Classification | [tasksource/acord](https://hf.co/datasets/tasksource/acord) | ✓ |
| 533 | [edos](src/tasksource/tasks.py#L2765) | Classification | [tasksource/edos](https://hf.co/datasets/tasksource/edos) | ✓ |
| 534 | [evidence-inference](src/tasksource/tasks.py#L2768) | Classification | [tasksource/evidence-inference](https://hf.co/datasets/tasksource/evidence-inference) | ✓ |
| 535 | [quac/answerability](src/tasksource/tasks.py#L2790) | Classification | [allenai/quac](https://hf.co/datasets/allenai/quac) | ✓ |
| 536 | [hatexplain/majority](src/tasksource/tasks.py#L2798) | Classification | [Hate-speech-CNERG/hatexplain](https://hf.co/datasets/Hate-speech-CNERG/hatexplain) | ✓ |
| 537 | [hotpotqa/supporting-fact](src/tasksource/tasks.py#L2815) | Classification | [hotpotqa/hotpot_qa](https://hf.co/datasets/hotpotqa/hotpot_qa) | ✓ |
| 538 | [multiwoz22/dialogue-act](src/tasksource/tasks.py#L2859) | Classification | [pfb30/multi_woz_v22](https://hf.co/datasets/pfb30/multi_woz_v22) | ✓ |
| 539 | [multiwoz22/categorical-slot](src/tasksource/tasks.py#L2862) | Classification | [pfb30/multi_woz_v22](https://hf.co/datasets/pfb30/multi_woz_v22) | ✓ |
| 540 | [schema-guided/intent](src/tasksource/tasks.py#L2865) | Classification | [google-research-datasets/schema_guided_dstc8](https://hf.co/datasets/google-research-datasets/schema_guided_dstc8) | ✓ |
| 541 | [coqa/yesno](src/tasksource/tasks.py#L2882) | Classification | [stanfordnlp/coqa](https://hf.co/datasets/stanfordnlp/coqa) | ✓ |
| 542 | [qasper/yesno](src/tasksource/tasks.py#L2903) | Classification | [allenai/qasper](https://hf.co/datasets/allenai/qasper) | ✓ |
| 543 | [topv2/intent](src/tasksource/tasks.py#L2915) | Classification | [WillHeld/top_v2](https://hf.co/datasets/WillHeld/top_v2) | ✓ |
| 544 | [slurp/intent](src/tasksource/tasks.py#L2923) | Classification | [tasksource/slurp](https://hf.co/datasets/tasksource/slurp) | ✓ |
| 545 | [mcscript2](src/tasksource/tasks.py#L2927) | MultipleChoice | [tasksource/mcscript2](https://hf.co/datasets/tasksource/mcscript2) |  |

## Soft labels

Annotations whose label is a distribution (annotator votes, rater shares, survey counts), loaded with `load_task(id, soft=True)`. Those with a hard view are also listed above, by their majority label; the others have soft labels only. `votes` are shares of annotators, `mean` a numeric value interpolated over score anchors (see each task for its meaning); annotators is the typical count per item (vote shares from fewer than five are coarse, and the Jev build leaves them out).

| id | kind | aggregation | annotators | default view | dataset |
|---|---|---|--:|---|---|
| [glue/stsb](src/tasksource/tasks.py#L46) | score | mean |  | regression | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |
| [sick/relatedness](src/tasksource/tasks.py#L85) | score | mean | 10 | regression | [tasksource/sick](https://hf.co/datasets/tasksource/sick) |
| [google_wellformed_query](src/tasksource/tasks.py#L846) | noul | votes | 5 | Classification | [tasksource/google_wellformed_query](https://hf.co/datasets/tasksource/google_wellformed_query) |
| [sts-companion](src/tasksource/tasks.py#L1033) | score | mean |  | regression | [tasksource/sts-companion](https://hf.co/datasets/tasksource/sts-companion) |
| [wouldyourather](src/tasksource/tasks.py#L1099) | choice | votes | 1000 | MultipleChoice | [tasksource/wouldyourather](https://hf.co/datasets/tasksource/wouldyourather) |
| [acceptability-prediction/rating_votes](src/tasksource/tasks.py#L1283) | score | votes | 15 |  | [tasksource/acceptability-prediction](https://hf.co/datasets/tasksource/acceptability-prediction) |
| [acceptability-prediction/binary_votes](src/tasksource/tasks.py#L1288) | noul | votes | 15 |  | [tasksource/acceptability-prediction](https://hf.co/datasets/tasksource/acceptability-prediction) |
| [HelpSteer3/individual_preferences](src/tasksource/tasks.py#L2095) | score | votes | 3 |  | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) |
| [HelpSteer3/feedback](src/tasksource/tasks.py#L2128) | score | votes | 3 | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) |
| [proto_qa/proto_qa](src/tasksource/tasks.py#L2222) | choice | votes | 100 |  | [community-datasets/proto_qa](https://hf.co/datasets/community-datasets/proto_qa) |
| [UNLI](src/tasksource/tasks.py#L2236) | noul | mean | 2 |  | [Zhengping/UNLI](https://hf.co/datasets/Zhengping/UNLI) |
| [chaos-mnli-ambiguity/votes](src/tasksource/tasks.py#L2240) | choice | votes | 100 |  | [tasksource/chaos-mnli-ambiguity](https://hf.co/datasets/tasksource/chaos-mnli-ambiguity) |
| [hate_speech_offensive/votes](src/tasksource/tasks.py#L2244) | choice | votes | 3 |  | [tdavidson/hate_speech_offensive](https://hf.co/datasets/tdavidson/hate_speech_offensive) |
| [scruples/verdict_votes](src/tasksource/tasks.py#L2252) | choice | votes | 8 |  | [tasksource/scruples](https://hf.co/datasets/tasksource/scruples) |
| [BeaverTails/unsafe_votes](src/tasksource/tasks.py#L2266) | noul | votes | 3 |  | [PKU-Alignment/BeaverTails](https://hf.co/datasets/PKU-Alignment/BeaverTails) |
| [dynasent/r1_votes](src/tasksource/tasks.py#L2277) | choice | votes | 5 |  | [dynabench/dynasent](https://hf.co/datasets/dynabench/dynasent) |
| [dynasent/r2_votes](src/tasksource/tasks.py#L2281) | choice | votes | 5 |  | [dynabench/dynasent](https://hf.co/datasets/dynabench/dynasent) |
| [hatexplain/votes](src/tasksource/tasks.py#L2286) | choice | votes | 3 |  | [Hate-speech-CNERG/hatexplain](https://hf.co/datasets/Hate-speech-CNERG/hatexplain) |
| [measuring-hate-speech/sentiment](src/tasksource/tasks.py#L2301) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/respect](src/tasksource/tasks.py#L2303) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/insult](src/tasksource/tasks.py#L2305) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/humiliate](src/tasksource/tasks.py#L2306) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/status](src/tasksource/tasks.py#L2307) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/dehumanize](src/tasksource/tasks.py#L2309) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/violence](src/tasksource/tasks.py#L2311) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/genocide](src/tasksource/tasks.py#L2312) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/attack_defend](src/tasksource/tasks.py#L2314) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/hatespeech](src/tasksource/tasks.py#L2316) | choice | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [wikipedia-detox/attack](src/tasksource/tasks.py#L2328) | choice | votes | 10 |  | [tasksource/wikipedia-detox-votes](https://hf.co/datasets/tasksource/wikipedia-detox-votes) |
| [wikipedia-detox/aggression](src/tasksource/tasks.py#L2330) | choice | votes | 10 |  | [tasksource/wikipedia-detox-votes](https://hf.co/datasets/tasksource/wikipedia-detox-votes) |
| [wikipedia-detox/aggression_score](src/tasksource/tasks.py#L2332) | score | votes | 10 |  | [tasksource/wikipedia-detox-votes](https://hf.co/datasets/tasksource/wikipedia-detox-votes) |
| [wikipedia-detox/toxicity](src/tasksource/tasks.py#L2336) | choice | votes | 10 |  | [tasksource/wikipedia-detox-votes](https://hf.co/datasets/tasksource/wikipedia-detox-votes) |
| [wikipedia-detox/toxicity_score](src/tasksource/tasks.py#L2338) | score | votes | 10 |  | [tasksource/wikipedia-detox-votes](https://hf.co/datasets/tasksource/wikipedia-detox-votes) |
| [lewidi/md_agreement](src/tasksource/tasks.py#L2353) | noul | votes | 5 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/hs_brexit](src/tasksource/tasks.py#L2354) | noul | votes | 6 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/armis](src/tasksource/tasks.py#L2356) | noul | votes | 3 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/conv_abuse](src/tasksource/tasks.py#L2358) | noul | votes | 3 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/mp](src/tasksource/tasks.py#L2360) | noul | votes | 5 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/csc](src/tasksource/tasks.py#L2361) | score | votes | 4 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/varierrnli/entailment](src/tasksource/tasks.py#L2369) | noul | votes | 4 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/varierrnli/neutral](src/tasksource/tasks.py#L2370) | noul | votes | 4 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/varierrnli/contradiction](src/tasksource/tasks.py#L2371) | noul | votes | 4 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [civil_comments/toxicity_share](src/tasksource/tasks.py#L2379) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/severe_toxicity_share](src/tasksource/tasks.py#L2380) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/obscene_share](src/tasksource/tasks.py#L2381) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/threat_share](src/tasksource/tasks.py#L2382) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/insult_share](src/tasksource/tasks.py#L2383) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/identity_attack_share](src/tasksource/tasks.py#L2384) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/sexual_explicit_share](src/tasksource/tasks.py#L2385) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [oasst2/quality](src/tasksource/tasks.py#L2402) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/helpfulness](src/tasksource/tasks.py#L2403) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/creativity](src/tasksource/tasks.py#L2404) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/humor](src/tasksource/tasks.py#L2405) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/toxicity](src/tasksource/tasks.py#L2406) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/violence](src/tasksource/tasks.py#L2407) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/spam](src/tasksource/tasks.py#L2408) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/fails_task](src/tasksource/tasks.py#L2409) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/not_appropriate](src/tasksource/tasks.py#L2410) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/hate_speech](src/tasksource/tasks.py#L2411) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/sexual_content](src/tasksource/tasks.py#L2412) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/pii](src/tasksource/tasks.py#L2413) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/lang_mismatch](src/tasksource/tasks.py#L2414) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [scirepeval/search](src/tasksource/tasks.py#L2517) | score | mean |  | regression | [tasksource/scirepeval-search](https://hf.co/datasets/tasksource/scirepeval-search) |
