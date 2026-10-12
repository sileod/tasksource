560 English tasks. Load one with `load_task(id)`; the annotations are in [tasks.py](src/tasksource/tasks.py). Evaluation benchmarks are in [eval_only.py](src/tasksource/eval_only.py); other excluded annotations are in [parked.py](src/tasksource/parked.py).

Browse the [English example gallery](docs/examples/english/README.md) for formatted inputs, questions and gold answers.

| # | id | type | dataset | question |
|--:|---|---|---|:-:|
| 1 | [glue/mnli](src/tasksource/tasks.py#L35) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 2 | [glue/qnli](src/tasksource/tasks.py#L36) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) | ✓ |
| 3 | [glue/rte](src/tasksource/tasks.py#L38) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 4 | [glue/wnli](src/tasksource/tasks.py#L39) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 5 | [glue/mrpc](src/tasksource/tasks.py#L41) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 6 | [glue/qqp](src/tasksource/tasks.py#L42) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 7 | [glue/stsb](src/tasksource/tasks.py#L48) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) | ✓ |
| 8 | [super_glue/boolq](src/tasksource/tasks.py#L53) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 9 | [super_glue/boolq_passage](src/tasksource/tasks.py#L54) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 10 | [super_glue/cb](src/tasksource/tasks.py#L56) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 11 | [super_glue/multirc](src/tasksource/tasks.py#L57) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 12 | [super_glue/wic](src/tasksource/tasks.py#L62) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 13 | [super_glue/axg](src/tasksource/tasks.py#L67) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 14 | [anli/a1](src/tasksource/tasks.py#L70) | Classification | [facebook/anli](https://hf.co/datasets/facebook/anli) |  |
| 15 | [anli/a2](src/tasksource/tasks.py#L71) | Classification | [facebook/anli](https://hf.co/datasets/facebook/anli) |  |
| 16 | [anli/a3](src/tasksource/tasks.py#L72) | Classification | [facebook/anli](https://hf.co/datasets/facebook/anli) |  |
| 17 | [babi_nli/basic-deduction](src/tasksource/tasks.py#L75) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 18 | [babi_nli/conjunction](src/tasksource/tasks.py#L75) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 19 | [babi_nli/compound-coreference](src/tasksource/tasks.py#L75) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 20 | [babi_nli/lists-sets](src/tasksource/tasks.py#L75) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 21 | [babi_nli/basic-coreference](src/tasksource/tasks.py#L75) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 22 | [babi_nli/path-finding](src/tasksource/tasks.py#L75) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 23 | [babi_nli/positional-reasoning](src/tasksource/tasks.py#L75) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 24 | [babi_nli/simple-negation](src/tasksource/tasks.py#L75) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 25 | [babi_nli/size-reasoning](src/tasksource/tasks.py#L75) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 26 | [babi_nli/counting](src/tasksource/tasks.py#L75) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 27 | [babi_nli/three-arg-relations](src/tasksource/tasks.py#L75) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 28 | [babi_nli/three-supporting-facts](src/tasksource/tasks.py#L75) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 29 | [babi_nli/time-reasoning](src/tasksource/tasks.py#L75) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 30 | [babi_nli/two-arg-relations](src/tasksource/tasks.py#L75) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 31 | [babi_nli/two-supporting-facts](src/tasksource/tasks.py#L75) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 32 | [babi_nli/yes-no-questions](src/tasksource/tasks.py#L75) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 33 | [babi_nli/single-supporting-fact](src/tasksource/tasks.py#L75) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 34 | [babi_nli/indefinite-knowledge](src/tasksource/tasks.py#L75) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 35 | [babi_nli/basic-induction](src/tasksource/tasks.py#L80) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) | ✓ |
| 36 | [sick/label](src/tasksource/tasks.py#L85) | Classification | [tasksource/sick](https://hf.co/datasets/tasksource/sick) |  |
| 37 | [sick/relatedness](src/tasksource/tasks.py#L87) | Classification | [tasksource/sick](https://hf.co/datasets/tasksource/sick) | ✓ |
| 38 | [snli](src/tasksource/tasks.py#L145) | Classification | [stanfordnlp/snli](https://hf.co/datasets/stanfordnlp/snli) |  |
| 39 | [scitail/snli_format](src/tasksource/tasks.py#L148) | Classification | [allenai/scitail](https://hf.co/datasets/allenai/scitail) |  |
| 40 | [hans](src/tasksource/tasks.py#L150) | Classification | [tasksource/hans](https://hf.co/datasets/tasksource/hans) |  |
| 41 | [WANLI](src/tasksource/tasks.py#L153) | Classification | [alisawuffles/WANLI](https://hf.co/datasets/alisawuffles/WANLI) |  |
| 42 | [recast/recast_sentiment](src/tasksource/tasks.py#L155) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 43 | [recast/recast_factuality](src/tasksource/tasks.py#L155) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 44 | [recast/recast_verbnet](src/tasksource/tasks.py#L155) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 45 | [recast/recast_verbcorner](src/tasksource/tasks.py#L155) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 46 | [recast/recast_ner](src/tasksource/tasks.py#L155) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 47 | [recast/recast_puns](src/tasksource/tasks.py#L155) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 48 | [recast/recast_megaveridicality](src/tasksource/tasks.py#L155) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 49 | [probability_words_nli/reasoning_2hop](src/tasksource/tasks.py#L160) | Classification | [sileod/probability_words_nli](https://hf.co/datasets/sileod/probability_words_nli) | ✓ |
| 50 | [probability_words_nli/usnli](src/tasksource/tasks.py#L160) | Classification | [sileod/probability_words_nli](https://hf.co/datasets/sileod/probability_words_nli) | ✓ |
| 51 | [probability_words_nli/reasoning_1hop](src/tasksource/tasks.py#L160) | Classification | [sileod/probability_words_nli](https://hf.co/datasets/sileod/probability_words_nli) | ✓ |
| 52 | [nan-nli](src/tasksource/tasks.py#L165) | Classification | [joey234/nan-nli](https://hf.co/datasets/joey234/nan-nli) |  |
| 53 | [nli_fever](src/tasksource/tasks.py#L168) | Classification | [pietrolesci/nli_fever](https://hf.co/datasets/pietrolesci/nli_fever) |  |
| 54 | [breaking_nli](src/tasksource/tasks.py#L171) | Classification | [pietrolesci/breaking_nli](https://hf.co/datasets/pietrolesci/breaking_nli) |  |
| 55 | [conj_nli](src/tasksource/tasks.py#L175) | Classification | [pietrolesci/conj_nli](https://hf.co/datasets/pietrolesci/conj_nli) |  |
| 56 | [fracas](src/tasksource/tasks.py#L179) | Classification | [pietrolesci/fracas](https://hf.co/datasets/pietrolesci/fracas) |  |
| 57 | [dialogue_nli](src/tasksource/tasks.py#L182) | Classification | [pietrolesci/dialogue_nli](https://hf.co/datasets/pietrolesci/dialogue_nli) |  |
| 58 | [mpe](src/tasksource/tasks.py#L185) | Classification | [pietrolesci/mpe](https://hf.co/datasets/pietrolesci/mpe) |  |
| 59 | [dnc](src/tasksource/tasks.py#L189) | Classification | [pietrolesci/dnc](https://hf.co/datasets/pietrolesci/dnc) |  |
| 60 | [recast_white/fnplus](src/tasksource/tasks.py#L193) | Classification | [pietrolesci/recast_white](https://hf.co/datasets/pietrolesci/recast_white) |  |
| 61 | [recast_white/sprl](src/tasksource/tasks.py#L196) | Classification | [pietrolesci/recast_white](https://hf.co/datasets/pietrolesci/recast_white) |  |
| 62 | [recast_white/dpr](src/tasksource/tasks.py#L199) | Classification | [pietrolesci/recast_white](https://hf.co/datasets/pietrolesci/recast_white) |  |
| 63 | [joci](src/tasksource/tasks.py#L203) | Classification | [pietrolesci/joci](https://hf.co/datasets/pietrolesci/joci) | ✓ |
| 64 | [robust_nli/IS_CS](src/tasksource/tasks.py#L210) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 65 | [robust_nli/LI_LI](src/tasksource/tasks.py#L212) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 66 | [robust_nli/ST_WO](src/tasksource/tasks.py#L214) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 67 | [robust_nli/PI_SP](src/tasksource/tasks.py#L216) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 68 | [robust_nli/PI_CD](src/tasksource/tasks.py#L218) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 69 | [robust_nli/ST_SE](src/tasksource/tasks.py#L220) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 70 | [robust_nli/ST_NE](src/tasksource/tasks.py#L222) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 71 | [robust_nli/ST_LM](src/tasksource/tasks.py#L224) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 72 | [robust_nli_is_sd](src/tasksource/tasks.py#L226) | Classification | [pietrolesci/robust_nli_is_sd](https://hf.co/datasets/pietrolesci/robust_nli_is_sd) |  |
| 73 | [robust_nli_li_ts](src/tasksource/tasks.py#L229) | Classification | [pietrolesci/robust_nli_li_ts](https://hf.co/datasets/pietrolesci/robust_nli_li_ts) |  |
| 74 | [gen_debiased_nli/snli_seq_z](src/tasksource/tasks.py#L233) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| 75 | [gen_debiased_nli/snli_z_aug](src/tasksource/tasks.py#L235) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| 76 | [gen_debiased_nli/snli_par_z](src/tasksource/tasks.py#L237) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| 77 | [gen_debiased_nli/mnli_par_z](src/tasksource/tasks.py#L239) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| 78 | [gen_debiased_nli/mnli_z_aug](src/tasksource/tasks.py#L241) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| 79 | [gen_debiased_nli/mnli_seq_z](src/tasksource/tasks.py#L243) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| 80 | [add_one_rte](src/tasksource/tasks.py#L246) | Classification | [pietrolesci/add_one_rte](https://hf.co/datasets/pietrolesci/add_one_rte) |  |
| 81 | [hlgd](src/tasksource/tasks.py#L250) | Classification | [tasksource/hlgd](https://hf.co/datasets/tasksource/hlgd) | ✓ |
| 82 | [paws/labeled_final](src/tasksource/tasks.py#L254) | Classification | [google-research-datasets/paws](https://hf.co/datasets/google-research-datasets/paws) |  |
| 83 | [paws/labeled_swap](src/tasksource/tasks.py#L255) | Classification | [google-research-datasets/paws](https://hf.co/datasets/google-research-datasets/paws) |  |
| 84 | [medical_questions_pairs](src/tasksource/tasks.py#L257) | Classification | [curaihealth/medical_questions_pairs](https://hf.co/datasets/curaihealth/medical_questions_pairs) | ✓ |
| 85 | [conll2003/pos_tags](src/tasksource/tasks.py#L263) | TokenClassification | [tomaarsen/conll2003](https://hf.co/datasets/tomaarsen/conll2003) |  |
| 86 | [conll2003/chunk_tags](src/tasksource/tasks.py#L264) | TokenClassification | [tomaarsen/conll2003](https://hf.co/datasets/tomaarsen/conll2003) |  |
| 87 | [conll2003/ner_tags](src/tasksource/tasks.py#L265) | TokenClassification | [tomaarsen/conll2003](https://hf.co/datasets/tomaarsen/conll2003) |  |
| 88 | [fig-qa](src/tasksource/tasks.py#L271) | MultipleChoice | [nightingal3/fig-qa](https://hf.co/datasets/nightingal3/fig-qa) |  |
| 89 | [cos_e/v1.0](src/tasksource/tasks.py#L280) | MultipleChoice | [Salesforce/cos_e](https://hf.co/datasets/Salesforce/cos_e) |  |
| 90 | [cosmos_qa](src/tasksource/tasks.py#L285) | MultipleChoice | [Samsoup/cosmos_qa](https://hf.co/datasets/Samsoup/cosmos_qa) |  |
| 91 | [dream](src/tasksource/tasks.py#L288) | MultipleChoice | [dataset-org/dream](https://hf.co/datasets/dataset-org/dream) |  |
| 92 | [openbookqa](src/tasksource/tasks.py#L295) | MultipleChoice | [allenai/openbookqa](https://hf.co/datasets/allenai/openbookqa) |  |
| 93 | [qasc](src/tasksource/tasks.py#L301) | MultipleChoice | [allenai/qasc](https://hf.co/datasets/allenai/qasc) |  |
| 94 | [quartz](src/tasksource/tasks.py#L309) | MultipleChoice | [allenai/quartz](https://hf.co/datasets/allenai/quartz) |  |
| 95 | [quail](src/tasksource/tasks.py#L314) | MultipleChoice | [textmachinelab/quail](https://hf.co/datasets/textmachinelab/quail) |  |
| 96 | [head_qa/en](src/tasksource/tasks.py#L320) | MultipleChoice | [EleutherAI/headqa](https://hf.co/datasets/EleutherAI/headqa) |  |
| 97 | [sciq](src/tasksource/tasks.py#L328) | MultipleChoice | [allenai/sciq](https://hf.co/datasets/allenai/sciq) |  |
| 98 | [social_i_qa](src/tasksource/tasks.py#L333) | MultipleChoice | [tasksource/social_i_qa](https://hf.co/datasets/tasksource/social_i_qa) |  |
| 99 | [wiki_hop/original](src/tasksource/tasks.py#L339) | MultipleChoice | [MoE-UNC/wikihop](https://hf.co/datasets/MoE-UNC/wikihop) |  |
| 100 | [wiqa](src/tasksource/tasks.py#L352) | MultipleChoice | parquet |  |
| 101 | [piqa](src/tasksource/tasks.py#L359) | MultipleChoice | [baber/piqa](https://hf.co/datasets/baber/piqa) |  |
| 102 | [hellaswag](src/tasksource/tasks.py#L367) | MultipleChoice | [Rowan/hellaswag](https://hf.co/datasets/Rowan/hellaswag) |  |
| 103 | [super_glue/copa](src/tasksource/tasks.py#L376) | MultipleChoice | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 104 | [balanced-copa](src/tasksource/tasks.py#L378) | MultipleChoice | [pkavumba/balanced-copa](https://hf.co/datasets/pkavumba/balanced-copa) |  |
| 105 | [e-CARE](src/tasksource/tasks.py#L381) | MultipleChoice | [12ml/e-CARE](https://hf.co/datasets/12ml/e-CARE) |  |
| 106 | [art](src/tasksource/tasks.py#L384) | MultipleChoice | [allenai/art](https://hf.co/datasets/allenai/art) | ✓ |
| 107 | [winogrande/winogrande_xl](src/tasksource/tasks.py#L392) | MultipleChoice | [allenai/winogrande](https://hf.co/datasets/allenai/winogrande) |  |
| 108 | [codah/codah](src/tasksource/tasks.py#L395) | MultipleChoice | [jaredfern/codah](https://hf.co/datasets/jaredfern/codah) |  |
| 109 | [ai2_arc/ARC-Challenge/challenge](src/tasksource/tasks.py#L397) | MultipleChoice | [allenai/ai2_arc](https://hf.co/datasets/allenai/ai2_arc) |  |
| 110 | [ai2_arc/ARC-Easy/challenge](src/tasksource/tasks.py#L397) | MultipleChoice | [allenai/ai2_arc](https://hf.co/datasets/allenai/ai2_arc) |  |
| 111 | [definite_pronoun_resolution](src/tasksource/tasks.py#L402) | MultipleChoice | [community-datasets/definite_pronoun_resolution](https://hf.co/datasets/community-datasets/definite_pronoun_resolution) |  |
| 112 | [swag/regular](src/tasksource/tasks.py#L408) | MultipleChoice | [allenai/swag](https://hf.co/datasets/allenai/swag) |  |
| 113 | [math_qa](src/tasksource/tasks.py#L414) | MultipleChoice | [tasksource/math_qa](https://hf.co/datasets/tasksource/math_qa) |  |
| 114 | [glue/cola](src/tasksource/tasks.py#L423) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 115 | [glue/sst2](src/tasksource/tasks.py#L424) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 116 | [utilitarianism](src/tasksource/tasks.py#L438) | Classification | csv |  |
| 117 | [amazon_counterfactual/en](src/tasksource/tasks.py#L446) | Classification | [mteb/amazon_counterfactual](https://hf.co/datasets/mteb/amazon_counterfactual) |  |
| 118 | [insincere-questions](src/tasksource/tasks.py#L451) | Classification | [SetFit/insincere-questions](https://hf.co/datasets/SetFit/insincere-questions) |  |
| 119 | [toxic_conversations](src/tasksource/tasks.py#L455) | Classification | [SetFit/toxic_conversations](https://hf.co/datasets/SetFit/toxic_conversations) |  |
| 120 | [TuringBench](src/tasksource/tasks.py#L459) | Classification | csv | ✓ |
| 121 | [trec](src/tasksource/tasks.py#L469) | Classification | [tasksource/trec](https://hf.co/datasets/tasksource/trec) |  |
| 122 | [vitaminc](src/tasksource/tasks.py#L472) | Classification | [tals/vitaminc](https://hf.co/datasets/tals/vitaminc) |  |
| 123 | [hope_edi/english](src/tasksource/tasks.py#L474) | Classification | csv |  |
| 124 | [rumoureval_2019/RumourEval2019](src/tasksource/tasks.py#L488) | Classification | csv |  |
| 125 | [ethos/binary](src/tasksource/tasks.py#L502) | Classification | [SetFit/ethos_binary](https://hf.co/datasets/SetFit/ethos_binary) |  |
| 126 | [ethos/multilabel](src/tasksource/tasks.py#L526) | Classification | [tasksource/ethos](https://hf.co/datasets/tasksource/ethos) |  |
| 127 | [tweet_eval/offensive](src/tasksource/tasks.py#L529) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 128 | [tweet_eval/irony](src/tasksource/tasks.py#L529) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 129 | [tweet_eval/hate](src/tasksource/tasks.py#L529) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 130 | [tweet_eval/emotion](src/tasksource/tasks.py#L529) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 131 | [tweet_eval/sentiment](src/tasksource/tasks.py#L529) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 132 | [tweet_eval/emoji](src/tasksource/tasks.py#L529) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 133 | [tweet_eval/stance_abortion](src/tasksource/tasks.py#L544) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 134 | [tweet_eval/stance_atheism](src/tasksource/tasks.py#L545) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 135 | [tweet_eval/stance_climate](src/tasksource/tasks.py#L546) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 136 | [tweet_eval/stance_feminist](src/tasksource/tasks.py#L547) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 137 | [tweet_eval/stance_hillary](src/tasksource/tasks.py#L548) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 138 | [discovery/discovery](src/tasksource/tasks.py#L551) | Classification | [sileod/discovery](https://hf.co/datasets/sileod/discovery) | ✓ |
| 139 | [pragmeval/mrda](src/tasksource/tasks.py#L554) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 140 | [pragmeval/switchboard](src/tasksource/tasks.py#L554) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 141 | [pragmeval/verifiability](src/tasksource/tasks.py#L554) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 142 | [pragmeval/persuasiveness-premisetype](src/tasksource/tasks.py#L558) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 143 | [pragmeval/sarcasm](src/tasksource/tasks.py#L558) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 144 | [pragmeval/stac](src/tasksource/tasks.py#L558) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 145 | [pragmeval/emergent](src/tasksource/tasks.py#L558) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 146 | [pragmeval/persuasiveness-claimtype](src/tasksource/tasks.py#L558) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 147 | [pragmeval/gum](src/tasksource/tasks.py#L558) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 148 | [pragmeval/pdtb](src/tasksource/tasks.py#L558) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 149 | [pragmeval/emobank-arousal](src/tasksource/tasks.py#L571) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 150 | [pragmeval/emobank-dominance](src/tasksource/tasks.py#L572) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 151 | [pragmeval/emobank-valence](src/tasksource/tasks.py#L573) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 152 | [pragmeval/squinky-formality](src/tasksource/tasks.py#L574) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 153 | [pragmeval/squinky-implicature](src/tasksource/tasks.py#L575) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 154 | [pragmeval/squinky-informativeness](src/tasksource/tasks.py#L576) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 155 | [pragmeval/persuasiveness-eloquence](src/tasksource/tasks.py#L577) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 156 | [pragmeval/persuasiveness-relevance](src/tasksource/tasks.py#L578) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 157 | [pragmeval/persuasiveness-specificity](src/tasksource/tasks.py#L579) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 158 | [pragmeval/persuasiveness-strength](src/tasksource/tasks.py#L580) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 159 | [silicone/dyda_da](src/tasksource/tasks.py#L582) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| 160 | [silicone/dyda_e](src/tasksource/tasks.py#L582) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| 161 | [silicone/maptask](src/tasksource/tasks.py#L582) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| 162 | [silicone/sem](src/tasksource/tasks.py#L582) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| 163 | [silicone/oasis](src/tasksource/tasks.py#L582) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| 164 | [silicone/meld_s](src/tasksource/tasks.py#L582) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| 165 | [silicone/meld_e](src/tasksource/tasks.py#L582) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| 166 | [silicone/iemocap](src/tasksource/tasks.py#L589) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) |  |
| 167 | [lex_glue/eurlex](src/tasksource/tasks.py#L598) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) | ✓ |
| 168 | [lex_glue/scotus](src/tasksource/tasks.py#L602) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) |  |
| 169 | [lex_glue/ledgar](src/tasksource/tasks.py#L605) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) |  |
| 170 | [lex_glue/unfair_tos](src/tasksource/tasks.py#L607) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) | ✓ |
| 171 | [lex_glue/case_hold](src/tasksource/tasks.py#L610) | MultipleChoice | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) |  |
| 172 | [language-identification](src/tasksource/tasks.py#L618) | Classification | [papluca/language-identification](https://hf.co/datasets/papluca/language-identification) | ✓ |
| 173 | [imdb](src/tasksource/tasks.py#L623) | Classification | [stanfordnlp/imdb](https://hf.co/datasets/stanfordnlp/imdb) |  |
| 174 | [rotten_tomatoes](src/tasksource/tasks.py#L625) | Classification | [cornell-movie-review-data/rotten_tomatoes](https://hf.co/datasets/cornell-movie-review-data/rotten_tomatoes) |  |
| 175 | [ag_news](src/tasksource/tasks.py#L627) | Classification | [fancyzhx/ag_news](https://hf.co/datasets/fancyzhx/ag_news) |  |
| 176 | [yelp_review_full/yelp_review_full](src/tasksource/tasks.py#L629) | Classification | [Yelp/yelp_review_full](https://hf.co/datasets/Yelp/yelp_review_full) | ✓ |
| 177 | [financial_phrasebank/sentences_allagree](src/tasksource/tasks.py#L634) | Classification | [ghbacct/financial-phrasebank-all-agree-classification](https://hf.co/datasets/ghbacct/financial-phrasebank-all-agree-classification) |  |
| 178 | [poem_sentiment](src/tasksource/tasks.py#L639) | Classification | [google-research-datasets/poem_sentiment](https://hf.co/datasets/google-research-datasets/poem_sentiment) |  |
| 179 | [emotion](src/tasksource/tasks.py#L641) | Classification | [dair-ai/emotion](https://hf.co/datasets/dair-ai/emotion) |  |
| 180 | [dbpedia_14/dbpedia_14](src/tasksource/tasks.py#L643) | Classification | [fancyzhx/dbpedia_14](https://hf.co/datasets/fancyzhx/dbpedia_14) |  |
| 181 | [amazon_polarity/amazon_polarity](src/tasksource/tasks.py#L645) | Classification | [fancyzhx/amazon_polarity](https://hf.co/datasets/fancyzhx/amazon_polarity) |  |
| 182 | [app_reviews](src/tasksource/tasks.py#L647) | Classification | [sealuzh/app_reviews](https://hf.co/datasets/sealuzh/app_reviews) | ✓ |
| 183 | [hate_speech18](src/tasksource/tasks.py#L652) | Classification | [tasksource/hate_speech18](https://hf.co/datasets/tasksource/hate_speech18) |  |
| 184 | [sms_spam](src/tasksource/tasks.py#L658) | Classification | [ucirvine/sms_spam](https://hf.co/datasets/ucirvine/sms_spam) |  |
| 185 | [humicroedit/subtask-1](src/tasksource/tasks.py#L661) | Classification | [tasksource/humicroedit](https://hf.co/datasets/tasksource/humicroedit) | ✓ |
| 186 | [humicroedit/subtask-2](src/tasksource/tasks.py#L667) | Classification | [tasksource/humicroedit](https://hf.co/datasets/tasksource/humicroedit) | ✓ |
| 187 | [snips_built_in_intents](src/tasksource/tasks.py#L672) | Classification | [sonos-nlu-benchmark/snips_built_in_intents](https://hf.co/datasets/sonos-nlu-benchmark/snips_built_in_intents) |  |
| 188 | [hate_speech_offensive](src/tasksource/tasks.py#L676) | Classification | [tdavidson/hate_speech_offensive](https://hf.co/datasets/tdavidson/hate_speech_offensive) |  |
| 189 | [yahoo_answers_topics](src/tasksource/tasks.py#L678) | Classification | [community-datasets/yahoo_answers_topics](https://hf.co/datasets/community-datasets/yahoo_answers_topics) |  |
| 190 | [stackoverflow-questions](src/tasksource/tasks.py#L682) | Classification | [pacovaldez/stackoverflow-questions](https://hf.co/datasets/pacovaldez/stackoverflow-questions) | ✓ |
| 191 | [hyperpartisan_news](src/tasksource/tasks.py#L688) | Classification | [zapsdcn/hyperpartisan_news](https://hf.co/datasets/zapsdcn/hyperpartisan_news) |  |
| 192 | [sciie](src/tasksource/tasks.py#L693) | Classification | [zapsdcn/sciie](https://hf.co/datasets/zapsdcn/sciie) |  |
| 193 | [citation_intent](src/tasksource/tasks.py#L694) | Classification | [zapsdcn/citation_intent](https://hf.co/datasets/zapsdcn/citation_intent) |  |
| 194 | [go_emotions/simplified](src/tasksource/tasks.py#L696) | Classification | [google-research-datasets/go_emotions](https://hf.co/datasets/google-research-datasets/go_emotions) |  |
| 195 | [scicite](src/tasksource/tasks.py#L700) | Classification | [tasksource/scicite](https://hf.co/datasets/tasksource/scicite) |  |
| 196 | [liar](src/tasksource/tasks.py#L702) | Classification | [tasksource/liar](https://hf.co/datasets/tasksource/liar) | ✓ |
| 197 | [lexical_relation_classification/K&H+N](src/tasksource/tasks.py#L713) | Classification | json | ✓ |
| 198 | [lexical_relation_classification/BLESS](src/tasksource/tasks.py#L713) | Classification | json | ✓ |
| 199 | [lexical_relation_classification/EVALution](src/tasksource/tasks.py#L713) | Classification | json | ✓ |
| 200 | [lexical_relation_classification/ROOT09](src/tasksource/tasks.py#L713) | Classification | json | ✓ |
| 201 | [lexical_relation_classification/CogALexV](src/tasksource/tasks.py#L739) | Classification | json | ✓ |
| 202 | [linguisticprobing/subj_number](src/tasksource/tasks.py#L766) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 203 | [linguisticprobing/obj_number](src/tasksource/tasks.py#L767) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 204 | [linguisticprobing/past_present](src/tasksource/tasks.py#L768) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 205 | [linguisticprobing/top_constituents](src/tasksource/tasks.py#L769) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | ✓ |
| 206 | [linguisticprobing/tree_depth](src/tasksource/tasks.py#L771) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | ✓ |
| 207 | [linguisticprobing/coordination_inversion](src/tasksource/tasks.py#L772) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | ✓ |
| 208 | [linguisticprobing/odd_man_out](src/tasksource/tasks.py#L774) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | ✓ |
| 209 | [linguisticprobing/bigram_shift](src/tasksource/tasks.py#L775) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | ✓ |
| 210 | [crowdflower/text_emotion](src/tasksource/tasks.py#L777) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 211 | [crowdflower/economic-news](src/tasksource/tasks.py#L777) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 212 | [crowdflower/corporate-messaging](src/tasksource/tasks.py#L777) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 213 | [crowdflower/political-media-bias](src/tasksource/tasks.py#L777) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 214 | [crowdflower/political-media-audience](src/tasksource/tasks.py#L777) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 215 | [crowdflower/sentiment_nuclear_power](src/tasksource/tasks.py#L777) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 216 | [crowdflower/tweet_global_warming](src/tasksource/tasks.py#L777) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 217 | [crowdflower/airline-sentiment](src/tasksource/tasks.py#L777) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 218 | [crowdflower/political-media-message](src/tasksource/tasks.py#L777) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 219 | [ethics/commonsense](src/tasksource/tasks.py#L813) | Classification | csv |  |
| 220 | [ethics/deontology](src/tasksource/tasks.py#L821) | Classification | csv | ✓ |
| 221 | [ethics/justice](src/tasksource/tasks.py#L830) | Classification | csv |  |
| 222 | [ethics/virtue](src/tasksource/tasks.py#L838) | Classification | [hendrycks/ethics](https://hf.co/datasets/hendrycks/ethics) |  |
| 223 | [emo/emo2019](src/tasksource/tasks.py#L847) | Classification | [oneonlee/cleansed_emocontext](https://hf.co/datasets/oneonlee/cleansed_emocontext) |  |
| 224 | [google_wellformed_query](src/tasksource/tasks.py#L853) | Classification | [tasksource/google_wellformed_query](https://hf.co/datasets/tasksource/google_wellformed_query) | ✓ |
| 225 | [tweets_hate_speech_detection](src/tasksource/tasks.py#L859) | Classification | [tweets-hate-speech-detection/tweets_hate_speech_detection](https://hf.co/datasets/tweets-hate-speech-detection/tweets_hate_speech_detection) |  |
| 226 | [wnut_17/wnut_17](src/tasksource/tasks.py#L863) | TokenClassification | [flaitenberger/wnut_17](https://hf.co/datasets/flaitenberger/wnut_17) |  |
| 227 | [ncbi_disease/ncbi_disease](src/tasksource/tasks.py#L866) | TokenClassification | [ncbi/ncbi_disease](https://hf.co/datasets/ncbi/ncbi_disease) |  |
| 228 | [acronym_identification](src/tasksource/tasks.py#L869) | TokenClassification | [amirveyseh/acronym_identification](https://hf.co/datasets/amirveyseh/acronym_identification) |  |
| 229 | [jnlpba/jnlpba](src/tasksource/tasks.py#L872) | TokenClassification | [jnlpba/jnlpba](https://hf.co/datasets/jnlpba/jnlpba) |  |
| 230 | [ontonotes_english/SpeedOfMagic--ontonotes_english](src/tasksource/tasks.py#L879) | TokenClassification | [SpeedOfMagic/ontonotes_english](https://hf.co/datasets/SpeedOfMagic/ontonotes_english) |  |
| 231 | [blog_authorship_corpus/gender](src/tasksource/tasks.py#L883) | Classification | [tasksource/blog_authorship_corpus](https://hf.co/datasets/tasksource/blog_authorship_corpus) | ✓ |
| 232 | [blog_authorship_corpus/age](src/tasksource/tasks.py#L885) | Classification | [tasksource/blog_authorship_corpus](https://hf.co/datasets/tasksource/blog_authorship_corpus) | ✓ |
| 233 | [blog_authorship_corpus/job](src/tasksource/tasks.py#L888) | Classification | [tasksource/blog_authorship_corpus](https://hf.co/datasets/tasksource/blog_authorship_corpus) | ✓ |
| 234 | [open_question_type](src/tasksource/tasks.py#L899) | Classification | [Korea-MES/open_question_type](https://hf.co/datasets/Korea-MES/open_question_type) |  |
| 235 | [health_fact](src/tasksource/tasks.py#L901) | Classification | [marcov/health_fact_promptsource](https://hf.co/datasets/marcov/health_fact_promptsource) |  |
| 236 | [commonsense_qa](src/tasksource/tasks.py#L905) | MultipleChoice | [tau/commonsense_qa](https://hf.co/datasets/tau/commonsense_qa) |  |
| 237 | [mc_taco](src/tasksource/tasks.py#L911) | Classification | [marcov/mc_taco_promptsource](https://hf.co/datasets/marcov/mc_taco_promptsource) | ✓ |
| 238 | [ade_corpus_v2/Ade_corpus_v2_classification](src/tasksource/tasks.py#L918) | Classification | [ade-benchmark-corpus/ade_corpus_v2](https://hf.co/datasets/ade-benchmark-corpus/ade_corpus_v2) | ✓ |
| 239 | [discosense](src/tasksource/tasks.py#L921) | MultipleChoice | json |  |
| 240 | [circa](src/tasksource/tasks.py#L928) | Classification | [google-research-datasets/circa](https://hf.co/datasets/google-research-datasets/circa) |  |
| 241 | [code_x_glue_cc_defect_detection](src/tasksource/tasks.py#L933) | Classification | [google/code_x_glue_cc_defect_detection](https://hf.co/datasets/google/code_x_glue_cc_defect_detection) | ✓ |
| 242 | [phrase_similarity](src/tasksource/tasks.py#L938) | Classification | [Deehan1866/processed_phrase_similarity](https://hf.co/datasets/Deehan1866/processed_phrase_similarity) |  |
| 243 | [scientific-exaggeration-detection](src/tasksource/tasks.py#L946) | Classification | [copenlu/scientific-exaggeration-detection](https://hf.co/datasets/copenlu/scientific-exaggeration-detection) |  |
| 244 | [quarel](src/tasksource/tasks.py#L960) | MultipleChoice | [community-datasets/quarel](https://hf.co/datasets/community-datasets/quarel) | ✓ |
| 245 | [fever-evidence-related](src/tasksource/tasks.py#L967) | Classification | [mwong/fever-evidence-related](https://hf.co/datasets/mwong/fever-evidence-related) |  |
| 246 | [numer_sense](src/tasksource/tasks.py#L970) | Classification | [tasksource/numer_sense](https://hf.co/datasets/tasksource/numer_sense) |  |
| 247 | [dynasent/dynabench.dynasent.r1.all/r1](src/tasksource/tasks.py#L977) | Classification | [tasksource/dynasent](https://hf.co/datasets/tasksource/dynasent) |  |
| 248 | [dynasent/dynabench.dynasent.r2.all/r2](src/tasksource/tasks.py#L981) | Classification | [tasksource/dynasent](https://hf.co/datasets/tasksource/dynasent) |  |
| 249 | [Sarcasm_News_Headline](src/tasksource/tasks.py#L986) | Classification | [raquiba/Sarcasm_News_Headline](https://hf.co/datasets/raquiba/Sarcasm_News_Headline) |  |
| 250 | [sem_eval_2010_task_8](src/tasksource/tasks.py#L989) | Classification | [SemEvalWorkshop/sem_eval_2010_task_8](https://hf.co/datasets/SemEvalWorkshop/sem_eval_2010_task_8) |  |
| 251 | [auditor_review](src/tasksource/tasks.py#L991) | Classification | [demo-org/auditor_review](https://hf.co/datasets/demo-org/auditor_review) |  |
| 252 | [medmcqa](src/tasksource/tasks.py#L995) | MultipleChoice | [openlifescienceai/medmcqa](https://hf.co/datasets/openlifescienceai/medmcqa) |  |
| 253 | [Dynasent_Disagreement](src/tasksource/tasks.py#L1012) | Classification | [RuyuanWan/Dynasent_Disagreement](https://hf.co/datasets/RuyuanWan/Dynasent_Disagreement) | ✓ |
| 254 | [Politeness_Disagreement](src/tasksource/tasks.py#L1014) | Classification | [RuyuanWan/Politeness_Disagreement](https://hf.co/datasets/RuyuanWan/Politeness_Disagreement) | ✓ |
| 255 | [SBIC_Disagreement](src/tasksource/tasks.py#L1016) | Classification | [RuyuanWan/SBIC_Disagreement](https://hf.co/datasets/RuyuanWan/SBIC_Disagreement) | ✓ |
| 256 | [SChem_Disagreement](src/tasksource/tasks.py#L1018) | Classification | [RuyuanWan/SChem_Disagreement](https://hf.co/datasets/RuyuanWan/SChem_Disagreement) | ✓ |
| 257 | [Dilemmas_Disagreement](src/tasksource/tasks.py#L1020) | Classification | [RuyuanWan/Dilemmas_Disagreement](https://hf.co/datasets/RuyuanWan/Dilemmas_Disagreement) | ✓ |
| 258 | [logiqa](src/tasksource/tasks.py#L1023) | MultipleChoice | [fireworks-ai/logiqa](https://hf.co/datasets/fireworks-ai/logiqa) |  |
| 259 | [wiki_qa](src/tasksource/tasks.py#L1032) | Classification | [microsoft/wiki_qa](https://hf.co/datasets/microsoft/wiki_qa) | ✓ |
| 260 | [cycic_classification](src/tasksource/tasks.py#L1034) | Classification | [tasksource/cycic_classification](https://hf.co/datasets/tasksource/cycic_classification) |  |
| 261 | [cycic_multiplechoice](src/tasksource/tasks.py#L1036) | MultipleChoice | [tasksource/cycic_multiplechoice](https://hf.co/datasets/tasksource/cycic_multiplechoice) |  |
| 262 | [sts-companion](src/tasksource/tasks.py#L1040) | Classification | [tasksource/sts-companion](https://hf.co/datasets/tasksource/sts-companion) |  |
| 263 | [commonsense_qa_2.0](src/tasksource/tasks.py#L1043) | Classification | [tasksource/commonsense_qa_2.0](https://hf.co/datasets/tasksource/commonsense_qa_2.0) |  |
| 264 | [lingnli](src/tasksource/tasks.py#L1046) | Classification | [tasksource/lingnli](https://hf.co/datasets/tasksource/lingnli) |  |
| 265 | [monotonicity-entailment](src/tasksource/tasks.py#L1048) | Classification | [tasksource/monotonicity-entailment](https://hf.co/datasets/tasksource/monotonicity-entailment) |  |
| 266 | [arct](src/tasksource/tasks.py#L1051) | MultipleChoice | [tasksource/arct](https://hf.co/datasets/tasksource/arct) |  |
| 267 | [scinli](src/tasksource/tasks.py#L1054) | Classification | [tasksource/scinli](https://hf.co/datasets/tasksource/scinli) |  |
| 268 | [naturallogic](src/tasksource/tasks.py#L1058) | Classification | [tasksource/naturallogic](https://hf.co/datasets/tasksource/naturallogic) |  |
| 269 | [onestop_qa](src/tasksource/tasks.py#L1060) | MultipleChoice | [malmaud/onestop_qa](https://hf.co/datasets/malmaud/onestop_qa) |  |
| 270 | [moral_stories/full](src/tasksource/tasks.py#L1063) | MultipleChoice | [LabHC/moral_stories](https://hf.co/datasets/LabHC/moral_stories) |  |
| 271 | [prost](src/tasksource/tasks.py#L1071) | MultipleChoice | json |  |
| 272 | [dynahate](src/tasksource/tasks.py#L1076) | Classification | [tasksource/dynahate](https://hf.co/datasets/tasksource/dynahate) |  |
| 273 | [syntactic-augmentation-nli](src/tasksource/tasks.py#L1078) | Classification | [tasksource/syntactic-augmentation-nli](https://hf.co/datasets/tasksource/syntactic-augmentation-nli) |  |
| 274 | [autotnli](src/tasksource/tasks.py#L1080) | Classification | [tasksource/autotnli](https://hf.co/datasets/tasksource/autotnli) |  |
| 275 | [CONDAQA](src/tasksource/tasks.py#L1082) | Classification | [lasha-nlp/CONDAQA](https://hf.co/datasets/lasha-nlp/CONDAQA) |  |
| 276 | [webgpt_comparisons](src/tasksource/tasks.py#L1092) | MultipleChoice | [heegyu/webgpt_comparisons_ko](https://hf.co/datasets/heegyu/webgpt_comparisons_ko) | ✓ |
| 277 | [synthetic-instruct-gptj-pairwise](src/tasksource/tasks.py#L1100) | MultipleChoice | [Dahoas/synthetic-instruct-gptj-pairwise](https://hf.co/datasets/Dahoas/synthetic-instruct-gptj-pairwise) | ✓ |
| 278 | [scruples](src/tasksource/tasks.py#L1103) | Classification | [tasksource/scruples](https://hf.co/datasets/tasksource/scruples) | ✓ |
| 279 | [defeasible-nli/atomic](src/tasksource/tasks.py#L1108) | Classification | [tasksource/defeasible-nli](https://hf.co/datasets/tasksource/defeasible-nli) |  |
| 280 | [defeasible-nli/snli](src/tasksource/tasks.py#L1108) | Classification | [tasksource/defeasible-nli](https://hf.co/datasets/tasksource/defeasible-nli) |  |
| 281 | [defeasible-nli/social](src/tasksource/tasks.py#L1111) | Classification | [tasksource/defeasible-nli](https://hf.co/datasets/tasksource/defeasible-nli) |  |
| 282 | [help-nli](src/tasksource/tasks.py#L1114) | Classification | [tasksource/help-nli](https://hf.co/datasets/tasksource/help-nli) |  |
| 283 | [nli-veridicality-transitivity](src/tasksource/tasks.py#L1117) | Classification | [tasksource/nli-veridicality-transitivity](https://hf.co/datasets/tasksource/nli-veridicality-transitivity) |  |
| 284 | [lonli](src/tasksource/tasks.py#L1120) | Classification | [tasksource/lonli](https://hf.co/datasets/tasksource/lonli) |  |
| 285 | [dadc-limit-nli](src/tasksource/tasks.py#L1123) | Classification | [tasksource/dadc-limit-nli](https://hf.co/datasets/tasksource/dadc-limit-nli) |  |
| 286 | [FLUTE](src/tasksource/tasks.py#L1126) | Classification | [ColumbiaNLP/FLUTE](https://hf.co/datasets/ColumbiaNLP/FLUTE) |  |
| 287 | [strategy-qa](src/tasksource/tasks.py#L1129) | Classification | [tasksource/strategy-qa](https://hf.co/datasets/tasksource/strategy-qa) |  |
| 288 | [summarize_from_feedback/comparisons](src/tasksource/tasks.py#L1132) | MultipleChoice | [vwxyzjn/summarize_from_feedback_oai_preprocessing](https://hf.co/datasets/vwxyzjn/summarize_from_feedback_oai_preprocessing) | ✓ |
| 289 | [folio](src/tasksource/tasks.py#L1140) | Classification | [tasksource/folio](https://hf.co/datasets/tasksource/folio) |  |
| 290 | [tomi-nli](src/tasksource/tasks.py#L1167) | Classification | [tasksource/tomi-nli](https://hf.co/datasets/tasksource/tomi-nli) |  |
| 291 | [avicenna](src/tasksource/tasks.py#L1170) | Classification | [tasksource/avicenna](https://hf.co/datasets/tasksource/avicenna) | ✓ |
| 292 | [SHP](src/tasksource/tasks.py#L1173) | MultipleChoice | [stanfordnlp/SHP](https://hf.co/datasets/stanfordnlp/SHP) | ✓ |
| 293 | [MedQA-USMLE-4-options-hf](src/tasksource/tasks.py#L1181) | MultipleChoice | [GBaker/MedQA-USMLE-4-options-hf](https://hf.co/datasets/GBaker/MedQA-USMLE-4-options-hf) |  |
| 294 | [wikimedqa/medwiki](src/tasksource/tasks.py#L1184) | MultipleChoice | [sileod/wikimedqa](https://hf.co/datasets/sileod/wikimedqa) |  |
| 295 | [cicero](src/tasksource/tasks.py#L1193) | MultipleChoice | [declare-lab/cicero](https://hf.co/datasets/declare-lab/cicero) |  |
| 296 | [CREAK](src/tasksource/tasks.py#L1197) | Classification | [amydeng2000/CREAK](https://hf.co/datasets/amydeng2000/CREAK) |  |
| 297 | [mutual](src/tasksource/tasks.py#L1200) | MultipleChoice | [tasksource/mutual](https://hf.co/datasets/tasksource/mutual) |  |
| 298 | [implicatures](src/tasksource/tasks.py#L1205) | MultipleChoice | [tasksource/implicatures](https://hf.co/datasets/tasksource/implicatures) |  |
| 299 | [race/high](src/tasksource/tasks.py#L1210) | MultipleChoice | [ehovy/race](https://hf.co/datasets/ehovy/race) |  |
| 300 | [race/middle](src/tasksource/tasks.py#L1210) | MultipleChoice | [ehovy/race](https://hf.co/datasets/ehovy/race) |  |
| 301 | [race-c](src/tasksource/tasks.py#L1214) | MultipleChoice | [tasksource/race-c](https://hf.co/datasets/tasksource/race-c) |  |
| 302 | [spartqa-yn](src/tasksource/tasks.py#L1217) | Classification | [tasksource/spartqa-yn](https://hf.co/datasets/tasksource/spartqa-yn) |  |
| 303 | [spartqa-mchoice](src/tasksource/tasks.py#L1220) | MultipleChoice | [tasksource/spartqa-mchoice](https://hf.co/datasets/tasksource/spartqa-mchoice) |  |
| 304 | [temporal-nli](src/tasksource/tasks.py#L1223) | Classification | [tasksource/temporal-nli](https://hf.co/datasets/tasksource/temporal-nli) |  |
| 305 | [riddle_sense](src/tasksource/tasks.py#L1226) | MultipleChoice | [jeggers/riddle_sense](https://hf.co/datasets/jeggers/riddle_sense) |  |
| 306 | [clcd-english](src/tasksource/tasks.py#L1231) | Classification | [tasksource/clcd-english](https://hf.co/datasets/tasksource/clcd-english) |  |
| 307 | [twentyquestions](src/tasksource/tasks.py#L1243) | Classification | [tasksource/twentyquestions](https://hf.co/datasets/tasksource/twentyquestions) |  |
| 308 | [reclor](src/tasksource/tasks.py#L1248) | MultipleChoice | [tasksource/reclor](https://hf.co/datasets/tasksource/reclor) |  |
| 309 | [counterfactually-augmented-imdb](src/tasksource/tasks.py#L1251) | Classification | [tasksource/counterfactually-augmented-imdb](https://hf.co/datasets/tasksource/counterfactually-augmented-imdb) |  |
| 310 | [counterfactually-augmented-snli](src/tasksource/tasks.py#L1254) | Classification | [tasksource/counterfactually-augmented-snli](https://hf.co/datasets/tasksource/counterfactually-augmented-snli) |  |
| 311 | [cnli](src/tasksource/tasks.py#L1257) | Classification | [tasksource/cnli](https://hf.co/datasets/tasksource/cnli) |  |
| 312 | [boolq-natural-perturbations](src/tasksource/tasks.py#L1260) | Classification | [tasksource/boolq-natural-perturbations](https://hf.co/datasets/tasksource/boolq-natural-perturbations) |  |
| 313 | [acceptability-prediction](src/tasksource/tasks.py#L1265) | Classification | [tasksource/acceptability-prediction](https://hf.co/datasets/tasksource/acceptability-prediction) | ✓ |
| 314 | [equate](src/tasksource/tasks.py#L1291) | Classification | [tasksource/equate](https://hf.co/datasets/tasksource/equate) |  |
| 315 | [ScienceQA_text_only](src/tasksource/tasks.py#L1294) | MultipleChoice | [tasksource/ScienceQA_text_only](https://hf.co/datasets/tasksource/ScienceQA_text_only) |  |
| 316 | [ekar_english](src/tasksource/tasks.py#L1297) | MultipleChoice | [Jiangjie/ekar_english](https://hf.co/datasets/Jiangjie/ekar_english) | ✓ |
| 317 | [implicit-hate-stg1](src/tasksource/tasks.py#L1301) | Classification | [tasksource/implicit-hate-stg1](https://hf.co/datasets/tasksource/implicit-hate-stg1) | ✓ |
| 318 | [chaos-mnli-ambiguity](src/tasksource/tasks.py#L1305) | Classification | [tasksource/chaos-mnli-ambiguity](https://hf.co/datasets/tasksource/chaos-mnli-ambiguity) | ✓ |
| 319 | [headline_cause/en_simple](src/tasksource/tasks.py#L1309) | Classification | json |  |
| 320 | [logiqa-2.0-nli](src/tasksource/tasks.py#L1314) | Classification | [tasksource/logiqa-2.0-nli](https://hf.co/datasets/tasksource/logiqa-2.0-nli) |  |
| 321 | [oasst2_dense_flat/quality](src/tasksource/tasks.py#L1319) | Classification | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) | ✓ |
| 322 | [oasst2_dense_flat/toxicity](src/tasksource/tasks.py#L1321) | Classification | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) | ✓ |
| 323 | [oasst2_dense_flat/helpfulness](src/tasksource/tasks.py#L1323) | Classification | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) | ✓ |
| 324 | [mindgames](src/tasksource/tasks.py#L1326) | Classification | [sileod/mindgames](https://hf.co/datasets/sileod/mindgames) |  |
| 325 | [universal_dependencies/en_partut/deprel](src/tasksource/tasks.py#L1340) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| 326 | [universal_dependencies/en_lines/deprel](src/tasksource/tasks.py#L1340) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| 327 | [universal_dependencies/en_gum/deprel](src/tasksource/tasks.py#L1340) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| 328 | [universal_dependencies/en_ewt/deprel](src/tasksource/tasks.py#L1340) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| 329 | [ambient](src/tasksource/tasks.py#L1346) | Classification | [tasksource/ambient](https://hf.co/datasets/tasksource/ambient) | ✓ |
| 330 | [path-naturalness-prediction](src/tasksource/tasks.py#L1351) | MultipleChoice | [tasksource/path-naturalness-prediction](https://hf.co/datasets/tasksource/path-naturalness-prediction) | ✓ |
| 331 | [civil_comments/toxicity](src/tasksource/tasks.py#L1361) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 332 | [civil_comments/severe_toxicity](src/tasksource/tasks.py#L1362) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 333 | [civil_comments/obscene](src/tasksource/tasks.py#L1363) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 334 | [civil_comments/threat](src/tasksource/tasks.py#L1364) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 335 | [civil_comments/insult](src/tasksource/tasks.py#L1365) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 336 | [civil_comments/identity_attack](src/tasksource/tasks.py#L1366) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 337 | [civil_comments/sexual_explicit](src/tasksource/tasks.py#L1367) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 338 | [cloth](src/tasksource/tasks.py#L1369) | MultipleChoice | [AndyChiang/cloth](https://hf.co/datasets/AndyChiang/cloth) |  |
| 339 | [dgen](src/tasksource/tasks.py#L1370) | MultipleChoice | [AndyChiang/dgen](https://hf.co/datasets/AndyChiang/dgen) |  |
| 340 | [I2D2](src/tasksource/tasks.py#L1372) | Classification | [tasksource/I2D2](https://hf.co/datasets/tasksource/I2D2) | ✓ |
| 341 | [args_me](src/tasksource/tasks.py#L1377) | Classification | [webis/args_me](https://hf.co/datasets/webis/args_me) | ✓ |
| 342 | [Touche23-ValueEval](src/tasksource/tasks.py#L1382) | Classification | csv | ✓ |
| 343 | [starcon](src/tasksource/tasks.py#L1391) | Classification | [tasksource/starcon](https://hf.co/datasets/tasksource/starcon) | ✓ |
| 344 | [banking77](src/tasksource/tasks.py#L1396) | Classification | [legacy-datasets/banking77](https://hf.co/datasets/legacy-datasets/banking77) |  |
| 345 | [it-support-tickets](src/tasksource/tasks.py#L1398) | Classification | [tasksource/it-support-tickets](https://hf.co/datasets/tasksource/it-support-tickets) |  |
| 346 | [ConTRoL-nli](src/tasksource/tasks.py#L1402) | Classification | [tasksource/ConTRoL-nli](https://hf.co/datasets/tasksource/ConTRoL-nli) |  |
| 347 | [tracie](src/tasksource/tasks.py#L1404) | Classification | [tasksource/tracie](https://hf.co/datasets/tasksource/tracie) | ✓ |
| 348 | [sherliic](src/tasksource/tasks.py#L1406) | Classification | [tasksource/sherliic](https://hf.co/datasets/tasksource/sherliic) |  |
| 349 | [sen-making/1](src/tasksource/tasks.py#L1408) | MultipleChoice | [tasksource/sen-making](https://hf.co/datasets/tasksource/sen-making) | ✓ |
| 350 | [sen-making/2](src/tasksource/tasks.py#L1412) | MultipleChoice | [tasksource/sen-making](https://hf.co/datasets/tasksource/sen-making) | ✓ |
| 351 | [winowhy](src/tasksource/tasks.py#L1415) | Classification | [tasksource/winowhy](https://hf.co/datasets/tasksource/winowhy) | ✓ |
| 352 | [robustLR](src/tasksource/tasks.py#L1419) | Classification | [tasksource/robustLR](https://hf.co/datasets/tasksource/robustLR) |  |
| 353 | [clutrr](src/tasksource/tasks.py#L1421) | Classification | [tasksource/clutrr](https://hf.co/datasets/tasksource/clutrr) |  |
| 354 | [logical-fallacy](src/tasksource/tasks.py#L1423) | Classification | [tasksource/logical-fallacy](https://hf.co/datasets/tasksource/logical-fallacy) |  |
| 355 | [parade](src/tasksource/tasks.py#L1425) | Classification | [tasksource/parade](https://hf.co/datasets/tasksource/parade) |  |
| 356 | [subjectivity](src/tasksource/tasks.py#L1428) | Classification | [tasksource/subjectivity](https://hf.co/datasets/tasksource/subjectivity) | ✓ |
| 357 | [MOH](src/tasksource/tasks.py#L1434) | Classification | [tasksource/MOH](https://hf.co/datasets/tasksource/MOH) | ✓ |
| 358 | [VUAC](src/tasksource/tasks.py#L1435) | Classification | [tasksource/VUAC](https://hf.co/datasets/tasksource/VUAC) | ✓ |
| 359 | [TroFi](src/tasksource/tasks.py#L1436) | Classification | parquet | ✓ |
| 360 | [sharc](src/tasksource/tasks.py#L1444) | Classification | [tasksource/sharc](https://hf.co/datasets/tasksource/sharc) |  |
| 361 | [conceptrules_v2](src/tasksource/tasks.py#L1448) | Classification | [tasksource/conceptrules_v2](https://hf.co/datasets/tasksource/conceptrules_v2) | ✓ |
| 362 | [disrpt/eng.dep.scidtb.rels](src/tasksource/tasks.py#L1450) | Classification | [multilingual-discourse-hub/disrpt](https://hf.co/datasets/multilingual-discourse-hub/disrpt) | ✓ |
| 363 | [conll2000](src/tasksource/tasks.py#L1453) | TokenClassification | [eriktks/conll2000](https://hf.co/datasets/eriktks/conll2000) |  |
| 364 | [few-nerd/supervised](src/tasksource/tasks.py#L1456) | TokenClassification | [DFKI-SLT/few-nerd](https://hf.co/datasets/DFKI-SLT/few-nerd) |  |
| 365 | [finer-139](src/tasksource/tasks.py#L1457) | TokenClassification | [nlpaueb/finer-139](https://hf.co/datasets/nlpaueb/finer-139) |  |
| 366 | [zero-shot-label-nli](src/tasksource/tasks.py#L1460) | Classification | [tasksource/zero-shot-label-nli](https://hf.co/datasets/tasksource/zero-shot-label-nli) |  |
| 367 | [com2sense](src/tasksource/tasks.py#L1462) | Classification | [tasksource/com2sense](https://hf.co/datasets/tasksource/com2sense) |  |
| 368 | [scone](src/tasksource/tasks.py#L1464) | Classification | [tasksource/scone](https://hf.co/datasets/tasksource/scone) |  |
| 369 | [winodict](src/tasksource/tasks.py#L1466) | MultipleChoice | [tasksource/winodict](https://hf.co/datasets/tasksource/winodict) |  |
| 370 | [fool-me-twice](src/tasksource/tasks.py#L1468) | Classification | [tasksource/fool-me-twice](https://hf.co/datasets/tasksource/fool-me-twice) |  |
| 371 | [monli](src/tasksource/tasks.py#L1472) | Classification | [tasksource/monli](https://hf.co/datasets/tasksource/monli) |  |
| 372 | [corr2cause](src/tasksource/tasks.py#L1474) | Classification | [tasksource/corr2cause](https://hf.co/datasets/tasksource/corr2cause) |  |
| 373 | [lsat_qa/all](src/tasksource/tasks.py#L1476) | MultipleChoice | [lighteval/lsat_qa](https://hf.co/datasets/lighteval/lsat_qa) |  |
| 374 | [apt](src/tasksource/tasks.py#L1478) | Classification | [tasksource/apt](https://hf.co/datasets/tasksource/apt) |  |
| 375 | [twitter-financial-news-sentiment](src/tasksource/tasks.py#L1481) | Classification | [zeroshot/twitter-financial-news-sentiment](https://hf.co/datasets/zeroshot/twitter-financial-news-sentiment) |  |
| 376 | [icl-symbol-tuning-instruct](src/tasksource/tasks.py#L1488) | Classification | [tasksource/icl-symbol-tuning-instruct](https://hf.co/datasets/tasksource/icl-symbol-tuning-instruct) | ✓ |
| 377 | [SpaceNLI](src/tasksource/tasks.py#L1494) | Classification | [tasksource/SpaceNLI](https://hf.co/datasets/tasksource/SpaceNLI) |  |
| 378 | [propsegment/nli](src/tasksource/tasks.py#L1496) | Classification | json |  |
| 379 | [HatemojiBuild](src/tasksource/tasks.py#L1505) | Classification | [HannahRoseKirk/HatemojiBuild](https://hf.co/datasets/HannahRoseKirk/HatemojiBuild) |  |
| 380 | [regset](src/tasksource/tasks.py#L1508) | Classification | [tasksource/regset](https://hf.co/datasets/tasksource/regset) | ✓ |
| 381 | [esci](src/tasksource/tasks.py#L1514) | Classification | [tasksource/esci](https://hf.co/datasets/tasksource/esci) |  |
| 382 | [chatbot_arena_conversations](src/tasksource/tasks.py#L1533) | MultipleChoice | [lmsys/chatbot_arena_conversations](https://hf.co/datasets/lmsys/chatbot_arena_conversations) | ✓ |
| 383 | [dnd_style_intents](src/tasksource/tasks.py#L1539) | Classification | [neurae/dnd_style_intents](https://hf.co/datasets/neurae/dnd_style_intents) |  |
| 384 | [FLD.v2/default](src/tasksource/tasks.py#L1542) | Classification | [hitachi-nlp/FLD.v2](https://hf.co/datasets/hitachi-nlp/FLD.v2) | ✓ |
| 385 | [FLD.v2/star](src/tasksource/tasks.py#L1546) | Classification | [hitachi-nlp/FLD.v2](https://hf.co/datasets/hitachi-nlp/FLD.v2) | ✓ |
| 386 | [SDOH-NLI](src/tasksource/tasks.py#L1550) | Classification | [tasksource/SDOH-NLI](https://hf.co/datasets/tasksource/SDOH-NLI) |  |
| 387 | [scifact_entailment](src/tasksource/tasks.py#L1553) | Classification | [tasksource/scifact_entailment](https://hf.co/datasets/tasksource/scifact_entailment) |  |
| 388 | [feasibilityQA](src/tasksource/tasks.py#L1557) | Classification | [tasksource/feasibilityQA](https://hf.co/datasets/tasksource/feasibilityQA) |  |
| 389 | [simple_pair](src/tasksource/tasks.py#L1560) | Classification | [tasksource/simple_pair](https://hf.co/datasets/tasksource/simple_pair) |  |
| 390 | [AdjectiveScaleProbe-nli](src/tasksource/tasks.py#L1561) | Classification | [tasksource/AdjectiveScaleProbe-nli](https://hf.co/datasets/tasksource/AdjectiveScaleProbe-nli) |  |
| 391 | [resnli](src/tasksource/tasks.py#L1562) | Classification | [tasksource/resnli](https://hf.co/datasets/tasksource/resnli) |  |
| 392 | [SpaRTUN](src/tasksource/tasks.py#L1564) | MultipleChoice | [tasksource/SpaRTUN](https://hf.co/datasets/tasksource/SpaRTUN) |  |
| 393 | [ReSQ](src/tasksource/tasks.py#L1569) | MultipleChoice | [tasksource/ReSQ](https://hf.co/datasets/tasksource/ReSQ) |  |
| 394 | [semantic_fragments_nli](src/tasksource/tasks.py#L1574) | Classification | [tasksource/semantic_fragments_nli](https://hf.co/datasets/tasksource/semantic_fragments_nli) |  |
| 395 | [dataset_train_nli](src/tasksource/tasks.py#L1577) | Classification | [MoritzLaurer/dataset_train_nli](https://hf.co/datasets/MoritzLaurer/dataset_train_nli) |  |
| 396 | [stepgame](src/tasksource/tasks.py#L1582) | Classification | [tasksource/stepgame](https://hf.co/datasets/tasksource/stepgame) |  |
| 397 | [nlgraph](src/tasksource/tasks.py#L1590) | Classification | [tasksource/nlgraph](https://hf.co/datasets/tasksource/nlgraph) |  |
| 398 | [oasst2_pairwise_rlhf_reward](src/tasksource/tasks.py#L1594) | MultipleChoice | [tasksource/oasst2_pairwise_rlhf_reward](https://hf.co/datasets/tasksource/oasst2_pairwise_rlhf_reward) | ✓ |
| 399 | [hh-rlhf/helpful-rejection-sampled](src/tasksource/tasks.py#L1605) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| 400 | [hh-rlhf/helpful-base](src/tasksource/tasks.py#L1605) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| 401 | [hh-rlhf/helpful-online](src/tasksource/tasks.py#L1605) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| 402 | [hh-rlhf/harmless-base](src/tasksource/tasks.py#L1609) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| 403 | [ruletaker](src/tasksource/tasks.py#L1613) | Classification | [tasksource/ruletaker](https://hf.co/datasets/tasksource/ruletaker) | ✓ |
| 404 | [PARARULE-Plus](src/tasksource/tasks.py#L1617) | Classification | [qbao775/PARARULE-Plus](https://hf.co/datasets/qbao775/PARARULE-Plus) | ✓ |
| 405 | [proofwriter](src/tasksource/tasks.py#L1621) | Classification | [tasksource/proofwriter](https://hf.co/datasets/tasksource/proofwriter) |  |
| 406 | [logical-entailment](src/tasksource/tasks.py#L1624) | Classification | [tasksource/logical-entailment](https://hf.co/datasets/tasksource/logical-entailment) | ✓ |
| 407 | [nope](src/tasksource/tasks.py#L1627) | Classification | [tasksource/nope](https://hf.co/datasets/tasksource/nope) |  |
| 408 | [LogicNLI](src/tasksource/tasks.py#L1631) | Classification | [tasksource/LogicNLI](https://hf.co/datasets/tasksource/LogicNLI) | ✓ |
| 409 | [contract-nli/contractnli_a/seg](src/tasksource/tasks.py#L1634) | Classification | [tasksource/contract-nli](https://hf.co/datasets/tasksource/contract-nli) |  |
| 410 | [contract-nli/contractnli_b/full](src/tasksource/tasks.py#L1636) | Classification | [tasksource/contract-nli](https://hf.co/datasets/tasksource/contract-nli) |  |
| 411 | [nli4ct_semeval2024](src/tasksource/tasks.py#L1638) | Classification | [AshtonIsNotHere/nli4ct_semeval2024](https://hf.co/datasets/AshtonIsNotHere/nli4ct_semeval2024) |  |
| 412 | [lsat-ar](src/tasksource/tasks.py#L1641) | MultipleChoice | [tasksource/lsat-ar](https://hf.co/datasets/tasksource/lsat-ar) |  |
| 413 | [lsat-rc](src/tasksource/tasks.py#L1646) | MultipleChoice | [tasksource/lsat-rc](https://hf.co/datasets/tasksource/lsat-rc) |  |
| 414 | [biosift-nli](src/tasksource/tasks.py#L1651) | Classification | [AshtonIsNotHere/biosift-nli](https://hf.co/datasets/AshtonIsNotHere/biosift-nli) |  |
| 415 | [brainteasers/WP](src/tasksource/tasks.py#L1655) | MultipleChoice | [tasksource/brainteasers](https://hf.co/datasets/tasksource/brainteasers) |  |
| 416 | [brainteasers/SP](src/tasksource/tasks.py#L1655) | MultipleChoice | [tasksource/brainteasers](https://hf.co/datasets/tasksource/brainteasers) |  |
| 417 | [toxigen-data/annotated](src/tasksource/tasks.py#L1661) | Classification | [skg/toxigen-data](https://hf.co/datasets/skg/toxigen-data) |  |
| 418 | [persuasion](src/tasksource/tasks.py#L1672) | Classification | [Anthropic/persuasion](https://hf.co/datasets/Anthropic/persuasion) |  |
| 419 | [AmbigNQ-clarifying-question](src/tasksource/tasks.py#L1678) | Classification | [erbacher/AmbigNQ-clarifying-question](https://hf.co/datasets/erbacher/AmbigNQ-clarifying-question) | ✓ |
| 420 | [SIGA-nli](src/tasksource/tasks.py#L1683) | Classification | [tasksource/SIGA-nli](https://hf.co/datasets/tasksource/SIGA-nli) |  |
| 421 | [FOL-nli](src/tasksource/tasks.py#L1685) | Classification | [unigram/FOL-nli](https://hf.co/datasets/unigram/FOL-nli) |  |
| 422 | [goal-step-wikihow/goal](src/tasksource/tasks.py#L1687) | MultipleChoice | [tasksource/goal-step-wikihow](https://hf.co/datasets/tasksource/goal-step-wikihow) |  |
| 423 | [goal-step-wikihow/step](src/tasksource/tasks.py#L1690) | MultipleChoice | [tasksource/goal-step-wikihow](https://hf.co/datasets/tasksource/goal-step-wikihow) |  |
| 424 | [goal-step-wikihow/order](src/tasksource/tasks.py#L1693) | MultipleChoice | [tasksource/goal-step-wikihow](https://hf.co/datasets/tasksource/goal-step-wikihow) | ✓ |
| 425 | [PARADISE](src/tasksource/tasks.py#L1697) | MultipleChoice | [GGLab/PARADISE](https://hf.co/datasets/GGLab/PARADISE) |  |
| 426 | [doc-nli](src/tasksource/tasks.py#L1700) | Classification | [tasksource/doc-nli](https://hf.co/datasets/tasksource/doc-nli) |  |
| 427 | [mctest-nli](src/tasksource/tasks.py#L1702) | Classification | [tasksource/mctest-nli](https://hf.co/datasets/tasksource/mctest-nli) |  |
| 428 | [patent-phrase-similarity](src/tasksource/tasks.py#L1704) | Classification | [tasksource/patent-phrase-similarity](https://hf.co/datasets/tasksource/patent-phrase-similarity) | ✓ |
| 429 | [natural-language-satisfiability](src/tasksource/tasks.py#L1707) | Classification | [tasksource/natural-language-satisfiability](https://hf.co/datasets/tasksource/natural-language-satisfiability) |  |
| 430 | [idioms-nli](src/tasksource/tasks.py#L1709) | Classification | [tasksource/idioms-nli](https://hf.co/datasets/tasksource/idioms-nli) |  |
| 431 | [lifecycle-entailment](src/tasksource/tasks.py#L1711) | Classification | [tasksource/lifecycle-entailment](https://hf.co/datasets/tasksource/lifecycle-entailment) |  |
| 432 | [safe-guard-prompt-injection](src/tasksource/tasks.py#L1718) | Classification | [xTRam1/safe-guard-prompt-injection](https://hf.co/datasets/xTRam1/safe-guard-prompt-injection) | ✓ |
| 433 | [prompt-injections](src/tasksource/tasks.py#L1724) | Classification | [deepset/prompt-injections](https://hf.co/datasets/deepset/prompt-injections) | ✓ |
| 434 | [prompt-injection-dataset](src/tasksource/tasks.py#L1730) | Classification | [S-Labs/prompt-injection-dataset](https://hf.co/datasets/S-Labs/prompt-injection-dataset) | ✓ |
| 435 | [Prompt-injection-dataset/full](src/tasksource/tasks.py#L1736) | Classification | [neuralchemy/Prompt-injection-dataset](https://hf.co/datasets/neuralchemy/Prompt-injection-dataset) | ✓ |
| 436 | [PromptShield](src/tasksource/tasks.py#L1742) | Classification | [hendzh/PromptShield](https://hf.co/datasets/hendzh/PromptShield) | ✓ |
| 437 | [shell-safety-v2](src/tasksource/tasks.py#L1748) | Classification | [tomngdev/shell-safety-v2](https://hf.co/datasets/tomngdev/shell-safety-v2) | ✓ |
| 438 | [agent_action_safety](src/tasksource/tasks.py#L1753) | Classification | json | ✓ |
| 439 | [ShellRisk-Bench](src/tasksource/tasks.py#L1768) | Classification | [kontext-security/ShellRisk-Bench](https://hf.co/datasets/kontext-security/ShellRisk-Bench) | ✓ |
| 440 | [wildguardmix-cleaned/prompt_harm](src/tasksource/tasks.py#L1793) | Classification | [bogdanminko/wildguardmix-cleaned](https://hf.co/datasets/bogdanminko/wildguardmix-cleaned) | ✓ |
| 441 | [wildguardmix-cleaned/response_harm](src/tasksource/tasks.py#L1799) | Classification | [bogdanminko/wildguardmix-cleaned](https://hf.co/datasets/bogdanminko/wildguardmix-cleaned) | ✓ |
| 442 | [wildguardmix-cleaned/response_refusal](src/tasksource/tasks.py#L1804) | Classification | [bogdanminko/wildguardmix-cleaned](https://hf.co/datasets/bogdanminko/wildguardmix-cleaned) | ✓ |
| 443 | [BeaverTails](src/tasksource/tasks.py#L1820) | Classification | [PKU-Alignment/BeaverTails](https://hf.co/datasets/PKU-Alignment/BeaverTails) | ✓ |
| 444 | [privacy-200k-Mistral-Large-3](src/tasksource/tasks.py#L1836) | Classification | [gabrielloiseau/privacy-200k-Mistral-Large-3](https://hf.co/datasets/gabrielloiseau/privacy-200k-Mistral-Large-3) | ✓ |
| 445 | [toxic-chat/toxicchat0124/toxicity](src/tasksource/tasks.py#L1845) | Classification | [lmsys/toxic-chat](https://hf.co/datasets/lmsys/toxic-chat) | ✓ |
| 446 | [toxic-chat/toxicchat0124/jailbreaking](src/tasksource/tasks.py#L1850) | Classification | [lmsys/toxic-chat](https://hf.co/datasets/lmsys/toxic-chat) | ✓ |
| 447 | [clinc_oos/plus](src/tasksource/tasks.py#L1856) | Classification | [clinc/clinc_oos](https://hf.co/datasets/clinc/clinc_oos) |  |
| 448 | [IntentGrasp/all](src/tasksource/tasks.py#L1874) | MultipleChoice | [yuweiyin/IntentGrasp](https://hf.co/datasets/yuweiyin/IntentGrasp) |  |
| 449 | [few_rel/default](src/tasksource/tasks.py#L1912) | Classification | [tasksource/few_rel](https://hf.co/datasets/tasksource/few_rel) |  |
| 450 | [docred](src/tasksource/tasks.py#L1990) | Classification | json |  |
| 451 | [chemprot/chemprot_full_source](src/tasksource/tasks.py#L2025) | Classification | [bigbio/chemprot](https://hf.co/datasets/bigbio/chemprot) | ✓ |
| 452 | [PKU-SafeRLHF/helpfulness](src/tasksource/tasks.py#L2031) | MultipleChoice | [PKU-Alignment/PKU-SafeRLHF](https://hf.co/datasets/PKU-Alignment/PKU-SafeRLHF) | ✓ |
| 453 | [PKU-SafeRLHF/safety](src/tasksource/tasks.py#L2036) | MultipleChoice | [PKU-Alignment/PKU-SafeRLHF](https://hf.co/datasets/PKU-Alignment/PKU-SafeRLHF) | ✓ |
| 454 | [HelpSteer/helpfulness](src/tasksource/tasks.py#L2058) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 455 | [HelpSteer/correctness](src/tasksource/tasks.py#L2059) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 456 | [HelpSteer/coherence](src/tasksource/tasks.py#L2060) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 457 | [HelpSteer/complexity](src/tasksource/tasks.py#L2061) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 458 | [HelpSteer/verbosity](src/tasksource/tasks.py#L2062) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 459 | [HelpSteer2/helpfulness](src/tasksource/tasks.py#L2064) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 460 | [HelpSteer2/correctness](src/tasksource/tasks.py#L2065) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 461 | [HelpSteer2/coherence](src/tasksource/tasks.py#L2066) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 462 | [HelpSteer2/complexity](src/tasksource/tasks.py#L2067) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 463 | [HelpSteer2/verbosity](src/tasksource/tasks.py#L2068) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 464 | [HelpSteer3/preference](src/tasksource/tasks.py#L2073) | MultipleChoice | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | ✓ |
| 465 | [HelpSteer3/preference_strength](src/tasksource/tasks.py#L2086) | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | ✓ |
| 466 | [HelpSteer3/principle](src/tasksource/tasks.py#L2098) | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) |  |
| 467 | [HelpSteer3/edit_quality](src/tasksource/tasks.py#L2103) | MultipleChoice | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | ✓ |
| 468 | [HelpSteer3/feedback](src/tasksource/tasks.py#L2125) | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | ✓ |
| 469 | [MSciNLI](src/tasksource/tasks.py#L2131) | Classification | [sadat2307/MSciNLI](https://hf.co/datasets/sadat2307/MSciNLI) |  |
| 470 | [UltraFeedback-paired](src/tasksource/tasks.py#L2134) | MultipleChoice | [pushpdeep/UltraFeedback-paired](https://hf.co/datasets/pushpdeep/UltraFeedback-paired) | ✓ |
| 471 | [prm800k_dpo/solution](src/tasksource/tasks.py#L2138) | MultipleChoice | [tasksource/prm800k_dpo](https://hf.co/datasets/tasksource/prm800k_dpo) | ✓ |
| 472 | [prm800k_dpo/step](src/tasksource/tasks.py#L2141) | MultipleChoice | [tasksource/prm800k_dpo](https://hf.co/datasets/tasksource/prm800k_dpo) | ✓ |
| 473 | [AES2-essay-scoring](src/tasksource/tasks.py#L2145) | Classification | [tasksource/AES2-essay-scoring](https://hf.co/datasets/tasksource/AES2-essay-scoring) | ✓ |
| 474 | [argument-feedback](src/tasksource/tasks.py#L2149) | Classification | [tasksource/argument-feedback](https://hf.co/datasets/tasksource/argument-feedback) | ✓ |
| 475 | [english-grading/cohesion](src/tasksource/tasks.py#L2156) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 476 | [english-grading/syntax](src/tasksource/tasks.py#L2157) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 477 | [english-grading/vocabulary](src/tasksource/tasks.py#L2158) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 478 | [english-grading/phraseology](src/tasksource/tasks.py#L2159) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 479 | [english-grading/grammar](src/tasksource/tasks.py#L2160) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 480 | [english-grading/conventions](src/tasksource/tasks.py#L2161) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 481 | [wice](src/tasksource/tasks.py#L2163) | Classification | [tasksource/wice](https://hf.co/datasets/tasksource/wice) |  |
| 482 | [hover](src/tasksource/tasks.py#L2166) | Classification | [Dzeniks/hover](https://hf.co/datasets/Dzeniks/hover) |  |
| 483 | [hover-3way/nli](src/tasksource/tasks.py#L2170) | Classification | [Dzeniks/hover-3way](https://hf.co/datasets/Dzeniks/hover-3way) |  |
| 484 | [tasksource_dpo_pairs](src/tasksource/tasks.py#L2173) | MultipleChoice | [tasksource/tasksource_dpo_pairs](https://hf.co/datasets/tasksource/tasksource_dpo_pairs) | ✓ |
| 485 | [seahorse_summarization_evaluation](src/tasksource/tasks.py#L2176) | Classification | [tasksource/seahorse_summarization_evaluation](https://hf.co/datasets/tasksource/seahorse_summarization_evaluation) |  |
| 486 | [missing-item-prediction/contrastive](src/tasksource/tasks.py#L2188) | Classification | [sileod/missing-item-prediction](https://hf.co/datasets/sileod/missing-item-prediction) | ✓ |
| 487 | [jigsaw_toxicity](src/tasksource/tasks.py#L2194) | Classification | [tasksource/jigsaw_toxicity](https://hf.co/datasets/tasksource/jigsaw_toxicity) |  |
| 488 | [Pol_NLI](src/tasksource/tasks.py#L2197) | Classification | [mlburnham/Pol_NLI](https://hf.co/datasets/mlburnham/Pol_NLI) |  |
| 489 | [synthetic-retrieval-NLI/position](src/tasksource/tasks.py#L2200) | Classification | [tasksource/synthetic-retrieval-NLI](https://hf.co/datasets/tasksource/synthetic-retrieval-NLI) |  |
| 490 | [synthetic-retrieval-NLI/count](src/tasksource/tasks.py#L2200) | Classification | [tasksource/synthetic-retrieval-NLI](https://hf.co/datasets/tasksource/synthetic-retrieval-NLI) |  |
| 491 | [synthetic-retrieval-NLI/binary](src/tasksource/tasks.py#L2200) | Classification | [tasksource/synthetic-retrieval-NLI](https://hf.co/datasets/tasksource/synthetic-retrieval-NLI) |  |
| 492 | [github-issue-similarity](src/tasksource/tasks.py#L2209) | Classification | [WhereIsAI/github-issue-similarity](https://hf.co/datasets/WhereIsAI/github-issue-similarity) |  |
| 493 | [webinstruct/mc](src/tasksource/tasks.py#L2414) | MultipleChoice | [tasksource/webinstruct](https://hf.co/datasets/tasksource/webinstruct) |  |
| 494 | [webinstruct/binary](src/tasksource/tasks.py#L2416) | Classification | [tasksource/webinstruct](https://hf.co/datasets/tasksource/webinstruct) |  |
| 495 | [weblinx/action](src/tasksource/tasks.py#L2480) | Classification | [McGill-NLP/WebLINX](https://hf.co/datasets/McGill-NLP/WebLINX) | ✓ |
| 496 | [weblinx/dom-element](src/tasksource/tasks.py#L2483) | MultipleChoice | [McGill-NLP/WebLINX](https://hf.co/datasets/McGill-NLP/WebLINX) | ✓ |
| 497 | [websrc/yesno](src/tasksource/tasks.py#L2489) | Classification | [tasksource/websrc](https://hf.co/datasets/tasksource/websrc) |  |
| 498 | [websrc/element](src/tasksource/tasks.py#L2491) | MultipleChoice | [tasksource/websrc](https://hf.co/datasets/tasksource/websrc) | ✓ |
| 499 | [mind2web/action](src/tasksource/tasks.py#L2498) | Classification | [tasksource/mind2web-dom](https://hf.co/datasets/tasksource/mind2web-dom) | ✓ |
| 500 | [mind2web/dom-element](src/tasksource/tasks.py#L2501) | MultipleChoice | [tasksource/mind2web-dom](https://hf.co/datasets/tasksource/mind2web-dom) | ✓ |
| 501 | [sentfin](src/tasksource/tasks.py#L2506) | Classification | [tasksource/sentfin](https://hf.co/datasets/tasksource/sentfin) | ✓ |
| 502 | [wands](src/tasksource/tasks.py#L2510) | Classification | [tasksource/wands](https://hf.co/datasets/tasksource/wands) | ✓ |
| 503 | [scirepeval/search](src/tasksource/tasks.py#L2514) | Classification | [tasksource/scirepeval-search](https://hf.co/datasets/tasksource/scirepeval-search) | ✓ |
| 504 | [maud/consideration](src/tasksource/tasks.py#L2566) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 505 | [maud/bringdown-timing](src/tasksource/tasks.py#L2570) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 506 | [maud/ordinary-course-consent](src/tasksource/tasks.py#L2574) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 507 | [maud/ordinary-course-efforts](src/tasksource/tasks.py#L2578) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 508 | [maud/antitrust-efforts](src/tasksource/tasks.py#L2582) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 509 | [maud/specific-performance](src/tasksource/tasks.py#L2586) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 510 | [ddi/relation](src/tasksource/tasks.py#L2591) | Classification | [OpenMed/DDI-Corpus-Processed](https://hf.co/datasets/OpenMed/DDI-Corpus-Processed) | ✓ |
| 511 | [patent-classification/abstract](src/tasksource/tasks.py#L2598) | Classification | [ccdv/patent-classification](https://hf.co/datasets/ccdv/patent-classification) | ✓ |
| 512 | [tabfact](src/tasksource/tasks.py#L2605) | Classification | [wenhu/tab_fact](https://hf.co/datasets/wenhu/tab_fact) | ✓ |
| 513 | [infotabs](src/tasksource/tasks.py#L2613) | Classification | [table-benchmark/infotabs](https://hf.co/datasets/table-benchmark/infotabs) | ✓ |
| 514 | [squad-v2/answerability](src/tasksource/tasks.py#L2625) | Classification | [rajpurkar/squad_v2](https://hf.co/datasets/rajpurkar/squad_v2) | ✓ |
| 515 | [cuad/answerability](src/tasksource/tasks.py#L2630) | Classification | [theatticusproject/cuad-qa](https://hf.co/datasets/theatticusproject/cuad-qa) | ✓ |
| 516 | [hallmarks-of-cancer/yesno](src/tasksource/tasks.py#L2649) | Classification | [bigbio/hallmarks_of_cancer](https://hf.co/datasets/bigbio/hallmarks_of_cancer) | ✓ |
| 517 | [semeval2014-absa/restaurants](src/tasksource/tasks.py#L2671) | Classification | [alexcadillon/SemEval2014Task4](https://hf.co/datasets/alexcadillon/SemEval2014Task4) | ✓ |
| 518 | [semeval2014-absa/laptops](src/tasksource/tasks.py#L2671) | Classification | [alexcadillon/SemEval2014Task4](https://hf.co/datasets/alexcadillon/SemEval2014Task4) | ✓ |
| 519 | [record/candidate](src/tasksource/tasks.py#L2693) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) | ✓ |
| 520 | [multidogo/airline/intent](src/tasksource/tasks.py#L2716) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 521 | [multidogo/finance/intent](src/tasksource/tasks.py#L2716) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 522 | [multidogo/insurance/intent](src/tasksource/tasks.py#L2716) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 523 | [multidogo/media/intent](src/tasksource/tasks.py#L2716) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 524 | [multidogo/fastfood/intent](src/tasksource/tasks.py#L2716) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 525 | [multidogo/software/intent](src/tasksource/tasks.py#L2716) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 526 | [wikisql/aggregation](src/tasksource/tasks.py#L2744) | Classification | [Salesforce/wikisql](https://hf.co/datasets/Salesforce/wikisql) | ✓ |
| 527 | [wikisql/column](src/tasksource/tasks.py#L2749) | MultipleChoice | [Salesforce/wikisql](https://hf.co/datasets/Salesforce/wikisql) | ✓ |
| 528 | [wikisql/operator](src/tasksource/tasks.py#L2754) | Classification | [Salesforce/wikisql](https://hf.co/datasets/Salesforce/wikisql) | ✓ |
| 529 | [acord](src/tasksource/tasks.py#L2759) | Classification | [tasksource/acord](https://hf.co/datasets/tasksource/acord) | ✓ |
| 530 | [edos](src/tasksource/tasks.py#L2763) | Classification | [tasksource/edos](https://hf.co/datasets/tasksource/edos) | ✓ |
| 531 | [evidence-inference](src/tasksource/tasks.py#L2766) | Classification | [tasksource/evidence-inference](https://hf.co/datasets/tasksource/evidence-inference) | ✓ |
| 532 | [quac/answerability](src/tasksource/tasks.py#L2788) | Classification | [allenai/quac](https://hf.co/datasets/allenai/quac) | ✓ |
| 533 | [hatexplain/majority](src/tasksource/tasks.py#L2796) | Classification | [Hate-speech-CNERG/hatexplain](https://hf.co/datasets/Hate-speech-CNERG/hatexplain) | ✓ |
| 534 | [hotpotqa/supporting-fact](src/tasksource/tasks.py#L2813) | Classification | [hotpotqa/hotpot_qa](https://hf.co/datasets/hotpotqa/hotpot_qa) | ✓ |
| 535 | [multiwoz22/dialogue-act](src/tasksource/tasks.py#L2857) | Classification | [pfb30/multi_woz_v22](https://hf.co/datasets/pfb30/multi_woz_v22) | ✓ |
| 536 | [multiwoz22/categorical-slot](src/tasksource/tasks.py#L2860) | Classification | [pfb30/multi_woz_v22](https://hf.co/datasets/pfb30/multi_woz_v22) | ✓ |
| 537 | [schema-guided/intent](src/tasksource/tasks.py#L2863) | Classification | [google-research-datasets/schema_guided_dstc8](https://hf.co/datasets/google-research-datasets/schema_guided_dstc8) | ✓ |
| 538 | [coqa/yesno](src/tasksource/tasks.py#L2880) | Classification | [stanfordnlp/coqa](https://hf.co/datasets/stanfordnlp/coqa) | ✓ |
| 539 | [qasper/yesno](src/tasksource/tasks.py#L2901) | Classification | [allenai/qasper](https://hf.co/datasets/allenai/qasper) | ✓ |
| 540 | [topv2/intent](src/tasksource/tasks.py#L2913) | Classification | [WillHeld/top_v2](https://hf.co/datasets/WillHeld/top_v2) | ✓ |
| 541 | [slurp/intent](src/tasksource/tasks.py#L2921) | Classification | [tasksource/slurp](https://hf.co/datasets/tasksource/slurp) | ✓ |
| 542 | [mcscript2](src/tasksource/tasks.py#L2925) | MultipleChoice | [tasksource/mcscript2](https://hf.co/datasets/tasksource/mcscript2) |  |
| 543 | [fnc1/stance](src/tasksource/tasks.py#L2930) | Classification | [tasksource/fnc1](https://hf.co/datasets/tasksource/fnc1) | ✓ |
| 544 | [perspectrum/stance](src/tasksource/tasks.py#L2935) | Classification | [tasksource/perspectrum](https://hf.co/datasets/tasksource/perspectrum) | ✓ |
| 545 | [ukpconvarg/convincingness](src/tasksource/tasks.py#L2940) | MultipleChoice | [tasksource/ukpconvarg](https://hf.co/datasets/tasksource/ukpconvarg) | ✓ |
| 546 | [discogem/relation](src/tasksource/tasks.py#L2945) | Classification | [tasksource/discogem](https://hf.co/datasets/tasksource/discogem) | ✓ |
| 547 | [pubmed-rct/role](src/tasksource/tasks.py#L2953) | Classification | [tasksource/pubmed-rct](https://hf.co/datasets/tasksource/pubmed-rct) | ✓ |
| 548 | [maven-arg/role](src/tasksource/tasks.py#L2981) | Classification | [mneb/maven-arg](https://hf.co/datasets/mneb/maven-arg) | ✓ |
| 549 | [maven/event](src/tasksource/tasks.py#L3012) | Classification | [mneb/maven](https://hf.co/datasets/mneb/maven) | ✓ |
| 550 | [empathetic-dialogues/emotion](src/tasksource/tasks.py#L3016) | Classification | [facebook/empathetic_dialogues](https://hf.co/datasets/facebook/empathetic_dialogues) | ✓ |
| 551 | [olid/offensive](src/tasksource/tasks.py#L3024) | Classification | [christophsonntag/OLID](https://hf.co/datasets/christophsonntag/OLID) | ✓ |
| 552 | [isarcasm/intended](src/tasksource/tasks.py#L3030) | Classification | [viethq1906/isarcasm_2022_taskA_En](https://hf.co/datasets/viethq1906/isarcasm_2022_taskA_En) | ✓ |
| 553 | [magpie/idiomatic](src/tasksource/tasks.py#L3055) | Classification | [idioms-collaboration/magpie](https://hf.co/datasets/idioms-collaboration/magpie) | ✓ |
| 554 | [preco/coreference](src/tasksource/tasks.py#L3091) | Classification | [coref-data/preco_raw](https://hf.co/datasets/coref-data/preco_raw) | ✓ |
| 555 | [redial/preference](src/tasksource/tasks.py#L3111) | Classification | [community-datasets/re_dial](https://hf.co/datasets/community-datasets/re_dial) | ✓ |
| 556 | [sst5](src/tasksource/tasks.py#L3116) | Classification | [SetFit/sst5](https://hf.co/datasets/SetFit/sst5) | ✓ |
| 557 | [spider/column](src/tasksource/tasks.py#L3124) | MultipleChoice | [tasksource/spider](https://hf.co/datasets/tasksource/spider) | ✓ |
| 558 | [semcor](src/tasksource/tasks.py#L3128) | MultipleChoice | [tasksource/semcor](https://hf.co/datasets/tasksource/semcor) | ✓ |
| 559 | [streusle/noun](src/tasksource/tasks.py#L3131) | Classification | [tasksource/streusle](https://hf.co/datasets/tasksource/streusle) | ✓ |
| 560 | [streusle/verb](src/tasksource/tasks.py#L3135) | Classification | [tasksource/streusle](https://hf.co/datasets/tasksource/streusle) | ✓ |

## Soft labels

Annotations whose label is a distribution (annotator votes, rater shares, survey counts), loaded with `load_task(id, soft=True)`. Those with a hard view are also listed above, by their majority label; the others have soft labels only. `votes` are shares of annotators, `mean` a numeric value interpolated over score anchors (see each task for its meaning); annotators is the typical count per item (vote shares from fewer than five are coarse, and the Jev build leaves them out).

| id | kind | aggregation | annotators | default view | dataset |
|---|---|---|--:|---|---|
| [glue/stsb](src/tasksource/tasks.py#L48) | score | mean |  | regression | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |
| [sick/relatedness](src/tasksource/tasks.py#L87) | score | mean | 10 | regression | [tasksource/sick](https://hf.co/datasets/tasksource/sick) |
| [google_wellformed_query](src/tasksource/tasks.py#L853) | noul | votes | 5 | Classification | [tasksource/google_wellformed_query](https://hf.co/datasets/tasksource/google_wellformed_query) |
| [sts-companion](src/tasksource/tasks.py#L1040) | score | mean |  | regression | [tasksource/sts-companion](https://hf.co/datasets/tasksource/sts-companion) |
| [acceptability-prediction/rating_votes](src/tasksource/tasks.py#L1280) | score | votes | 15 |  | [tasksource/acceptability-prediction](https://hf.co/datasets/tasksource/acceptability-prediction) |
| [acceptability-prediction/binary_votes](src/tasksource/tasks.py#L1285) | noul | votes | 15 |  | [tasksource/acceptability-prediction](https://hf.co/datasets/tasksource/acceptability-prediction) |
| [HelpSteer3/individual_preferences](src/tasksource/tasks.py#L2092) | score | votes | 3 |  | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) |
| [HelpSteer3/feedback](src/tasksource/tasks.py#L2125) | score | votes | 3 | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) |
| [proto_qa/proto_qa](src/tasksource/tasks.py#L2219) | choice | votes | 100 |  | [community-datasets/proto_qa](https://hf.co/datasets/community-datasets/proto_qa) |
| [UNLI](src/tasksource/tasks.py#L2233) | noul | mean | 2 |  | [Zhengping/UNLI](https://hf.co/datasets/Zhengping/UNLI) |
| [chaos-mnli-ambiguity/votes](src/tasksource/tasks.py#L2237) | choice | votes | 100 |  | [tasksource/chaos-mnli-ambiguity](https://hf.co/datasets/tasksource/chaos-mnli-ambiguity) |
| [hate_speech_offensive/votes](src/tasksource/tasks.py#L2241) | choice | votes | 3 |  | [tdavidson/hate_speech_offensive](https://hf.co/datasets/tdavidson/hate_speech_offensive) |
| [scruples/verdict_votes](src/tasksource/tasks.py#L2249) | choice | votes | 8 |  | [tasksource/scruples](https://hf.co/datasets/tasksource/scruples) |
| [BeaverTails/unsafe_votes](src/tasksource/tasks.py#L2263) | noul | votes | 3 |  | [PKU-Alignment/BeaverTails](https://hf.co/datasets/PKU-Alignment/BeaverTails) |
| [dynasent/r1_votes](src/tasksource/tasks.py#L2274) | choice | votes | 5 |  | [dynabench/dynasent](https://hf.co/datasets/dynabench/dynasent) |
| [dynasent/r2_votes](src/tasksource/tasks.py#L2278) | choice | votes | 5 |  | [dynabench/dynasent](https://hf.co/datasets/dynabench/dynasent) |
| [hatexplain/votes](src/tasksource/tasks.py#L2283) | choice | votes | 3 |  | [Hate-speech-CNERG/hatexplain](https://hf.co/datasets/Hate-speech-CNERG/hatexplain) |
| [measuring-hate-speech/sentiment](src/tasksource/tasks.py#L2298) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/respect](src/tasksource/tasks.py#L2300) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/insult](src/tasksource/tasks.py#L2302) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/humiliate](src/tasksource/tasks.py#L2303) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/status](src/tasksource/tasks.py#L2304) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/dehumanize](src/tasksource/tasks.py#L2306) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/violence](src/tasksource/tasks.py#L2308) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/genocide](src/tasksource/tasks.py#L2309) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/attack_defend](src/tasksource/tasks.py#L2311) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/hatespeech](src/tasksource/tasks.py#L2313) | choice | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [wikipedia-detox/attack](src/tasksource/tasks.py#L2325) | choice | votes | 10 |  | [tasksource/wikipedia-detox-votes](https://hf.co/datasets/tasksource/wikipedia-detox-votes) |
| [wikipedia-detox/aggression](src/tasksource/tasks.py#L2327) | choice | votes | 10 |  | [tasksource/wikipedia-detox-votes](https://hf.co/datasets/tasksource/wikipedia-detox-votes) |
| [wikipedia-detox/aggression_score](src/tasksource/tasks.py#L2329) | score | votes | 10 |  | [tasksource/wikipedia-detox-votes](https://hf.co/datasets/tasksource/wikipedia-detox-votes) |
| [wikipedia-detox/toxicity](src/tasksource/tasks.py#L2333) | choice | votes | 10 |  | [tasksource/wikipedia-detox-votes](https://hf.co/datasets/tasksource/wikipedia-detox-votes) |
| [wikipedia-detox/toxicity_score](src/tasksource/tasks.py#L2335) | score | votes | 10 |  | [tasksource/wikipedia-detox-votes](https://hf.co/datasets/tasksource/wikipedia-detox-votes) |
| [lewidi/md_agreement](src/tasksource/tasks.py#L2350) | noul | votes | 5 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/hs_brexit](src/tasksource/tasks.py#L2351) | noul | votes | 6 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/armis](src/tasksource/tasks.py#L2353) | noul | votes | 3 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/conv_abuse](src/tasksource/tasks.py#L2355) | noul | votes | 3 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/mp](src/tasksource/tasks.py#L2357) | noul | votes | 5 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/csc](src/tasksource/tasks.py#L2358) | score | votes | 4 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/varierrnli/entailment](src/tasksource/tasks.py#L2366) | noul | votes | 4 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/varierrnli/neutral](src/tasksource/tasks.py#L2367) | noul | votes | 4 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/varierrnli/contradiction](src/tasksource/tasks.py#L2368) | noul | votes | 4 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [civil_comments/toxicity_share](src/tasksource/tasks.py#L2376) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/severe_toxicity_share](src/tasksource/tasks.py#L2377) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/obscene_share](src/tasksource/tasks.py#L2378) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/threat_share](src/tasksource/tasks.py#L2379) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/insult_share](src/tasksource/tasks.py#L2380) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/identity_attack_share](src/tasksource/tasks.py#L2381) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/sexual_explicit_share](src/tasksource/tasks.py#L2382) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [oasst2/quality](src/tasksource/tasks.py#L2399) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/helpfulness](src/tasksource/tasks.py#L2400) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/creativity](src/tasksource/tasks.py#L2401) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/humor](src/tasksource/tasks.py#L2402) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/toxicity](src/tasksource/tasks.py#L2403) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/violence](src/tasksource/tasks.py#L2404) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/spam](src/tasksource/tasks.py#L2405) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/fails_task](src/tasksource/tasks.py#L2406) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/not_appropriate](src/tasksource/tasks.py#L2407) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/hate_speech](src/tasksource/tasks.py#L2408) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/sexual_content](src/tasksource/tasks.py#L2409) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/pii](src/tasksource/tasks.py#L2410) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/lang_mismatch](src/tasksource/tasks.py#L2411) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [scirepeval/search](src/tasksource/tasks.py#L2514) | score | mean |  | regression | [tasksource/scirepeval-search](https://hf.co/datasets/tasksource/scirepeval-search) |
| [discogem/relation](src/tasksource/tasks.py#L2945) | choice | votes | 10 | Classification | [tasksource/discogem](https://hf.co/datasets/tasksource/discogem) |
