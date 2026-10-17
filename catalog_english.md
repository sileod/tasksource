578 English tasks. Load one with `load_task(id)`; the annotations are in [tasks.py](src/tasksource/tasks.py). Evaluation benchmarks are in [eval_only.py](src/tasksource/eval_only.py); other excluded annotations are in [parked.py](src/tasksource/parked.py).

Browse the [example catalog](docs/examples.md): one real example per task.

| # | id | type | dataset | question |
|--:|---|---|---|:-:|
| 1 | [glue/mnli](src/tasksource/tasks.py#L36) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 2 | [glue/qnli](src/tasksource/tasks.py#L37) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) | ✓ |
| 3 | [glue/rte](src/tasksource/tasks.py#L39) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 4 | [glue/wnli](src/tasksource/tasks.py#L40) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 5 | [glue/mrpc](src/tasksource/tasks.py#L42) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 6 | [glue/qqp](src/tasksource/tasks.py#L43) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 7 | [glue/stsb](src/tasksource/tasks.py#L49) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) | ✓ |
| 8 | [super_glue/boolq](src/tasksource/tasks.py#L54) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 9 | [super_glue/boolq_passage](src/tasksource/tasks.py#L55) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 10 | [super_glue/cb](src/tasksource/tasks.py#L57) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 11 | [super_glue/multirc](src/tasksource/tasks.py#L58) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) | ✓ |
| 12 | [super_glue/wic](src/tasksource/tasks.py#L64) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 13 | [super_glue/axg](src/tasksource/tasks.py#L69) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 14 | [anli/a1](src/tasksource/tasks.py#L72) | Classification | [facebook/anli](https://hf.co/datasets/facebook/anli) |  |
| 15 | [anli/a2](src/tasksource/tasks.py#L73) | Classification | [facebook/anli](https://hf.co/datasets/facebook/anli) |  |
| 16 | [anli/a3](src/tasksource/tasks.py#L74) | Classification | [facebook/anli](https://hf.co/datasets/facebook/anli) |  |
| 17 | [babi_nli/conjunction](src/tasksource/tasks.py#L77) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 18 | [babi_nli/simple-negation](src/tasksource/tasks.py#L77) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 19 | [babi_nli/positional-reasoning](src/tasksource/tasks.py#L77) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 20 | [babi_nli/size-reasoning](src/tasksource/tasks.py#L77) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 21 | [babi_nli/three-arg-relations](src/tasksource/tasks.py#L77) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 22 | [babi_nli/three-supporting-facts](src/tasksource/tasks.py#L77) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 23 | [babi_nli/time-reasoning](src/tasksource/tasks.py#L77) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 24 | [babi_nli/two-arg-relations](src/tasksource/tasks.py#L77) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 25 | [babi_nli/path-finding](src/tasksource/tasks.py#L77) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 26 | [babi_nli/lists-sets](src/tasksource/tasks.py#L77) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 27 | [babi_nli/yes-no-questions](src/tasksource/tasks.py#L77) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 28 | [babi_nli/indefinite-knowledge](src/tasksource/tasks.py#L77) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 29 | [babi_nli/counting](src/tasksource/tasks.py#L77) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 30 | [babi_nli/single-supporting-fact](src/tasksource/tasks.py#L77) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 31 | [babi_nli/compound-coreference](src/tasksource/tasks.py#L77) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 32 | [babi_nli/two-supporting-facts](src/tasksource/tasks.py#L77) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 33 | [babi_nli/basic-coreference](src/tasksource/tasks.py#L77) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 34 | [babi_nli/basic-deduction](src/tasksource/tasks.py#L77) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 35 | [babi_nli/basic-induction](src/tasksource/tasks.py#L82) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) | ✓ |
| 36 | [sick/label](src/tasksource/tasks.py#L87) | Classification | [tasksource/sick](https://hf.co/datasets/tasksource/sick) |  |
| 37 | [sick/relatedness](src/tasksource/tasks.py#L89) | Classification | [tasksource/sick](https://hf.co/datasets/tasksource/sick) | ✓ |
| 38 | [snli](src/tasksource/tasks.py#L147) | Classification | [stanfordnlp/snli](https://hf.co/datasets/stanfordnlp/snli) |  |
| 39 | [scitail/snli_format](src/tasksource/tasks.py#L150) | Classification | [allenai/scitail](https://hf.co/datasets/allenai/scitail) |  |
| 40 | [hans](src/tasksource/tasks.py#L152) | Classification | [tasksource/hans](https://hf.co/datasets/tasksource/hans) |  |
| 41 | [WANLI](src/tasksource/tasks.py#L155) | Classification | [alisawuffles/WANLI](https://hf.co/datasets/alisawuffles/WANLI) |  |
| 42 | [recast/recast_factuality](src/tasksource/tasks.py#L157) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 43 | [recast/recast_ner](src/tasksource/tasks.py#L157) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 44 | [recast/recast_sentiment](src/tasksource/tasks.py#L157) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 45 | [recast/recast_megaveridicality](src/tasksource/tasks.py#L157) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 46 | [recast/recast_puns](src/tasksource/tasks.py#L157) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 47 | [recast/recast_verbnet](src/tasksource/tasks.py#L157) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 48 | [recast/recast_verbcorner](src/tasksource/tasks.py#L161) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 49 | [probability_words_nli/usnli](src/tasksource/tasks.py#L165) | Classification | [sileod/probability_words_nli](https://hf.co/datasets/sileod/probability_words_nli) | ✓ |
| 50 | [probability_words_nli/reasoning_2hop](src/tasksource/tasks.py#L165) | Classification | [sileod/probability_words_nli](https://hf.co/datasets/sileod/probability_words_nli) | ✓ |
| 51 | [probability_words_nli/reasoning_1hop](src/tasksource/tasks.py#L165) | Classification | [sileod/probability_words_nli](https://hf.co/datasets/sileod/probability_words_nli) | ✓ |
| 52 | [nan-nli](src/tasksource/tasks.py#L170) | Classification | [joey234/nan-nli](https://hf.co/datasets/joey234/nan-nli) |  |
| 53 | [nli_fever](src/tasksource/tasks.py#L173) | Classification | [pietrolesci/nli_fever](https://hf.co/datasets/pietrolesci/nli_fever) |  |
| 54 | [breaking_nli](src/tasksource/tasks.py#L176) | Classification | [pietrolesci/breaking_nli](https://hf.co/datasets/pietrolesci/breaking_nli) |  |
| 55 | [conj_nli](src/tasksource/tasks.py#L180) | Classification | [pietrolesci/conj_nli](https://hf.co/datasets/pietrolesci/conj_nli) |  |
| 56 | [fracas](src/tasksource/tasks.py#L184) | Classification | [pietrolesci/fracas](https://hf.co/datasets/pietrolesci/fracas) |  |
| 57 | [dialogue_nli](src/tasksource/tasks.py#L187) | Classification | [pietrolesci/dialogue_nli](https://hf.co/datasets/pietrolesci/dialogue_nli) | ✓ |
| 58 | [mpe](src/tasksource/tasks.py#L190) | Classification | [pietrolesci/mpe](https://hf.co/datasets/pietrolesci/mpe) |  |
| 59 | [dnc](src/tasksource/tasks.py#L194) | Classification | [pietrolesci/dnc](https://hf.co/datasets/pietrolesci/dnc) |  |
| 60 | [recast_white/fnplus](src/tasksource/tasks.py#L198) | Classification | [pietrolesci/recast_white](https://hf.co/datasets/pietrolesci/recast_white) |  |
| 61 | [recast_white/sprl](src/tasksource/tasks.py#L201) | Classification | [pietrolesci/recast_white](https://hf.co/datasets/pietrolesci/recast_white) |  |
| 62 | [recast_white/dpr](src/tasksource/tasks.py#L204) | Classification | [pietrolesci/recast_white](https://hf.co/datasets/pietrolesci/recast_white) |  |
| 63 | [joci](src/tasksource/tasks.py#L208) | Classification | [pietrolesci/joci](https://hf.co/datasets/pietrolesci/joci) | ✓ |
| 64 | [robust_nli/IS_CS](src/tasksource/tasks.py#L216) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 65 | [robust_nli/LI_LI](src/tasksource/tasks.py#L218) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 66 | [robust_nli/ST_WO](src/tasksource/tasks.py#L220) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 67 | [robust_nli/PI_SP](src/tasksource/tasks.py#L222) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 68 | [robust_nli/PI_CD](src/tasksource/tasks.py#L224) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 69 | [robust_nli/ST_SE](src/tasksource/tasks.py#L226) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 70 | [robust_nli/ST_NE](src/tasksource/tasks.py#L228) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 71 | [robust_nli/ST_LM](src/tasksource/tasks.py#L230) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 72 | [robust_nli_is_sd](src/tasksource/tasks.py#L232) | Classification | [pietrolesci/robust_nli_is_sd](https://hf.co/datasets/pietrolesci/robust_nli_is_sd) |  |
| 73 | [robust_nli_li_ts](src/tasksource/tasks.py#L235) | Classification | [pietrolesci/robust_nli_li_ts](https://hf.co/datasets/pietrolesci/robust_nli_li_ts) |  |
| 74 | [add_one_rte](src/tasksource/tasks.py#L240) | Classification | [pietrolesci/add_one_rte](https://hf.co/datasets/pietrolesci/add_one_rte) |  |
| 75 | [paws/labeled_final](src/tasksource/tasks.py#L244) | Classification | [google-research-datasets/paws](https://hf.co/datasets/google-research-datasets/paws) |  |
| 76 | [paws/labeled_swap](src/tasksource/tasks.py#L245) | Classification | [google-research-datasets/paws](https://hf.co/datasets/google-research-datasets/paws) |  |
| 77 | [medical_questions_pairs](src/tasksource/tasks.py#L247) | Classification | [curaihealth/medical_questions_pairs](https://hf.co/datasets/curaihealth/medical_questions_pairs) | ✓ |
| 78 | [conll2003/pos_tags](src/tasksource/tasks.py#L253) | TokenClassification | [tomaarsen/conll2003](https://hf.co/datasets/tomaarsen/conll2003) | ✓ |
| 79 | [conll2003/chunk_tags](src/tasksource/tasks.py#L255) | TokenClassification | [tomaarsen/conll2003](https://hf.co/datasets/tomaarsen/conll2003) | ✓ |
| 80 | [conll2003/ner_tags](src/tasksource/tasks.py#L257) | TokenClassification | [tomaarsen/conll2003](https://hf.co/datasets/tomaarsen/conll2003) |  |
| 81 | [fig-qa](src/tasksource/tasks.py#L263) | MultipleChoice | [nightingal3/fig-qa](https://hf.co/datasets/nightingal3/fig-qa) |  |
| 82 | [cosmos_qa](src/tasksource/tasks.py#L273) | MultipleChoice | [Samsoup/cosmos_qa](https://hf.co/datasets/Samsoup/cosmos_qa) |  |
| 83 | [dream](src/tasksource/tasks.py#L276) | MultipleChoice | [dataset-org/dream](https://hf.co/datasets/dataset-org/dream) |  |
| 84 | [openbookqa](src/tasksource/tasks.py#L283) | MultipleChoice | [allenai/openbookqa](https://hf.co/datasets/allenai/openbookqa) |  |
| 85 | [qasc](src/tasksource/tasks.py#L289) | MultipleChoice | [allenai/qasc](https://hf.co/datasets/allenai/qasc) |  |
| 86 | [quartz](src/tasksource/tasks.py#L297) | MultipleChoice | [allenai/quartz](https://hf.co/datasets/allenai/quartz) |  |
| 87 | [quail](src/tasksource/tasks.py#L302) | MultipleChoice | [textmachinelab/quail](https://hf.co/datasets/textmachinelab/quail) |  |
| 88 | [head_qa/en](src/tasksource/tasks.py#L308) | MultipleChoice | [EleutherAI/headqa](https://hf.co/datasets/EleutherAI/headqa) |  |
| 89 | [sciq](src/tasksource/tasks.py#L316) | MultipleChoice | [allenai/sciq](https://hf.co/datasets/allenai/sciq) |  |
| 90 | [social_i_qa](src/tasksource/tasks.py#L321) | MultipleChoice | [tasksource/social_i_qa](https://hf.co/datasets/tasksource/social_i_qa) |  |
| 91 | [wiki_hop/original](src/tasksource/tasks.py#L327) | MultipleChoice | [MoE-UNC/wikihop](https://hf.co/datasets/MoE-UNC/wikihop) |  |
| 92 | [wiqa](src/tasksource/tasks.py#L340) | MultipleChoice | [allenai/wiqa](https://hf.co/datasets/allenai/wiqa) (`parquet` loader) |  |
| 93 | [piqa](src/tasksource/tasks.py#L347) | MultipleChoice | [baber/piqa](https://hf.co/datasets/baber/piqa) |  |
| 94 | [hellaswag](src/tasksource/tasks.py#L355) | MultipleChoice | [Rowan/hellaswag](https://hf.co/datasets/Rowan/hellaswag) |  |
| 95 | [super_glue/copa](src/tasksource/tasks.py#L364) | MultipleChoice | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 96 | [balanced-copa](src/tasksource/tasks.py#L366) | MultipleChoice | [pkavumba/balanced-copa](https://hf.co/datasets/pkavumba/balanced-copa) |  |
| 97 | [e-CARE](src/tasksource/tasks.py#L369) | MultipleChoice | [12ml/e-CARE](https://hf.co/datasets/12ml/e-CARE) |  |
| 98 | [art](src/tasksource/tasks.py#L372) | MultipleChoice | [allenai/art](https://hf.co/datasets/allenai/art) | ✓ |
| 99 | [winogrande/winogrande_xl](src/tasksource/tasks.py#L380) | MultipleChoice | [allenai/winogrande](https://hf.co/datasets/allenai/winogrande) |  |
| 100 | [codah/codah](src/tasksource/tasks.py#L383) | MultipleChoice | [jaredfern/codah](https://hf.co/datasets/jaredfern/codah) |  |
| 101 | [ai2_arc/ARC-Challenge/challenge](src/tasksource/tasks.py#L385) | MultipleChoice | [allenai/ai2_arc](https://hf.co/datasets/allenai/ai2_arc) |  |
| 102 | [ai2_arc/ARC-Easy/challenge](src/tasksource/tasks.py#L385) | MultipleChoice | [allenai/ai2_arc](https://hf.co/datasets/allenai/ai2_arc) |  |
| 103 | [definite_pronoun_resolution](src/tasksource/tasks.py#L390) | MultipleChoice | [community-datasets/definite_pronoun_resolution](https://hf.co/datasets/community-datasets/definite_pronoun_resolution) |  |
| 104 | [swag/regular](src/tasksource/tasks.py#L396) | MultipleChoice | [allenai/swag](https://hf.co/datasets/allenai/swag) |  |
| 105 | [math_qa](src/tasksource/tasks.py#L402) | MultipleChoice | [tasksource/math_qa](https://hf.co/datasets/tasksource/math_qa) |  |
| 106 | [glue/cola](src/tasksource/tasks.py#L411) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 107 | [glue/sst2](src/tasksource/tasks.py#L412) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 108 | [utilitarianism](src/tasksource/tasks.py#L426) | Classification | [hendrycks/ethics](https://hf.co/datasets/hendrycks/ethics) (`csv` loader) |  |
| 109 | [amazon_counterfactual/en](src/tasksource/tasks.py#L434) | Classification | [mteb/amazon_counterfactual](https://hf.co/datasets/mteb/amazon_counterfactual) |  |
| 110 | [insincere-questions](src/tasksource/tasks.py#L439) | Classification | [SetFit/insincere-questions](https://hf.co/datasets/SetFit/insincere-questions) |  |
| 111 | [toxic_conversations](src/tasksource/tasks.py#L443) | Classification | [SetFit/toxic_conversations](https://hf.co/datasets/SetFit/toxic_conversations) |  |
| 112 | [TuringBench](src/tasksource/tasks.py#L447) | Classification | [jana4/turingbench-humanized](https://hf.co/datasets/jana4/turingbench-humanized), [turingbench/TuringBench](https://hf.co/datasets/turingbench/TuringBench) (`csv` loader) | ✓ |
| 113 | [trec](src/tasksource/tasks.py#L466) | Classification | [tasksource/trec](https://hf.co/datasets/tasksource/trec) | ✓ |
| 114 | [vitaminc](src/tasksource/tasks.py#L470) | Classification | [tals/vitaminc](https://hf.co/datasets/tals/vitaminc) |  |
| 115 | [hope_edi/english](src/tasksource/tasks.py#L472) | Classification | [dravidianlangtech/hope_edi](https://hf.co/datasets/dravidianlangtech/hope_edi) (`csv` loader) |  |
| 116 | [rumoureval_2019/RumourEval2019](src/tasksource/tasks.py#L486) | Classification | [strombergnlp/rumoureval_2019](https://hf.co/datasets/strombergnlp/rumoureval_2019) (`csv` loader) |  |
| 117 | [ethos/binary](src/tasksource/tasks.py#L500) | Classification | [SetFit/ethos_binary](https://hf.co/datasets/SetFit/ethos_binary) |  |
| 118 | [ethos/multilabel](src/tasksource/tasks.py#L524) | Classification | [tasksource/ethos](https://hf.co/datasets/tasksource/ethos) |  |
| 119 | [tweet_eval/irony](src/tasksource/tasks.py#L527) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 120 | [tweet_eval/emoji](src/tasksource/tasks.py#L527) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 121 | [tweet_eval/emotion](src/tasksource/tasks.py#L527) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 122 | [tweet_eval/hate](src/tasksource/tasks.py#L527) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 123 | [tweet_eval/sentiment](src/tasksource/tasks.py#L527) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 124 | [tweet_eval/offensive](src/tasksource/tasks.py#L527) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 125 | [tweet_eval/stance_abortion](src/tasksource/tasks.py#L542) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 126 | [tweet_eval/stance_atheism](src/tasksource/tasks.py#L543) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 127 | [tweet_eval/stance_climate](src/tasksource/tasks.py#L544) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 128 | [tweet_eval/stance_feminist](src/tasksource/tasks.py#L545) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 129 | [tweet_eval/stance_hillary](src/tasksource/tasks.py#L546) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 130 | [discovery/discovery](src/tasksource/tasks.py#L549) | Classification | [sileod/discovery](https://hf.co/datasets/sileod/discovery) | ✓ |
| 131 | [pragmeval/verifiability](src/tasksource/tasks.py#L552) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 132 | [pragmeval/switchboard](src/tasksource/tasks.py#L552) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 133 | [pragmeval/emergent](src/tasksource/tasks.py#L556) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 134 | [pragmeval/gum](src/tasksource/tasks.py#L556) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 135 | [pragmeval/pdtb](src/tasksource/tasks.py#L556) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 136 | [pragmeval/stac](src/tasksource/tasks.py#L556) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 137 | [pragmeval/persuasiveness-claimtype](src/tasksource/tasks.py#L556) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 138 | [pragmeval/persuasiveness-premisetype](src/tasksource/tasks.py#L556) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 139 | [pragmeval/sarcasm](src/tasksource/tasks.py#L556) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 140 | [pragmeval/emobank-dominance](src/tasksource/tasks.py#L571) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 141 | [pragmeval/squinky-formality](src/tasksource/tasks.py#L572) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 142 | [pragmeval/squinky-implicature](src/tasksource/tasks.py#L573) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 143 | [pragmeval/squinky-informativeness](src/tasksource/tasks.py#L574) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 144 | [pragmeval/persuasiveness-eloquence](src/tasksource/tasks.py#L575) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 145 | [pragmeval/persuasiveness-relevance](src/tasksource/tasks.py#L576) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 146 | [pragmeval/persuasiveness-specificity](src/tasksource/tasks.py#L577) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 147 | [pragmeval/persuasiveness-strength](src/tasksource/tasks.py#L578) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 148 | [silicone/dyda_da](src/tasksource/tasks.py#L580) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | ✓ |
| 149 | [silicone/meld_s](src/tasksource/tasks.py#L580) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | ✓ |
| 150 | [silicone/maptask](src/tasksource/tasks.py#L580) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | ✓ |
| 151 | [silicone/meld_e](src/tasksource/tasks.py#L580) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | ✓ |
| 152 | [silicone/oasis](src/tasksource/tasks.py#L580) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | ✓ |
| 153 | [silicone/sem](src/tasksource/tasks.py#L580) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | ✓ |
| 154 | [silicone/dyda_e](src/tasksource/tasks.py#L580) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | ✓ |
| 155 | [silicone/iemocap](src/tasksource/tasks.py#L590) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | ✓ |
| 156 | [lex_glue/eurlex](src/tasksource/tasks.py#L600) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) | ✓ |
| 157 | [lex_glue/scotus](src/tasksource/tasks.py#L604) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) |  |
| 158 | [lex_glue/ledgar](src/tasksource/tasks.py#L607) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) |  |
| 159 | [lex_glue/unfair_tos](src/tasksource/tasks.py#L609) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) | ✓ |
| 160 | [lex_glue/case_hold](src/tasksource/tasks.py#L612) | MultipleChoice | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) |  |
| 161 | [language-identification](src/tasksource/tasks.py#L620) | Classification | [papluca/language-identification](https://hf.co/datasets/papluca/language-identification) | ✓ |
| 162 | [imdb](src/tasksource/tasks.py#L625) | Classification | [stanfordnlp/imdb](https://hf.co/datasets/stanfordnlp/imdb) |  |
| 163 | [rotten_tomatoes](src/tasksource/tasks.py#L627) | Classification | [cornell-movie-review-data/rotten_tomatoes](https://hf.co/datasets/cornell-movie-review-data/rotten_tomatoes) |  |
| 164 | [ag_news](src/tasksource/tasks.py#L629) | Classification | [fancyzhx/ag_news](https://hf.co/datasets/fancyzhx/ag_news) |  |
| 165 | [yelp_review_full/yelp_review_full](src/tasksource/tasks.py#L631) | Classification | [Yelp/yelp_review_full](https://hf.co/datasets/Yelp/yelp_review_full) | ✓ |
| 166 | [financial_phrasebank/sentences_allagree](src/tasksource/tasks.py#L636) | Classification | [ghbacct/financial-phrasebank-all-agree-classification](https://hf.co/datasets/ghbacct/financial-phrasebank-all-agree-classification) |  |
| 167 | [poem_sentiment](src/tasksource/tasks.py#L641) | Classification | [google-research-datasets/poem_sentiment](https://hf.co/datasets/google-research-datasets/poem_sentiment) |  |
| 168 | [dbpedia_14/dbpedia_14](src/tasksource/tasks.py#L643) | Classification | [fancyzhx/dbpedia_14](https://hf.co/datasets/fancyzhx/dbpedia_14) |  |
| 169 | [amazon_polarity/amazon_polarity](src/tasksource/tasks.py#L645) | Classification | [fancyzhx/amazon_polarity](https://hf.co/datasets/fancyzhx/amazon_polarity) |  |
| 170 | [hate_speech18](src/tasksource/tasks.py#L649) | Classification | [tasksource/hate_speech18](https://hf.co/datasets/tasksource/hate_speech18) |  |
| 171 | [sms_spam](src/tasksource/tasks.py#L655) | Classification | [ucirvine/sms_spam](https://hf.co/datasets/ucirvine/sms_spam) |  |
| 172 | [humicroedit/subtask-1](src/tasksource/tasks.py#L658) | Classification | [tasksource/humicroedit](https://hf.co/datasets/tasksource/humicroedit) | ✓ |
| 173 | [humicroedit/subtask-2](src/tasksource/tasks.py#L664) | Classification | [tasksource/humicroedit](https://hf.co/datasets/tasksource/humicroedit) | ✓ |
| 174 | [snips_built_in_intents](src/tasksource/tasks.py#L669) | Classification | [sonos-nlu-benchmark/snips_built_in_intents](https://hf.co/datasets/sonos-nlu-benchmark/snips_built_in_intents) |  |
| 175 | [hate_speech_offensive](src/tasksource/tasks.py#L673) | Classification | [tdavidson/hate_speech_offensive](https://hf.co/datasets/tdavidson/hate_speech_offensive) |  |
| 176 | [yahoo_answers_topics](src/tasksource/tasks.py#L675) | Classification | [community-datasets/yahoo_answers_topics](https://hf.co/datasets/community-datasets/yahoo_answers_topics) |  |
| 177 | [stackoverflow-questions](src/tasksource/tasks.py#L679) | Classification | [pacovaldez/stackoverflow-questions](https://hf.co/datasets/pacovaldez/stackoverflow-questions) | ✓ |
| 178 | [hyperpartisan_news](src/tasksource/tasks.py#L685) | Classification | [zapsdcn/hyperpartisan_news](https://hf.co/datasets/zapsdcn/hyperpartisan_news) |  |
| 179 | [sciie](src/tasksource/tasks.py#L690) | Classification | [zapsdcn/sciie](https://hf.co/datasets/zapsdcn/sciie) |  |
| 180 | [citation_intent](src/tasksource/tasks.py#L691) | Classification | [zapsdcn/citation_intent](https://hf.co/datasets/zapsdcn/citation_intent) |  |
| 181 | [go_emotions/simplified](src/tasksource/tasks.py#L693) | Classification | [google-research-datasets/go_emotions](https://hf.co/datasets/google-research-datasets/go_emotions) |  |
| 182 | [scicite](src/tasksource/tasks.py#L697) | Classification | [tasksource/scicite](https://hf.co/datasets/tasksource/scicite) |  |
| 183 | [liar](src/tasksource/tasks.py#L699) | Classification | [tasksource/liar](https://hf.co/datasets/tasksource/liar) | ✓ |
| 184 | [lexical_relation_classification/ROOT09](src/tasksource/tasks.py#L710) | Classification | [relbert/lexical_relation_classification](https://hf.co/datasets/relbert/lexical_relation_classification) (`json` loader) | ✓ |
| 185 | [lexical_relation_classification/BLESS](src/tasksource/tasks.py#L710) | Classification | [relbert/lexical_relation_classification](https://hf.co/datasets/relbert/lexical_relation_classification) (`json` loader) | ✓ |
| 186 | [lexical_relation_classification/K&H+N](src/tasksource/tasks.py#L710) | Classification | [relbert/lexical_relation_classification](https://hf.co/datasets/relbert/lexical_relation_classification) (`json` loader) | ✓ |
| 187 | [lexical_relation_classification/CogALexV](src/tasksource/tasks.py#L736) | Classification | [relbert/lexical_relation_classification](https://hf.co/datasets/relbert/lexical_relation_classification) (`json` loader) | ✓ |
| 188 | [linguisticprobing/subj_number](src/tasksource/tasks.py#L763) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 189 | [linguisticprobing/obj_number](src/tasksource/tasks.py#L764) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 190 | [linguisticprobing/past_present](src/tasksource/tasks.py#L765) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 191 | [linguisticprobing/top_constituents](src/tasksource/tasks.py#L766) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | ✓ |
| 192 | [linguisticprobing/tree_depth](src/tasksource/tasks.py#L768) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | ✓ |
| 193 | [linguisticprobing/coordination_inversion](src/tasksource/tasks.py#L769) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | ✓ |
| 194 | [linguisticprobing/odd_man_out](src/tasksource/tasks.py#L771) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | ✓ |
| 195 | [linguisticprobing/bigram_shift](src/tasksource/tasks.py#L772) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | ✓ |
| 196 | [crowdflower/corporate-messaging](src/tasksource/tasks.py#L774) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 197 | [crowdflower/economic-news](src/tasksource/tasks.py#L774) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 198 | [crowdflower/tweet_global_warming](src/tasksource/tasks.py#L774) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 199 | [crowdflower/airline-sentiment](src/tasksource/tasks.py#L774) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 200 | [crowdflower/political-media-audience](src/tasksource/tasks.py#L789) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 201 | [crowdflower/political-media-bias](src/tasksource/tasks.py#L789) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 202 | [crowdflower/political-media-message](src/tasksource/tasks.py#L789) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 203 | [ethics/commonsense](src/tasksource/tasks.py#L812) | Classification | [hendrycks/ethics](https://hf.co/datasets/hendrycks/ethics) (`csv` loader) |  |
| 204 | [ethics/deontology](src/tasksource/tasks.py#L820) | Classification | [hendrycks/ethics](https://hf.co/datasets/hendrycks/ethics) (`csv` loader) | ✓ |
| 205 | [ethics/justice](src/tasksource/tasks.py#L829) | Classification | [hendrycks/ethics](https://hf.co/datasets/hendrycks/ethics) (`csv` loader) | ✓ |
| 206 | [ethics/virtue](src/tasksource/tasks.py#L838) | Classification | [hendrycks/ethics](https://hf.co/datasets/hendrycks/ethics) | ✓ |
| 207 | [emo/emo2019](src/tasksource/tasks.py#L848) | Classification | [oneonlee/cleansed_emocontext](https://hf.co/datasets/oneonlee/cleansed_emocontext) |  |
| 208 | [google_wellformed_query](src/tasksource/tasks.py#L854) | Classification | [tasksource/google_wellformed_query](https://hf.co/datasets/tasksource/google_wellformed_query) | ✓ |
| 209 | [tweets_hate_speech_detection](src/tasksource/tasks.py#L860) | Classification | [tweets-hate-speech-detection/tweets_hate_speech_detection](https://hf.co/datasets/tweets-hate-speech-detection/tweets_hate_speech_detection) |  |
| 210 | [wnut_17/wnut_17](src/tasksource/tasks.py#L864) | TokenClassification | [flaitenberger/wnut_17](https://hf.co/datasets/flaitenberger/wnut_17) |  |
| 211 | [ncbi_disease/ncbi_disease](src/tasksource/tasks.py#L867) | TokenClassification | [ncbi/ncbi_disease](https://hf.co/datasets/ncbi/ncbi_disease) | ✓ |
| 212 | [acronym_identification](src/tasksource/tasks.py#L872) | TokenClassification | [amirveyseh/acronym_identification](https://hf.co/datasets/amirveyseh/acronym_identification) | ✓ |
| 213 | [jnlpba/jnlpba](src/tasksource/tasks.py#L876) | TokenClassification | [jnlpba/jnlpba](https://hf.co/datasets/jnlpba/jnlpba) | ✓ |
| 214 | [ontonotes_english/SpeedOfMagic--ontonotes_english](src/tasksource/tasks.py#L885) | TokenClassification | [SpeedOfMagic/ontonotes_english](https://hf.co/datasets/SpeedOfMagic/ontonotes_english) | ✓ |
| 215 | [blog_authorship_corpus/gender](src/tasksource/tasks.py#L891) | Classification | [tasksource/blog_authorship_corpus](https://hf.co/datasets/tasksource/blog_authorship_corpus) | ✓ |
| 216 | [blog_authorship_corpus/age](src/tasksource/tasks.py#L893) | Classification | [tasksource/blog_authorship_corpus](https://hf.co/datasets/tasksource/blog_authorship_corpus) | ✓ |
| 217 | [open_question_type](src/tasksource/tasks.py#L904) | Classification | [Korea-MES/open_question_type](https://hf.co/datasets/Korea-MES/open_question_type) |  |
| 218 | [health_fact](src/tasksource/tasks.py#L906) | Classification | [marcov/health_fact_promptsource](https://hf.co/datasets/marcov/health_fact_promptsource) |  |
| 219 | [commonsense_qa](src/tasksource/tasks.py#L910) | MultipleChoice | [tau/commonsense_qa](https://hf.co/datasets/tau/commonsense_qa) |  |
| 220 | [mc_taco](src/tasksource/tasks.py#L916) | Classification | [marcov/mc_taco_promptsource](https://hf.co/datasets/marcov/mc_taco_promptsource) | ✓ |
| 221 | [ade_corpus_v2/Ade_corpus_v2_classification](src/tasksource/tasks.py#L923) | Classification | [ade-benchmark-corpus/ade_corpus_v2](https://hf.co/datasets/ade-benchmark-corpus/ade_corpus_v2) | ✓ |
| 222 | [discosense](src/tasksource/tasks.py#L926) | MultipleChoice | [tasksource/discosense](https://hf.co/datasets/tasksource/discosense) | ✓ |
| 223 | [circa](src/tasksource/tasks.py#L933) | Classification | [google-research-datasets/circa](https://hf.co/datasets/google-research-datasets/circa) |  |
| 224 | [code_x_glue_cc_defect_detection](src/tasksource/tasks.py#L938) | Classification | [google/code_x_glue_cc_defect_detection](https://hf.co/datasets/google/code_x_glue_cc_defect_detection) | ✓ |
| 225 | [phrase_similarity](src/tasksource/tasks.py#L943) | Classification | [Deehan1866/processed_phrase_similarity](https://hf.co/datasets/Deehan1866/processed_phrase_similarity) |  |
| 226 | [scientific-exaggeration-detection](src/tasksource/tasks.py#L951) | Classification | [copenlu/scientific-exaggeration-detection](https://hf.co/datasets/copenlu/scientific-exaggeration-detection) |  |
| 227 | [quarel](src/tasksource/tasks.py#L965) | MultipleChoice | [community-datasets/quarel](https://hf.co/datasets/community-datasets/quarel) | ✓ |
| 228 | [fever-evidence-related](src/tasksource/tasks.py#L972) | Classification | [mwong/fever-evidence-related](https://hf.co/datasets/mwong/fever-evidence-related) |  |
| 229 | [numer_sense](src/tasksource/tasks.py#L975) | Classification | [tasksource/numer_sense](https://hf.co/datasets/tasksource/numer_sense) |  |
| 230 | [dynasent/dynabench.dynasent.r1.all/r1](src/tasksource/tasks.py#L982) | Classification | [tasksource/dynasent](https://hf.co/datasets/tasksource/dynasent) | ✓ |
| 231 | [dynasent/dynabench.dynasent.r2.all/r2](src/tasksource/tasks.py#L986) | Classification | [tasksource/dynasent](https://hf.co/datasets/tasksource/dynasent) | ✓ |
| 232 | [Sarcasm_News_Headline](src/tasksource/tasks.py#L991) | Classification | [raquiba/Sarcasm_News_Headline](https://hf.co/datasets/raquiba/Sarcasm_News_Headline) |  |
| 233 | [sem_eval_2010_task_8](src/tasksource/tasks.py#L1000) | Classification | [SemEvalWorkshop/sem_eval_2010_task_8](https://hf.co/datasets/SemEvalWorkshop/sem_eval_2010_task_8) |  |
| 234 | [auditor_review](src/tasksource/tasks.py#L1003) | Classification | [demo-org/auditor_review](https://hf.co/datasets/demo-org/auditor_review) |  |
| 235 | [medmcqa](src/tasksource/tasks.py#L1007) | MultipleChoice | [openlifescienceai/medmcqa](https://hf.co/datasets/openlifescienceai/medmcqa) |  |
| 236 | [Dynasent_Disagreement](src/tasksource/tasks.py#L1024) | Classification | [RuyuanWan/Dynasent_Disagreement](https://hf.co/datasets/RuyuanWan/Dynasent_Disagreement) | ✓ |
| 237 | [Politeness_Disagreement](src/tasksource/tasks.py#L1026) | Classification | [RuyuanWan/Politeness_Disagreement](https://hf.co/datasets/RuyuanWan/Politeness_Disagreement) | ✓ |
| 238 | [SBIC_Disagreement](src/tasksource/tasks.py#L1028) | Classification | [RuyuanWan/SBIC_Disagreement](https://hf.co/datasets/RuyuanWan/SBIC_Disagreement) | ✓ |
| 239 | [SChem_Disagreement](src/tasksource/tasks.py#L1030) | Classification | [RuyuanWan/SChem_Disagreement](https://hf.co/datasets/RuyuanWan/SChem_Disagreement) | ✓ |
| 240 | [Dilemmas_Disagreement](src/tasksource/tasks.py#L1032) | Classification | [RuyuanWan/Dilemmas_Disagreement](https://hf.co/datasets/RuyuanWan/Dilemmas_Disagreement) | ✓ |
| 241 | [logiqa](src/tasksource/tasks.py#L1035) | MultipleChoice | [fireworks-ai/logiqa](https://hf.co/datasets/fireworks-ai/logiqa) |  |
| 242 | [wiki_qa](src/tasksource/tasks.py#L1044) | Classification | [microsoft/wiki_qa](https://hf.co/datasets/microsoft/wiki_qa) | ✓ |
| 243 | [cycic_classification](src/tasksource/tasks.py#L1046) | Classification | [tasksource/cycic_classification](https://hf.co/datasets/tasksource/cycic_classification) |  |
| 244 | [cycic_multiplechoice](src/tasksource/tasks.py#L1048) | MultipleChoice | [tasksource/cycic_multiplechoice](https://hf.co/datasets/tasksource/cycic_multiplechoice) |  |
| 245 | [sts-companion](src/tasksource/tasks.py#L1052) | Classification | [tasksource/sts-companion](https://hf.co/datasets/tasksource/sts-companion) |  |
| 246 | [commonsense_qa_2.0](src/tasksource/tasks.py#L1055) | Classification | [tasksource/commonsense_qa_2.0](https://hf.co/datasets/tasksource/commonsense_qa_2.0) |  |
| 247 | [lingnli](src/tasksource/tasks.py#L1058) | Classification | [tasksource/lingnli](https://hf.co/datasets/tasksource/lingnli) |  |
| 248 | [monotonicity-entailment](src/tasksource/tasks.py#L1060) | Classification | [tasksource/monotonicity-entailment](https://hf.co/datasets/tasksource/monotonicity-entailment) |  |
| 249 | [arct](src/tasksource/tasks.py#L1063) | MultipleChoice | [tasksource/arct](https://hf.co/datasets/tasksource/arct) |  |
| 250 | [scinli](src/tasksource/tasks.py#L1066) | Classification | [tasksource/scinli](https://hf.co/datasets/tasksource/scinli) |  |
| 251 | [naturallogic](src/tasksource/tasks.py#L1070) | Classification | [tasksource/naturallogic](https://hf.co/datasets/tasksource/naturallogic) |  |
| 252 | [onestop_qa](src/tasksource/tasks.py#L1072) | MultipleChoice | [malmaud/onestop_qa](https://hf.co/datasets/malmaud/onestop_qa) |  |
| 253 | [moral_stories/full](src/tasksource/tasks.py#L1075) | MultipleChoice | [LabHC/moral_stories](https://hf.co/datasets/LabHC/moral_stories) |  |
| 254 | [dynahate](src/tasksource/tasks.py#L1079) | Classification | [tasksource/dynahate](https://hf.co/datasets/tasksource/dynahate) |  |
| 255 | [syntactic-augmentation-nli](src/tasksource/tasks.py#L1081) | Classification | [tasksource/syntactic-augmentation-nli](https://hf.co/datasets/tasksource/syntactic-augmentation-nli) |  |
| 256 | [autotnli](src/tasksource/tasks.py#L1083) | Classification | [tasksource/autotnli](https://hf.co/datasets/tasksource/autotnli) |  |
| 257 | [CONDAQA](src/tasksource/tasks.py#L1085) | Classification | [lasha-nlp/CONDAQA](https://hf.co/datasets/lasha-nlp/CONDAQA) |  |
| 258 | [webgpt_comparisons](src/tasksource/tasks.py#L1095) | MultipleChoice | [heegyu/webgpt_comparisons_ko](https://hf.co/datasets/heegyu/webgpt_comparisons_ko) | ✓ |
| 259 | [synthetic-instruct-gptj-pairwise](src/tasksource/tasks.py#L1103) | MultipleChoice | [Dahoas/synthetic-instruct-gptj-pairwise](https://hf.co/datasets/Dahoas/synthetic-instruct-gptj-pairwise) | ✓ |
| 260 | [scruples](src/tasksource/tasks.py#L1106) | Classification | [tasksource/scruples](https://hf.co/datasets/tasksource/scruples) | ✓ |
| 261 | [defeasible-nli/atomic](src/tasksource/tasks.py#L1111) | Classification | [tasksource/defeasible-nli](https://hf.co/datasets/tasksource/defeasible-nli) |  |
| 262 | [defeasible-nli/snli](src/tasksource/tasks.py#L1111) | Classification | [tasksource/defeasible-nli](https://hf.co/datasets/tasksource/defeasible-nli) |  |
| 263 | [defeasible-nli/social](src/tasksource/tasks.py#L1115) | Classification | [tasksource/defeasible-nli](https://hf.co/datasets/tasksource/defeasible-nli) |  |
| 264 | [help-nli](src/tasksource/tasks.py#L1119) | Classification | [tasksource/help-nli](https://hf.co/datasets/tasksource/help-nli) |  |
| 265 | [nli-veridicality-transitivity](src/tasksource/tasks.py#L1122) | Classification | [tasksource/nli-veridicality-transitivity](https://hf.co/datasets/tasksource/nli-veridicality-transitivity) |  |
| 266 | [lonli](src/tasksource/tasks.py#L1125) | Classification | [tasksource/lonli](https://hf.co/datasets/tasksource/lonli) |  |
| 267 | [dadc-limit-nli](src/tasksource/tasks.py#L1128) | Classification | [tasksource/dadc-limit-nli](https://hf.co/datasets/tasksource/dadc-limit-nli) |  |
| 268 | [FLUTE](src/tasksource/tasks.py#L1131) | Classification | [ColumbiaNLP/FLUTE](https://hf.co/datasets/ColumbiaNLP/FLUTE) |  |
| 269 | [strategy-qa](src/tasksource/tasks.py#L1134) | Classification | [tasksource/strategy-qa](https://hf.co/datasets/tasksource/strategy-qa) |  |
| 270 | [summarize_from_feedback/comparisons](src/tasksource/tasks.py#L1137) | MultipleChoice | [vwxyzjn/summarize_from_feedback_oai_preprocessing](https://hf.co/datasets/vwxyzjn/summarize_from_feedback_oai_preprocessing) | ✓ |
| 271 | [folio](src/tasksource/tasks.py#L1145) | Classification | [tasksource/folio](https://hf.co/datasets/tasksource/folio) |  |
| 272 | [tomi-nli](src/tasksource/tasks.py#L1172) | Classification | [tasksource/tomi-nli](https://hf.co/datasets/tasksource/tomi-nli) |  |
| 273 | [avicenna](src/tasksource/tasks.py#L1175) | Classification | [tasksource/avicenna](https://hf.co/datasets/tasksource/avicenna) | ✓ |
| 274 | [SHP](src/tasksource/tasks.py#L1178) | MultipleChoice | [stanfordnlp/SHP](https://hf.co/datasets/stanfordnlp/SHP) | ✓ |
| 275 | [MedQA-USMLE-4-options-hf](src/tasksource/tasks.py#L1186) | MultipleChoice | [GBaker/MedQA-USMLE-4-options-hf](https://hf.co/datasets/GBaker/MedQA-USMLE-4-options-hf) |  |
| 276 | [wikimedqa/medwiki](src/tasksource/tasks.py#L1189) | MultipleChoice | [sileod/wikimedqa](https://hf.co/datasets/sileod/wikimedqa) |  |
| 277 | [cicero](src/tasksource/tasks.py#L1198) | MultipleChoice | [declare-lab/cicero](https://hf.co/datasets/declare-lab/cicero) |  |
| 278 | [CREAK](src/tasksource/tasks.py#L1202) | Classification | [amydeng2000/CREAK](https://hf.co/datasets/amydeng2000/CREAK) |  |
| 279 | [mutual](src/tasksource/tasks.py#L1205) | MultipleChoice | [tasksource/mutual](https://hf.co/datasets/tasksource/mutual) |  |
| 280 | [implicatures](src/tasksource/tasks.py#L1210) | MultipleChoice | [tasksource/implicatures](https://hf.co/datasets/tasksource/implicatures) |  |
| 281 | [race/middle](src/tasksource/tasks.py#L1215) | MultipleChoice | [ehovy/race](https://hf.co/datasets/ehovy/race) |  |
| 282 | [race/high](src/tasksource/tasks.py#L1215) | MultipleChoice | [ehovy/race](https://hf.co/datasets/ehovy/race) |  |
| 283 | [race-c](src/tasksource/tasks.py#L1219) | MultipleChoice | [tasksource/race-c](https://hf.co/datasets/tasksource/race-c) |  |
| 284 | [spartqa-yn](src/tasksource/tasks.py#L1222) | Classification | [tasksource/spartqa-yn](https://hf.co/datasets/tasksource/spartqa-yn) |  |
| 285 | [spartqa-mchoice](src/tasksource/tasks.py#L1225) | MultipleChoice | [tasksource/spartqa-mchoice](https://hf.co/datasets/tasksource/spartqa-mchoice) |  |
| 286 | [temporal-nli](src/tasksource/tasks.py#L1228) | Classification | [tasksource/temporal-nli](https://hf.co/datasets/tasksource/temporal-nli) |  |
| 287 | [riddle_sense](src/tasksource/tasks.py#L1231) | MultipleChoice | [jeggers/riddle_sense](https://hf.co/datasets/jeggers/riddle_sense) |  |
| 288 | [clcd-english](src/tasksource/tasks.py#L1236) | Classification | [tasksource/clcd-english](https://hf.co/datasets/tasksource/clcd-english) |  |
| 289 | [twentyquestions](src/tasksource/tasks.py#L1248) | Classification | [tasksource/twentyquestions](https://hf.co/datasets/tasksource/twentyquestions) |  |
| 290 | [reclor](src/tasksource/tasks.py#L1253) | MultipleChoice | [tasksource/reclor](https://hf.co/datasets/tasksource/reclor) |  |
| 291 | [counterfactually-augmented-imdb](src/tasksource/tasks.py#L1256) | Classification | [tasksource/counterfactually-augmented-imdb](https://hf.co/datasets/tasksource/counterfactually-augmented-imdb) |  |
| 292 | [counterfactually-augmented-snli](src/tasksource/tasks.py#L1259) | Classification | [tasksource/counterfactually-augmented-snli](https://hf.co/datasets/tasksource/counterfactually-augmented-snli) |  |
| 293 | [cnli](src/tasksource/tasks.py#L1262) | Classification | [tasksource/cnli](https://hf.co/datasets/tasksource/cnli) |  |
| 294 | [boolq-natural-perturbations](src/tasksource/tasks.py#L1265) | Classification | [tasksource/boolq-natural-perturbations](https://hf.co/datasets/tasksource/boolq-natural-perturbations) |  |
| 295 | [acceptability-prediction](src/tasksource/tasks.py#L1270) | Classification | [tasksource/acceptability-prediction](https://hf.co/datasets/tasksource/acceptability-prediction) | ✓ |
| 296 | [equate](src/tasksource/tasks.py#L1296) | Classification | [tasksource/equate](https://hf.co/datasets/tasksource/equate) |  |
| 297 | [ScienceQA_text_only](src/tasksource/tasks.py#L1299) | MultipleChoice | [tasksource/ScienceQA_text_only](https://hf.co/datasets/tasksource/ScienceQA_text_only) |  |
| 298 | [ekar_english](src/tasksource/tasks.py#L1302) | MultipleChoice | [Jiangjie/ekar_english](https://hf.co/datasets/Jiangjie/ekar_english) | ✓ |
| 299 | [implicit-hate-stg1](src/tasksource/tasks.py#L1306) | Classification | [tasksource/implicit-hate-stg1](https://hf.co/datasets/tasksource/implicit-hate-stg1) | ✓ |
| 300 | [chaos-mnli-ambiguity](src/tasksource/tasks.py#L1310) | Classification | [tasksource/chaos-mnli-ambiguity](https://hf.co/datasets/tasksource/chaos-mnli-ambiguity) | ✓ |
| 301 | [headline_cause/en_simple](src/tasksource/tasks.py#L1314) | Classification | [IlyaGusev/headline_cause](https://hf.co/datasets/IlyaGusev/headline_cause) (`json` loader) |  |
| 302 | [logiqa-2.0-nli](src/tasksource/tasks.py#L1319) | Classification | [tasksource/logiqa-2.0-nli](https://hf.co/datasets/tasksource/logiqa-2.0-nli) |  |
| 303 | [oasst2_dense_flat/quality](src/tasksource/tasks.py#L1324) | Classification | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) | ✓ |
| 304 | [oasst2_dense_flat/toxicity](src/tasksource/tasks.py#L1326) | Classification | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) | ✓ |
| 305 | [oasst2_dense_flat/helpfulness](src/tasksource/tasks.py#L1328) | Classification | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) | ✓ |
| 306 | [mindgames](src/tasksource/tasks.py#L1331) | Classification | [sileod/mindgames](https://hf.co/datasets/sileod/mindgames) |  |
| 307 | [universal_dependencies/en_lines/deprel](src/tasksource/tasks.py#L1345) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) | ✓ |
| 308 | [universal_dependencies/en_gum/deprel](src/tasksource/tasks.py#L1345) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) | ✓ |
| 309 | [universal_dependencies/en_ewt/deprel](src/tasksource/tasks.py#L1345) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) | ✓ |
| 310 | [universal_dependencies/en_partut/deprel](src/tasksource/tasks.py#L1345) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) | ✓ |
| 311 | [ambient](src/tasksource/tasks.py#L1352) | Classification | [tasksource/ambient](https://hf.co/datasets/tasksource/ambient) | ✓ |
| 312 | [path-naturalness-prediction](src/tasksource/tasks.py#L1357) | MultipleChoice | [tasksource/path-naturalness-prediction](https://hf.co/datasets/tasksource/path-naturalness-prediction) |  |
| 313 | [civil_comments/toxicity](src/tasksource/tasks.py#L1367) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 314 | [civil_comments/insult](src/tasksource/tasks.py#L1368) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 315 | [cloth](src/tasksource/tasks.py#L1370) | MultipleChoice | [AndyChiang/cloth](https://hf.co/datasets/AndyChiang/cloth) |  |
| 316 | [dgen](src/tasksource/tasks.py#L1371) | MultipleChoice | [AndyChiang/dgen](https://hf.co/datasets/AndyChiang/dgen) |  |
| 317 | [I2D2](src/tasksource/tasks.py#L1373) | Classification | [tasksource/I2D2](https://hf.co/datasets/tasksource/I2D2) | ✓ |
| 318 | [args_me](src/tasksource/tasks.py#L1378) | Classification | [webis/args_me](https://hf.co/datasets/webis/args_me) | ✓ |
| 319 | [Touche23-ValueEval](src/tasksource/tasks.py#L1383) | Classification | [webis/Touche23-ValueEval](https://hf.co/datasets/webis/Touche23-ValueEval) (`csv` loader) | ✓ |
| 320 | [starcon](src/tasksource/tasks.py#L1392) | Classification | [tasksource/starcon](https://hf.co/datasets/tasksource/starcon) | ✓ |
| 321 | [banking77](src/tasksource/tasks.py#L1397) | Classification | [legacy-datasets/banking77](https://hf.co/datasets/legacy-datasets/banking77) |  |
| 322 | [it-support-tickets](src/tasksource/tasks.py#L1399) | Classification | [tasksource/it-support-tickets](https://hf.co/datasets/tasksource/it-support-tickets) |  |
| 323 | [ConTRoL-nli](src/tasksource/tasks.py#L1403) | Classification | [tasksource/ConTRoL-nli](https://hf.co/datasets/tasksource/ConTRoL-nli) |  |
| 324 | [tracie](src/tasksource/tasks.py#L1405) | Classification | [tasksource/tracie](https://hf.co/datasets/tasksource/tracie) | ✓ |
| 325 | [sherliic](src/tasksource/tasks.py#L1407) | Classification | [tasksource/sherliic](https://hf.co/datasets/tasksource/sherliic) |  |
| 326 | [sen-making/1](src/tasksource/tasks.py#L1410) | MultipleChoice | [tasksource/sen-making](https://hf.co/datasets/tasksource/sen-making) |  |
| 327 | [sen-making/2](src/tasksource/tasks.py#L1413) | MultipleChoice | [tasksource/sen-making](https://hf.co/datasets/tasksource/sen-making) | ✓ |
| 328 | [winowhy](src/tasksource/tasks.py#L1416) | Classification | [tasksource/winowhy](https://hf.co/datasets/tasksource/winowhy) | ✓ |
| 329 | [robustLR](src/tasksource/tasks.py#L1420) | Classification | [tasksource/robustLR](https://hf.co/datasets/tasksource/robustLR) |  |
| 330 | [clutrr](src/tasksource/tasks.py#L1422) | Classification | [tasksource/clutrr](https://hf.co/datasets/tasksource/clutrr) |  |
| 331 | [logical-fallacy](src/tasksource/tasks.py#L1424) | Classification | [tasksource/logical-fallacy](https://hf.co/datasets/tasksource/logical-fallacy) |  |
| 332 | [parade](src/tasksource/tasks.py#L1427) | Classification | [tasksource/parade](https://hf.co/datasets/tasksource/parade) |  |
| 333 | [subjectivity](src/tasksource/tasks.py#L1430) | Classification | [tasksource/subjectivity](https://hf.co/datasets/tasksource/subjectivity) | ✓ |
| 334 | [MOH](src/tasksource/tasks.py#L1436) | Classification | [tasksource/MOH](https://hf.co/datasets/tasksource/MOH) | ✓ |
| 335 | [VUAC](src/tasksource/tasks.py#L1437) | Classification | [tasksource/VUAC](https://hf.co/datasets/tasksource/VUAC) | ✓ |
| 336 | [TroFi](src/tasksource/tasks.py#L1438) | Classification | [tasksource/TroFi](https://hf.co/datasets/tasksource/TroFi) (`parquet` loader) | ✓ |
| 337 | [sharc](src/tasksource/tasks.py#L1446) | Classification | [tasksource/sharc](https://hf.co/datasets/tasksource/sharc) |  |
| 338 | [conceptrules_v2](src/tasksource/tasks.py#L1450) | Classification | [tasksource/conceptrules_v2](https://hf.co/datasets/tasksource/conceptrules_v2) | ✓ |
| 339 | [disrpt/eng.dep.scidtb.rels](src/tasksource/tasks.py#L1453) | Classification | [multilingual-discourse-hub/disrpt](https://hf.co/datasets/multilingual-discourse-hub/disrpt) | ✓ |
| 340 | [conll2000](src/tasksource/tasks.py#L1456) | TokenClassification | [eriktks/conll2000](https://hf.co/datasets/eriktks/conll2000) |  |
| 341 | [few-nerd/supervised](src/tasksource/tasks.py#L1459) | TokenClassification | [DFKI-SLT/few-nerd](https://hf.co/datasets/DFKI-SLT/few-nerd) |  |
| 342 | [finer-139](src/tasksource/tasks.py#L1460) | TokenClassification | [nlpaueb/finer-139](https://hf.co/datasets/nlpaueb/finer-139) |  |
| 343 | [zero-shot-label-nli](src/tasksource/tasks.py#L1463) | Classification | [tasksource/zero-shot-label-nli](https://hf.co/datasets/tasksource/zero-shot-label-nli) | ✓ |
| 344 | [com2sense](src/tasksource/tasks.py#L1465) | Classification | [tasksource/com2sense](https://hf.co/datasets/tasksource/com2sense) |  |
| 345 | [scone](src/tasksource/tasks.py#L1468) | Classification | [tasksource/scone](https://hf.co/datasets/tasksource/scone) |  |
| 346 | [winodict](src/tasksource/tasks.py#L1470) | MultipleChoice | [tasksource/winodict](https://hf.co/datasets/tasksource/winodict) |  |
| 347 | [fool-me-twice](src/tasksource/tasks.py#L1472) | Classification | [tasksource/fool-me-twice](https://hf.co/datasets/tasksource/fool-me-twice) |  |
| 348 | [monli](src/tasksource/tasks.py#L1476) | Classification | [tasksource/monli](https://hf.co/datasets/tasksource/monli) |  |
| 349 | [corr2cause](src/tasksource/tasks.py#L1478) | Classification | [tasksource/corr2cause](https://hf.co/datasets/tasksource/corr2cause) |  |
| 350 | [lsat_qa/all](src/tasksource/tasks.py#L1480) | MultipleChoice | [lighteval/lsat_qa](https://hf.co/datasets/lighteval/lsat_qa) |  |
| 351 | [apt](src/tasksource/tasks.py#L1482) | Classification | [tasksource/apt](https://hf.co/datasets/tasksource/apt) |  |
| 352 | [twitter-financial-news-sentiment](src/tasksource/tasks.py#L1485) | Classification | [zeroshot/twitter-financial-news-sentiment](https://hf.co/datasets/zeroshot/twitter-financial-news-sentiment) |  |
| 353 | [icl-symbol-tuning-instruct](src/tasksource/tasks.py#L1492) | Classification | [tasksource/icl-symbol-tuning-instruct](https://hf.co/datasets/tasksource/icl-symbol-tuning-instruct) | ✓ |
| 354 | [SpaceNLI](src/tasksource/tasks.py#L1498) | Classification | [tasksource/SpaceNLI](https://hf.co/datasets/tasksource/SpaceNLI) |  |
| 355 | [propsegment/nli](src/tasksource/tasks.py#L1500) | Classification | [sihaochen/propsegment](https://hf.co/datasets/sihaochen/propsegment) (`json` loader) | ✓ |
| 356 | [HatemojiBuild](src/tasksource/tasks.py#L1510) | Classification | [HannahRoseKirk/HatemojiBuild](https://hf.co/datasets/HannahRoseKirk/HatemojiBuild) |  |
| 357 | [regset](src/tasksource/tasks.py#L1513) | Classification | [tasksource/regset](https://hf.co/datasets/tasksource/regset) | ✓ |
| 358 | [esci](src/tasksource/tasks.py#L1520) | Classification | [tasksource/esci](https://hf.co/datasets/tasksource/esci) |  |
| 359 | [chatbot_arena_conversations](src/tasksource/tasks.py#L1541) | MultipleChoice | [lmsys/chatbot_arena_conversations](https://hf.co/datasets/lmsys/chatbot_arena_conversations) | ✓ |
| 360 | [dnd_style_intents](src/tasksource/tasks.py#L1547) | Classification | [neurae/dnd_style_intents](https://hf.co/datasets/neurae/dnd_style_intents) | ✓ |
| 361 | [FLD.v2/default](src/tasksource/tasks.py#L1552) | Classification | [hitachi-nlp/FLD.v2](https://hf.co/datasets/hitachi-nlp/FLD.v2) | ✓ |
| 362 | [FLD.v2/star](src/tasksource/tasks.py#L1556) | Classification | [hitachi-nlp/FLD.v2](https://hf.co/datasets/hitachi-nlp/FLD.v2) | ✓ |
| 363 | [SDOH-NLI](src/tasksource/tasks.py#L1560) | Classification | [tasksource/SDOH-NLI](https://hf.co/datasets/tasksource/SDOH-NLI) |  |
| 364 | [scifact_entailment](src/tasksource/tasks.py#L1563) | Classification | [tasksource/scifact_entailment](https://hf.co/datasets/tasksource/scifact_entailment) |  |
| 365 | [feasibilityQA](src/tasksource/tasks.py#L1567) | Classification | [tasksource/feasibilityQA](https://hf.co/datasets/tasksource/feasibilityQA) |  |
| 366 | [simple_pair](src/tasksource/tasks.py#L1570) | Classification | [tasksource/simple_pair](https://hf.co/datasets/tasksource/simple_pair) |  |
| 367 | [AdjectiveScaleProbe-nli](src/tasksource/tasks.py#L1571) | Classification | [tasksource/AdjectiveScaleProbe-nli](https://hf.co/datasets/tasksource/AdjectiveScaleProbe-nli) |  |
| 368 | [resnli](src/tasksource/tasks.py#L1572) | Classification | [tasksource/resnli](https://hf.co/datasets/tasksource/resnli) |  |
| 369 | [SpaRTUN](src/tasksource/tasks.py#L1574) | MultipleChoice | [tasksource/SpaRTUN](https://hf.co/datasets/tasksource/SpaRTUN) |  |
| 370 | [ReSQ](src/tasksource/tasks.py#L1579) | MultipleChoice | [tasksource/ReSQ](https://hf.co/datasets/tasksource/ReSQ) |  |
| 371 | [semantic_fragments_nli](src/tasksource/tasks.py#L1584) | Classification | [tasksource/semantic_fragments_nli](https://hf.co/datasets/tasksource/semantic_fragments_nli) |  |
| 372 | [dataset_train_nli](src/tasksource/tasks.py#L1587) | Classification | [MoritzLaurer/dataset_train_nli](https://hf.co/datasets/MoritzLaurer/dataset_train_nli) |  |
| 373 | [stepgame](src/tasksource/tasks.py#L1592) | Classification | [tasksource/stepgame](https://hf.co/datasets/tasksource/stepgame) |  |
| 374 | [nlgraph](src/tasksource/tasks.py#L1600) | Classification | [tasksource/nlgraph](https://hf.co/datasets/tasksource/nlgraph) |  |
| 375 | [oasst2_pairwise_rlhf_reward](src/tasksource/tasks.py#L1604) | MultipleChoice | [tasksource/oasst2_pairwise_rlhf_reward](https://hf.co/datasets/tasksource/oasst2_pairwise_rlhf_reward) | ✓ |
| 376 | [hh-rlhf/helpful-online](src/tasksource/tasks.py#L1615) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| 377 | [hh-rlhf/helpful-rejection-sampled](src/tasksource/tasks.py#L1615) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| 378 | [hh-rlhf/helpful-base](src/tasksource/tasks.py#L1615) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| 379 | [hh-rlhf/harmless-base](src/tasksource/tasks.py#L1619) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| 380 | [ruletaker](src/tasksource/tasks.py#L1623) | Classification | [tasksource/ruletaker](https://hf.co/datasets/tasksource/ruletaker) | ✓ |
| 381 | [PARARULE-Plus](src/tasksource/tasks.py#L1627) | Classification | [qbao775/PARARULE-Plus](https://hf.co/datasets/qbao775/PARARULE-Plus) | ✓ |
| 382 | [proofwriter](src/tasksource/tasks.py#L1631) | Classification | [tasksource/proofwriter](https://hf.co/datasets/tasksource/proofwriter) |  |
| 383 | [logical-entailment](src/tasksource/tasks.py#L1634) | Classification | [tasksource/logical-entailment](https://hf.co/datasets/tasksource/logical-entailment) | ✓ |
| 384 | [nope](src/tasksource/tasks.py#L1637) | Classification | [tasksource/nope](https://hf.co/datasets/tasksource/nope) |  |
| 385 | [LogicNLI](src/tasksource/tasks.py#L1641) | Classification | [tasksource/LogicNLI](https://hf.co/datasets/tasksource/LogicNLI) | ✓ |
| 386 | [contract-nli/contractnli_a/seg](src/tasksource/tasks.py#L1644) | Classification | [tasksource/contract-nli](https://hf.co/datasets/tasksource/contract-nli) |  |
| 387 | [contract-nli/contractnli_b/full](src/tasksource/tasks.py#L1646) | Classification | [tasksource/contract-nli](https://hf.co/datasets/tasksource/contract-nli) |  |
| 388 | [nli4ct_semeval2024](src/tasksource/tasks.py#L1648) | Classification | [AshtonIsNotHere/nli4ct_semeval2024](https://hf.co/datasets/AshtonIsNotHere/nli4ct_semeval2024) |  |
| 389 | [lsat-ar](src/tasksource/tasks.py#L1651) | MultipleChoice | [tasksource/lsat-ar](https://hf.co/datasets/tasksource/lsat-ar) |  |
| 390 | [lsat-rc](src/tasksource/tasks.py#L1656) | MultipleChoice | [tasksource/lsat-rc](https://hf.co/datasets/tasksource/lsat-rc) |  |
| 391 | [biosift-nli](src/tasksource/tasks.py#L1661) | Classification | [AshtonIsNotHere/biosift-nli](https://hf.co/datasets/AshtonIsNotHere/biosift-nli) |  |
| 392 | [brainteasers/WP](src/tasksource/tasks.py#L1665) | MultipleChoice | [tasksource/brainteasers](https://hf.co/datasets/tasksource/brainteasers) |  |
| 393 | [toxigen-data/annotated](src/tasksource/tasks.py#L1671) | Classification | [skg/toxigen-data](https://hf.co/datasets/skg/toxigen-data) |  |
| 394 | [persuasion](src/tasksource/tasks.py#L1682) | Classification | [Anthropic/persuasion](https://hf.co/datasets/Anthropic/persuasion) |  |
| 395 | [AmbigNQ-clarifying-question](src/tasksource/tasks.py#L1688) | Classification | [erbacher/AmbigNQ-clarifying-question](https://hf.co/datasets/erbacher/AmbigNQ-clarifying-question) | ✓ |
| 396 | [SIGA-nli](src/tasksource/tasks.py#L1693) | Classification | [tasksource/SIGA-nli](https://hf.co/datasets/tasksource/SIGA-nli) | ✓ |
| 397 | [FOL-nli](src/tasksource/tasks.py#L1695) | Classification | [unigram/FOL-nli](https://hf.co/datasets/unigram/FOL-nli) |  |
| 398 | [goal-step-wikihow/goal](src/tasksource/tasks.py#L1697) | MultipleChoice | [tasksource/goal-step-wikihow](https://hf.co/datasets/tasksource/goal-step-wikihow) |  |
| 399 | [goal-step-wikihow/step](src/tasksource/tasks.py#L1700) | MultipleChoice | [tasksource/goal-step-wikihow](https://hf.co/datasets/tasksource/goal-step-wikihow) |  |
| 400 | [goal-step-wikihow/order](src/tasksource/tasks.py#L1703) | MultipleChoice | [tasksource/goal-step-wikihow](https://hf.co/datasets/tasksource/goal-step-wikihow) | ✓ |
| 401 | [PARADISE](src/tasksource/tasks.py#L1707) | MultipleChoice | [GGLab/PARADISE](https://hf.co/datasets/GGLab/PARADISE) |  |
| 402 | [doc-nli](src/tasksource/tasks.py#L1710) | Classification | [tasksource/doc-nli](https://hf.co/datasets/tasksource/doc-nli) |  |
| 403 | [mctest-nli](src/tasksource/tasks.py#L1712) | Classification | [tasksource/mctest-nli](https://hf.co/datasets/tasksource/mctest-nli) |  |
| 404 | [patent-phrase-similarity](src/tasksource/tasks.py#L1714) | Classification | [tasksource/patent-phrase-similarity](https://hf.co/datasets/tasksource/patent-phrase-similarity) | ✓ |
| 405 | [natural-language-satisfiability](src/tasksource/tasks.py#L1717) | Classification | [tasksource/natural-language-satisfiability](https://hf.co/datasets/tasksource/natural-language-satisfiability) |  |
| 406 | [idioms-nli](src/tasksource/tasks.py#L1719) | Classification | [tasksource/idioms-nli](https://hf.co/datasets/tasksource/idioms-nli) |  |
| 407 | [lifecycle-entailment](src/tasksource/tasks.py#L1721) | Classification | [tasksource/lifecycle-entailment](https://hf.co/datasets/tasksource/lifecycle-entailment) |  |
| 408 | [safe-guard-prompt-injection](src/tasksource/tasks.py#L1728) | Classification | [xTRam1/safe-guard-prompt-injection](https://hf.co/datasets/xTRam1/safe-guard-prompt-injection) | ✓ |
| 409 | [prompt-injections](src/tasksource/tasks.py#L1734) | Classification | [deepset/prompt-injections](https://hf.co/datasets/deepset/prompt-injections) | ✓ |
| 410 | [prompt-injection-dataset](src/tasksource/tasks.py#L1740) | Classification | [S-Labs/prompt-injection-dataset](https://hf.co/datasets/S-Labs/prompt-injection-dataset) | ✓ |
| 411 | [Prompt-injection-dataset/full](src/tasksource/tasks.py#L1746) | Classification | [neuralchemy/Prompt-injection-dataset](https://hf.co/datasets/neuralchemy/Prompt-injection-dataset) | ✓ |
| 412 | [PromptShield](src/tasksource/tasks.py#L1752) | Classification | [hendzh/PromptShield](https://hf.co/datasets/hendzh/PromptShield) | ✓ |
| 413 | [shell-safety-v2](src/tasksource/tasks.py#L1758) | Classification | [tomngdev/shell-safety-v2](https://hf.co/datasets/tomngdev/shell-safety-v2) | ✓ |
| 414 | [agent_action_safety](src/tasksource/tasks.py#L1763) | Classification | [karanxa/agent-action-safety-dataset](https://hf.co/datasets/karanxa/agent-action-safety-dataset) (`json` loader) | ✓ |
| 415 | [ShellRisk-Bench](src/tasksource/tasks.py#L1778) | Classification | [kontext-security/ShellRisk-Bench](https://hf.co/datasets/kontext-security/ShellRisk-Bench) | ✓ |
| 416 | [wildguardmix-cleaned/prompt_harm](src/tasksource/tasks.py#L1803) | Classification | [bogdanminko/wildguardmix-cleaned](https://hf.co/datasets/bogdanminko/wildguardmix-cleaned) | ✓ |
| 417 | [wildguardmix-cleaned/response_harm](src/tasksource/tasks.py#L1809) | Classification | [bogdanminko/wildguardmix-cleaned](https://hf.co/datasets/bogdanminko/wildguardmix-cleaned) | ✓ |
| 418 | [wildguardmix-cleaned/response_refusal](src/tasksource/tasks.py#L1814) | Classification | [bogdanminko/wildguardmix-cleaned](https://hf.co/datasets/bogdanminko/wildguardmix-cleaned) | ✓ |
| 419 | [BeaverTails](src/tasksource/tasks.py#L1830) | Classification | [PKU-Alignment/BeaverTails](https://hf.co/datasets/PKU-Alignment/BeaverTails) | ✓ |
| 420 | [privacy-200k-Mistral-Large-3](src/tasksource/tasks.py#L1846) | Classification | [gabrielloiseau/privacy-200k-Mistral-Large-3](https://hf.co/datasets/gabrielloiseau/privacy-200k-Mistral-Large-3) | ✓ |
| 421 | [toxic-chat/toxicchat0124/toxicity](src/tasksource/tasks.py#L1855) | Classification | [lmsys/toxic-chat](https://hf.co/datasets/lmsys/toxic-chat) | ✓ |
| 422 | [toxic-chat/toxicchat0124/jailbreaking](src/tasksource/tasks.py#L1860) | Classification | [lmsys/toxic-chat](https://hf.co/datasets/lmsys/toxic-chat) | ✓ |
| 423 | [clinc_oos/plus](src/tasksource/tasks.py#L1866) | Classification | [clinc/clinc_oos](https://hf.co/datasets/clinc/clinc_oos) |  |
| 424 | [IntentGrasp/all](src/tasksource/tasks.py#L1884) | MultipleChoice | [yuweiyin/IntentGrasp](https://hf.co/datasets/yuweiyin/IntentGrasp) |  |
| 425 | [few_rel/default](src/tasksource/tasks.py#L1922) | Classification | [tasksource/few_rel](https://hf.co/datasets/tasksource/few_rel) |  |
| 426 | [docred](src/tasksource/tasks.py#L2000) | Classification | [YufeiHFUT/DocRED_origin](https://hf.co/datasets/YufeiHFUT/DocRED_origin), [thunlp/docred](https://hf.co/datasets/thunlp/docred) (`json` loader) |  |
| 427 | [chemprot/chemprot_full_source](src/tasksource/tasks.py#L2035) | Classification | [bigbio/chemprot](https://hf.co/datasets/bigbio/chemprot) | ✓ |
| 428 | [PKU-SafeRLHF/helpfulness](src/tasksource/tasks.py#L2041) | MultipleChoice | [PKU-Alignment/PKU-SafeRLHF](https://hf.co/datasets/PKU-Alignment/PKU-SafeRLHF) | ✓ |
| 429 | [PKU-SafeRLHF/safety](src/tasksource/tasks.py#L2046) | MultipleChoice | [PKU-Alignment/PKU-SafeRLHF](https://hf.co/datasets/PKU-Alignment/PKU-SafeRLHF) | ✓ |
| 430 | [HelpSteer/helpfulness](src/tasksource/tasks.py#L2068) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 431 | [HelpSteer/correctness](src/tasksource/tasks.py#L2069) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 432 | [HelpSteer/coherence](src/tasksource/tasks.py#L2070) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 433 | [HelpSteer/complexity](src/tasksource/tasks.py#L2071) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 434 | [HelpSteer/verbosity](src/tasksource/tasks.py#L2072) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 435 | [HelpSteer2/helpfulness](src/tasksource/tasks.py#L2074) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 436 | [HelpSteer2/correctness](src/tasksource/tasks.py#L2075) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 437 | [HelpSteer2/coherence](src/tasksource/tasks.py#L2076) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 438 | [HelpSteer2/complexity](src/tasksource/tasks.py#L2077) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 439 | [HelpSteer2/verbosity](src/tasksource/tasks.py#L2078) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 440 | [HelpSteer3/preference](src/tasksource/tasks.py#L2083) | MultipleChoice | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | ✓ |
| 441 | [HelpSteer3/preference_strength](src/tasksource/tasks.py#L2096) | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | ✓ |
| 442 | [HelpSteer3/principle](src/tasksource/tasks.py#L2108) | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) |  |
| 443 | [HelpSteer3/edit_quality](src/tasksource/tasks.py#L2113) | MultipleChoice | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | ✓ |
| 444 | [HelpSteer3/feedback](src/tasksource/tasks.py#L2135) | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | ✓ |
| 445 | [MSciNLI](src/tasksource/tasks.py#L2141) | Classification | [sadat2307/MSciNLI](https://hf.co/datasets/sadat2307/MSciNLI) |  |
| 446 | [UltraFeedback-paired](src/tasksource/tasks.py#L2144) | MultipleChoice | [pushpdeep/UltraFeedback-paired](https://hf.co/datasets/pushpdeep/UltraFeedback-paired) | ✓ |
| 447 | [prm800k_dpo/solution](src/tasksource/tasks.py#L2148) | MultipleChoice | [tasksource/prm800k_dpo](https://hf.co/datasets/tasksource/prm800k_dpo) | ✓ |
| 448 | [prm800k_dpo/step](src/tasksource/tasks.py#L2151) | MultipleChoice | [tasksource/prm800k_dpo](https://hf.co/datasets/tasksource/prm800k_dpo) | ✓ |
| 449 | [AES2-essay-scoring](src/tasksource/tasks.py#L2155) | Classification | [tasksource/AES2-essay-scoring](https://hf.co/datasets/tasksource/AES2-essay-scoring) | ✓ |
| 450 | [argument-feedback](src/tasksource/tasks.py#L2159) | Classification | [tasksource/argument-feedback](https://hf.co/datasets/tasksource/argument-feedback) | ✓ |
| 451 | [english-grading/cohesion](src/tasksource/tasks.py#L2166) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 452 | [english-grading/syntax](src/tasksource/tasks.py#L2167) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 453 | [english-grading/vocabulary](src/tasksource/tasks.py#L2168) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 454 | [english-grading/phraseology](src/tasksource/tasks.py#L2169) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 455 | [english-grading/grammar](src/tasksource/tasks.py#L2170) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 456 | [english-grading/conventions](src/tasksource/tasks.py#L2171) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 457 | [wice](src/tasksource/tasks.py#L2173) | Classification | [tasksource/wice](https://hf.co/datasets/tasksource/wice) |  |
| 458 | [hover](src/tasksource/tasks.py#L2176) | Classification | [Dzeniks/hover](https://hf.co/datasets/Dzeniks/hover) |  |
| 459 | [hover-3way/nli](src/tasksource/tasks.py#L2180) | Classification | [Dzeniks/hover-3way](https://hf.co/datasets/Dzeniks/hover-3way) | ✓ |
| 460 | [tasksource_dpo_pairs](src/tasksource/tasks.py#L2183) | MultipleChoice | [tasksource/tasksource_dpo_pairs](https://hf.co/datasets/tasksource/tasksource_dpo_pairs) | ✓ |
| 461 | [seahorse_summarization_evaluation](src/tasksource/tasks.py#L2186) | Classification | [tasksource/seahorse_summarization_evaluation](https://hf.co/datasets/tasksource/seahorse_summarization_evaluation) |  |
| 462 | [missing-item-prediction/contrastive](src/tasksource/tasks.py#L2198) | Classification | [sileod/missing-item-prediction](https://hf.co/datasets/sileod/missing-item-prediction) | ✓ |
| 463 | [jigsaw_toxicity](src/tasksource/tasks.py#L2205) | Classification | [tasksource/jigsaw_toxicity](https://hf.co/datasets/tasksource/jigsaw_toxicity) |  |
| 464 | [Pol_NLI](src/tasksource/tasks.py#L2208) | Classification | [mlburnham/Pol_NLI](https://hf.co/datasets/mlburnham/Pol_NLI) |  |
| 465 | [synthetic-retrieval-NLI/position](src/tasksource/tasks.py#L2211) | Classification | [tasksource/synthetic-retrieval-NLI](https://hf.co/datasets/tasksource/synthetic-retrieval-NLI) |  |
| 466 | [synthetic-retrieval-NLI/count](src/tasksource/tasks.py#L2211) | Classification | [tasksource/synthetic-retrieval-NLI](https://hf.co/datasets/tasksource/synthetic-retrieval-NLI) |  |
| 467 | [synthetic-retrieval-NLI/binary](src/tasksource/tasks.py#L2211) | Classification | [tasksource/synthetic-retrieval-NLI](https://hf.co/datasets/tasksource/synthetic-retrieval-NLI) |  |
| 468 | [github-issue-similarity](src/tasksource/tasks.py#L2220) | Classification | [WhereIsAI/github-issue-similarity](https://hf.co/datasets/WhereIsAI/github-issue-similarity) |  |
| 469 | [webinstruct/mc](src/tasksource/tasks.py#L2425) | MultipleChoice | [tasksource/webinstruct](https://hf.co/datasets/tasksource/webinstruct) |  |
| 470 | [webinstruct/binary](src/tasksource/tasks.py#L2427) | Classification | [tasksource/webinstruct](https://hf.co/datasets/tasksource/webinstruct) |  |
| 471 | [weblinx/action](src/tasksource/tasks.py#L2491) | Classification | [McGill-NLP/WebLINX](https://hf.co/datasets/McGill-NLP/WebLINX) | ✓ |
| 472 | [weblinx/dom-element](src/tasksource/tasks.py#L2494) | MultipleChoice | [McGill-NLP/WebLINX](https://hf.co/datasets/McGill-NLP/WebLINX) | ✓ |
| 473 | [websrc/yesno](src/tasksource/tasks.py#L2500) | Classification | [tasksource/websrc](https://hf.co/datasets/tasksource/websrc) |  |
| 474 | [websrc/element](src/tasksource/tasks.py#L2502) | MultipleChoice | [tasksource/websrc](https://hf.co/datasets/tasksource/websrc) | ✓ |
| 475 | [mind2web/action](src/tasksource/tasks.py#L2509) | Classification | [tasksource/mind2web-dom](https://hf.co/datasets/tasksource/mind2web-dom) | ✓ |
| 476 | [mind2web/dom-element](src/tasksource/tasks.py#L2512) | MultipleChoice | [tasksource/mind2web-dom](https://hf.co/datasets/tasksource/mind2web-dom) | ✓ |
| 477 | [sentfin](src/tasksource/tasks.py#L2517) | Classification | [tasksource/sentfin](https://hf.co/datasets/tasksource/sentfin) | ✓ |
| 478 | [wands](src/tasksource/tasks.py#L2521) | Classification | [tasksource/wands](https://hf.co/datasets/tasksource/wands) | ✓ |
| 479 | [scirepeval/search](src/tasksource/tasks.py#L2525) | Classification | [tasksource/scirepeval-search](https://hf.co/datasets/tasksource/scirepeval-search) | ✓ |
| 480 | [maud/consideration](src/tasksource/tasks.py#L2577) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 481 | [maud/bringdown-timing](src/tasksource/tasks.py#L2581) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 482 | [maud/ordinary-course-consent](src/tasksource/tasks.py#L2585) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 483 | [maud/ordinary-course-efforts](src/tasksource/tasks.py#L2589) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 484 | [maud/antitrust-efforts](src/tasksource/tasks.py#L2593) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 485 | [maud/specific-performance](src/tasksource/tasks.py#L2597) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 486 | [ddi/relation](src/tasksource/tasks.py#L2602) | Classification | [OpenMed/DDI-Corpus-Processed](https://hf.co/datasets/OpenMed/DDI-Corpus-Processed) | ✓ |
| 487 | [patent-classification/abstract](src/tasksource/tasks.py#L2609) | Classification | [ccdv/patent-classification](https://hf.co/datasets/ccdv/patent-classification) | ✓ |
| 488 | [tabfact](src/tasksource/tasks.py#L2616) | Classification | [wenhu/tab_fact](https://hf.co/datasets/wenhu/tab_fact) | ✓ |
| 489 | [infotabs](src/tasksource/tasks.py#L2624) | Classification | [table-benchmark/infotabs](https://hf.co/datasets/table-benchmark/infotabs) | ✓ |
| 490 | [squad-v2/answerability](src/tasksource/tasks.py#L2636) | Classification | [rajpurkar/squad_v2](https://hf.co/datasets/rajpurkar/squad_v2) | ✓ |
| 491 | [cuad/answerability](src/tasksource/tasks.py#L2641) | Classification | [theatticusproject/cuad-qa](https://hf.co/datasets/theatticusproject/cuad-qa) | ✓ |
| 492 | [hallmarks-of-cancer/yesno](src/tasksource/tasks.py#L2660) | Classification | [bigbio/hallmarks_of_cancer](https://hf.co/datasets/bigbio/hallmarks_of_cancer) | ✓ |
| 493 | [semeval2014-absa/laptops](src/tasksource/tasks.py#L2682) | Classification | [alexcadillon/SemEval2014Task4](https://hf.co/datasets/alexcadillon/SemEval2014Task4) | ✓ |
| 494 | [semeval2014-absa/restaurants](src/tasksource/tasks.py#L2682) | Classification | [alexcadillon/SemEval2014Task4](https://hf.co/datasets/alexcadillon/SemEval2014Task4) | ✓ |
| 495 | [record/candidate](src/tasksource/tasks.py#L2704) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) | ✓ |
| 496 | [multidogo/software/intent](src/tasksource/tasks.py#L2727) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 497 | [multidogo/media/intent](src/tasksource/tasks.py#L2727) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 498 | [multidogo/insurance/intent](src/tasksource/tasks.py#L2727) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 499 | [multidogo/finance/intent](src/tasksource/tasks.py#L2727) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 500 | [multidogo/fastfood/intent](src/tasksource/tasks.py#L2727) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 501 | [multidogo/airline/intent](src/tasksource/tasks.py#L2727) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 502 | [wikisql/aggregation](src/tasksource/tasks.py#L2755) | Classification | [Salesforce/wikisql](https://hf.co/datasets/Salesforce/wikisql) | ✓ |
| 503 | [wikisql/column](src/tasksource/tasks.py#L2760) | MultipleChoice | [Salesforce/wikisql](https://hf.co/datasets/Salesforce/wikisql) | ✓ |
| 504 | [wikisql/operator](src/tasksource/tasks.py#L2765) | Classification | [Salesforce/wikisql](https://hf.co/datasets/Salesforce/wikisql) | ✓ |
| 505 | [acord](src/tasksource/tasks.py#L2770) | Classification | [tasksource/acord](https://hf.co/datasets/tasksource/acord) | ✓ |
| 506 | [edos](src/tasksource/tasks.py#L2774) | Classification | [tasksource/edos](https://hf.co/datasets/tasksource/edos) | ✓ |
| 507 | [evidence-inference](src/tasksource/tasks.py#L2777) | Classification | [tasksource/evidence-inference](https://hf.co/datasets/tasksource/evidence-inference) | ✓ |
| 508 | [quac/answerability](src/tasksource/tasks.py#L2799) | Classification | [allenai/quac](https://hf.co/datasets/allenai/quac) | ✓ |
| 509 | [hatexplain/majority](src/tasksource/tasks.py#L2807) | Classification | [Hate-speech-CNERG/hatexplain](https://hf.co/datasets/Hate-speech-CNERG/hatexplain) | ✓ |
| 510 | [hotpotqa/supporting-fact](src/tasksource/tasks.py#L2824) | Classification | [hotpotqa/hotpot_qa](https://hf.co/datasets/hotpotqa/hotpot_qa) | ✓ |
| 511 | [multiwoz22/dialogue-act](src/tasksource/tasks.py#L2868) | Classification | [pfb30/multi_woz_v22](https://hf.co/datasets/pfb30/multi_woz_v22) | ✓ |
| 512 | [multiwoz22/categorical-slot](src/tasksource/tasks.py#L2871) | Classification | [pfb30/multi_woz_v22](https://hf.co/datasets/pfb30/multi_woz_v22) | ✓ |
| 513 | [schema-guided/intent](src/tasksource/tasks.py#L2874) | Classification | [google-research-datasets/schema_guided_dstc8](https://hf.co/datasets/google-research-datasets/schema_guided_dstc8) | ✓ |
| 514 | [coqa/yesno](src/tasksource/tasks.py#L2891) | Classification | [stanfordnlp/coqa](https://hf.co/datasets/stanfordnlp/coqa) | ✓ |
| 515 | [qasper/yesno](src/tasksource/tasks.py#L2912) | Classification | [allenai/qasper](https://hf.co/datasets/allenai/qasper) | ✓ |
| 516 | [topv2/intent](src/tasksource/tasks.py#L2924) | Classification | [WillHeld/top_v2](https://hf.co/datasets/WillHeld/top_v2) | ✓ |
| 517 | [slurp/intent](src/tasksource/tasks.py#L2932) | Classification | [tasksource/slurp](https://hf.co/datasets/tasksource/slurp) | ✓ |
| 518 | [mcscript2](src/tasksource/tasks.py#L2936) | MultipleChoice | [tasksource/mcscript2](https://hf.co/datasets/tasksource/mcscript2) |  |
| 519 | [fnc1/stance](src/tasksource/tasks.py#L2941) | Classification | [tasksource/fnc1](https://hf.co/datasets/tasksource/fnc1) | ✓ |
| 520 | [perspectrum/stance](src/tasksource/tasks.py#L2946) | Classification | [tasksource/perspectrum](https://hf.co/datasets/tasksource/perspectrum) | ✓ |
| 521 | [ukpconvarg/convincingness](src/tasksource/tasks.py#L2951) | MultipleChoice | [tasksource/ukpconvarg](https://hf.co/datasets/tasksource/ukpconvarg) | ✓ |
| 522 | [discogem/relation](src/tasksource/tasks.py#L2956) | Classification | [tasksource/discogem](https://hf.co/datasets/tasksource/discogem) | ✓ |
| 523 | [pubmed-rct/role](src/tasksource/tasks.py#L2964) | Classification | [tasksource/pubmed-rct](https://hf.co/datasets/tasksource/pubmed-rct) | ✓ |
| 524 | [maven-arg/role](src/tasksource/tasks.py#L2992) | Classification | [mneb/maven-arg](https://hf.co/datasets/mneb/maven-arg) | ✓ |
| 525 | [maven/event](src/tasksource/tasks.py#L3023) | Classification | [mneb/maven](https://hf.co/datasets/mneb/maven) | ✓ |
| 526 | [empathetic-dialogues/emotion](src/tasksource/tasks.py#L3027) | Classification | [facebook/empathetic_dialogues](https://hf.co/datasets/facebook/empathetic_dialogues) | ✓ |
| 527 | [olid/offensive](src/tasksource/tasks.py#L3035) | Classification | [christophsonntag/OLID](https://hf.co/datasets/christophsonntag/OLID) | ✓ |
| 528 | [isarcasm/intended](src/tasksource/tasks.py#L3041) | Classification | [viethq1906/isarcasm_2022_taskA_En](https://hf.co/datasets/viethq1906/isarcasm_2022_taskA_En) | ✓ |
| 529 | [magpie/idiomatic](src/tasksource/tasks.py#L3066) | Classification | [idioms-collaboration/magpie](https://hf.co/datasets/idioms-collaboration/magpie) | ✓ |
| 530 | [preco/coreference](src/tasksource/tasks.py#L3102) | Classification | [coref-data/preco_raw](https://hf.co/datasets/coref-data/preco_raw) | ✓ |
| 531 | [redial/preference](src/tasksource/tasks.py#L3122) | Classification | [community-datasets/re_dial](https://hf.co/datasets/community-datasets/re_dial) | ✓ |
| 532 | [sst5](src/tasksource/tasks.py#L3127) | Classification | [SetFit/sst5](https://hf.co/datasets/SetFit/sst5) | ✓ |
| 533 | [spider/column](src/tasksource/tasks.py#L3135) | MultipleChoice | [tasksource/spider](https://hf.co/datasets/tasksource/spider) | ✓ |
| 534 | [semcor](src/tasksource/tasks.py#L3139) | MultipleChoice | [tasksource/semcor](https://hf.co/datasets/tasksource/semcor) | ✓ |
| 535 | [streusle/noun](src/tasksource/tasks.py#L3142) | Classification | [tasksource/streusle](https://hf.co/datasets/tasksource/streusle) | ✓ |
| 536 | [streusle/verb](src/tasksource/tasks.py#L3146) | Classification | [tasksource/streusle](https://hf.co/datasets/tasksource/streusle) | ✓ |
| 537 | [maven-ere/causal](src/tasksource/tasks.py#L3189) | Classification | [Nofing/Maven-ERE-span](https://hf.co/datasets/Nofing/Maven-ERE-span) | ✓ |
| 538 | [biored/relation](src/tasksource/tasks.py#L3211) | Classification | [gyorilab/BioRED](https://hf.co/datasets/gyorilab/BioRED) | ✓ |
| 539 | [matres/temporal](src/tasksource/tasks.py#L3218) | Classification | [tasksource/matres](https://hf.co/datasets/tasksource/matres) | ✓ |
| 540 | [causal-news](src/tasksource/tasks.py#L3227) | Classification | [thagen/CNCv2](https://hf.co/datasets/thagen/CNCv2) | ✓ |
| 541 | [slurp/slot-type](src/tasksource/tasks.py#L3274) | Classification | [tasksource/slurp](https://hf.co/datasets/tasksource/slurp) | ✓ |
| 542 | [topv2/slot-type](src/tasksource/tasks.py#L3279) | Classification | [WillHeld/top_v2](https://hf.co/datasets/WillHeld/top_v2) | ✓ |
| 543 | [wikiqa](src/tasksource/tasks.py#L3292) | Classification | [microsoft/wiki_qa](https://hf.co/datasets/microsoft/wiki_qa) | ✓ |
| 544 | [nevir](src/tasksource/tasks.py#L3307) | MultipleChoice | [orionweller/NevIR](https://hf.co/datasets/orionweller/NevIR) | ✓ |
| 545 | [antique](src/tasksource/tasks.py#L3315) | Classification | [tasksource/antique](https://hf.co/datasets/tasksource/antique) | ✓ |
| 546 | [causal-reasoning/transitivity](src/tasksource/tasks.py#L3341) | Classification | [ludwigw/causal-reasoning-benchmarks](https://hf.co/datasets/ludwigw/causal-reasoning-benchmarks) | ✓ |
| 547 | [cebab/rating](src/tasksource/tasks.py#L3358) | Classification | [EleutherAI/CEBaB](https://hf.co/datasets/EleutherAI/CEBaB) | ✓ |
| 548 | [cebab/food](src/tasksource/tasks.py#L3362) | Classification | [EleutherAI/CEBaB](https://hf.co/datasets/EleutherAI/CEBaB) | ✓ |
| 549 | [cebab/ambiance](src/tasksource/tasks.py#L3365) | Classification | [EleutherAI/CEBaB](https://hf.co/datasets/EleutherAI/CEBaB) | ✓ |
| 550 | [cebab/service](src/tasksource/tasks.py#L3368) | Classification | [EleutherAI/CEBaB](https://hf.co/datasets/EleutherAI/CEBaB) | ✓ |
| 551 | [cebab/noise](src/tasksource/tasks.py#L3371) | Classification | [EleutherAI/CEBaB](https://hf.co/datasets/EleutherAI/CEBaB) | ✓ |
| 552 | [fabsa/sentiment](src/tasksource/tasks.py#L3384) | Classification | [jordiclive/FABSA](https://hf.co/datasets/jordiclive/FABSA) | ✓ |
| 553 | [tab/identifier](src/tasksource/tasks.py#L3401) | Classification | [doshimit3015/text-anonymization-benchmark](https://hf.co/datasets/doshimit3015/text-anonymization-benchmark) | ✓ |
| 554 | [newsmtsc](src/tasksource/tasks.py#L3416) | Classification | [fhamborg/news_sentiment_newsmtsc](https://hf.co/datasets/fhamborg/news_sentiment_newsmtsc) | ✓ |
| 555 | [sentihood/sentiment](src/tasksource/tasks.py#L3444) | Classification | [bhavnicksm/sentihood](https://hf.co/datasets/bhavnicksm/sentihood) | ✓ |
| 556 | [mams/atsa](src/tasksource/tasks.py#L3461) | Classification | [NEUDM/mams](https://hf.co/datasets/NEUDM/mams) | ✓ |
| 557 | [quality](src/tasksource/tasks.py#L3469) | MultipleChoice | [emozilla/quality](https://hf.co/datasets/emozilla/quality) |  |
| 558 | [clear/readability](src/tasksource/tasks.py#L3478) | Classification | [casey-martin/CommonLit-Ease-of-Readability](https://hf.co/datasets/casey-martin/CommonLit-Ease-of-Readability) | ✓ |
| 559 | [esconv/strategy](src/tasksource/tasks.py#L3491) | Classification | [tasksource/esconv](https://hf.co/datasets/tasksource/esconv) | ✓ |
| 560 | [prosocial-dialog/safety](src/tasksource/tasks.py#L3512) | Classification | [allenai/prosocial-dialog](https://hf.co/datasets/allenai/prosocial-dialog) | ✓ |
| 561 | [gap](src/tasksource/tasks.py#L3532) | Classification | [coref-data/gap_raw](https://hf.co/datasets/coref-data/gap_raw) | ✓ |
| 562 | [trip/cloze](src/tasksource/tasks.py#L3540) | MultipleChoice | [sled-umich/TRIP](https://hf.co/datasets/sled-umich/TRIP) | ✓ |
| 563 | [trip/order](src/tasksource/tasks.py#L3549) | MultipleChoice | [sled-umich/TRIP](https://hf.co/datasets/sled-umich/TRIP) | ✓ |
| 564 | [sbic/offensiveness](src/tasksource/tasks.py#L3559) | Classification | [tasksource/sbic](https://hf.co/datasets/tasksource/sbic) | ✓ |
| 565 | [sbic/intent](src/tasksource/tasks.py#L3566) | Classification | [tasksource/sbic](https://hf.co/datasets/tasksource/sbic) | ✓ |
| 566 | [faithdial/grounding](src/tasksource/tasks.py#L3588) | Classification | [McGill-NLP/FaithDial](https://hf.co/datasets/McGill-NLP/FaithDial) | ✓ |
| 567 | [ragtruth/grounding](src/tasksource/tasks.py#L3605) | Classification | [wandb/RAGTruth-processed](https://hf.co/datasets/wandb/RAGTruth-processed) | ✓ |
| 568 | [comve](src/tasksource/tasks.py#L3617) | MultipleChoice | [tasksource/comve](https://hf.co/datasets/tasksource/comve) |  |
| 569 | [cbt/ne](src/tasksource/tasks.py#L3634) | MultipleChoice | [cam-cst/cbt](https://hf.co/datasets/cam-cst/cbt) | ✓ |
| 570 | [cbt/cn](src/tasksource/tasks.py#L3642) | MultipleChoice | [cam-cst/cbt](https://hf.co/datasets/cam-cst/cbt) | ✓ |
| 571 | [recam](src/tasksource/tasks.py#L3657) | MultipleChoice | [holylovenia/recam](https://hf.co/datasets/holylovenia/recam) | ✓ |
| 572 | [knowref60k](src/tasksource/tasks.py#L3672) | MultipleChoice | [coref-data/knowref_60k_raw](https://hf.co/datasets/coref-data/knowref_60k_raw) | ✓ |
| 573 | [tabmwp](src/tasksource/tasks.py#L3683) | MultipleChoice | [tasksource/tabmwp](https://hf.co/datasets/tasksource/tabmwp) |  |
| 574 | [procedural-choices/reasoning_gym](src/tasksource/tasks.py#L3697) | MultipleChoice | [reasoning-core/procedural-choices](https://hf.co/datasets/reasoning-core/procedural-choices) |  |
| 575 | [procedural-choices/reasoning_core](src/tasksource/tasks.py#L3697) | MultipleChoice | [reasoning-core/procedural-choices](https://hf.co/datasets/reasoning-core/procedural-choices) |  |
| 576 | [ems-mcqa](src/tasksource/tasks.py#L3724) | MultipleChoice | [Xueren/EMS-MCQA](https://hf.co/datasets/Xueren/EMS-MCQA) | ✓ |
| 577 | [soul](src/tasksource/tasks.py#L3734) | Classification | [DAMO-NLP-SG/SOUL](https://hf.co/datasets/DAMO-NLP-SG/SOUL) | ✓ |
| 578 | [checkthat2024/check-worthiness](src/tasksource/tasks.py#L3746) | Classification | [OpenFact/CLEF24-CheckThat-Task1-en-all](https://hf.co/datasets/OpenFact/CLEF24-CheckThat-Task1-en-all) | ✓ |

## Soft labels

Annotations whose label is a distribution (annotator votes, rater shares, survey counts), loaded with `load_task(id, soft=True)`. Those with a hard view are also listed above, by their majority label; the others have soft labels only. `votes` are shares of annotators, `mean` a numeric value interpolated over score anchors (see each task for its meaning); annotators is the typical count per item (vote shares from fewer than five are coarse, and the Jev build leaves them out).

| id | kind | aggregation | annotators | default view | dataset |
|---|---|---|--:|---|---|
| [glue/stsb](src/tasksource/tasks.py#L49) | score | mean |  | regression | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |
| [sick/relatedness](src/tasksource/tasks.py#L89) | score | mean | 10 | regression | [tasksource/sick](https://hf.co/datasets/tasksource/sick) |
| [google_wellformed_query](src/tasksource/tasks.py#L854) | noul | votes | 5 | Classification | [tasksource/google_wellformed_query](https://hf.co/datasets/tasksource/google_wellformed_query) |
| [sts-companion](src/tasksource/tasks.py#L1052) | score | mean |  | regression | [tasksource/sts-companion](https://hf.co/datasets/tasksource/sts-companion) |
| [acceptability-prediction/rating_votes](src/tasksource/tasks.py#L1285) | score | votes | 15 |  | [tasksource/acceptability-prediction](https://hf.co/datasets/tasksource/acceptability-prediction) |
| [acceptability-prediction/binary_votes](src/tasksource/tasks.py#L1290) | noul | votes | 15 |  | [tasksource/acceptability-prediction](https://hf.co/datasets/tasksource/acceptability-prediction) |
| [HelpSteer3/individual_preferences](src/tasksource/tasks.py#L2102) | score | votes | 3 |  | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) |
| [HelpSteer3/feedback](src/tasksource/tasks.py#L2135) | score | votes | 3 | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) |
| [proto_qa/proto_qa](src/tasksource/tasks.py#L2230) | choice | votes | 100 |  | [community-datasets/proto_qa](https://hf.co/datasets/community-datasets/proto_qa) |
| [UNLI](src/tasksource/tasks.py#L2244) | noul | mean | 2 |  | [Zhengping/UNLI](https://hf.co/datasets/Zhengping/UNLI) |
| [chaos-mnli-ambiguity/votes](src/tasksource/tasks.py#L2248) | choice | votes | 100 |  | [tasksource/chaos-mnli-ambiguity](https://hf.co/datasets/tasksource/chaos-mnli-ambiguity) |
| [hate_speech_offensive/votes](src/tasksource/tasks.py#L2252) | choice | votes | 3 |  | [tdavidson/hate_speech_offensive](https://hf.co/datasets/tdavidson/hate_speech_offensive) |
| [scruples/verdict_votes](src/tasksource/tasks.py#L2260) | choice | votes | 8 |  | [tasksource/scruples](https://hf.co/datasets/tasksource/scruples) |
| [BeaverTails/unsafe_votes](src/tasksource/tasks.py#L2274) | noul | votes | 3 |  | [PKU-Alignment/BeaverTails](https://hf.co/datasets/PKU-Alignment/BeaverTails) |
| [dynasent/r1_votes](src/tasksource/tasks.py#L2285) | choice | votes | 5 |  | [dynabench/dynasent](https://hf.co/datasets/dynabench/dynasent) |
| [dynasent/r2_votes](src/tasksource/tasks.py#L2289) | choice | votes | 5 |  | [dynabench/dynasent](https://hf.co/datasets/dynabench/dynasent) |
| [hatexplain/votes](src/tasksource/tasks.py#L2294) | choice | votes | 3 |  | [Hate-speech-CNERG/hatexplain](https://hf.co/datasets/Hate-speech-CNERG/hatexplain) |
| [measuring-hate-speech/sentiment](src/tasksource/tasks.py#L2309) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/respect](src/tasksource/tasks.py#L2311) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/insult](src/tasksource/tasks.py#L2313) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/humiliate](src/tasksource/tasks.py#L2314) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/status](src/tasksource/tasks.py#L2315) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/dehumanize](src/tasksource/tasks.py#L2317) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/violence](src/tasksource/tasks.py#L2319) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/genocide](src/tasksource/tasks.py#L2320) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/attack_defend](src/tasksource/tasks.py#L2322) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/hatespeech](src/tasksource/tasks.py#L2324) | choice | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [wikipedia-detox/attack](src/tasksource/tasks.py#L2336) | choice | votes | 10 |  | [tasksource/wikipedia-detox-votes](https://hf.co/datasets/tasksource/wikipedia-detox-votes) |
| [wikipedia-detox/aggression](src/tasksource/tasks.py#L2338) | choice | votes | 10 |  | [tasksource/wikipedia-detox-votes](https://hf.co/datasets/tasksource/wikipedia-detox-votes) |
| [wikipedia-detox/aggression_score](src/tasksource/tasks.py#L2340) | score | votes | 10 |  | [tasksource/wikipedia-detox-votes](https://hf.co/datasets/tasksource/wikipedia-detox-votes) |
| [wikipedia-detox/toxicity](src/tasksource/tasks.py#L2344) | choice | votes | 10 |  | [tasksource/wikipedia-detox-votes](https://hf.co/datasets/tasksource/wikipedia-detox-votes) |
| [wikipedia-detox/toxicity_score](src/tasksource/tasks.py#L2346) | score | votes | 10 |  | [tasksource/wikipedia-detox-votes](https://hf.co/datasets/tasksource/wikipedia-detox-votes) |
| [lewidi/md_agreement](src/tasksource/tasks.py#L2361) | noul | votes | 5 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/hs_brexit](src/tasksource/tasks.py#L2362) | noul | votes | 6 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/armis](src/tasksource/tasks.py#L2364) | noul | votes | 3 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/conv_abuse](src/tasksource/tasks.py#L2366) | noul | votes | 3 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/mp](src/tasksource/tasks.py#L2368) | noul | votes | 5 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/csc](src/tasksource/tasks.py#L2369) | score | votes | 4 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/varierrnli/entailment](src/tasksource/tasks.py#L2377) | noul | votes | 4 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/varierrnli/neutral](src/tasksource/tasks.py#L2378) | noul | votes | 4 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/varierrnli/contradiction](src/tasksource/tasks.py#L2379) | noul | votes | 4 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [civil_comments/toxicity_share](src/tasksource/tasks.py#L2387) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/severe_toxicity_share](src/tasksource/tasks.py#L2388) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/obscene_share](src/tasksource/tasks.py#L2389) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/threat_share](src/tasksource/tasks.py#L2390) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/insult_share](src/tasksource/tasks.py#L2391) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/identity_attack_share](src/tasksource/tasks.py#L2392) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/sexual_explicit_share](src/tasksource/tasks.py#L2393) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [oasst2/quality](src/tasksource/tasks.py#L2410) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/helpfulness](src/tasksource/tasks.py#L2411) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/creativity](src/tasksource/tasks.py#L2412) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/humor](src/tasksource/tasks.py#L2413) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/toxicity](src/tasksource/tasks.py#L2414) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/violence](src/tasksource/tasks.py#L2415) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/spam](src/tasksource/tasks.py#L2416) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/fails_task](src/tasksource/tasks.py#L2417) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/not_appropriate](src/tasksource/tasks.py#L2418) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/hate_speech](src/tasksource/tasks.py#L2419) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/sexual_content](src/tasksource/tasks.py#L2420) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/pii](src/tasksource/tasks.py#L2421) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/lang_mismatch](src/tasksource/tasks.py#L2422) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [scirepeval/search](src/tasksource/tasks.py#L2525) | score | mean |  | regression | [tasksource/scirepeval-search](https://hf.co/datasets/tasksource/scirepeval-search) |
| [discogem/relation](src/tasksource/tasks.py#L2956) | choice | votes | 10 | Classification | [tasksource/discogem](https://hf.co/datasets/tasksource/discogem) |
| [clear/readability](src/tasksource/tasks.py#L3478) | score | mean |  | regression | [casey-martin/CommonLit-Ease-of-Readability](https://hf.co/datasets/casey-martin/CommonLit-Ease-of-Readability) |
