584 English tasks. Load one with `load_task(id)`; the annotations are in [tasks.py](src/tasksource/tasks.py). Evaluation benchmarks are in [eval_only.py](src/tasksource/eval_only.py); other excluded annotations are in [parked.py](src/tasksource/parked.py).

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
| 11 | [super_glue/multirc](src/tasksource/tasks.py#L58) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 12 | [super_glue/wic](src/tasksource/tasks.py#L63) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 13 | [super_glue/axg](src/tasksource/tasks.py#L68) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 14 | [anli/a1](src/tasksource/tasks.py#L71) | Classification | [facebook/anli](https://hf.co/datasets/facebook/anli) |  |
| 15 | [anli/a2](src/tasksource/tasks.py#L72) | Classification | [facebook/anli](https://hf.co/datasets/facebook/anli) |  |
| 16 | [anli/a3](src/tasksource/tasks.py#L73) | Classification | [facebook/anli](https://hf.co/datasets/facebook/anli) |  |
| 17 | [babi_nli/indefinite-knowledge](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 18 | [babi_nli/positional-reasoning](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 19 | [babi_nli/lists-sets](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 20 | [babi_nli/path-finding](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 21 | [babi_nli/basic-deduction](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 22 | [babi_nli/compound-coreference](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 23 | [babi_nli/basic-coreference](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 24 | [babi_nli/simple-negation](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 25 | [babi_nli/two-supporting-facts](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 26 | [babi_nli/size-reasoning](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 27 | [babi_nli/three-arg-relations](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 28 | [babi_nli/three-supporting-facts](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 29 | [babi_nli/time-reasoning](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 30 | [babi_nli/two-arg-relations](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 31 | [babi_nli/conjunction](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 32 | [babi_nli/yes-no-questions](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 33 | [babi_nli/single-supporting-fact](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 34 | [babi_nli/counting](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 35 | [babi_nli/basic-induction](src/tasksource/tasks.py#L81) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) | ✓ |
| 36 | [sick/label](src/tasksource/tasks.py#L86) | Classification | [tasksource/sick](https://hf.co/datasets/tasksource/sick) |  |
| 37 | [sick/relatedness](src/tasksource/tasks.py#L88) | Classification | [tasksource/sick](https://hf.co/datasets/tasksource/sick) | ✓ |
| 38 | [snli](src/tasksource/tasks.py#L146) | Classification | [stanfordnlp/snli](https://hf.co/datasets/stanfordnlp/snli) |  |
| 39 | [scitail/snli_format](src/tasksource/tasks.py#L149) | Classification | [allenai/scitail](https://hf.co/datasets/allenai/scitail) |  |
| 40 | [hans](src/tasksource/tasks.py#L151) | Classification | [tasksource/hans](https://hf.co/datasets/tasksource/hans) |  |
| 41 | [WANLI](src/tasksource/tasks.py#L154) | Classification | [alisawuffles/WANLI](https://hf.co/datasets/alisawuffles/WANLI) |  |
| 42 | [recast/recast_megaveridicality](src/tasksource/tasks.py#L156) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 43 | [recast/recast_sentiment](src/tasksource/tasks.py#L156) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 44 | [recast/recast_ner](src/tasksource/tasks.py#L156) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 45 | [recast/recast_verbcorner](src/tasksource/tasks.py#L156) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 46 | [recast/recast_verbnet](src/tasksource/tasks.py#L156) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 47 | [recast/recast_factuality](src/tasksource/tasks.py#L156) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 48 | [recast/recast_puns](src/tasksource/tasks.py#L156) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 49 | [probability_words_nli/reasoning_1hop](src/tasksource/tasks.py#L161) | Classification | [sileod/probability_words_nli](https://hf.co/datasets/sileod/probability_words_nli) | ✓ |
| 50 | [probability_words_nli/usnli](src/tasksource/tasks.py#L161) | Classification | [sileod/probability_words_nli](https://hf.co/datasets/sileod/probability_words_nli) | ✓ |
| 51 | [probability_words_nli/reasoning_2hop](src/tasksource/tasks.py#L161) | Classification | [sileod/probability_words_nli](https://hf.co/datasets/sileod/probability_words_nli) | ✓ |
| 52 | [nan-nli](src/tasksource/tasks.py#L166) | Classification | [joey234/nan-nli](https://hf.co/datasets/joey234/nan-nli) |  |
| 53 | [nli_fever](src/tasksource/tasks.py#L169) | Classification | [pietrolesci/nli_fever](https://hf.co/datasets/pietrolesci/nli_fever) |  |
| 54 | [breaking_nli](src/tasksource/tasks.py#L172) | Classification | [pietrolesci/breaking_nli](https://hf.co/datasets/pietrolesci/breaking_nli) |  |
| 55 | [conj_nli](src/tasksource/tasks.py#L176) | Classification | [pietrolesci/conj_nli](https://hf.co/datasets/pietrolesci/conj_nli) |  |
| 56 | [fracas](src/tasksource/tasks.py#L180) | Classification | [pietrolesci/fracas](https://hf.co/datasets/pietrolesci/fracas) |  |
| 57 | [dialogue_nli](src/tasksource/tasks.py#L183) | Classification | [pietrolesci/dialogue_nli](https://hf.co/datasets/pietrolesci/dialogue_nli) | ✓ |
| 58 | [mpe](src/tasksource/tasks.py#L186) | Classification | [pietrolesci/mpe](https://hf.co/datasets/pietrolesci/mpe) |  |
| 59 | [dnc](src/tasksource/tasks.py#L190) | Classification | [pietrolesci/dnc](https://hf.co/datasets/pietrolesci/dnc) |  |
| 60 | [recast_white/fnplus](src/tasksource/tasks.py#L194) | Classification | [pietrolesci/recast_white](https://hf.co/datasets/pietrolesci/recast_white) |  |
| 61 | [recast_white/sprl](src/tasksource/tasks.py#L197) | Classification | [pietrolesci/recast_white](https://hf.co/datasets/pietrolesci/recast_white) |  |
| 62 | [recast_white/dpr](src/tasksource/tasks.py#L200) | Classification | [pietrolesci/recast_white](https://hf.co/datasets/pietrolesci/recast_white) |  |
| 63 | [joci](src/tasksource/tasks.py#L204) | Classification | [pietrolesci/joci](https://hf.co/datasets/pietrolesci/joci) | ✓ |
| 64 | [robust_nli/IS_CS](src/tasksource/tasks.py#L211) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 65 | [robust_nli/LI_LI](src/tasksource/tasks.py#L213) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 66 | [robust_nli/ST_WO](src/tasksource/tasks.py#L215) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 67 | [robust_nli/PI_SP](src/tasksource/tasks.py#L217) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 68 | [robust_nli/PI_CD](src/tasksource/tasks.py#L219) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 69 | [robust_nli/ST_SE](src/tasksource/tasks.py#L221) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 70 | [robust_nli/ST_NE](src/tasksource/tasks.py#L223) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 71 | [robust_nli/ST_LM](src/tasksource/tasks.py#L225) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 72 | [robust_nli_is_sd](src/tasksource/tasks.py#L227) | Classification | [pietrolesci/robust_nli_is_sd](https://hf.co/datasets/pietrolesci/robust_nli_is_sd) |  |
| 73 | [robust_nli_li_ts](src/tasksource/tasks.py#L230) | Classification | [pietrolesci/robust_nli_li_ts](https://hf.co/datasets/pietrolesci/robust_nli_li_ts) |  |
| 74 | [gen_debiased_nli/snli_seq_z](src/tasksource/tasks.py#L234) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| 75 | [gen_debiased_nli/snli_z_aug](src/tasksource/tasks.py#L236) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| 76 | [gen_debiased_nli/snli_par_z](src/tasksource/tasks.py#L238) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| 77 | [gen_debiased_nli/mnli_par_z](src/tasksource/tasks.py#L240) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| 78 | [gen_debiased_nli/mnli_z_aug](src/tasksource/tasks.py#L242) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| 79 | [gen_debiased_nli/mnli_seq_z](src/tasksource/tasks.py#L244) | Classification | [pietrolesci/gen_debiased_nli](https://hf.co/datasets/pietrolesci/gen_debiased_nli) |  |
| 80 | [add_one_rte](src/tasksource/tasks.py#L247) | Classification | [pietrolesci/add_one_rte](https://hf.co/datasets/pietrolesci/add_one_rte) |  |
| 81 | [hlgd](src/tasksource/tasks.py#L251) | Classification | [tasksource/hlgd](https://hf.co/datasets/tasksource/hlgd) | ✓ |
| 82 | [paws/labeled_final](src/tasksource/tasks.py#L255) | Classification | [google-research-datasets/paws](https://hf.co/datasets/google-research-datasets/paws) |  |
| 83 | [paws/labeled_swap](src/tasksource/tasks.py#L256) | Classification | [google-research-datasets/paws](https://hf.co/datasets/google-research-datasets/paws) |  |
| 84 | [medical_questions_pairs](src/tasksource/tasks.py#L258) | Classification | [curaihealth/medical_questions_pairs](https://hf.co/datasets/curaihealth/medical_questions_pairs) | ✓ |
| 85 | [conll2003/pos_tags](src/tasksource/tasks.py#L264) | TokenClassification | [tomaarsen/conll2003](https://hf.co/datasets/tomaarsen/conll2003) | ✓ |
| 86 | [conll2003/chunk_tags](src/tasksource/tasks.py#L266) | TokenClassification | [tomaarsen/conll2003](https://hf.co/datasets/tomaarsen/conll2003) | ✓ |
| 87 | [conll2003/ner_tags](src/tasksource/tasks.py#L268) | TokenClassification | [tomaarsen/conll2003](https://hf.co/datasets/tomaarsen/conll2003) |  |
| 88 | [fig-qa](src/tasksource/tasks.py#L274) | MultipleChoice | [nightingal3/fig-qa](https://hf.co/datasets/nightingal3/fig-qa) |  |
| 89 | [cos_e/v1.0](src/tasksource/tasks.py#L283) | MultipleChoice | [Salesforce/cos_e](https://hf.co/datasets/Salesforce/cos_e) |  |
| 90 | [cosmos_qa](src/tasksource/tasks.py#L288) | MultipleChoice | [Samsoup/cosmos_qa](https://hf.co/datasets/Samsoup/cosmos_qa) |  |
| 91 | [dream](src/tasksource/tasks.py#L291) | MultipleChoice | [dataset-org/dream](https://hf.co/datasets/dataset-org/dream) |  |
| 92 | [openbookqa](src/tasksource/tasks.py#L298) | MultipleChoice | [allenai/openbookqa](https://hf.co/datasets/allenai/openbookqa) |  |
| 93 | [qasc](src/tasksource/tasks.py#L304) | MultipleChoice | [allenai/qasc](https://hf.co/datasets/allenai/qasc) |  |
| 94 | [quartz](src/tasksource/tasks.py#L312) | MultipleChoice | [allenai/quartz](https://hf.co/datasets/allenai/quartz) |  |
| 95 | [quail](src/tasksource/tasks.py#L317) | MultipleChoice | [textmachinelab/quail](https://hf.co/datasets/textmachinelab/quail) |  |
| 96 | [head_qa/en](src/tasksource/tasks.py#L323) | MultipleChoice | [EleutherAI/headqa](https://hf.co/datasets/EleutherAI/headqa) |  |
| 97 | [sciq](src/tasksource/tasks.py#L331) | MultipleChoice | [allenai/sciq](https://hf.co/datasets/allenai/sciq) |  |
| 98 | [social_i_qa](src/tasksource/tasks.py#L336) | MultipleChoice | [tasksource/social_i_qa](https://hf.co/datasets/tasksource/social_i_qa) |  |
| 99 | [wiki_hop/original](src/tasksource/tasks.py#L342) | MultipleChoice | [MoE-UNC/wikihop](https://hf.co/datasets/MoE-UNC/wikihop) |  |
| 100 | [wiqa](src/tasksource/tasks.py#L355) | MultipleChoice | [allenai/wiqa](https://hf.co/datasets/allenai/wiqa) (`parquet` loader) |  |
| 101 | [piqa](src/tasksource/tasks.py#L362) | MultipleChoice | [baber/piqa](https://hf.co/datasets/baber/piqa) |  |
| 102 | [hellaswag](src/tasksource/tasks.py#L370) | MultipleChoice | [Rowan/hellaswag](https://hf.co/datasets/Rowan/hellaswag) |  |
| 103 | [super_glue/copa](src/tasksource/tasks.py#L379) | MultipleChoice | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 104 | [balanced-copa](src/tasksource/tasks.py#L381) | MultipleChoice | [pkavumba/balanced-copa](https://hf.co/datasets/pkavumba/balanced-copa) |  |
| 105 | [e-CARE](src/tasksource/tasks.py#L384) | MultipleChoice | [12ml/e-CARE](https://hf.co/datasets/12ml/e-CARE) |  |
| 106 | [art](src/tasksource/tasks.py#L387) | MultipleChoice | [allenai/art](https://hf.co/datasets/allenai/art) | ✓ |
| 107 | [winogrande/winogrande_xl](src/tasksource/tasks.py#L395) | MultipleChoice | [allenai/winogrande](https://hf.co/datasets/allenai/winogrande) |  |
| 108 | [codah/codah](src/tasksource/tasks.py#L398) | MultipleChoice | [jaredfern/codah](https://hf.co/datasets/jaredfern/codah) |  |
| 109 | [ai2_arc/ARC-Easy/challenge](src/tasksource/tasks.py#L400) | MultipleChoice | [allenai/ai2_arc](https://hf.co/datasets/allenai/ai2_arc) |  |
| 110 | [ai2_arc/ARC-Challenge/challenge](src/tasksource/tasks.py#L400) | MultipleChoice | [allenai/ai2_arc](https://hf.co/datasets/allenai/ai2_arc) |  |
| 111 | [definite_pronoun_resolution](src/tasksource/tasks.py#L405) | MultipleChoice | [community-datasets/definite_pronoun_resolution](https://hf.co/datasets/community-datasets/definite_pronoun_resolution) |  |
| 112 | [swag/regular](src/tasksource/tasks.py#L411) | MultipleChoice | [allenai/swag](https://hf.co/datasets/allenai/swag) |  |
| 113 | [math_qa](src/tasksource/tasks.py#L417) | MultipleChoice | [tasksource/math_qa](https://hf.co/datasets/tasksource/math_qa) |  |
| 114 | [glue/cola](src/tasksource/tasks.py#L426) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 115 | [glue/sst2](src/tasksource/tasks.py#L427) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 116 | [utilitarianism](src/tasksource/tasks.py#L441) | Classification | [hendrycks/ethics](https://hf.co/datasets/hendrycks/ethics) (`csv` loader) |  |
| 117 | [amazon_counterfactual/en](src/tasksource/tasks.py#L449) | Classification | [mteb/amazon_counterfactual](https://hf.co/datasets/mteb/amazon_counterfactual) |  |
| 118 | [insincere-questions](src/tasksource/tasks.py#L454) | Classification | [SetFit/insincere-questions](https://hf.co/datasets/SetFit/insincere-questions) |  |
| 119 | [toxic_conversations](src/tasksource/tasks.py#L458) | Classification | [SetFit/toxic_conversations](https://hf.co/datasets/SetFit/toxic_conversations) |  |
| 120 | [TuringBench](src/tasksource/tasks.py#L462) | Classification | [jana4/turingbench-humanized](https://hf.co/datasets/jana4/turingbench-humanized), [turingbench/TuringBench](https://hf.co/datasets/turingbench/TuringBench) (`csv` loader) | ✓ |
| 121 | [trec](src/tasksource/tasks.py#L472) | Classification | [tasksource/trec](https://hf.co/datasets/tasksource/trec) |  |
| 122 | [vitaminc](src/tasksource/tasks.py#L475) | Classification | [tals/vitaminc](https://hf.co/datasets/tals/vitaminc) |  |
| 123 | [hope_edi/english](src/tasksource/tasks.py#L477) | Classification | [dravidianlangtech/hope_edi](https://hf.co/datasets/dravidianlangtech/hope_edi) (`csv` loader) |  |
| 124 | [rumoureval_2019/RumourEval2019](src/tasksource/tasks.py#L491) | Classification | [strombergnlp/rumoureval_2019](https://hf.co/datasets/strombergnlp/rumoureval_2019) (`csv` loader) |  |
| 125 | [ethos/binary](src/tasksource/tasks.py#L505) | Classification | [SetFit/ethos_binary](https://hf.co/datasets/SetFit/ethos_binary) |  |
| 126 | [ethos/multilabel](src/tasksource/tasks.py#L529) | Classification | [tasksource/ethos](https://hf.co/datasets/tasksource/ethos) |  |
| 127 | [tweet_eval/sentiment](src/tasksource/tasks.py#L532) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 128 | [tweet_eval/irony](src/tasksource/tasks.py#L532) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 129 | [tweet_eval/hate](src/tasksource/tasks.py#L532) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 130 | [tweet_eval/emotion](src/tasksource/tasks.py#L532) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 131 | [tweet_eval/emoji](src/tasksource/tasks.py#L532) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 132 | [tweet_eval/offensive](src/tasksource/tasks.py#L532) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 133 | [tweet_eval/stance_abortion](src/tasksource/tasks.py#L547) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 134 | [tweet_eval/stance_atheism](src/tasksource/tasks.py#L548) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 135 | [tweet_eval/stance_climate](src/tasksource/tasks.py#L549) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 136 | [tweet_eval/stance_feminist](src/tasksource/tasks.py#L550) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 137 | [tweet_eval/stance_hillary](src/tasksource/tasks.py#L551) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 138 | [discovery/discovery](src/tasksource/tasks.py#L554) | Classification | [sileod/discovery](https://hf.co/datasets/sileod/discovery) | ✓ |
| 139 | [pragmeval/verifiability](src/tasksource/tasks.py#L557) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 140 | [pragmeval/mrda](src/tasksource/tasks.py#L557) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 141 | [pragmeval/switchboard](src/tasksource/tasks.py#L557) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 142 | [pragmeval/persuasiveness-claimtype](src/tasksource/tasks.py#L561) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 143 | [pragmeval/persuasiveness-premisetype](src/tasksource/tasks.py#L561) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 144 | [pragmeval/sarcasm](src/tasksource/tasks.py#L561) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 145 | [pragmeval/stac](src/tasksource/tasks.py#L561) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 146 | [pragmeval/gum](src/tasksource/tasks.py#L561) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 147 | [pragmeval/emergent](src/tasksource/tasks.py#L561) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 148 | [pragmeval/pdtb](src/tasksource/tasks.py#L561) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 149 | [pragmeval/emobank-arousal](src/tasksource/tasks.py#L574) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 150 | [pragmeval/emobank-dominance](src/tasksource/tasks.py#L575) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 151 | [pragmeval/emobank-valence](src/tasksource/tasks.py#L576) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 152 | [pragmeval/squinky-formality](src/tasksource/tasks.py#L577) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 153 | [pragmeval/squinky-implicature](src/tasksource/tasks.py#L578) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 154 | [pragmeval/squinky-informativeness](src/tasksource/tasks.py#L579) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 155 | [pragmeval/persuasiveness-eloquence](src/tasksource/tasks.py#L580) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 156 | [pragmeval/persuasiveness-relevance](src/tasksource/tasks.py#L581) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 157 | [pragmeval/persuasiveness-specificity](src/tasksource/tasks.py#L582) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 158 | [pragmeval/persuasiveness-strength](src/tasksource/tasks.py#L583) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 159 | [silicone/maptask](src/tasksource/tasks.py#L585) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | ✓ |
| 160 | [silicone/dyda_e](src/tasksource/tasks.py#L585) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | ✓ |
| 161 | [silicone/meld_e](src/tasksource/tasks.py#L585) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | ✓ |
| 162 | [silicone/meld_s](src/tasksource/tasks.py#L585) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | ✓ |
| 163 | [silicone/sem](src/tasksource/tasks.py#L585) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | ✓ |
| 164 | [silicone/oasis](src/tasksource/tasks.py#L585) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | ✓ |
| 165 | [silicone/dyda_da](src/tasksource/tasks.py#L585) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | ✓ |
| 166 | [silicone/iemocap](src/tasksource/tasks.py#L595) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | ✓ |
| 167 | [lex_glue/eurlex](src/tasksource/tasks.py#L605) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) | ✓ |
| 168 | [lex_glue/scotus](src/tasksource/tasks.py#L609) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) |  |
| 169 | [lex_glue/ledgar](src/tasksource/tasks.py#L612) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) |  |
| 170 | [lex_glue/unfair_tos](src/tasksource/tasks.py#L614) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) | ✓ |
| 171 | [lex_glue/case_hold](src/tasksource/tasks.py#L617) | MultipleChoice | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) |  |
| 172 | [language-identification](src/tasksource/tasks.py#L625) | Classification | [papluca/language-identification](https://hf.co/datasets/papluca/language-identification) | ✓ |
| 173 | [imdb](src/tasksource/tasks.py#L630) | Classification | [stanfordnlp/imdb](https://hf.co/datasets/stanfordnlp/imdb) |  |
| 174 | [rotten_tomatoes](src/tasksource/tasks.py#L632) | Classification | [cornell-movie-review-data/rotten_tomatoes](https://hf.co/datasets/cornell-movie-review-data/rotten_tomatoes) |  |
| 175 | [ag_news](src/tasksource/tasks.py#L634) | Classification | [fancyzhx/ag_news](https://hf.co/datasets/fancyzhx/ag_news) |  |
| 176 | [yelp_review_full/yelp_review_full](src/tasksource/tasks.py#L636) | Classification | [Yelp/yelp_review_full](https://hf.co/datasets/Yelp/yelp_review_full) | ✓ |
| 177 | [financial_phrasebank/sentences_allagree](src/tasksource/tasks.py#L641) | Classification | [ghbacct/financial-phrasebank-all-agree-classification](https://hf.co/datasets/ghbacct/financial-phrasebank-all-agree-classification) |  |
| 178 | [poem_sentiment](src/tasksource/tasks.py#L646) | Classification | [google-research-datasets/poem_sentiment](https://hf.co/datasets/google-research-datasets/poem_sentiment) |  |
| 179 | [dbpedia_14/dbpedia_14](src/tasksource/tasks.py#L648) | Classification | [fancyzhx/dbpedia_14](https://hf.co/datasets/fancyzhx/dbpedia_14) |  |
| 180 | [amazon_polarity/amazon_polarity](src/tasksource/tasks.py#L650) | Classification | [fancyzhx/amazon_polarity](https://hf.co/datasets/fancyzhx/amazon_polarity) |  |
| 181 | [app_reviews](src/tasksource/tasks.py#L652) | Classification | [sealuzh/app_reviews](https://hf.co/datasets/sealuzh/app_reviews) | ✓ |
| 182 | [hate_speech18](src/tasksource/tasks.py#L659) | Classification | [tasksource/hate_speech18](https://hf.co/datasets/tasksource/hate_speech18) |  |
| 183 | [sms_spam](src/tasksource/tasks.py#L665) | Classification | [ucirvine/sms_spam](https://hf.co/datasets/ucirvine/sms_spam) |  |
| 184 | [humicroedit/subtask-1](src/tasksource/tasks.py#L668) | Classification | [tasksource/humicroedit](https://hf.co/datasets/tasksource/humicroedit) | ✓ |
| 185 | [humicroedit/subtask-2](src/tasksource/tasks.py#L674) | Classification | [tasksource/humicroedit](https://hf.co/datasets/tasksource/humicroedit) | ✓ |
| 186 | [snips_built_in_intents](src/tasksource/tasks.py#L679) | Classification | [sonos-nlu-benchmark/snips_built_in_intents](https://hf.co/datasets/sonos-nlu-benchmark/snips_built_in_intents) |  |
| 187 | [hate_speech_offensive](src/tasksource/tasks.py#L683) | Classification | [tdavidson/hate_speech_offensive](https://hf.co/datasets/tdavidson/hate_speech_offensive) |  |
| 188 | [yahoo_answers_topics](src/tasksource/tasks.py#L685) | Classification | [community-datasets/yahoo_answers_topics](https://hf.co/datasets/community-datasets/yahoo_answers_topics) |  |
| 189 | [stackoverflow-questions](src/tasksource/tasks.py#L689) | Classification | [pacovaldez/stackoverflow-questions](https://hf.co/datasets/pacovaldez/stackoverflow-questions) | ✓ |
| 190 | [hyperpartisan_news](src/tasksource/tasks.py#L695) | Classification | [zapsdcn/hyperpartisan_news](https://hf.co/datasets/zapsdcn/hyperpartisan_news) |  |
| 191 | [sciie](src/tasksource/tasks.py#L700) | Classification | [zapsdcn/sciie](https://hf.co/datasets/zapsdcn/sciie) |  |
| 192 | [citation_intent](src/tasksource/tasks.py#L701) | Classification | [zapsdcn/citation_intent](https://hf.co/datasets/zapsdcn/citation_intent) |  |
| 193 | [go_emotions/simplified](src/tasksource/tasks.py#L703) | Classification | [google-research-datasets/go_emotions](https://hf.co/datasets/google-research-datasets/go_emotions) |  |
| 194 | [scicite](src/tasksource/tasks.py#L707) | Classification | [tasksource/scicite](https://hf.co/datasets/tasksource/scicite) |  |
| 195 | [liar](src/tasksource/tasks.py#L709) | Classification | [tasksource/liar](https://hf.co/datasets/tasksource/liar) | ✓ |
| 196 | [lexical_relation_classification/K&H+N](src/tasksource/tasks.py#L720) | Classification | [relbert/lexical_relation_classification](https://hf.co/datasets/relbert/lexical_relation_classification) (`json` loader) | ✓ |
| 197 | [lexical_relation_classification/BLESS](src/tasksource/tasks.py#L720) | Classification | [relbert/lexical_relation_classification](https://hf.co/datasets/relbert/lexical_relation_classification) (`json` loader) | ✓ |
| 198 | [lexical_relation_classification/EVALution](src/tasksource/tasks.py#L720) | Classification | [relbert/lexical_relation_classification](https://hf.co/datasets/relbert/lexical_relation_classification) (`json` loader) | ✓ |
| 199 | [lexical_relation_classification/ROOT09](src/tasksource/tasks.py#L720) | Classification | [relbert/lexical_relation_classification](https://hf.co/datasets/relbert/lexical_relation_classification) (`json` loader) | ✓ |
| 200 | [lexical_relation_classification/CogALexV](src/tasksource/tasks.py#L746) | Classification | [relbert/lexical_relation_classification](https://hf.co/datasets/relbert/lexical_relation_classification) (`json` loader) | ✓ |
| 201 | [linguisticprobing/subj_number](src/tasksource/tasks.py#L773) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 202 | [linguisticprobing/obj_number](src/tasksource/tasks.py#L774) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 203 | [linguisticprobing/past_present](src/tasksource/tasks.py#L775) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 204 | [linguisticprobing/top_constituents](src/tasksource/tasks.py#L776) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | ✓ |
| 205 | [linguisticprobing/tree_depth](src/tasksource/tasks.py#L778) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | ✓ |
| 206 | [linguisticprobing/coordination_inversion](src/tasksource/tasks.py#L779) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | ✓ |
| 207 | [linguisticprobing/odd_man_out](src/tasksource/tasks.py#L781) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | ✓ |
| 208 | [linguisticprobing/bigram_shift](src/tasksource/tasks.py#L782) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | ✓ |
| 209 | [crowdflower/political-media-audience](src/tasksource/tasks.py#L784) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 210 | [crowdflower/political-media-message](src/tasksource/tasks.py#L784) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 211 | [crowdflower/sentiment_nuclear_power](src/tasksource/tasks.py#L784) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 212 | [crowdflower/tweet_global_warming](src/tasksource/tasks.py#L784) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 213 | [crowdflower/airline-sentiment](src/tasksource/tasks.py#L784) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 214 | [crowdflower/political-media-bias](src/tasksource/tasks.py#L784) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 215 | [crowdflower/corporate-messaging](src/tasksource/tasks.py#L784) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 216 | [crowdflower/economic-news](src/tasksource/tasks.py#L784) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 217 | [ethics/commonsense](src/tasksource/tasks.py#L818) | Classification | [hendrycks/ethics](https://hf.co/datasets/hendrycks/ethics) (`csv` loader) |  |
| 218 | [ethics/deontology](src/tasksource/tasks.py#L826) | Classification | [hendrycks/ethics](https://hf.co/datasets/hendrycks/ethics) (`csv` loader) | ✓ |
| 219 | [ethics/justice](src/tasksource/tasks.py#L835) | Classification | [hendrycks/ethics](https://hf.co/datasets/hendrycks/ethics) (`csv` loader) |  |
| 220 | [ethics/virtue](src/tasksource/tasks.py#L843) | Classification | [hendrycks/ethics](https://hf.co/datasets/hendrycks/ethics) |  |
| 221 | [emo/emo2019](src/tasksource/tasks.py#L852) | Classification | [oneonlee/cleansed_emocontext](https://hf.co/datasets/oneonlee/cleansed_emocontext) |  |
| 222 | [google_wellformed_query](src/tasksource/tasks.py#L858) | Classification | [tasksource/google_wellformed_query](https://hf.co/datasets/tasksource/google_wellformed_query) | ✓ |
| 223 | [tweets_hate_speech_detection](src/tasksource/tasks.py#L864) | Classification | [tweets-hate-speech-detection/tweets_hate_speech_detection](https://hf.co/datasets/tweets-hate-speech-detection/tweets_hate_speech_detection) |  |
| 224 | [wnut_17/wnut_17](src/tasksource/tasks.py#L868) | TokenClassification | [flaitenberger/wnut_17](https://hf.co/datasets/flaitenberger/wnut_17) |  |
| 225 | [ncbi_disease/ncbi_disease](src/tasksource/tasks.py#L871) | TokenClassification | [ncbi/ncbi_disease](https://hf.co/datasets/ncbi/ncbi_disease) | ✓ |
| 226 | [acronym_identification](src/tasksource/tasks.py#L876) | TokenClassification | [amirveyseh/acronym_identification](https://hf.co/datasets/amirveyseh/acronym_identification) | ✓ |
| 227 | [jnlpba/jnlpba](src/tasksource/tasks.py#L880) | TokenClassification | [jnlpba/jnlpba](https://hf.co/datasets/jnlpba/jnlpba) | ✓ |
| 228 | [ontonotes_english/SpeedOfMagic--ontonotes_english](src/tasksource/tasks.py#L889) | TokenClassification | [SpeedOfMagic/ontonotes_english](https://hf.co/datasets/SpeedOfMagic/ontonotes_english) | ✓ |
| 229 | [blog_authorship_corpus/gender](src/tasksource/tasks.py#L895) | Classification | [tasksource/blog_authorship_corpus](https://hf.co/datasets/tasksource/blog_authorship_corpus) | ✓ |
| 230 | [blog_authorship_corpus/age](src/tasksource/tasks.py#L897) | Classification | [tasksource/blog_authorship_corpus](https://hf.co/datasets/tasksource/blog_authorship_corpus) | ✓ |
| 231 | [blog_authorship_corpus/job](src/tasksource/tasks.py#L900) | Classification | [tasksource/blog_authorship_corpus](https://hf.co/datasets/tasksource/blog_authorship_corpus) | ✓ |
| 232 | [open_question_type](src/tasksource/tasks.py#L911) | Classification | [Korea-MES/open_question_type](https://hf.co/datasets/Korea-MES/open_question_type) |  |
| 233 | [health_fact](src/tasksource/tasks.py#L913) | Classification | [marcov/health_fact_promptsource](https://hf.co/datasets/marcov/health_fact_promptsource) |  |
| 234 | [commonsense_qa](src/tasksource/tasks.py#L917) | MultipleChoice | [tau/commonsense_qa](https://hf.co/datasets/tau/commonsense_qa) |  |
| 235 | [mc_taco](src/tasksource/tasks.py#L923) | Classification | [marcov/mc_taco_promptsource](https://hf.co/datasets/marcov/mc_taco_promptsource) | ✓ |
| 236 | [ade_corpus_v2/Ade_corpus_v2_classification](src/tasksource/tasks.py#L930) | Classification | [ade-benchmark-corpus/ade_corpus_v2](https://hf.co/datasets/ade-benchmark-corpus/ade_corpus_v2) | ✓ |
| 237 | [discosense](src/tasksource/tasks.py#L933) | MultipleChoice | [tasksource/discosense](https://hf.co/datasets/tasksource/discosense) | ✓ |
| 238 | [circa](src/tasksource/tasks.py#L940) | Classification | [google-research-datasets/circa](https://hf.co/datasets/google-research-datasets/circa) |  |
| 239 | [code_x_glue_cc_defect_detection](src/tasksource/tasks.py#L945) | Classification | [google/code_x_glue_cc_defect_detection](https://hf.co/datasets/google/code_x_glue_cc_defect_detection) | ✓ |
| 240 | [phrase_similarity](src/tasksource/tasks.py#L950) | Classification | [Deehan1866/processed_phrase_similarity](https://hf.co/datasets/Deehan1866/processed_phrase_similarity) |  |
| 241 | [scientific-exaggeration-detection](src/tasksource/tasks.py#L958) | Classification | [copenlu/scientific-exaggeration-detection](https://hf.co/datasets/copenlu/scientific-exaggeration-detection) |  |
| 242 | [quarel](src/tasksource/tasks.py#L972) | MultipleChoice | [community-datasets/quarel](https://hf.co/datasets/community-datasets/quarel) | ✓ |
| 243 | [fever-evidence-related](src/tasksource/tasks.py#L979) | Classification | [mwong/fever-evidence-related](https://hf.co/datasets/mwong/fever-evidence-related) |  |
| 244 | [numer_sense](src/tasksource/tasks.py#L982) | Classification | [tasksource/numer_sense](https://hf.co/datasets/tasksource/numer_sense) |  |
| 245 | [dynasent/dynabench.dynasent.r1.all/r1](src/tasksource/tasks.py#L989) | Classification | [tasksource/dynasent](https://hf.co/datasets/tasksource/dynasent) | ✓ |
| 246 | [dynasent/dynabench.dynasent.r2.all/r2](src/tasksource/tasks.py#L993) | Classification | [tasksource/dynasent](https://hf.co/datasets/tasksource/dynasent) | ✓ |
| 247 | [Sarcasm_News_Headline](src/tasksource/tasks.py#L998) | Classification | [raquiba/Sarcasm_News_Headline](https://hf.co/datasets/raquiba/Sarcasm_News_Headline) |  |
| 248 | [sem_eval_2010_task_8](src/tasksource/tasks.py#L1001) | Classification | [SemEvalWorkshop/sem_eval_2010_task_8](https://hf.co/datasets/SemEvalWorkshop/sem_eval_2010_task_8) |  |
| 249 | [auditor_review](src/tasksource/tasks.py#L1003) | Classification | [demo-org/auditor_review](https://hf.co/datasets/demo-org/auditor_review) |  |
| 250 | [medmcqa](src/tasksource/tasks.py#L1007) | MultipleChoice | [openlifescienceai/medmcqa](https://hf.co/datasets/openlifescienceai/medmcqa) |  |
| 251 | [Dynasent_Disagreement](src/tasksource/tasks.py#L1024) | Classification | [RuyuanWan/Dynasent_Disagreement](https://hf.co/datasets/RuyuanWan/Dynasent_Disagreement) | ✓ |
| 252 | [Politeness_Disagreement](src/tasksource/tasks.py#L1026) | Classification | [RuyuanWan/Politeness_Disagreement](https://hf.co/datasets/RuyuanWan/Politeness_Disagreement) | ✓ |
| 253 | [SBIC_Disagreement](src/tasksource/tasks.py#L1028) | Classification | [RuyuanWan/SBIC_Disagreement](https://hf.co/datasets/RuyuanWan/SBIC_Disagreement) | ✓ |
| 254 | [SChem_Disagreement](src/tasksource/tasks.py#L1030) | Classification | [RuyuanWan/SChem_Disagreement](https://hf.co/datasets/RuyuanWan/SChem_Disagreement) | ✓ |
| 255 | [Dilemmas_Disagreement](src/tasksource/tasks.py#L1032) | Classification | [RuyuanWan/Dilemmas_Disagreement](https://hf.co/datasets/RuyuanWan/Dilemmas_Disagreement) | ✓ |
| 256 | [logiqa](src/tasksource/tasks.py#L1035) | MultipleChoice | [fireworks-ai/logiqa](https://hf.co/datasets/fireworks-ai/logiqa) |  |
| 257 | [wiki_qa](src/tasksource/tasks.py#L1044) | Classification | [microsoft/wiki_qa](https://hf.co/datasets/microsoft/wiki_qa) | ✓ |
| 258 | [cycic_classification](src/tasksource/tasks.py#L1046) | Classification | [tasksource/cycic_classification](https://hf.co/datasets/tasksource/cycic_classification) |  |
| 259 | [cycic_multiplechoice](src/tasksource/tasks.py#L1048) | MultipleChoice | [tasksource/cycic_multiplechoice](https://hf.co/datasets/tasksource/cycic_multiplechoice) |  |
| 260 | [sts-companion](src/tasksource/tasks.py#L1052) | Classification | [tasksource/sts-companion](https://hf.co/datasets/tasksource/sts-companion) |  |
| 261 | [commonsense_qa_2.0](src/tasksource/tasks.py#L1055) | Classification | [tasksource/commonsense_qa_2.0](https://hf.co/datasets/tasksource/commonsense_qa_2.0) |  |
| 262 | [lingnli](src/tasksource/tasks.py#L1058) | Classification | [tasksource/lingnli](https://hf.co/datasets/tasksource/lingnli) |  |
| 263 | [monotonicity-entailment](src/tasksource/tasks.py#L1060) | Classification | [tasksource/monotonicity-entailment](https://hf.co/datasets/tasksource/monotonicity-entailment) |  |
| 264 | [arct](src/tasksource/tasks.py#L1063) | MultipleChoice | [tasksource/arct](https://hf.co/datasets/tasksource/arct) |  |
| 265 | [scinli](src/tasksource/tasks.py#L1066) | Classification | [tasksource/scinli](https://hf.co/datasets/tasksource/scinli) |  |
| 266 | [naturallogic](src/tasksource/tasks.py#L1070) | Classification | [tasksource/naturallogic](https://hf.co/datasets/tasksource/naturallogic) |  |
| 267 | [onestop_qa](src/tasksource/tasks.py#L1072) | MultipleChoice | [malmaud/onestop_qa](https://hf.co/datasets/malmaud/onestop_qa) |  |
| 268 | [moral_stories/full](src/tasksource/tasks.py#L1075) | MultipleChoice | [LabHC/moral_stories](https://hf.co/datasets/LabHC/moral_stories) |  |
| 269 | [dynahate](src/tasksource/tasks.py#L1079) | Classification | [tasksource/dynahate](https://hf.co/datasets/tasksource/dynahate) |  |
| 270 | [syntactic-augmentation-nli](src/tasksource/tasks.py#L1081) | Classification | [tasksource/syntactic-augmentation-nli](https://hf.co/datasets/tasksource/syntactic-augmentation-nli) |  |
| 271 | [autotnli](src/tasksource/tasks.py#L1083) | Classification | [tasksource/autotnli](https://hf.co/datasets/tasksource/autotnli) |  |
| 272 | [CONDAQA](src/tasksource/tasks.py#L1085) | Classification | [lasha-nlp/CONDAQA](https://hf.co/datasets/lasha-nlp/CONDAQA) |  |
| 273 | [webgpt_comparisons](src/tasksource/tasks.py#L1095) | MultipleChoice | [heegyu/webgpt_comparisons_ko](https://hf.co/datasets/heegyu/webgpt_comparisons_ko) | ✓ |
| 274 | [synthetic-instruct-gptj-pairwise](src/tasksource/tasks.py#L1103) | MultipleChoice | [Dahoas/synthetic-instruct-gptj-pairwise](https://hf.co/datasets/Dahoas/synthetic-instruct-gptj-pairwise) | ✓ |
| 275 | [scruples](src/tasksource/tasks.py#L1106) | Classification | [tasksource/scruples](https://hf.co/datasets/tasksource/scruples) | ✓ |
| 276 | [defeasible-nli/snli](src/tasksource/tasks.py#L1111) | Classification | [tasksource/defeasible-nli](https://hf.co/datasets/tasksource/defeasible-nli) |  |
| 277 | [defeasible-nli/atomic](src/tasksource/tasks.py#L1111) | Classification | [tasksource/defeasible-nli](https://hf.co/datasets/tasksource/defeasible-nli) |  |
| 278 | [defeasible-nli/social](src/tasksource/tasks.py#L1114) | Classification | [tasksource/defeasible-nli](https://hf.co/datasets/tasksource/defeasible-nli) |  |
| 279 | [help-nli](src/tasksource/tasks.py#L1117) | Classification | [tasksource/help-nli](https://hf.co/datasets/tasksource/help-nli) |  |
| 280 | [nli-veridicality-transitivity](src/tasksource/tasks.py#L1120) | Classification | [tasksource/nli-veridicality-transitivity](https://hf.co/datasets/tasksource/nli-veridicality-transitivity) |  |
| 281 | [lonli](src/tasksource/tasks.py#L1123) | Classification | [tasksource/lonli](https://hf.co/datasets/tasksource/lonli) |  |
| 282 | [dadc-limit-nli](src/tasksource/tasks.py#L1126) | Classification | [tasksource/dadc-limit-nli](https://hf.co/datasets/tasksource/dadc-limit-nli) |  |
| 283 | [FLUTE](src/tasksource/tasks.py#L1129) | Classification | [ColumbiaNLP/FLUTE](https://hf.co/datasets/ColumbiaNLP/FLUTE) |  |
| 284 | [strategy-qa](src/tasksource/tasks.py#L1132) | Classification | [tasksource/strategy-qa](https://hf.co/datasets/tasksource/strategy-qa) |  |
| 285 | [summarize_from_feedback/comparisons](src/tasksource/tasks.py#L1135) | MultipleChoice | [vwxyzjn/summarize_from_feedback_oai_preprocessing](https://hf.co/datasets/vwxyzjn/summarize_from_feedback_oai_preprocessing) | ✓ |
| 286 | [folio](src/tasksource/tasks.py#L1143) | Classification | [tasksource/folio](https://hf.co/datasets/tasksource/folio) |  |
| 287 | [tomi-nli](src/tasksource/tasks.py#L1170) | Classification | [tasksource/tomi-nli](https://hf.co/datasets/tasksource/tomi-nli) |  |
| 288 | [avicenna](src/tasksource/tasks.py#L1173) | Classification | [tasksource/avicenna](https://hf.co/datasets/tasksource/avicenna) | ✓ |
| 289 | [SHP](src/tasksource/tasks.py#L1176) | MultipleChoice | [stanfordnlp/SHP](https://hf.co/datasets/stanfordnlp/SHP) | ✓ |
| 290 | [MedQA-USMLE-4-options-hf](src/tasksource/tasks.py#L1184) | MultipleChoice | [GBaker/MedQA-USMLE-4-options-hf](https://hf.co/datasets/GBaker/MedQA-USMLE-4-options-hf) |  |
| 291 | [wikimedqa/medwiki](src/tasksource/tasks.py#L1187) | MultipleChoice | [sileod/wikimedqa](https://hf.co/datasets/sileod/wikimedqa) |  |
| 292 | [cicero](src/tasksource/tasks.py#L1196) | MultipleChoice | [declare-lab/cicero](https://hf.co/datasets/declare-lab/cicero) |  |
| 293 | [CREAK](src/tasksource/tasks.py#L1200) | Classification | [amydeng2000/CREAK](https://hf.co/datasets/amydeng2000/CREAK) |  |
| 294 | [mutual](src/tasksource/tasks.py#L1203) | MultipleChoice | [tasksource/mutual](https://hf.co/datasets/tasksource/mutual) |  |
| 295 | [implicatures](src/tasksource/tasks.py#L1208) | MultipleChoice | [tasksource/implicatures](https://hf.co/datasets/tasksource/implicatures) |  |
| 296 | [race/middle](src/tasksource/tasks.py#L1213) | MultipleChoice | [ehovy/race](https://hf.co/datasets/ehovy/race) |  |
| 297 | [race/high](src/tasksource/tasks.py#L1213) | MultipleChoice | [ehovy/race](https://hf.co/datasets/ehovy/race) |  |
| 298 | [race-c](src/tasksource/tasks.py#L1217) | MultipleChoice | [tasksource/race-c](https://hf.co/datasets/tasksource/race-c) |  |
| 299 | [spartqa-yn](src/tasksource/tasks.py#L1220) | Classification | [tasksource/spartqa-yn](https://hf.co/datasets/tasksource/spartqa-yn) |  |
| 300 | [spartqa-mchoice](src/tasksource/tasks.py#L1223) | MultipleChoice | [tasksource/spartqa-mchoice](https://hf.co/datasets/tasksource/spartqa-mchoice) |  |
| 301 | [temporal-nli](src/tasksource/tasks.py#L1226) | Classification | [tasksource/temporal-nli](https://hf.co/datasets/tasksource/temporal-nli) |  |
| 302 | [riddle_sense](src/tasksource/tasks.py#L1229) | MultipleChoice | [jeggers/riddle_sense](https://hf.co/datasets/jeggers/riddle_sense) |  |
| 303 | [clcd-english](src/tasksource/tasks.py#L1234) | Classification | [tasksource/clcd-english](https://hf.co/datasets/tasksource/clcd-english) |  |
| 304 | [twentyquestions](src/tasksource/tasks.py#L1246) | Classification | [tasksource/twentyquestions](https://hf.co/datasets/tasksource/twentyquestions) |  |
| 305 | [reclor](src/tasksource/tasks.py#L1251) | MultipleChoice | [tasksource/reclor](https://hf.co/datasets/tasksource/reclor) |  |
| 306 | [counterfactually-augmented-imdb](src/tasksource/tasks.py#L1254) | Classification | [tasksource/counterfactually-augmented-imdb](https://hf.co/datasets/tasksource/counterfactually-augmented-imdb) |  |
| 307 | [counterfactually-augmented-snli](src/tasksource/tasks.py#L1257) | Classification | [tasksource/counterfactually-augmented-snli](https://hf.co/datasets/tasksource/counterfactually-augmented-snli) |  |
| 308 | [cnli](src/tasksource/tasks.py#L1260) | Classification | [tasksource/cnli](https://hf.co/datasets/tasksource/cnli) |  |
| 309 | [boolq-natural-perturbations](src/tasksource/tasks.py#L1263) | Classification | [tasksource/boolq-natural-perturbations](https://hf.co/datasets/tasksource/boolq-natural-perturbations) |  |
| 310 | [acceptability-prediction](src/tasksource/tasks.py#L1268) | Classification | [tasksource/acceptability-prediction](https://hf.co/datasets/tasksource/acceptability-prediction) | ✓ |
| 311 | [equate](src/tasksource/tasks.py#L1294) | Classification | [tasksource/equate](https://hf.co/datasets/tasksource/equate) |  |
| 312 | [ScienceQA_text_only](src/tasksource/tasks.py#L1297) | MultipleChoice | [tasksource/ScienceQA_text_only](https://hf.co/datasets/tasksource/ScienceQA_text_only) |  |
| 313 | [ekar_english](src/tasksource/tasks.py#L1300) | MultipleChoice | [Jiangjie/ekar_english](https://hf.co/datasets/Jiangjie/ekar_english) | ✓ |
| 314 | [implicit-hate-stg1](src/tasksource/tasks.py#L1304) | Classification | [tasksource/implicit-hate-stg1](https://hf.co/datasets/tasksource/implicit-hate-stg1) | ✓ |
| 315 | [chaos-mnli-ambiguity](src/tasksource/tasks.py#L1308) | Classification | [tasksource/chaos-mnli-ambiguity](https://hf.co/datasets/tasksource/chaos-mnli-ambiguity) | ✓ |
| 316 | [headline_cause/en_simple](src/tasksource/tasks.py#L1312) | Classification | [IlyaGusev/headline_cause](https://hf.co/datasets/IlyaGusev/headline_cause) (`json` loader) |  |
| 317 | [logiqa-2.0-nli](src/tasksource/tasks.py#L1317) | Classification | [tasksource/logiqa-2.0-nli](https://hf.co/datasets/tasksource/logiqa-2.0-nli) |  |
| 318 | [oasst2_dense_flat/quality](src/tasksource/tasks.py#L1322) | Classification | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) | ✓ |
| 319 | [oasst2_dense_flat/toxicity](src/tasksource/tasks.py#L1324) | Classification | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) | ✓ |
| 320 | [oasst2_dense_flat/helpfulness](src/tasksource/tasks.py#L1326) | Classification | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) | ✓ |
| 321 | [mindgames](src/tasksource/tasks.py#L1329) | Classification | [sileod/mindgames](https://hf.co/datasets/sileod/mindgames) |  |
| 322 | [universal_dependencies/en_gum/deprel](src/tasksource/tasks.py#L1343) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) | ✓ |
| 323 | [universal_dependencies/en_lines/deprel](src/tasksource/tasks.py#L1343) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) | ✓ |
| 324 | [universal_dependencies/en_partut/deprel](src/tasksource/tasks.py#L1343) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) | ✓ |
| 325 | [universal_dependencies/en_ewt/deprel](src/tasksource/tasks.py#L1343) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) | ✓ |
| 326 | [ambient](src/tasksource/tasks.py#L1350) | Classification | [tasksource/ambient](https://hf.co/datasets/tasksource/ambient) | ✓ |
| 327 | [path-naturalness-prediction](src/tasksource/tasks.py#L1355) | MultipleChoice | [tasksource/path-naturalness-prediction](https://hf.co/datasets/tasksource/path-naturalness-prediction) | ✓ |
| 328 | [civil_comments/toxicity](src/tasksource/tasks.py#L1365) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 329 | [civil_comments/severe_toxicity](src/tasksource/tasks.py#L1366) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 330 | [civil_comments/obscene](src/tasksource/tasks.py#L1367) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 331 | [civil_comments/threat](src/tasksource/tasks.py#L1368) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 332 | [civil_comments/insult](src/tasksource/tasks.py#L1369) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 333 | [civil_comments/identity_attack](src/tasksource/tasks.py#L1370) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 334 | [civil_comments/sexual_explicit](src/tasksource/tasks.py#L1371) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 335 | [cloth](src/tasksource/tasks.py#L1373) | MultipleChoice | [AndyChiang/cloth](https://hf.co/datasets/AndyChiang/cloth) |  |
| 336 | [dgen](src/tasksource/tasks.py#L1374) | MultipleChoice | [AndyChiang/dgen](https://hf.co/datasets/AndyChiang/dgen) |  |
| 337 | [I2D2](src/tasksource/tasks.py#L1376) | Classification | [tasksource/I2D2](https://hf.co/datasets/tasksource/I2D2) | ✓ |
| 338 | [args_me](src/tasksource/tasks.py#L1381) | Classification | [webis/args_me](https://hf.co/datasets/webis/args_me) | ✓ |
| 339 | [Touche23-ValueEval](src/tasksource/tasks.py#L1386) | Classification | [webis/Touche23-ValueEval](https://hf.co/datasets/webis/Touche23-ValueEval) (`csv` loader) | ✓ |
| 340 | [starcon](src/tasksource/tasks.py#L1395) | Classification | [tasksource/starcon](https://hf.co/datasets/tasksource/starcon) | ✓ |
| 341 | [banking77](src/tasksource/tasks.py#L1400) | Classification | [legacy-datasets/banking77](https://hf.co/datasets/legacy-datasets/banking77) |  |
| 342 | [it-support-tickets](src/tasksource/tasks.py#L1402) | Classification | [tasksource/it-support-tickets](https://hf.co/datasets/tasksource/it-support-tickets) |  |
| 343 | [ConTRoL-nli](src/tasksource/tasks.py#L1406) | Classification | [tasksource/ConTRoL-nli](https://hf.co/datasets/tasksource/ConTRoL-nli) |  |
| 344 | [tracie](src/tasksource/tasks.py#L1408) | Classification | [tasksource/tracie](https://hf.co/datasets/tasksource/tracie) | ✓ |
| 345 | [sherliic](src/tasksource/tasks.py#L1410) | Classification | [tasksource/sherliic](https://hf.co/datasets/tasksource/sherliic) |  |
| 346 | [sen-making/1](src/tasksource/tasks.py#L1412) | MultipleChoice | [tasksource/sen-making](https://hf.co/datasets/tasksource/sen-making) | ✓ |
| 347 | [sen-making/2](src/tasksource/tasks.py#L1416) | MultipleChoice | [tasksource/sen-making](https://hf.co/datasets/tasksource/sen-making) | ✓ |
| 348 | [winowhy](src/tasksource/tasks.py#L1419) | Classification | [tasksource/winowhy](https://hf.co/datasets/tasksource/winowhy) | ✓ |
| 349 | [robustLR](src/tasksource/tasks.py#L1423) | Classification | [tasksource/robustLR](https://hf.co/datasets/tasksource/robustLR) |  |
| 350 | [clutrr](src/tasksource/tasks.py#L1425) | Classification | [tasksource/clutrr](https://hf.co/datasets/tasksource/clutrr) |  |
| 351 | [logical-fallacy](src/tasksource/tasks.py#L1427) | Classification | [tasksource/logical-fallacy](https://hf.co/datasets/tasksource/logical-fallacy) |  |
| 352 | [parade](src/tasksource/tasks.py#L1429) | Classification | [tasksource/parade](https://hf.co/datasets/tasksource/parade) |  |
| 353 | [subjectivity](src/tasksource/tasks.py#L1432) | Classification | [tasksource/subjectivity](https://hf.co/datasets/tasksource/subjectivity) | ✓ |
| 354 | [MOH](src/tasksource/tasks.py#L1438) | Classification | [tasksource/MOH](https://hf.co/datasets/tasksource/MOH) | ✓ |
| 355 | [VUAC](src/tasksource/tasks.py#L1439) | Classification | [tasksource/VUAC](https://hf.co/datasets/tasksource/VUAC) | ✓ |
| 356 | [TroFi](src/tasksource/tasks.py#L1440) | Classification | [tasksource/TroFi](https://hf.co/datasets/tasksource/TroFi) (`parquet` loader) | ✓ |
| 357 | [sharc](src/tasksource/tasks.py#L1448) | Classification | [tasksource/sharc](https://hf.co/datasets/tasksource/sharc) |  |
| 358 | [conceptrules_v2](src/tasksource/tasks.py#L1452) | Classification | [tasksource/conceptrules_v2](https://hf.co/datasets/tasksource/conceptrules_v2) | ✓ |
| 359 | [disrpt/eng.dep.scidtb.rels](src/tasksource/tasks.py#L1454) | Classification | [multilingual-discourse-hub/disrpt](https://hf.co/datasets/multilingual-discourse-hub/disrpt) | ✓ |
| 360 | [conll2000](src/tasksource/tasks.py#L1457) | TokenClassification | [eriktks/conll2000](https://hf.co/datasets/eriktks/conll2000) |  |
| 361 | [few-nerd/supervised](src/tasksource/tasks.py#L1460) | TokenClassification | [DFKI-SLT/few-nerd](https://hf.co/datasets/DFKI-SLT/few-nerd) |  |
| 362 | [finer-139](src/tasksource/tasks.py#L1461) | TokenClassification | [nlpaueb/finer-139](https://hf.co/datasets/nlpaueb/finer-139) |  |
| 363 | [zero-shot-label-nli](src/tasksource/tasks.py#L1464) | Classification | [tasksource/zero-shot-label-nli](https://hf.co/datasets/tasksource/zero-shot-label-nli) | ✓ |
| 364 | [com2sense](src/tasksource/tasks.py#L1466) | Classification | [tasksource/com2sense](https://hf.co/datasets/tasksource/com2sense) |  |
| 365 | [scone](src/tasksource/tasks.py#L1468) | Classification | [tasksource/scone](https://hf.co/datasets/tasksource/scone) |  |
| 366 | [winodict](src/tasksource/tasks.py#L1470) | MultipleChoice | [tasksource/winodict](https://hf.co/datasets/tasksource/winodict) |  |
| 367 | [fool-me-twice](src/tasksource/tasks.py#L1472) | Classification | [tasksource/fool-me-twice](https://hf.co/datasets/tasksource/fool-me-twice) |  |
| 368 | [monli](src/tasksource/tasks.py#L1476) | Classification | [tasksource/monli](https://hf.co/datasets/tasksource/monli) |  |
| 369 | [corr2cause](src/tasksource/tasks.py#L1478) | Classification | [tasksource/corr2cause](https://hf.co/datasets/tasksource/corr2cause) |  |
| 370 | [lsat_qa/all](src/tasksource/tasks.py#L1480) | MultipleChoice | [lighteval/lsat_qa](https://hf.co/datasets/lighteval/lsat_qa) |  |
| 371 | [apt](src/tasksource/tasks.py#L1482) | Classification | [tasksource/apt](https://hf.co/datasets/tasksource/apt) |  |
| 372 | [twitter-financial-news-sentiment](src/tasksource/tasks.py#L1485) | Classification | [zeroshot/twitter-financial-news-sentiment](https://hf.co/datasets/zeroshot/twitter-financial-news-sentiment) |  |
| 373 | [icl-symbol-tuning-instruct](src/tasksource/tasks.py#L1492) | Classification | [tasksource/icl-symbol-tuning-instruct](https://hf.co/datasets/tasksource/icl-symbol-tuning-instruct) | ✓ |
| 374 | [SpaceNLI](src/tasksource/tasks.py#L1498) | Classification | [tasksource/SpaceNLI](https://hf.co/datasets/tasksource/SpaceNLI) |  |
| 375 | [propsegment/nli](src/tasksource/tasks.py#L1500) | Classification | [sihaochen/propsegment](https://hf.co/datasets/sihaochen/propsegment) (`json` loader) | ✓ |
| 376 | [HatemojiBuild](src/tasksource/tasks.py#L1510) | Classification | [HannahRoseKirk/HatemojiBuild](https://hf.co/datasets/HannahRoseKirk/HatemojiBuild) |  |
| 377 | [regset](src/tasksource/tasks.py#L1513) | Classification | [tasksource/regset](https://hf.co/datasets/tasksource/regset) | ✓ |
| 378 | [esci](src/tasksource/tasks.py#L1519) | Classification | [tasksource/esci](https://hf.co/datasets/tasksource/esci) |  |
| 379 | [chatbot_arena_conversations](src/tasksource/tasks.py#L1538) | MultipleChoice | [lmsys/chatbot_arena_conversations](https://hf.co/datasets/lmsys/chatbot_arena_conversations) | ✓ |
| 380 | [dnd_style_intents](src/tasksource/tasks.py#L1544) | Classification | [neurae/dnd_style_intents](https://hf.co/datasets/neurae/dnd_style_intents) |  |
| 381 | [FLD.v2/default](src/tasksource/tasks.py#L1547) | Classification | [hitachi-nlp/FLD.v2](https://hf.co/datasets/hitachi-nlp/FLD.v2) | ✓ |
| 382 | [FLD.v2/star](src/tasksource/tasks.py#L1551) | Classification | [hitachi-nlp/FLD.v2](https://hf.co/datasets/hitachi-nlp/FLD.v2) | ✓ |
| 383 | [SDOH-NLI](src/tasksource/tasks.py#L1555) | Classification | [tasksource/SDOH-NLI](https://hf.co/datasets/tasksource/SDOH-NLI) |  |
| 384 | [scifact_entailment](src/tasksource/tasks.py#L1558) | Classification | [tasksource/scifact_entailment](https://hf.co/datasets/tasksource/scifact_entailment) |  |
| 385 | [feasibilityQA](src/tasksource/tasks.py#L1562) | Classification | [tasksource/feasibilityQA](https://hf.co/datasets/tasksource/feasibilityQA) |  |
| 386 | [simple_pair](src/tasksource/tasks.py#L1565) | Classification | [tasksource/simple_pair](https://hf.co/datasets/tasksource/simple_pair) |  |
| 387 | [AdjectiveScaleProbe-nli](src/tasksource/tasks.py#L1566) | Classification | [tasksource/AdjectiveScaleProbe-nli](https://hf.co/datasets/tasksource/AdjectiveScaleProbe-nli) |  |
| 388 | [resnli](src/tasksource/tasks.py#L1567) | Classification | [tasksource/resnli](https://hf.co/datasets/tasksource/resnli) |  |
| 389 | [SpaRTUN](src/tasksource/tasks.py#L1569) | MultipleChoice | [tasksource/SpaRTUN](https://hf.co/datasets/tasksource/SpaRTUN) |  |
| 390 | [ReSQ](src/tasksource/tasks.py#L1574) | MultipleChoice | [tasksource/ReSQ](https://hf.co/datasets/tasksource/ReSQ) |  |
| 391 | [semantic_fragments_nli](src/tasksource/tasks.py#L1579) | Classification | [tasksource/semantic_fragments_nli](https://hf.co/datasets/tasksource/semantic_fragments_nli) |  |
| 392 | [dataset_train_nli](src/tasksource/tasks.py#L1582) | Classification | [MoritzLaurer/dataset_train_nli](https://hf.co/datasets/MoritzLaurer/dataset_train_nli) |  |
| 393 | [stepgame](src/tasksource/tasks.py#L1587) | Classification | [tasksource/stepgame](https://hf.co/datasets/tasksource/stepgame) |  |
| 394 | [nlgraph](src/tasksource/tasks.py#L1595) | Classification | [tasksource/nlgraph](https://hf.co/datasets/tasksource/nlgraph) |  |
| 395 | [oasst2_pairwise_rlhf_reward](src/tasksource/tasks.py#L1599) | MultipleChoice | [tasksource/oasst2_pairwise_rlhf_reward](https://hf.co/datasets/tasksource/oasst2_pairwise_rlhf_reward) | ✓ |
| 396 | [hh-rlhf/helpful-rejection-sampled](src/tasksource/tasks.py#L1610) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| 397 | [hh-rlhf/helpful-online](src/tasksource/tasks.py#L1610) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| 398 | [hh-rlhf/helpful-base](src/tasksource/tasks.py#L1610) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| 399 | [hh-rlhf/harmless-base](src/tasksource/tasks.py#L1614) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| 400 | [ruletaker](src/tasksource/tasks.py#L1618) | Classification | [tasksource/ruletaker](https://hf.co/datasets/tasksource/ruletaker) | ✓ |
| 401 | [PARARULE-Plus](src/tasksource/tasks.py#L1622) | Classification | [qbao775/PARARULE-Plus](https://hf.co/datasets/qbao775/PARARULE-Plus) | ✓ |
| 402 | [proofwriter](src/tasksource/tasks.py#L1626) | Classification | [tasksource/proofwriter](https://hf.co/datasets/tasksource/proofwriter) |  |
| 403 | [logical-entailment](src/tasksource/tasks.py#L1629) | Classification | [tasksource/logical-entailment](https://hf.co/datasets/tasksource/logical-entailment) | ✓ |
| 404 | [nope](src/tasksource/tasks.py#L1632) | Classification | [tasksource/nope](https://hf.co/datasets/tasksource/nope) |  |
| 405 | [LogicNLI](src/tasksource/tasks.py#L1636) | Classification | [tasksource/LogicNLI](https://hf.co/datasets/tasksource/LogicNLI) | ✓ |
| 406 | [contract-nli/contractnli_a/seg](src/tasksource/tasks.py#L1639) | Classification | [tasksource/contract-nli](https://hf.co/datasets/tasksource/contract-nli) |  |
| 407 | [contract-nli/contractnli_b/full](src/tasksource/tasks.py#L1641) | Classification | [tasksource/contract-nli](https://hf.co/datasets/tasksource/contract-nli) |  |
| 408 | [nli4ct_semeval2024](src/tasksource/tasks.py#L1643) | Classification | [AshtonIsNotHere/nli4ct_semeval2024](https://hf.co/datasets/AshtonIsNotHere/nli4ct_semeval2024) |  |
| 409 | [lsat-ar](src/tasksource/tasks.py#L1646) | MultipleChoice | [tasksource/lsat-ar](https://hf.co/datasets/tasksource/lsat-ar) |  |
| 410 | [lsat-rc](src/tasksource/tasks.py#L1651) | MultipleChoice | [tasksource/lsat-rc](https://hf.co/datasets/tasksource/lsat-rc) |  |
| 411 | [biosift-nli](src/tasksource/tasks.py#L1656) | Classification | [AshtonIsNotHere/biosift-nli](https://hf.co/datasets/AshtonIsNotHere/biosift-nli) |  |
| 412 | [brainteasers/WP](src/tasksource/tasks.py#L1660) | MultipleChoice | [tasksource/brainteasers](https://hf.co/datasets/tasksource/brainteasers) |  |
| 413 | [brainteasers/SP](src/tasksource/tasks.py#L1660) | MultipleChoice | [tasksource/brainteasers](https://hf.co/datasets/tasksource/brainteasers) |  |
| 414 | [toxigen-data/annotated](src/tasksource/tasks.py#L1666) | Classification | [skg/toxigen-data](https://hf.co/datasets/skg/toxigen-data) |  |
| 415 | [persuasion](src/tasksource/tasks.py#L1677) | Classification | [Anthropic/persuasion](https://hf.co/datasets/Anthropic/persuasion) |  |
| 416 | [AmbigNQ-clarifying-question](src/tasksource/tasks.py#L1683) | Classification | [erbacher/AmbigNQ-clarifying-question](https://hf.co/datasets/erbacher/AmbigNQ-clarifying-question) | ✓ |
| 417 | [SIGA-nli](src/tasksource/tasks.py#L1688) | Classification | [tasksource/SIGA-nli](https://hf.co/datasets/tasksource/SIGA-nli) | ✓ |
| 418 | [FOL-nli](src/tasksource/tasks.py#L1690) | Classification | [unigram/FOL-nli](https://hf.co/datasets/unigram/FOL-nli) |  |
| 419 | [goal-step-wikihow/goal](src/tasksource/tasks.py#L1692) | MultipleChoice | [tasksource/goal-step-wikihow](https://hf.co/datasets/tasksource/goal-step-wikihow) |  |
| 420 | [goal-step-wikihow/step](src/tasksource/tasks.py#L1695) | MultipleChoice | [tasksource/goal-step-wikihow](https://hf.co/datasets/tasksource/goal-step-wikihow) |  |
| 421 | [goal-step-wikihow/order](src/tasksource/tasks.py#L1698) | MultipleChoice | [tasksource/goal-step-wikihow](https://hf.co/datasets/tasksource/goal-step-wikihow) | ✓ |
| 422 | [PARADISE](src/tasksource/tasks.py#L1702) | MultipleChoice | [GGLab/PARADISE](https://hf.co/datasets/GGLab/PARADISE) |  |
| 423 | [doc-nli](src/tasksource/tasks.py#L1705) | Classification | [tasksource/doc-nli](https://hf.co/datasets/tasksource/doc-nli) |  |
| 424 | [mctest-nli](src/tasksource/tasks.py#L1707) | Classification | [tasksource/mctest-nli](https://hf.co/datasets/tasksource/mctest-nli) |  |
| 425 | [patent-phrase-similarity](src/tasksource/tasks.py#L1709) | Classification | [tasksource/patent-phrase-similarity](https://hf.co/datasets/tasksource/patent-phrase-similarity) | ✓ |
| 426 | [natural-language-satisfiability](src/tasksource/tasks.py#L1712) | Classification | [tasksource/natural-language-satisfiability](https://hf.co/datasets/tasksource/natural-language-satisfiability) |  |
| 427 | [idioms-nli](src/tasksource/tasks.py#L1714) | Classification | [tasksource/idioms-nli](https://hf.co/datasets/tasksource/idioms-nli) |  |
| 428 | [lifecycle-entailment](src/tasksource/tasks.py#L1716) | Classification | [tasksource/lifecycle-entailment](https://hf.co/datasets/tasksource/lifecycle-entailment) |  |
| 429 | [safe-guard-prompt-injection](src/tasksource/tasks.py#L1723) | Classification | [xTRam1/safe-guard-prompt-injection](https://hf.co/datasets/xTRam1/safe-guard-prompt-injection) | ✓ |
| 430 | [prompt-injections](src/tasksource/tasks.py#L1729) | Classification | [deepset/prompt-injections](https://hf.co/datasets/deepset/prompt-injections) | ✓ |
| 431 | [prompt-injection-dataset](src/tasksource/tasks.py#L1735) | Classification | [S-Labs/prompt-injection-dataset](https://hf.co/datasets/S-Labs/prompt-injection-dataset) | ✓ |
| 432 | [Prompt-injection-dataset/full](src/tasksource/tasks.py#L1741) | Classification | [neuralchemy/Prompt-injection-dataset](https://hf.co/datasets/neuralchemy/Prompt-injection-dataset) | ✓ |
| 433 | [PromptShield](src/tasksource/tasks.py#L1747) | Classification | [hendzh/PromptShield](https://hf.co/datasets/hendzh/PromptShield) | ✓ |
| 434 | [shell-safety-v2](src/tasksource/tasks.py#L1753) | Classification | [tomngdev/shell-safety-v2](https://hf.co/datasets/tomngdev/shell-safety-v2) | ✓ |
| 435 | [agent_action_safety](src/tasksource/tasks.py#L1758) | Classification | [karanxa/agent-action-safety-dataset](https://hf.co/datasets/karanxa/agent-action-safety-dataset) (`json` loader) | ✓ |
| 436 | [ShellRisk-Bench](src/tasksource/tasks.py#L1773) | Classification | [kontext-security/ShellRisk-Bench](https://hf.co/datasets/kontext-security/ShellRisk-Bench) | ✓ |
| 437 | [wildguardmix-cleaned/prompt_harm](src/tasksource/tasks.py#L1798) | Classification | [bogdanminko/wildguardmix-cleaned](https://hf.co/datasets/bogdanminko/wildguardmix-cleaned) | ✓ |
| 438 | [wildguardmix-cleaned/response_harm](src/tasksource/tasks.py#L1804) | Classification | [bogdanminko/wildguardmix-cleaned](https://hf.co/datasets/bogdanminko/wildguardmix-cleaned) | ✓ |
| 439 | [wildguardmix-cleaned/response_refusal](src/tasksource/tasks.py#L1809) | Classification | [bogdanminko/wildguardmix-cleaned](https://hf.co/datasets/bogdanminko/wildguardmix-cleaned) | ✓ |
| 440 | [BeaverTails](src/tasksource/tasks.py#L1825) | Classification | [PKU-Alignment/BeaverTails](https://hf.co/datasets/PKU-Alignment/BeaverTails) | ✓ |
| 441 | [privacy-200k-Mistral-Large-3](src/tasksource/tasks.py#L1841) | Classification | [gabrielloiseau/privacy-200k-Mistral-Large-3](https://hf.co/datasets/gabrielloiseau/privacy-200k-Mistral-Large-3) | ✓ |
| 442 | [toxic-chat/toxicchat0124/toxicity](src/tasksource/tasks.py#L1850) | Classification | [lmsys/toxic-chat](https://hf.co/datasets/lmsys/toxic-chat) | ✓ |
| 443 | [toxic-chat/toxicchat0124/jailbreaking](src/tasksource/tasks.py#L1855) | Classification | [lmsys/toxic-chat](https://hf.co/datasets/lmsys/toxic-chat) | ✓ |
| 444 | [clinc_oos/plus](src/tasksource/tasks.py#L1861) | Classification | [clinc/clinc_oos](https://hf.co/datasets/clinc/clinc_oos) |  |
| 445 | [IntentGrasp/all](src/tasksource/tasks.py#L1879) | MultipleChoice | [yuweiyin/IntentGrasp](https://hf.co/datasets/yuweiyin/IntentGrasp) |  |
| 446 | [few_rel/default](src/tasksource/tasks.py#L1917) | Classification | [tasksource/few_rel](https://hf.co/datasets/tasksource/few_rel) |  |
| 447 | [docred](src/tasksource/tasks.py#L1995) | Classification | [YufeiHFUT/DocRED_origin](https://hf.co/datasets/YufeiHFUT/DocRED_origin), [thunlp/docred](https://hf.co/datasets/thunlp/docred) (`json` loader) |  |
| 448 | [chemprot/chemprot_full_source](src/tasksource/tasks.py#L2030) | Classification | [bigbio/chemprot](https://hf.co/datasets/bigbio/chemprot) | ✓ |
| 449 | [PKU-SafeRLHF/helpfulness](src/tasksource/tasks.py#L2036) | MultipleChoice | [PKU-Alignment/PKU-SafeRLHF](https://hf.co/datasets/PKU-Alignment/PKU-SafeRLHF) | ✓ |
| 450 | [PKU-SafeRLHF/safety](src/tasksource/tasks.py#L2041) | MultipleChoice | [PKU-Alignment/PKU-SafeRLHF](https://hf.co/datasets/PKU-Alignment/PKU-SafeRLHF) | ✓ |
| 451 | [HelpSteer/helpfulness](src/tasksource/tasks.py#L2063) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 452 | [HelpSteer/correctness](src/tasksource/tasks.py#L2064) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 453 | [HelpSteer/coherence](src/tasksource/tasks.py#L2065) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 454 | [HelpSteer/complexity](src/tasksource/tasks.py#L2066) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 455 | [HelpSteer/verbosity](src/tasksource/tasks.py#L2067) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 456 | [HelpSteer2/helpfulness](src/tasksource/tasks.py#L2069) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 457 | [HelpSteer2/correctness](src/tasksource/tasks.py#L2070) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 458 | [HelpSteer2/coherence](src/tasksource/tasks.py#L2071) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 459 | [HelpSteer2/complexity](src/tasksource/tasks.py#L2072) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 460 | [HelpSteer2/verbosity](src/tasksource/tasks.py#L2073) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 461 | [HelpSteer3/preference](src/tasksource/tasks.py#L2078) | MultipleChoice | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | ✓ |
| 462 | [HelpSteer3/preference_strength](src/tasksource/tasks.py#L2091) | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | ✓ |
| 463 | [HelpSteer3/principle](src/tasksource/tasks.py#L2103) | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) |  |
| 464 | [HelpSteer3/edit_quality](src/tasksource/tasks.py#L2108) | MultipleChoice | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | ✓ |
| 465 | [HelpSteer3/feedback](src/tasksource/tasks.py#L2130) | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | ✓ |
| 466 | [MSciNLI](src/tasksource/tasks.py#L2136) | Classification | [sadat2307/MSciNLI](https://hf.co/datasets/sadat2307/MSciNLI) |  |
| 467 | [UltraFeedback-paired](src/tasksource/tasks.py#L2139) | MultipleChoice | [pushpdeep/UltraFeedback-paired](https://hf.co/datasets/pushpdeep/UltraFeedback-paired) | ✓ |
| 468 | [prm800k_dpo/solution](src/tasksource/tasks.py#L2143) | MultipleChoice | [tasksource/prm800k_dpo](https://hf.co/datasets/tasksource/prm800k_dpo) | ✓ |
| 469 | [prm800k_dpo/step](src/tasksource/tasks.py#L2146) | MultipleChoice | [tasksource/prm800k_dpo](https://hf.co/datasets/tasksource/prm800k_dpo) | ✓ |
| 470 | [AES2-essay-scoring](src/tasksource/tasks.py#L2150) | Classification | [tasksource/AES2-essay-scoring](https://hf.co/datasets/tasksource/AES2-essay-scoring) | ✓ |
| 471 | [argument-feedback](src/tasksource/tasks.py#L2154) | Classification | [tasksource/argument-feedback](https://hf.co/datasets/tasksource/argument-feedback) | ✓ |
| 472 | [english-grading/cohesion](src/tasksource/tasks.py#L2161) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 473 | [english-grading/syntax](src/tasksource/tasks.py#L2162) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 474 | [english-grading/vocabulary](src/tasksource/tasks.py#L2163) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 475 | [english-grading/phraseology](src/tasksource/tasks.py#L2164) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 476 | [english-grading/grammar](src/tasksource/tasks.py#L2165) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 477 | [english-grading/conventions](src/tasksource/tasks.py#L2166) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 478 | [wice](src/tasksource/tasks.py#L2168) | Classification | [tasksource/wice](https://hf.co/datasets/tasksource/wice) |  |
| 479 | [hover](src/tasksource/tasks.py#L2171) | Classification | [Dzeniks/hover](https://hf.co/datasets/Dzeniks/hover) |  |
| 480 | [hover-3way/nli](src/tasksource/tasks.py#L2175) | Classification | [Dzeniks/hover-3way](https://hf.co/datasets/Dzeniks/hover-3way) | ✓ |
| 481 | [tasksource_dpo_pairs](src/tasksource/tasks.py#L2178) | MultipleChoice | [tasksource/tasksource_dpo_pairs](https://hf.co/datasets/tasksource/tasksource_dpo_pairs) | ✓ |
| 482 | [seahorse_summarization_evaluation](src/tasksource/tasks.py#L2181) | Classification | [tasksource/seahorse_summarization_evaluation](https://hf.co/datasets/tasksource/seahorse_summarization_evaluation) |  |
| 483 | [missing-item-prediction/contrastive](src/tasksource/tasks.py#L2193) | Classification | [sileod/missing-item-prediction](https://hf.co/datasets/sileod/missing-item-prediction) | ✓ |
| 484 | [jigsaw_toxicity](src/tasksource/tasks.py#L2199) | Classification | [tasksource/jigsaw_toxicity](https://hf.co/datasets/tasksource/jigsaw_toxicity) |  |
| 485 | [Pol_NLI](src/tasksource/tasks.py#L2202) | Classification | [mlburnham/Pol_NLI](https://hf.co/datasets/mlburnham/Pol_NLI) |  |
| 486 | [synthetic-retrieval-NLI/binary](src/tasksource/tasks.py#L2205) | Classification | [tasksource/synthetic-retrieval-NLI](https://hf.co/datasets/tasksource/synthetic-retrieval-NLI) |  |
| 487 | [synthetic-retrieval-NLI/position](src/tasksource/tasks.py#L2205) | Classification | [tasksource/synthetic-retrieval-NLI](https://hf.co/datasets/tasksource/synthetic-retrieval-NLI) |  |
| 488 | [synthetic-retrieval-NLI/count](src/tasksource/tasks.py#L2205) | Classification | [tasksource/synthetic-retrieval-NLI](https://hf.co/datasets/tasksource/synthetic-retrieval-NLI) |  |
| 489 | [github-issue-similarity](src/tasksource/tasks.py#L2214) | Classification | [WhereIsAI/github-issue-similarity](https://hf.co/datasets/WhereIsAI/github-issue-similarity) |  |
| 490 | [webinstruct/mc](src/tasksource/tasks.py#L2419) | MultipleChoice | [tasksource/webinstruct](https://hf.co/datasets/tasksource/webinstruct) |  |
| 491 | [webinstruct/binary](src/tasksource/tasks.py#L2421) | Classification | [tasksource/webinstruct](https://hf.co/datasets/tasksource/webinstruct) |  |
| 492 | [weblinx/action](src/tasksource/tasks.py#L2485) | Classification | [McGill-NLP/WebLINX](https://hf.co/datasets/McGill-NLP/WebLINX) | ✓ |
| 493 | [weblinx/dom-element](src/tasksource/tasks.py#L2488) | MultipleChoice | [McGill-NLP/WebLINX](https://hf.co/datasets/McGill-NLP/WebLINX) | ✓ |
| 494 | [websrc/yesno](src/tasksource/tasks.py#L2494) | Classification | [tasksource/websrc](https://hf.co/datasets/tasksource/websrc) |  |
| 495 | [websrc/element](src/tasksource/tasks.py#L2496) | MultipleChoice | [tasksource/websrc](https://hf.co/datasets/tasksource/websrc) | ✓ |
| 496 | [mind2web/action](src/tasksource/tasks.py#L2503) | Classification | [tasksource/mind2web-dom](https://hf.co/datasets/tasksource/mind2web-dom) | ✓ |
| 497 | [mind2web/dom-element](src/tasksource/tasks.py#L2506) | MultipleChoice | [tasksource/mind2web-dom](https://hf.co/datasets/tasksource/mind2web-dom) | ✓ |
| 498 | [sentfin](src/tasksource/tasks.py#L2511) | Classification | [tasksource/sentfin](https://hf.co/datasets/tasksource/sentfin) | ✓ |
| 499 | [wands](src/tasksource/tasks.py#L2515) | Classification | [tasksource/wands](https://hf.co/datasets/tasksource/wands) | ✓ |
| 500 | [scirepeval/search](src/tasksource/tasks.py#L2519) | Classification | [tasksource/scirepeval-search](https://hf.co/datasets/tasksource/scirepeval-search) | ✓ |
| 501 | [maud/consideration](src/tasksource/tasks.py#L2571) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 502 | [maud/bringdown-timing](src/tasksource/tasks.py#L2575) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 503 | [maud/ordinary-course-consent](src/tasksource/tasks.py#L2579) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 504 | [maud/ordinary-course-efforts](src/tasksource/tasks.py#L2583) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 505 | [maud/antitrust-efforts](src/tasksource/tasks.py#L2587) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 506 | [maud/specific-performance](src/tasksource/tasks.py#L2591) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 507 | [ddi/relation](src/tasksource/tasks.py#L2596) | Classification | [OpenMed/DDI-Corpus-Processed](https://hf.co/datasets/OpenMed/DDI-Corpus-Processed) | ✓ |
| 508 | [patent-classification/abstract](src/tasksource/tasks.py#L2603) | Classification | [ccdv/patent-classification](https://hf.co/datasets/ccdv/patent-classification) | ✓ |
| 509 | [tabfact](src/tasksource/tasks.py#L2610) | Classification | [wenhu/tab_fact](https://hf.co/datasets/wenhu/tab_fact) | ✓ |
| 510 | [infotabs](src/tasksource/tasks.py#L2618) | Classification | [table-benchmark/infotabs](https://hf.co/datasets/table-benchmark/infotabs) | ✓ |
| 511 | [squad-v2/answerability](src/tasksource/tasks.py#L2630) | Classification | [rajpurkar/squad_v2](https://hf.co/datasets/rajpurkar/squad_v2) | ✓ |
| 512 | [cuad/answerability](src/tasksource/tasks.py#L2635) | Classification | [theatticusproject/cuad-qa](https://hf.co/datasets/theatticusproject/cuad-qa) | ✓ |
| 513 | [hallmarks-of-cancer/yesno](src/tasksource/tasks.py#L2654) | Classification | [bigbio/hallmarks_of_cancer](https://hf.co/datasets/bigbio/hallmarks_of_cancer) | ✓ |
| 514 | [semeval2014-absa/restaurants](src/tasksource/tasks.py#L2676) | Classification | [alexcadillon/SemEval2014Task4](https://hf.co/datasets/alexcadillon/SemEval2014Task4) | ✓ |
| 515 | [semeval2014-absa/laptops](src/tasksource/tasks.py#L2676) | Classification | [alexcadillon/SemEval2014Task4](https://hf.co/datasets/alexcadillon/SemEval2014Task4) | ✓ |
| 516 | [record/candidate](src/tasksource/tasks.py#L2698) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) | ✓ |
| 517 | [multidogo/fastfood/intent](src/tasksource/tasks.py#L2721) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 518 | [multidogo/insurance/intent](src/tasksource/tasks.py#L2721) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 519 | [multidogo/media/intent](src/tasksource/tasks.py#L2721) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 520 | [multidogo/software/intent](src/tasksource/tasks.py#L2721) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 521 | [multidogo/finance/intent](src/tasksource/tasks.py#L2721) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 522 | [multidogo/airline/intent](src/tasksource/tasks.py#L2721) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 523 | [wikisql/aggregation](src/tasksource/tasks.py#L2749) | Classification | [Salesforce/wikisql](https://hf.co/datasets/Salesforce/wikisql) | ✓ |
| 524 | [wikisql/column](src/tasksource/tasks.py#L2754) | MultipleChoice | [Salesforce/wikisql](https://hf.co/datasets/Salesforce/wikisql) | ✓ |
| 525 | [wikisql/operator](src/tasksource/tasks.py#L2759) | Classification | [Salesforce/wikisql](https://hf.co/datasets/Salesforce/wikisql) | ✓ |
| 526 | [acord](src/tasksource/tasks.py#L2764) | Classification | [tasksource/acord](https://hf.co/datasets/tasksource/acord) | ✓ |
| 527 | [edos](src/tasksource/tasks.py#L2768) | Classification | [tasksource/edos](https://hf.co/datasets/tasksource/edos) | ✓ |
| 528 | [evidence-inference](src/tasksource/tasks.py#L2771) | Classification | [tasksource/evidence-inference](https://hf.co/datasets/tasksource/evidence-inference) | ✓ |
| 529 | [quac/answerability](src/tasksource/tasks.py#L2793) | Classification | [allenai/quac](https://hf.co/datasets/allenai/quac) | ✓ |
| 530 | [hatexplain/majority](src/tasksource/tasks.py#L2801) | Classification | [Hate-speech-CNERG/hatexplain](https://hf.co/datasets/Hate-speech-CNERG/hatexplain) | ✓ |
| 531 | [hotpotqa/supporting-fact](src/tasksource/tasks.py#L2818) | Classification | [hotpotqa/hotpot_qa](https://hf.co/datasets/hotpotqa/hotpot_qa) | ✓ |
| 532 | [multiwoz22/dialogue-act](src/tasksource/tasks.py#L2862) | Classification | [pfb30/multi_woz_v22](https://hf.co/datasets/pfb30/multi_woz_v22) | ✓ |
| 533 | [multiwoz22/categorical-slot](src/tasksource/tasks.py#L2865) | Classification | [pfb30/multi_woz_v22](https://hf.co/datasets/pfb30/multi_woz_v22) | ✓ |
| 534 | [schema-guided/intent](src/tasksource/tasks.py#L2868) | Classification | [google-research-datasets/schema_guided_dstc8](https://hf.co/datasets/google-research-datasets/schema_guided_dstc8) | ✓ |
| 535 | [coqa/yesno](src/tasksource/tasks.py#L2885) | Classification | [stanfordnlp/coqa](https://hf.co/datasets/stanfordnlp/coqa) | ✓ |
| 536 | [qasper/yesno](src/tasksource/tasks.py#L2906) | Classification | [allenai/qasper](https://hf.co/datasets/allenai/qasper) | ✓ |
| 537 | [topv2/intent](src/tasksource/tasks.py#L2918) | Classification | [WillHeld/top_v2](https://hf.co/datasets/WillHeld/top_v2) | ✓ |
| 538 | [slurp/intent](src/tasksource/tasks.py#L2926) | Classification | [tasksource/slurp](https://hf.co/datasets/tasksource/slurp) | ✓ |
| 539 | [mcscript2](src/tasksource/tasks.py#L2930) | MultipleChoice | [tasksource/mcscript2](https://hf.co/datasets/tasksource/mcscript2) |  |
| 540 | [fnc1/stance](src/tasksource/tasks.py#L2935) | Classification | [tasksource/fnc1](https://hf.co/datasets/tasksource/fnc1) | ✓ |
| 541 | [perspectrum/stance](src/tasksource/tasks.py#L2940) | Classification | [tasksource/perspectrum](https://hf.co/datasets/tasksource/perspectrum) | ✓ |
| 542 | [ukpconvarg/convincingness](src/tasksource/tasks.py#L2945) | MultipleChoice | [tasksource/ukpconvarg](https://hf.co/datasets/tasksource/ukpconvarg) | ✓ |
| 543 | [discogem/relation](src/tasksource/tasks.py#L2950) | Classification | [tasksource/discogem](https://hf.co/datasets/tasksource/discogem) | ✓ |
| 544 | [pubmed-rct/role](src/tasksource/tasks.py#L2958) | Classification | [tasksource/pubmed-rct](https://hf.co/datasets/tasksource/pubmed-rct) | ✓ |
| 545 | [maven-arg/role](src/tasksource/tasks.py#L2986) | Classification | [mneb/maven-arg](https://hf.co/datasets/mneb/maven-arg) | ✓ |
| 546 | [maven/event](src/tasksource/tasks.py#L3017) | Classification | [mneb/maven](https://hf.co/datasets/mneb/maven) | ✓ |
| 547 | [empathetic-dialogues/emotion](src/tasksource/tasks.py#L3021) | Classification | [facebook/empathetic_dialogues](https://hf.co/datasets/facebook/empathetic_dialogues) | ✓ |
| 548 | [olid/offensive](src/tasksource/tasks.py#L3029) | Classification | [christophsonntag/OLID](https://hf.co/datasets/christophsonntag/OLID) | ✓ |
| 549 | [isarcasm/intended](src/tasksource/tasks.py#L3035) | Classification | [viethq1906/isarcasm_2022_taskA_En](https://hf.co/datasets/viethq1906/isarcasm_2022_taskA_En) | ✓ |
| 550 | [magpie/idiomatic](src/tasksource/tasks.py#L3060) | Classification | [idioms-collaboration/magpie](https://hf.co/datasets/idioms-collaboration/magpie) | ✓ |
| 551 | [preco/coreference](src/tasksource/tasks.py#L3096) | Classification | [coref-data/preco_raw](https://hf.co/datasets/coref-data/preco_raw) | ✓ |
| 552 | [redial/preference](src/tasksource/tasks.py#L3116) | Classification | [community-datasets/re_dial](https://hf.co/datasets/community-datasets/re_dial) | ✓ |
| 553 | [sst5](src/tasksource/tasks.py#L3121) | Classification | [SetFit/sst5](https://hf.co/datasets/SetFit/sst5) | ✓ |
| 554 | [spider/column](src/tasksource/tasks.py#L3129) | MultipleChoice | [tasksource/spider](https://hf.co/datasets/tasksource/spider) | ✓ |
| 555 | [semcor](src/tasksource/tasks.py#L3133) | MultipleChoice | [tasksource/semcor](https://hf.co/datasets/tasksource/semcor) | ✓ |
| 556 | [streusle/noun](src/tasksource/tasks.py#L3136) | Classification | [tasksource/streusle](https://hf.co/datasets/tasksource/streusle) | ✓ |
| 557 | [streusle/verb](src/tasksource/tasks.py#L3140) | Classification | [tasksource/streusle](https://hf.co/datasets/tasksource/streusle) | ✓ |
| 558 | [maven-ere/causal](src/tasksource/tasks.py#L3183) | Classification | [Nofing/Maven-ERE-span](https://hf.co/datasets/Nofing/Maven-ERE-span) | ✓ |
| 559 | [biored/relation](src/tasksource/tasks.py#L3205) | Classification | [gyorilab/BioRED](https://hf.co/datasets/gyorilab/BioRED) | ✓ |
| 560 | [matres/temporal](src/tasksource/tasks.py#L3212) | Classification | [tasksource/matres](https://hf.co/datasets/tasksource/matres) | ✓ |
| 561 | [causal-news](src/tasksource/tasks.py#L3221) | Classification | [thagen/CNCv2](https://hf.co/datasets/thagen/CNCv2) | ✓ |
| 562 | [slurp/slot-type](src/tasksource/tasks.py#L3268) | Classification | [tasksource/slurp](https://hf.co/datasets/tasksource/slurp) | ✓ |
| 563 | [topv2/slot-type](src/tasksource/tasks.py#L3273) | Classification | [WillHeld/top_v2](https://hf.co/datasets/WillHeld/top_v2) | ✓ |
| 564 | [wikiqa](src/tasksource/tasks.py#L3286) | Classification | [microsoft/wiki_qa](https://hf.co/datasets/microsoft/wiki_qa) | ✓ |
| 565 | [nevir](src/tasksource/tasks.py#L3301) | MultipleChoice | [orionweller/NevIR](https://hf.co/datasets/orionweller/NevIR) | ✓ |
| 566 | [antique](src/tasksource/tasks.py#L3309) | Classification | [tasksource/antique](https://hf.co/datasets/tasksource/antique) | ✓ |
| 567 | [causal-reasoning/transitivity](src/tasksource/tasks.py#L3335) | Classification | [ludwigw/causal-reasoning-benchmarks](https://hf.co/datasets/ludwigw/causal-reasoning-benchmarks) | ✓ |
| 568 | [cebab/rating](src/tasksource/tasks.py#L3352) | Classification | [EleutherAI/CEBaB](https://hf.co/datasets/EleutherAI/CEBaB) | ✓ |
| 569 | [cebab/food](src/tasksource/tasks.py#L3356) | Classification | [EleutherAI/CEBaB](https://hf.co/datasets/EleutherAI/CEBaB) | ✓ |
| 570 | [cebab/ambiance](src/tasksource/tasks.py#L3359) | Classification | [EleutherAI/CEBaB](https://hf.co/datasets/EleutherAI/CEBaB) | ✓ |
| 571 | [cebab/service](src/tasksource/tasks.py#L3362) | Classification | [EleutherAI/CEBaB](https://hf.co/datasets/EleutherAI/CEBaB) | ✓ |
| 572 | [cebab/noise](src/tasksource/tasks.py#L3365) | Classification | [EleutherAI/CEBaB](https://hf.co/datasets/EleutherAI/CEBaB) | ✓ |
| 573 | [fabsa/sentiment](src/tasksource/tasks.py#L3378) | Classification | [jordiclive/FABSA](https://hf.co/datasets/jordiclive/FABSA) | ✓ |
| 574 | [tab/identifier](src/tasksource/tasks.py#L3395) | Classification | [doshimit3015/text-anonymization-benchmark](https://hf.co/datasets/doshimit3015/text-anonymization-benchmark) | ✓ |

| 575 | [newsmtsc](src/tasksource/tasks.py#L3410) | Classification | [fhamborg/news_sentiment_newsmtsc](https://hf.co/datasets/fhamborg/news_sentiment_newsmtsc) | ✓ |
| 576 | [sentihood/sentiment](src/tasksource/tasks.py#L3438) | Classification | [bhavnicksm/sentihood](https://hf.co/datasets/bhavnicksm/sentihood) | ✓ |
| 577 | [mams/atsa](src/tasksource/tasks.py#L3455) | Classification | [NEUDM/mams](https://hf.co/datasets/NEUDM/mams) | ✓ |
| 578 | [quality](src/tasksource/tasks.py#L3463) | MultipleChoice | [emozilla/quality](https://hf.co/datasets/emozilla/quality) |  |
| 579 | [clear/readability](src/tasksource/tasks.py#L3472) | Classification | [casey-martin/CommonLit-Ease-of-Readability](https://hf.co/datasets/casey-martin/CommonLit-Ease-of-Readability) | ✓ |
| 580 | [esconv/strategy](src/tasksource/tasks.py#L3485) | Classification | [tasksource/esconv](https://hf.co/datasets/tasksource/esconv) | ✓ |
| 581 | [prosocial-dialog/safety](src/tasksource/tasks.py#L3506) | Classification | [allenai/prosocial-dialog](https://hf.co/datasets/allenai/prosocial-dialog) | ✓ |
| 582 | [gap](src/tasksource/tasks.py#L3526) | Classification | [coref-data/gap_raw](https://hf.co/datasets/coref-data/gap_raw) | ✓ |
| 583 | [trip/cloze](src/tasksource/tasks.py#L3534) | MultipleChoice | [sled-umich/TRIP](https://hf.co/datasets/sled-umich/TRIP) | ✓ |
| 584 | [trip/order](src/tasksource/tasks.py#L3543) | MultipleChoice | [sled-umich/TRIP](https://hf.co/datasets/sled-umich/TRIP) | ✓ |

## Soft labels

Annotations whose label is a distribution (annotator votes, rater shares, survey counts), loaded with `load_task(id, soft=True)`. Those with a hard view are also listed above, by their majority label; the others have soft labels only. `votes` are shares of annotators, `mean` a numeric value interpolated over score anchors (see each task for its meaning); annotators is the typical count per item (vote shares from fewer than five are coarse, and the Jev build leaves them out).

| id | kind | aggregation | annotators | default view | dataset |
|---|---|---|--:|---|---|
| [glue/stsb](src/tasksource/tasks.py#L49) | score | mean |  | regression | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |
| [sick/relatedness](src/tasksource/tasks.py#L88) | score | mean | 10 | regression | [tasksource/sick](https://hf.co/datasets/tasksource/sick) |
| [google_wellformed_query](src/tasksource/tasks.py#L858) | noul | votes | 5 | Classification | [tasksource/google_wellformed_query](https://hf.co/datasets/tasksource/google_wellformed_query) |
| [sts-companion](src/tasksource/tasks.py#L1052) | score | mean |  | regression | [tasksource/sts-companion](https://hf.co/datasets/tasksource/sts-companion) |
| [acceptability-prediction/rating_votes](src/tasksource/tasks.py#L1283) | score | votes | 15 |  | [tasksource/acceptability-prediction](https://hf.co/datasets/tasksource/acceptability-prediction) |
| [acceptability-prediction/binary_votes](src/tasksource/tasks.py#L1288) | noul | votes | 15 |  | [tasksource/acceptability-prediction](https://hf.co/datasets/tasksource/acceptability-prediction) |
| [HelpSteer3/individual_preferences](src/tasksource/tasks.py#L2097) | score | votes | 3 |  | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) |
| [HelpSteer3/feedback](src/tasksource/tasks.py#L2130) | score | votes | 3 | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) |
| [proto_qa/proto_qa](src/tasksource/tasks.py#L2224) | choice | votes | 100 |  | [community-datasets/proto_qa](https://hf.co/datasets/community-datasets/proto_qa) |
| [UNLI](src/tasksource/tasks.py#L2238) | noul | mean | 2 |  | [Zhengping/UNLI](https://hf.co/datasets/Zhengping/UNLI) |
| [chaos-mnli-ambiguity/votes](src/tasksource/tasks.py#L2242) | choice | votes | 100 |  | [tasksource/chaos-mnli-ambiguity](https://hf.co/datasets/tasksource/chaos-mnli-ambiguity) |
| [hate_speech_offensive/votes](src/tasksource/tasks.py#L2246) | choice | votes | 3 |  | [tdavidson/hate_speech_offensive](https://hf.co/datasets/tdavidson/hate_speech_offensive) |
| [scruples/verdict_votes](src/tasksource/tasks.py#L2254) | choice | votes | 8 |  | [tasksource/scruples](https://hf.co/datasets/tasksource/scruples) |
| [BeaverTails/unsafe_votes](src/tasksource/tasks.py#L2268) | noul | votes | 3 |  | [PKU-Alignment/BeaverTails](https://hf.co/datasets/PKU-Alignment/BeaverTails) |
| [dynasent/r1_votes](src/tasksource/tasks.py#L2279) | choice | votes | 5 |  | [dynabench/dynasent](https://hf.co/datasets/dynabench/dynasent) |
| [dynasent/r2_votes](src/tasksource/tasks.py#L2283) | choice | votes | 5 |  | [dynabench/dynasent](https://hf.co/datasets/dynabench/dynasent) |
| [hatexplain/votes](src/tasksource/tasks.py#L2288) | choice | votes | 3 |  | [Hate-speech-CNERG/hatexplain](https://hf.co/datasets/Hate-speech-CNERG/hatexplain) |
| [measuring-hate-speech/sentiment](src/tasksource/tasks.py#L2303) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/respect](src/tasksource/tasks.py#L2305) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/insult](src/tasksource/tasks.py#L2307) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/humiliate](src/tasksource/tasks.py#L2308) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/status](src/tasksource/tasks.py#L2309) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/dehumanize](src/tasksource/tasks.py#L2311) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/violence](src/tasksource/tasks.py#L2313) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/genocide](src/tasksource/tasks.py#L2314) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/attack_defend](src/tasksource/tasks.py#L2316) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/hatespeech](src/tasksource/tasks.py#L2318) | choice | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [wikipedia-detox/attack](src/tasksource/tasks.py#L2330) | choice | votes | 10 |  | [tasksource/wikipedia-detox-votes](https://hf.co/datasets/tasksource/wikipedia-detox-votes) |
| [wikipedia-detox/aggression](src/tasksource/tasks.py#L2332) | choice | votes | 10 |  | [tasksource/wikipedia-detox-votes](https://hf.co/datasets/tasksource/wikipedia-detox-votes) |
| [wikipedia-detox/aggression_score](src/tasksource/tasks.py#L2334) | score | votes | 10 |  | [tasksource/wikipedia-detox-votes](https://hf.co/datasets/tasksource/wikipedia-detox-votes) |
| [wikipedia-detox/toxicity](src/tasksource/tasks.py#L2338) | choice | votes | 10 |  | [tasksource/wikipedia-detox-votes](https://hf.co/datasets/tasksource/wikipedia-detox-votes) |
| [wikipedia-detox/toxicity_score](src/tasksource/tasks.py#L2340) | score | votes | 10 |  | [tasksource/wikipedia-detox-votes](https://hf.co/datasets/tasksource/wikipedia-detox-votes) |
| [lewidi/md_agreement](src/tasksource/tasks.py#L2355) | noul | votes | 5 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/hs_brexit](src/tasksource/tasks.py#L2356) | noul | votes | 6 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/armis](src/tasksource/tasks.py#L2358) | noul | votes | 3 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/conv_abuse](src/tasksource/tasks.py#L2360) | noul | votes | 3 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/mp](src/tasksource/tasks.py#L2362) | noul | votes | 5 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/csc](src/tasksource/tasks.py#L2363) | score | votes | 4 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/varierrnli/entailment](src/tasksource/tasks.py#L2371) | noul | votes | 4 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/varierrnli/neutral](src/tasksource/tasks.py#L2372) | noul | votes | 4 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/varierrnli/contradiction](src/tasksource/tasks.py#L2373) | noul | votes | 4 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [civil_comments/toxicity_share](src/tasksource/tasks.py#L2381) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/severe_toxicity_share](src/tasksource/tasks.py#L2382) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/obscene_share](src/tasksource/tasks.py#L2383) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/threat_share](src/tasksource/tasks.py#L2384) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/insult_share](src/tasksource/tasks.py#L2385) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/identity_attack_share](src/tasksource/tasks.py#L2386) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/sexual_explicit_share](src/tasksource/tasks.py#L2387) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [oasst2/quality](src/tasksource/tasks.py#L2404) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/helpfulness](src/tasksource/tasks.py#L2405) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/creativity](src/tasksource/tasks.py#L2406) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/humor](src/tasksource/tasks.py#L2407) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/toxicity](src/tasksource/tasks.py#L2408) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/violence](src/tasksource/tasks.py#L2409) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/spam](src/tasksource/tasks.py#L2410) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/fails_task](src/tasksource/tasks.py#L2411) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/not_appropriate](src/tasksource/tasks.py#L2412) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/hate_speech](src/tasksource/tasks.py#L2413) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/sexual_content](src/tasksource/tasks.py#L2414) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/pii](src/tasksource/tasks.py#L2415) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/lang_mismatch](src/tasksource/tasks.py#L2416) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [scirepeval/search](src/tasksource/tasks.py#L2519) | score | mean |  | regression | [tasksource/scirepeval-search](https://hf.co/datasets/tasksource/scirepeval-search) |
| [discogem/relation](src/tasksource/tasks.py#L2950) | choice | votes | 10 | Classification | [tasksource/discogem](https://hf.co/datasets/tasksource/discogem) |
| [clear/readability](src/tasksource/tasks.py#L3472) | score | mean |  | regression | [casey-martin/CommonLit-Ease-of-Readability](https://hf.co/datasets/casey-martin/CommonLit-Ease-of-Readability) |
