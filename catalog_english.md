580 English tasks. Load one with `load_task(id)`; the annotations are in [tasks.py](src/tasksource/tasks.py). Evaluation benchmarks are in [eval_only.py](src/tasksource/eval_only.py); other excluded annotations are in [parked.py](src/tasksource/parked.py).

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
| 17 | [babi_nli/conjunction](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 18 | [babi_nli/indefinite-knowledge](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 19 | [babi_nli/lists-sets](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 20 | [babi_nli/path-finding](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 21 | [babi_nli/positional-reasoning](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 22 | [babi_nli/simple-negation](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 23 | [babi_nli/size-reasoning](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 24 | [babi_nli/single-supporting-fact](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 25 | [babi_nli/three-supporting-facts](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 26 | [babi_nli/time-reasoning](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 27 | [babi_nli/two-arg-relations](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 28 | [babi_nli/two-supporting-facts](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 29 | [babi_nli/yes-no-questions](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 30 | [babi_nli/compound-coreference](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 31 | [babi_nli/three-arg-relations](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 32 | [babi_nli/basic-deduction](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 33 | [babi_nli/counting](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 34 | [babi_nli/basic-coreference](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 35 | [babi_nli/basic-induction](src/tasksource/tasks.py#L81) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) | ✓ |
| 36 | [sick/label](src/tasksource/tasks.py#L86) | Classification | [tasksource/sick](https://hf.co/datasets/tasksource/sick) |  |
| 37 | [sick/relatedness](src/tasksource/tasks.py#L88) | Classification | [tasksource/sick](https://hf.co/datasets/tasksource/sick) | ✓ |
| 38 | [snli](src/tasksource/tasks.py#L146) | Classification | [stanfordnlp/snli](https://hf.co/datasets/stanfordnlp/snli) |  |
| 39 | [scitail/snli_format](src/tasksource/tasks.py#L149) | Classification | [allenai/scitail](https://hf.co/datasets/allenai/scitail) |  |
| 40 | [hans](src/tasksource/tasks.py#L151) | Classification | [tasksource/hans](https://hf.co/datasets/tasksource/hans) |  |
| 41 | [WANLI](src/tasksource/tasks.py#L154) | Classification | [alisawuffles/WANLI](https://hf.co/datasets/alisawuffles/WANLI) |  |
| 42 | [recast/recast_verbnet](src/tasksource/tasks.py#L156) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 43 | [recast/recast_megaveridicality](src/tasksource/tasks.py#L156) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 44 | [recast/recast_ner](src/tasksource/tasks.py#L156) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 45 | [recast/recast_verbcorner](src/tasksource/tasks.py#L156) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 46 | [recast/recast_sentiment](src/tasksource/tasks.py#L156) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 47 | [recast/recast_factuality](src/tasksource/tasks.py#L156) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 48 | [recast/recast_puns](src/tasksource/tasks.py#L156) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 49 | [probability_words_nli/reasoning_1hop](src/tasksource/tasks.py#L161) | Classification | [sileod/probability_words_nli](https://hf.co/datasets/sileod/probability_words_nli) | ✓ |
| 50 | [probability_words_nli/reasoning_2hop](src/tasksource/tasks.py#L161) | Classification | [sileod/probability_words_nli](https://hf.co/datasets/sileod/probability_words_nli) | ✓ |
| 51 | [probability_words_nli/usnli](src/tasksource/tasks.py#L161) | Classification | [sileod/probability_words_nli](https://hf.co/datasets/sileod/probability_words_nli) | ✓ |
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
| 64 | [robust_nli/IS_CS](src/tasksource/tasks.py#L212) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 65 | [robust_nli/LI_LI](src/tasksource/tasks.py#L214) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 66 | [robust_nli/ST_WO](src/tasksource/tasks.py#L216) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 67 | [robust_nli/PI_SP](src/tasksource/tasks.py#L218) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 68 | [robust_nli/PI_CD](src/tasksource/tasks.py#L220) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 69 | [robust_nli/ST_SE](src/tasksource/tasks.py#L222) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 70 | [robust_nli/ST_NE](src/tasksource/tasks.py#L224) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 71 | [robust_nli/ST_LM](src/tasksource/tasks.py#L226) | Classification | [pietrolesci/robust_nli](https://hf.co/datasets/pietrolesci/robust_nli) |  |
| 72 | [robust_nli_is_sd](src/tasksource/tasks.py#L228) | Classification | [pietrolesci/robust_nli_is_sd](https://hf.co/datasets/pietrolesci/robust_nli_is_sd) |  |
| 73 | [robust_nli_li_ts](src/tasksource/tasks.py#L231) | Classification | [pietrolesci/robust_nli_li_ts](https://hf.co/datasets/pietrolesci/robust_nli_li_ts) |  |
| 74 | [add_one_rte](src/tasksource/tasks.py#L236) | Classification | [pietrolesci/add_one_rte](https://hf.co/datasets/pietrolesci/add_one_rte) |  |
| 75 | [hlgd](src/tasksource/tasks.py#L240) | Classification | [tasksource/hlgd](https://hf.co/datasets/tasksource/hlgd) | ✓ |
| 76 | [paws/labeled_final](src/tasksource/tasks.py#L244) | Classification | [google-research-datasets/paws](https://hf.co/datasets/google-research-datasets/paws) |  |
| 77 | [paws/labeled_swap](src/tasksource/tasks.py#L245) | Classification | [google-research-datasets/paws](https://hf.co/datasets/google-research-datasets/paws) |  |
| 78 | [medical_questions_pairs](src/tasksource/tasks.py#L247) | Classification | [curaihealth/medical_questions_pairs](https://hf.co/datasets/curaihealth/medical_questions_pairs) | ✓ |
| 79 | [conll2003/pos_tags](src/tasksource/tasks.py#L253) | TokenClassification | [tomaarsen/conll2003](https://hf.co/datasets/tomaarsen/conll2003) | ✓ |
| 80 | [conll2003/chunk_tags](src/tasksource/tasks.py#L255) | TokenClassification | [tomaarsen/conll2003](https://hf.co/datasets/tomaarsen/conll2003) | ✓ |
| 81 | [conll2003/ner_tags](src/tasksource/tasks.py#L257) | TokenClassification | [tomaarsen/conll2003](https://hf.co/datasets/tomaarsen/conll2003) |  |
| 82 | [fig-qa](src/tasksource/tasks.py#L263) | MultipleChoice | [nightingal3/fig-qa](https://hf.co/datasets/nightingal3/fig-qa) |  |
| 83 | [cos_e/v1.0](src/tasksource/tasks.py#L272) | MultipleChoice | [Salesforce/cos_e](https://hf.co/datasets/Salesforce/cos_e) |  |
| 84 | [cosmos_qa](src/tasksource/tasks.py#L277) | MultipleChoice | [Samsoup/cosmos_qa](https://hf.co/datasets/Samsoup/cosmos_qa) |  |
| 85 | [dream](src/tasksource/tasks.py#L280) | MultipleChoice | [dataset-org/dream](https://hf.co/datasets/dataset-org/dream) |  |
| 86 | [openbookqa](src/tasksource/tasks.py#L287) | MultipleChoice | [allenai/openbookqa](https://hf.co/datasets/allenai/openbookqa) |  |
| 87 | [qasc](src/tasksource/tasks.py#L293) | MultipleChoice | [allenai/qasc](https://hf.co/datasets/allenai/qasc) |  |
| 88 | [quartz](src/tasksource/tasks.py#L301) | MultipleChoice | [allenai/quartz](https://hf.co/datasets/allenai/quartz) |  |
| 89 | [quail](src/tasksource/tasks.py#L306) | MultipleChoice | [textmachinelab/quail](https://hf.co/datasets/textmachinelab/quail) |  |
| 90 | [head_qa/en](src/tasksource/tasks.py#L312) | MultipleChoice | [EleutherAI/headqa](https://hf.co/datasets/EleutherAI/headqa) |  |
| 91 | [sciq](src/tasksource/tasks.py#L320) | MultipleChoice | [allenai/sciq](https://hf.co/datasets/allenai/sciq) |  |
| 92 | [social_i_qa](src/tasksource/tasks.py#L325) | MultipleChoice | [tasksource/social_i_qa](https://hf.co/datasets/tasksource/social_i_qa) |  |
| 93 | [wiki_hop/original](src/tasksource/tasks.py#L331) | MultipleChoice | [MoE-UNC/wikihop](https://hf.co/datasets/MoE-UNC/wikihop) |  |
| 94 | [wiqa](src/tasksource/tasks.py#L344) | MultipleChoice | [allenai/wiqa](https://hf.co/datasets/allenai/wiqa) (`parquet` loader) |  |
| 95 | [piqa](src/tasksource/tasks.py#L351) | MultipleChoice | [baber/piqa](https://hf.co/datasets/baber/piqa) |  |
| 96 | [hellaswag](src/tasksource/tasks.py#L359) | MultipleChoice | [Rowan/hellaswag](https://hf.co/datasets/Rowan/hellaswag) |  |
| 97 | [super_glue/copa](src/tasksource/tasks.py#L368) | MultipleChoice | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 98 | [balanced-copa](src/tasksource/tasks.py#L370) | MultipleChoice | [pkavumba/balanced-copa](https://hf.co/datasets/pkavumba/balanced-copa) |  |
| 99 | [e-CARE](src/tasksource/tasks.py#L373) | MultipleChoice | [12ml/e-CARE](https://hf.co/datasets/12ml/e-CARE) |  |
| 100 | [art](src/tasksource/tasks.py#L376) | MultipleChoice | [allenai/art](https://hf.co/datasets/allenai/art) | ✓ |
| 101 | [winogrande/winogrande_xl](src/tasksource/tasks.py#L384) | MultipleChoice | [allenai/winogrande](https://hf.co/datasets/allenai/winogrande) |  |
| 102 | [codah/codah](src/tasksource/tasks.py#L387) | MultipleChoice | [jaredfern/codah](https://hf.co/datasets/jaredfern/codah) |  |
| 103 | [ai2_arc/ARC-Easy/challenge](src/tasksource/tasks.py#L389) | MultipleChoice | [allenai/ai2_arc](https://hf.co/datasets/allenai/ai2_arc) |  |
| 104 | [ai2_arc/ARC-Challenge/challenge](src/tasksource/tasks.py#L389) | MultipleChoice | [allenai/ai2_arc](https://hf.co/datasets/allenai/ai2_arc) |  |
| 105 | [definite_pronoun_resolution](src/tasksource/tasks.py#L394) | MultipleChoice | [community-datasets/definite_pronoun_resolution](https://hf.co/datasets/community-datasets/definite_pronoun_resolution) |  |
| 106 | [swag/regular](src/tasksource/tasks.py#L400) | MultipleChoice | [allenai/swag](https://hf.co/datasets/allenai/swag) |  |
| 107 | [math_qa](src/tasksource/tasks.py#L406) | MultipleChoice | [tasksource/math_qa](https://hf.co/datasets/tasksource/math_qa) |  |
| 108 | [glue/cola](src/tasksource/tasks.py#L415) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 109 | [glue/sst2](src/tasksource/tasks.py#L416) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 110 | [utilitarianism](src/tasksource/tasks.py#L430) | Classification | [hendrycks/ethics](https://hf.co/datasets/hendrycks/ethics) (`csv` loader) |  |
| 111 | [amazon_counterfactual/en](src/tasksource/tasks.py#L438) | Classification | [mteb/amazon_counterfactual](https://hf.co/datasets/mteb/amazon_counterfactual) |  |
| 112 | [insincere-questions](src/tasksource/tasks.py#L443) | Classification | [SetFit/insincere-questions](https://hf.co/datasets/SetFit/insincere-questions) |  |
| 113 | [toxic_conversations](src/tasksource/tasks.py#L447) | Classification | [SetFit/toxic_conversations](https://hf.co/datasets/SetFit/toxic_conversations) |  |
| 114 | [TuringBench](src/tasksource/tasks.py#L451) | Classification | [jana4/turingbench-humanized](https://hf.co/datasets/jana4/turingbench-humanized), [turingbench/TuringBench](https://hf.co/datasets/turingbench/TuringBench) (`csv` loader) | ✓ |
| 115 | [trec](src/tasksource/tasks.py#L461) | Classification | [tasksource/trec](https://hf.co/datasets/tasksource/trec) |  |
| 116 | [vitaminc](src/tasksource/tasks.py#L464) | Classification | [tals/vitaminc](https://hf.co/datasets/tals/vitaminc) |  |
| 117 | [hope_edi/english](src/tasksource/tasks.py#L466) | Classification | [dravidianlangtech/hope_edi](https://hf.co/datasets/dravidianlangtech/hope_edi) (`csv` loader) |  |
| 118 | [rumoureval_2019/RumourEval2019](src/tasksource/tasks.py#L480) | Classification | [strombergnlp/rumoureval_2019](https://hf.co/datasets/strombergnlp/rumoureval_2019) (`csv` loader) |  |
| 119 | [ethos/binary](src/tasksource/tasks.py#L494) | Classification | [SetFit/ethos_binary](https://hf.co/datasets/SetFit/ethos_binary) |  |
| 120 | [ethos/multilabel](src/tasksource/tasks.py#L518) | Classification | [tasksource/ethos](https://hf.co/datasets/tasksource/ethos) |  |
| 121 | [tweet_eval/sentiment](src/tasksource/tasks.py#L521) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 122 | [tweet_eval/offensive](src/tasksource/tasks.py#L521) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 123 | [tweet_eval/irony](src/tasksource/tasks.py#L521) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 124 | [tweet_eval/emoji](src/tasksource/tasks.py#L521) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 125 | [tweet_eval/emotion](src/tasksource/tasks.py#L521) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 126 | [tweet_eval/hate](src/tasksource/tasks.py#L521) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 127 | [tweet_eval/stance_abortion](src/tasksource/tasks.py#L536) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 128 | [tweet_eval/stance_atheism](src/tasksource/tasks.py#L537) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 129 | [tweet_eval/stance_climate](src/tasksource/tasks.py#L538) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 130 | [tweet_eval/stance_feminist](src/tasksource/tasks.py#L539) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 131 | [tweet_eval/stance_hillary](src/tasksource/tasks.py#L540) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 132 | [discovery/discovery](src/tasksource/tasks.py#L543) | Classification | [sileod/discovery](https://hf.co/datasets/sileod/discovery) | ✓ |
| 133 | [pragmeval/switchboard](src/tasksource/tasks.py#L546) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 134 | [pragmeval/verifiability](src/tasksource/tasks.py#L546) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 135 | [pragmeval/persuasiveness-claimtype](src/tasksource/tasks.py#L550) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 136 | [pragmeval/emergent](src/tasksource/tasks.py#L550) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 137 | [pragmeval/sarcasm](src/tasksource/tasks.py#L550) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 138 | [pragmeval/gum](src/tasksource/tasks.py#L550) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 139 | [pragmeval/pdtb](src/tasksource/tasks.py#L550) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 140 | [pragmeval/stac](src/tasksource/tasks.py#L550) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 141 | [pragmeval/persuasiveness-premisetype](src/tasksource/tasks.py#L550) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 142 | [pragmeval/emobank-dominance](src/tasksource/tasks.py#L565) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 143 | [pragmeval/squinky-formality](src/tasksource/tasks.py#L566) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 144 | [pragmeval/squinky-implicature](src/tasksource/tasks.py#L567) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 145 | [pragmeval/squinky-informativeness](src/tasksource/tasks.py#L568) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 146 | [pragmeval/persuasiveness-eloquence](src/tasksource/tasks.py#L569) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 147 | [pragmeval/persuasiveness-relevance](src/tasksource/tasks.py#L570) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 148 | [pragmeval/persuasiveness-specificity](src/tasksource/tasks.py#L571) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 149 | [pragmeval/persuasiveness-strength](src/tasksource/tasks.py#L572) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 150 | [silicone/dyda_da](src/tasksource/tasks.py#L574) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | ✓ |
| 151 | [silicone/dyda_e](src/tasksource/tasks.py#L574) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | ✓ |
| 152 | [silicone/maptask](src/tasksource/tasks.py#L574) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | ✓ |
| 153 | [silicone/meld_e](src/tasksource/tasks.py#L574) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | ✓ |
| 154 | [silicone/meld_s](src/tasksource/tasks.py#L574) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | ✓ |
| 155 | [silicone/oasis](src/tasksource/tasks.py#L574) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | ✓ |
| 156 | [silicone/sem](src/tasksource/tasks.py#L574) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | ✓ |
| 157 | [silicone/iemocap](src/tasksource/tasks.py#L584) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | ✓ |
| 158 | [lex_glue/eurlex](src/tasksource/tasks.py#L594) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) | ✓ |
| 159 | [lex_glue/scotus](src/tasksource/tasks.py#L598) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) |  |
| 160 | [lex_glue/ledgar](src/tasksource/tasks.py#L601) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) |  |
| 161 | [lex_glue/unfair_tos](src/tasksource/tasks.py#L603) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) | ✓ |
| 162 | [lex_glue/case_hold](src/tasksource/tasks.py#L606) | MultipleChoice | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) |  |
| 163 | [language-identification](src/tasksource/tasks.py#L614) | Classification | [papluca/language-identification](https://hf.co/datasets/papluca/language-identification) | ✓ |
| 164 | [imdb](src/tasksource/tasks.py#L619) | Classification | [stanfordnlp/imdb](https://hf.co/datasets/stanfordnlp/imdb) |  |
| 165 | [rotten_tomatoes](src/tasksource/tasks.py#L621) | Classification | [cornell-movie-review-data/rotten_tomatoes](https://hf.co/datasets/cornell-movie-review-data/rotten_tomatoes) |  |
| 166 | [ag_news](src/tasksource/tasks.py#L623) | Classification | [fancyzhx/ag_news](https://hf.co/datasets/fancyzhx/ag_news) |  |
| 167 | [yelp_review_full/yelp_review_full](src/tasksource/tasks.py#L625) | Classification | [Yelp/yelp_review_full](https://hf.co/datasets/Yelp/yelp_review_full) | ✓ |
| 168 | [financial_phrasebank/sentences_allagree](src/tasksource/tasks.py#L630) | Classification | [ghbacct/financial-phrasebank-all-agree-classification](https://hf.co/datasets/ghbacct/financial-phrasebank-all-agree-classification) |  |
| 169 | [poem_sentiment](src/tasksource/tasks.py#L635) | Classification | [google-research-datasets/poem_sentiment](https://hf.co/datasets/google-research-datasets/poem_sentiment) |  |
| 170 | [dbpedia_14/dbpedia_14](src/tasksource/tasks.py#L637) | Classification | [fancyzhx/dbpedia_14](https://hf.co/datasets/fancyzhx/dbpedia_14) |  |
| 171 | [amazon_polarity/amazon_polarity](src/tasksource/tasks.py#L639) | Classification | [fancyzhx/amazon_polarity](https://hf.co/datasets/fancyzhx/amazon_polarity) |  |
| 172 | [app_reviews](src/tasksource/tasks.py#L641) | Classification | [sealuzh/app_reviews](https://hf.co/datasets/sealuzh/app_reviews) | ✓ |
| 173 | [hate_speech18](src/tasksource/tasks.py#L648) | Classification | [tasksource/hate_speech18](https://hf.co/datasets/tasksource/hate_speech18) |  |
| 174 | [sms_spam](src/tasksource/tasks.py#L654) | Classification | [ucirvine/sms_spam](https://hf.co/datasets/ucirvine/sms_spam) |  |
| 175 | [humicroedit/subtask-1](src/tasksource/tasks.py#L657) | Classification | [tasksource/humicroedit](https://hf.co/datasets/tasksource/humicroedit) | ✓ |
| 176 | [humicroedit/subtask-2](src/tasksource/tasks.py#L663) | Classification | [tasksource/humicroedit](https://hf.co/datasets/tasksource/humicroedit) | ✓ |
| 177 | [snips_built_in_intents](src/tasksource/tasks.py#L668) | Classification | [sonos-nlu-benchmark/snips_built_in_intents](https://hf.co/datasets/sonos-nlu-benchmark/snips_built_in_intents) |  |
| 178 | [hate_speech_offensive](src/tasksource/tasks.py#L672) | Classification | [tdavidson/hate_speech_offensive](https://hf.co/datasets/tdavidson/hate_speech_offensive) |  |
| 179 | [yahoo_answers_topics](src/tasksource/tasks.py#L674) | Classification | [community-datasets/yahoo_answers_topics](https://hf.co/datasets/community-datasets/yahoo_answers_topics) |  |
| 180 | [stackoverflow-questions](src/tasksource/tasks.py#L678) | Classification | [pacovaldez/stackoverflow-questions](https://hf.co/datasets/pacovaldez/stackoverflow-questions) | ✓ |
| 181 | [hyperpartisan_news](src/tasksource/tasks.py#L684) | Classification | [zapsdcn/hyperpartisan_news](https://hf.co/datasets/zapsdcn/hyperpartisan_news) |  |
| 182 | [sciie](src/tasksource/tasks.py#L689) | Classification | [zapsdcn/sciie](https://hf.co/datasets/zapsdcn/sciie) |  |
| 183 | [citation_intent](src/tasksource/tasks.py#L690) | Classification | [zapsdcn/citation_intent](https://hf.co/datasets/zapsdcn/citation_intent) |  |
| 184 | [go_emotions/simplified](src/tasksource/tasks.py#L692) | Classification | [google-research-datasets/go_emotions](https://hf.co/datasets/google-research-datasets/go_emotions) |  |
| 185 | [scicite](src/tasksource/tasks.py#L696) | Classification | [tasksource/scicite](https://hf.co/datasets/tasksource/scicite) |  |
| 186 | [liar](src/tasksource/tasks.py#L698) | Classification | [tasksource/liar](https://hf.co/datasets/tasksource/liar) | ✓ |
| 187 | [lexical_relation_classification/BLESS](src/tasksource/tasks.py#L709) | Classification | [relbert/lexical_relation_classification](https://hf.co/datasets/relbert/lexical_relation_classification) (`json` loader) | ✓ |
| 188 | [lexical_relation_classification/ROOT09](src/tasksource/tasks.py#L709) | Classification | [relbert/lexical_relation_classification](https://hf.co/datasets/relbert/lexical_relation_classification) (`json` loader) | ✓ |
| 189 | [lexical_relation_classification/EVALution](src/tasksource/tasks.py#L709) | Classification | [relbert/lexical_relation_classification](https://hf.co/datasets/relbert/lexical_relation_classification) (`json` loader) | ✓ |
| 190 | [lexical_relation_classification/K&H+N](src/tasksource/tasks.py#L709) | Classification | [relbert/lexical_relation_classification](https://hf.co/datasets/relbert/lexical_relation_classification) (`json` loader) | ✓ |
| 191 | [lexical_relation_classification/CogALexV](src/tasksource/tasks.py#L735) | Classification | [relbert/lexical_relation_classification](https://hf.co/datasets/relbert/lexical_relation_classification) (`json` loader) | ✓ |
| 192 | [linguisticprobing/subj_number](src/tasksource/tasks.py#L762) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 193 | [linguisticprobing/obj_number](src/tasksource/tasks.py#L763) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 194 | [linguisticprobing/past_present](src/tasksource/tasks.py#L764) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 195 | [linguisticprobing/top_constituents](src/tasksource/tasks.py#L765) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | ✓ |
| 196 | [linguisticprobing/tree_depth](src/tasksource/tasks.py#L767) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | ✓ |
| 197 | [linguisticprobing/coordination_inversion](src/tasksource/tasks.py#L768) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | ✓ |
| 198 | [linguisticprobing/odd_man_out](src/tasksource/tasks.py#L770) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | ✓ |
| 199 | [linguisticprobing/bigram_shift](src/tasksource/tasks.py#L771) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | ✓ |
| 200 | [crowdflower/airline-sentiment](src/tasksource/tasks.py#L773) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 201 | [crowdflower/corporate-messaging](src/tasksource/tasks.py#L773) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 202 | [crowdflower/tweet_global_warming](src/tasksource/tasks.py#L773) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 203 | [crowdflower/sentiment_nuclear_power](src/tasksource/tasks.py#L773) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 204 | [crowdflower/economic-news](src/tasksource/tasks.py#L773) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 205 | [crowdflower/political-media-audience](src/tasksource/tasks.py#L773) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 206 | [crowdflower/political-media-bias](src/tasksource/tasks.py#L773) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 207 | [crowdflower/political-media-message](src/tasksource/tasks.py#L773) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 208 | [ethics/commonsense](src/tasksource/tasks.py#L807) | Classification | [hendrycks/ethics](https://hf.co/datasets/hendrycks/ethics) (`csv` loader) |  |
| 209 | [ethics/deontology](src/tasksource/tasks.py#L815) | Classification | [hendrycks/ethics](https://hf.co/datasets/hendrycks/ethics) (`csv` loader) | ✓ |
| 210 | [ethics/justice](src/tasksource/tasks.py#L824) | Classification | [hendrycks/ethics](https://hf.co/datasets/hendrycks/ethics) (`csv` loader) |  |
| 211 | [ethics/virtue](src/tasksource/tasks.py#L832) | Classification | [hendrycks/ethics](https://hf.co/datasets/hendrycks/ethics) |  |
| 212 | [emo/emo2019](src/tasksource/tasks.py#L841) | Classification | [oneonlee/cleansed_emocontext](https://hf.co/datasets/oneonlee/cleansed_emocontext) |  |
| 213 | [google_wellformed_query](src/tasksource/tasks.py#L847) | Classification | [tasksource/google_wellformed_query](https://hf.co/datasets/tasksource/google_wellformed_query) | ✓ |
| 214 | [tweets_hate_speech_detection](src/tasksource/tasks.py#L853) | Classification | [tweets-hate-speech-detection/tweets_hate_speech_detection](https://hf.co/datasets/tweets-hate-speech-detection/tweets_hate_speech_detection) |  |
| 215 | [wnut_17/wnut_17](src/tasksource/tasks.py#L857) | TokenClassification | [flaitenberger/wnut_17](https://hf.co/datasets/flaitenberger/wnut_17) |  |
| 216 | [ncbi_disease/ncbi_disease](src/tasksource/tasks.py#L860) | TokenClassification | [ncbi/ncbi_disease](https://hf.co/datasets/ncbi/ncbi_disease) | ✓ |
| 217 | [acronym_identification](src/tasksource/tasks.py#L865) | TokenClassification | [amirveyseh/acronym_identification](https://hf.co/datasets/amirveyseh/acronym_identification) | ✓ |
| 218 | [jnlpba/jnlpba](src/tasksource/tasks.py#L869) | TokenClassification | [jnlpba/jnlpba](https://hf.co/datasets/jnlpba/jnlpba) | ✓ |
| 219 | [ontonotes_english/SpeedOfMagic--ontonotes_english](src/tasksource/tasks.py#L878) | TokenClassification | [SpeedOfMagic/ontonotes_english](https://hf.co/datasets/SpeedOfMagic/ontonotes_english) | ✓ |
| 220 | [blog_authorship_corpus/gender](src/tasksource/tasks.py#L884) | Classification | [tasksource/blog_authorship_corpus](https://hf.co/datasets/tasksource/blog_authorship_corpus) | ✓ |
| 221 | [blog_authorship_corpus/age](src/tasksource/tasks.py#L886) | Classification | [tasksource/blog_authorship_corpus](https://hf.co/datasets/tasksource/blog_authorship_corpus) | ✓ |
| 222 | [blog_authorship_corpus/job](src/tasksource/tasks.py#L889) | Classification | [tasksource/blog_authorship_corpus](https://hf.co/datasets/tasksource/blog_authorship_corpus) | ✓ |
| 223 | [open_question_type](src/tasksource/tasks.py#L900) | Classification | [Korea-MES/open_question_type](https://hf.co/datasets/Korea-MES/open_question_type) |  |
| 224 | [health_fact](src/tasksource/tasks.py#L902) | Classification | [marcov/health_fact_promptsource](https://hf.co/datasets/marcov/health_fact_promptsource) |  |
| 225 | [commonsense_qa](src/tasksource/tasks.py#L906) | MultipleChoice | [tau/commonsense_qa](https://hf.co/datasets/tau/commonsense_qa) |  |
| 226 | [mc_taco](src/tasksource/tasks.py#L912) | Classification | [marcov/mc_taco_promptsource](https://hf.co/datasets/marcov/mc_taco_promptsource) | ✓ |
| 227 | [ade_corpus_v2/Ade_corpus_v2_classification](src/tasksource/tasks.py#L919) | Classification | [ade-benchmark-corpus/ade_corpus_v2](https://hf.co/datasets/ade-benchmark-corpus/ade_corpus_v2) | ✓ |
| 228 | [discosense](src/tasksource/tasks.py#L922) | MultipleChoice | [tasksource/discosense](https://hf.co/datasets/tasksource/discosense) | ✓ |
| 229 | [circa](src/tasksource/tasks.py#L929) | Classification | [google-research-datasets/circa](https://hf.co/datasets/google-research-datasets/circa) |  |
| 230 | [code_x_glue_cc_defect_detection](src/tasksource/tasks.py#L934) | Classification | [google/code_x_glue_cc_defect_detection](https://hf.co/datasets/google/code_x_glue_cc_defect_detection) | ✓ |
| 231 | [phrase_similarity](src/tasksource/tasks.py#L939) | Classification | [Deehan1866/processed_phrase_similarity](https://hf.co/datasets/Deehan1866/processed_phrase_similarity) |  |
| 232 | [scientific-exaggeration-detection](src/tasksource/tasks.py#L947) | Classification | [copenlu/scientific-exaggeration-detection](https://hf.co/datasets/copenlu/scientific-exaggeration-detection) |  |
| 233 | [quarel](src/tasksource/tasks.py#L961) | MultipleChoice | [community-datasets/quarel](https://hf.co/datasets/community-datasets/quarel) | ✓ |
| 234 | [fever-evidence-related](src/tasksource/tasks.py#L968) | Classification | [mwong/fever-evidence-related](https://hf.co/datasets/mwong/fever-evidence-related) |  |
| 235 | [numer_sense](src/tasksource/tasks.py#L971) | Classification | [tasksource/numer_sense](https://hf.co/datasets/tasksource/numer_sense) |  |
| 236 | [dynasent/dynabench.dynasent.r1.all/r1](src/tasksource/tasks.py#L978) | Classification | [tasksource/dynasent](https://hf.co/datasets/tasksource/dynasent) | ✓ |
| 237 | [dynasent/dynabench.dynasent.r2.all/r2](src/tasksource/tasks.py#L982) | Classification | [tasksource/dynasent](https://hf.co/datasets/tasksource/dynasent) | ✓ |
| 238 | [Sarcasm_News_Headline](src/tasksource/tasks.py#L987) | Classification | [raquiba/Sarcasm_News_Headline](https://hf.co/datasets/raquiba/Sarcasm_News_Headline) |  |
| 239 | [sem_eval_2010_task_8](src/tasksource/tasks.py#L990) | Classification | [SemEvalWorkshop/sem_eval_2010_task_8](https://hf.co/datasets/SemEvalWorkshop/sem_eval_2010_task_8) |  |
| 240 | [auditor_review](src/tasksource/tasks.py#L992) | Classification | [demo-org/auditor_review](https://hf.co/datasets/demo-org/auditor_review) |  |
| 241 | [medmcqa](src/tasksource/tasks.py#L996) | MultipleChoice | [openlifescienceai/medmcqa](https://hf.co/datasets/openlifescienceai/medmcqa) |  |
| 242 | [Dynasent_Disagreement](src/tasksource/tasks.py#L1013) | Classification | [RuyuanWan/Dynasent_Disagreement](https://hf.co/datasets/RuyuanWan/Dynasent_Disagreement) | ✓ |
| 243 | [Politeness_Disagreement](src/tasksource/tasks.py#L1015) | Classification | [RuyuanWan/Politeness_Disagreement](https://hf.co/datasets/RuyuanWan/Politeness_Disagreement) | ✓ |
| 244 | [SBIC_Disagreement](src/tasksource/tasks.py#L1017) | Classification | [RuyuanWan/SBIC_Disagreement](https://hf.co/datasets/RuyuanWan/SBIC_Disagreement) | ✓ |
| 245 | [SChem_Disagreement](src/tasksource/tasks.py#L1019) | Classification | [RuyuanWan/SChem_Disagreement](https://hf.co/datasets/RuyuanWan/SChem_Disagreement) | ✓ |
| 246 | [Dilemmas_Disagreement](src/tasksource/tasks.py#L1021) | Classification | [RuyuanWan/Dilemmas_Disagreement](https://hf.co/datasets/RuyuanWan/Dilemmas_Disagreement) | ✓ |
| 247 | [logiqa](src/tasksource/tasks.py#L1024) | MultipleChoice | [fireworks-ai/logiqa](https://hf.co/datasets/fireworks-ai/logiqa) |  |
| 248 | [wiki_qa](src/tasksource/tasks.py#L1033) | Classification | [microsoft/wiki_qa](https://hf.co/datasets/microsoft/wiki_qa) | ✓ |
| 249 | [cycic_classification](src/tasksource/tasks.py#L1035) | Classification | [tasksource/cycic_classification](https://hf.co/datasets/tasksource/cycic_classification) |  |
| 250 | [cycic_multiplechoice](src/tasksource/tasks.py#L1037) | MultipleChoice | [tasksource/cycic_multiplechoice](https://hf.co/datasets/tasksource/cycic_multiplechoice) |  |
| 251 | [sts-companion](src/tasksource/tasks.py#L1041) | Classification | [tasksource/sts-companion](https://hf.co/datasets/tasksource/sts-companion) |  |
| 252 | [commonsense_qa_2.0](src/tasksource/tasks.py#L1044) | Classification | [tasksource/commonsense_qa_2.0](https://hf.co/datasets/tasksource/commonsense_qa_2.0) |  |
| 253 | [lingnli](src/tasksource/tasks.py#L1047) | Classification | [tasksource/lingnli](https://hf.co/datasets/tasksource/lingnli) |  |
| 254 | [monotonicity-entailment](src/tasksource/tasks.py#L1049) | Classification | [tasksource/monotonicity-entailment](https://hf.co/datasets/tasksource/monotonicity-entailment) |  |
| 255 | [arct](src/tasksource/tasks.py#L1052) | MultipleChoice | [tasksource/arct](https://hf.co/datasets/tasksource/arct) |  |
| 256 | [scinli](src/tasksource/tasks.py#L1055) | Classification | [tasksource/scinli](https://hf.co/datasets/tasksource/scinli) |  |
| 257 | [naturallogic](src/tasksource/tasks.py#L1059) | Classification | [tasksource/naturallogic](https://hf.co/datasets/tasksource/naturallogic) |  |
| 258 | [onestop_qa](src/tasksource/tasks.py#L1061) | MultipleChoice | [malmaud/onestop_qa](https://hf.co/datasets/malmaud/onestop_qa) |  |
| 259 | [moral_stories/full](src/tasksource/tasks.py#L1064) | MultipleChoice | [LabHC/moral_stories](https://hf.co/datasets/LabHC/moral_stories) |  |
| 260 | [dynahate](src/tasksource/tasks.py#L1068) | Classification | [tasksource/dynahate](https://hf.co/datasets/tasksource/dynahate) |  |
| 261 | [syntactic-augmentation-nli](src/tasksource/tasks.py#L1070) | Classification | [tasksource/syntactic-augmentation-nli](https://hf.co/datasets/tasksource/syntactic-augmentation-nli) |  |
| 262 | [autotnli](src/tasksource/tasks.py#L1072) | Classification | [tasksource/autotnli](https://hf.co/datasets/tasksource/autotnli) |  |
| 263 | [CONDAQA](src/tasksource/tasks.py#L1074) | Classification | [lasha-nlp/CONDAQA](https://hf.co/datasets/lasha-nlp/CONDAQA) |  |
| 264 | [webgpt_comparisons](src/tasksource/tasks.py#L1084) | MultipleChoice | [heegyu/webgpt_comparisons_ko](https://hf.co/datasets/heegyu/webgpt_comparisons_ko) | ✓ |
| 265 | [synthetic-instruct-gptj-pairwise](src/tasksource/tasks.py#L1092) | MultipleChoice | [Dahoas/synthetic-instruct-gptj-pairwise](https://hf.co/datasets/Dahoas/synthetic-instruct-gptj-pairwise) | ✓ |
| 266 | [scruples](src/tasksource/tasks.py#L1095) | Classification | [tasksource/scruples](https://hf.co/datasets/tasksource/scruples) | ✓ |
| 267 | [defeasible-nli/atomic](src/tasksource/tasks.py#L1100) | Classification | [tasksource/defeasible-nli](https://hf.co/datasets/tasksource/defeasible-nli) |  |
| 268 | [defeasible-nli/snli](src/tasksource/tasks.py#L1100) | Classification | [tasksource/defeasible-nli](https://hf.co/datasets/tasksource/defeasible-nli) |  |
| 269 | [defeasible-nli/social](src/tasksource/tasks.py#L1103) | Classification | [tasksource/defeasible-nli](https://hf.co/datasets/tasksource/defeasible-nli) |  |
| 270 | [help-nli](src/tasksource/tasks.py#L1106) | Classification | [tasksource/help-nli](https://hf.co/datasets/tasksource/help-nli) |  |
| 271 | [nli-veridicality-transitivity](src/tasksource/tasks.py#L1109) | Classification | [tasksource/nli-veridicality-transitivity](https://hf.co/datasets/tasksource/nli-veridicality-transitivity) |  |
| 272 | [lonli](src/tasksource/tasks.py#L1112) | Classification | [tasksource/lonli](https://hf.co/datasets/tasksource/lonli) |  |
| 273 | [dadc-limit-nli](src/tasksource/tasks.py#L1115) | Classification | [tasksource/dadc-limit-nli](https://hf.co/datasets/tasksource/dadc-limit-nli) |  |
| 274 | [FLUTE](src/tasksource/tasks.py#L1118) | Classification | [ColumbiaNLP/FLUTE](https://hf.co/datasets/ColumbiaNLP/FLUTE) |  |
| 275 | [strategy-qa](src/tasksource/tasks.py#L1121) | Classification | [tasksource/strategy-qa](https://hf.co/datasets/tasksource/strategy-qa) |  |
| 276 | [summarize_from_feedback/comparisons](src/tasksource/tasks.py#L1124) | MultipleChoice | [vwxyzjn/summarize_from_feedback_oai_preprocessing](https://hf.co/datasets/vwxyzjn/summarize_from_feedback_oai_preprocessing) | ✓ |
| 277 | [folio](src/tasksource/tasks.py#L1132) | Classification | [tasksource/folio](https://hf.co/datasets/tasksource/folio) |  |
| 278 | [tomi-nli](src/tasksource/tasks.py#L1159) | Classification | [tasksource/tomi-nli](https://hf.co/datasets/tasksource/tomi-nli) |  |
| 279 | [avicenna](src/tasksource/tasks.py#L1162) | Classification | [tasksource/avicenna](https://hf.co/datasets/tasksource/avicenna) | ✓ |
| 280 | [SHP](src/tasksource/tasks.py#L1165) | MultipleChoice | [stanfordnlp/SHP](https://hf.co/datasets/stanfordnlp/SHP) | ✓ |
| 281 | [MedQA-USMLE-4-options-hf](src/tasksource/tasks.py#L1173) | MultipleChoice | [GBaker/MedQA-USMLE-4-options-hf](https://hf.co/datasets/GBaker/MedQA-USMLE-4-options-hf) |  |
| 282 | [wikimedqa/medwiki](src/tasksource/tasks.py#L1176) | MultipleChoice | [sileod/wikimedqa](https://hf.co/datasets/sileod/wikimedqa) |  |
| 283 | [cicero](src/tasksource/tasks.py#L1185) | MultipleChoice | [declare-lab/cicero](https://hf.co/datasets/declare-lab/cicero) |  |
| 284 | [CREAK](src/tasksource/tasks.py#L1189) | Classification | [amydeng2000/CREAK](https://hf.co/datasets/amydeng2000/CREAK) |  |
| 285 | [mutual](src/tasksource/tasks.py#L1192) | MultipleChoice | [tasksource/mutual](https://hf.co/datasets/tasksource/mutual) |  |
| 286 | [implicatures](src/tasksource/tasks.py#L1197) | MultipleChoice | [tasksource/implicatures](https://hf.co/datasets/tasksource/implicatures) |  |
| 287 | [race/middle](src/tasksource/tasks.py#L1202) | MultipleChoice | [ehovy/race](https://hf.co/datasets/ehovy/race) |  |
| 288 | [race/high](src/tasksource/tasks.py#L1202) | MultipleChoice | [ehovy/race](https://hf.co/datasets/ehovy/race) |  |
| 289 | [race-c](src/tasksource/tasks.py#L1206) | MultipleChoice | [tasksource/race-c](https://hf.co/datasets/tasksource/race-c) |  |
| 290 | [spartqa-yn](src/tasksource/tasks.py#L1209) | Classification | [tasksource/spartqa-yn](https://hf.co/datasets/tasksource/spartqa-yn) |  |
| 291 | [spartqa-mchoice](src/tasksource/tasks.py#L1212) | MultipleChoice | [tasksource/spartqa-mchoice](https://hf.co/datasets/tasksource/spartqa-mchoice) |  |
| 292 | [temporal-nli](src/tasksource/tasks.py#L1215) | Classification | [tasksource/temporal-nli](https://hf.co/datasets/tasksource/temporal-nli) |  |
| 293 | [riddle_sense](src/tasksource/tasks.py#L1218) | MultipleChoice | [jeggers/riddle_sense](https://hf.co/datasets/jeggers/riddle_sense) |  |
| 294 | [clcd-english](src/tasksource/tasks.py#L1223) | Classification | [tasksource/clcd-english](https://hf.co/datasets/tasksource/clcd-english) |  |
| 295 | [twentyquestions](src/tasksource/tasks.py#L1235) | Classification | [tasksource/twentyquestions](https://hf.co/datasets/tasksource/twentyquestions) |  |
| 296 | [reclor](src/tasksource/tasks.py#L1240) | MultipleChoice | [tasksource/reclor](https://hf.co/datasets/tasksource/reclor) |  |
| 297 | [counterfactually-augmented-imdb](src/tasksource/tasks.py#L1243) | Classification | [tasksource/counterfactually-augmented-imdb](https://hf.co/datasets/tasksource/counterfactually-augmented-imdb) |  |
| 298 | [counterfactually-augmented-snli](src/tasksource/tasks.py#L1246) | Classification | [tasksource/counterfactually-augmented-snli](https://hf.co/datasets/tasksource/counterfactually-augmented-snli) |  |
| 299 | [cnli](src/tasksource/tasks.py#L1249) | Classification | [tasksource/cnli](https://hf.co/datasets/tasksource/cnli) |  |
| 300 | [boolq-natural-perturbations](src/tasksource/tasks.py#L1252) | Classification | [tasksource/boolq-natural-perturbations](https://hf.co/datasets/tasksource/boolq-natural-perturbations) |  |
| 301 | [acceptability-prediction](src/tasksource/tasks.py#L1257) | Classification | [tasksource/acceptability-prediction](https://hf.co/datasets/tasksource/acceptability-prediction) | ✓ |
| 302 | [equate](src/tasksource/tasks.py#L1283) | Classification | [tasksource/equate](https://hf.co/datasets/tasksource/equate) |  |
| 303 | [ScienceQA_text_only](src/tasksource/tasks.py#L1286) | MultipleChoice | [tasksource/ScienceQA_text_only](https://hf.co/datasets/tasksource/ScienceQA_text_only) |  |
| 304 | [ekar_english](src/tasksource/tasks.py#L1289) | MultipleChoice | [Jiangjie/ekar_english](https://hf.co/datasets/Jiangjie/ekar_english) | ✓ |
| 305 | [implicit-hate-stg1](src/tasksource/tasks.py#L1293) | Classification | [tasksource/implicit-hate-stg1](https://hf.co/datasets/tasksource/implicit-hate-stg1) | ✓ |
| 306 | [chaos-mnli-ambiguity](src/tasksource/tasks.py#L1297) | Classification | [tasksource/chaos-mnli-ambiguity](https://hf.co/datasets/tasksource/chaos-mnli-ambiguity) | ✓ |
| 307 | [headline_cause/en_simple](src/tasksource/tasks.py#L1301) | Classification | [IlyaGusev/headline_cause](https://hf.co/datasets/IlyaGusev/headline_cause) (`json` loader) |  |
| 308 | [logiqa-2.0-nli](src/tasksource/tasks.py#L1306) | Classification | [tasksource/logiqa-2.0-nli](https://hf.co/datasets/tasksource/logiqa-2.0-nli) |  |
| 309 | [oasst2_dense_flat/quality](src/tasksource/tasks.py#L1311) | Classification | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) | ✓ |
| 310 | [oasst2_dense_flat/toxicity](src/tasksource/tasks.py#L1313) | Classification | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) | ✓ |
| 311 | [oasst2_dense_flat/helpfulness](src/tasksource/tasks.py#L1315) | Classification | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) | ✓ |
| 312 | [mindgames](src/tasksource/tasks.py#L1318) | Classification | [sileod/mindgames](https://hf.co/datasets/sileod/mindgames) |  |
| 313 | [universal_dependencies/en_partut/deprel](src/tasksource/tasks.py#L1332) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) | ✓ |
| 314 | [universal_dependencies/en_lines/deprel](src/tasksource/tasks.py#L1332) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) | ✓ |
| 315 | [universal_dependencies/en_gum/deprel](src/tasksource/tasks.py#L1332) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) | ✓ |
| 316 | [universal_dependencies/en_ewt/deprel](src/tasksource/tasks.py#L1332) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) | ✓ |
| 317 | [ambient](src/tasksource/tasks.py#L1339) | Classification | [tasksource/ambient](https://hf.co/datasets/tasksource/ambient) | ✓ |
| 318 | [path-naturalness-prediction](src/tasksource/tasks.py#L1344) | MultipleChoice | [tasksource/path-naturalness-prediction](https://hf.co/datasets/tasksource/path-naturalness-prediction) | ✓ |
| 319 | [civil_comments/toxicity](src/tasksource/tasks.py#L1354) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 320 | [civil_comments/severe_toxicity](src/tasksource/tasks.py#L1355) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 321 | [civil_comments/obscene](src/tasksource/tasks.py#L1356) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 322 | [civil_comments/threat](src/tasksource/tasks.py#L1357) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 323 | [civil_comments/insult](src/tasksource/tasks.py#L1358) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 324 | [civil_comments/identity_attack](src/tasksource/tasks.py#L1359) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 325 | [civil_comments/sexual_explicit](src/tasksource/tasks.py#L1360) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 326 | [cloth](src/tasksource/tasks.py#L1362) | MultipleChoice | [AndyChiang/cloth](https://hf.co/datasets/AndyChiang/cloth) |  |
| 327 | [dgen](src/tasksource/tasks.py#L1363) | MultipleChoice | [AndyChiang/dgen](https://hf.co/datasets/AndyChiang/dgen) |  |
| 328 | [I2D2](src/tasksource/tasks.py#L1365) | Classification | [tasksource/I2D2](https://hf.co/datasets/tasksource/I2D2) | ✓ |
| 329 | [args_me](src/tasksource/tasks.py#L1370) | Classification | [webis/args_me](https://hf.co/datasets/webis/args_me) | ✓ |
| 330 | [Touche23-ValueEval](src/tasksource/tasks.py#L1375) | Classification | [webis/Touche23-ValueEval](https://hf.co/datasets/webis/Touche23-ValueEval) (`csv` loader) | ✓ |
| 331 | [starcon](src/tasksource/tasks.py#L1384) | Classification | [tasksource/starcon](https://hf.co/datasets/tasksource/starcon) | ✓ |
| 332 | [banking77](src/tasksource/tasks.py#L1389) | Classification | [legacy-datasets/banking77](https://hf.co/datasets/legacy-datasets/banking77) |  |
| 333 | [it-support-tickets](src/tasksource/tasks.py#L1391) | Classification | [tasksource/it-support-tickets](https://hf.co/datasets/tasksource/it-support-tickets) |  |
| 334 | [ConTRoL-nli](src/tasksource/tasks.py#L1395) | Classification | [tasksource/ConTRoL-nli](https://hf.co/datasets/tasksource/ConTRoL-nli) |  |
| 335 | [tracie](src/tasksource/tasks.py#L1397) | Classification | [tasksource/tracie](https://hf.co/datasets/tasksource/tracie) | ✓ |
| 336 | [sherliic](src/tasksource/tasks.py#L1399) | Classification | [tasksource/sherliic](https://hf.co/datasets/tasksource/sherliic) |  |
| 337 | [sen-making/1](src/tasksource/tasks.py#L1401) | MultipleChoice | [tasksource/sen-making](https://hf.co/datasets/tasksource/sen-making) | ✓ |
| 338 | [sen-making/2](src/tasksource/tasks.py#L1405) | MultipleChoice | [tasksource/sen-making](https://hf.co/datasets/tasksource/sen-making) | ✓ |
| 339 | [winowhy](src/tasksource/tasks.py#L1408) | Classification | [tasksource/winowhy](https://hf.co/datasets/tasksource/winowhy) | ✓ |
| 340 | [robustLR](src/tasksource/tasks.py#L1412) | Classification | [tasksource/robustLR](https://hf.co/datasets/tasksource/robustLR) |  |
| 341 | [clutrr](src/tasksource/tasks.py#L1414) | Classification | [tasksource/clutrr](https://hf.co/datasets/tasksource/clutrr) |  |
| 342 | [logical-fallacy](src/tasksource/tasks.py#L1416) | Classification | [tasksource/logical-fallacy](https://hf.co/datasets/tasksource/logical-fallacy) |  |
| 343 | [parade](src/tasksource/tasks.py#L1418) | Classification | [tasksource/parade](https://hf.co/datasets/tasksource/parade) |  |
| 344 | [subjectivity](src/tasksource/tasks.py#L1421) | Classification | [tasksource/subjectivity](https://hf.co/datasets/tasksource/subjectivity) | ✓ |
| 345 | [MOH](src/tasksource/tasks.py#L1427) | Classification | [tasksource/MOH](https://hf.co/datasets/tasksource/MOH) | ✓ |
| 346 | [VUAC](src/tasksource/tasks.py#L1428) | Classification | [tasksource/VUAC](https://hf.co/datasets/tasksource/VUAC) | ✓ |
| 347 | [TroFi](src/tasksource/tasks.py#L1429) | Classification | [tasksource/TroFi](https://hf.co/datasets/tasksource/TroFi) (`parquet` loader) | ✓ |
| 348 | [sharc](src/tasksource/tasks.py#L1437) | Classification | [tasksource/sharc](https://hf.co/datasets/tasksource/sharc) |  |
| 349 | [conceptrules_v2](src/tasksource/tasks.py#L1441) | Classification | [tasksource/conceptrules_v2](https://hf.co/datasets/tasksource/conceptrules_v2) | ✓ |
| 350 | [disrpt/eng.dep.scidtb.rels](src/tasksource/tasks.py#L1443) | Classification | [multilingual-discourse-hub/disrpt](https://hf.co/datasets/multilingual-discourse-hub/disrpt) | ✓ |
| 351 | [conll2000](src/tasksource/tasks.py#L1446) | TokenClassification | [eriktks/conll2000](https://hf.co/datasets/eriktks/conll2000) |  |
| 352 | [few-nerd/supervised](src/tasksource/tasks.py#L1449) | TokenClassification | [DFKI-SLT/few-nerd](https://hf.co/datasets/DFKI-SLT/few-nerd) |  |
| 353 | [finer-139](src/tasksource/tasks.py#L1450) | TokenClassification | [nlpaueb/finer-139](https://hf.co/datasets/nlpaueb/finer-139) |  |
| 354 | [zero-shot-label-nli](src/tasksource/tasks.py#L1453) | Classification | [tasksource/zero-shot-label-nli](https://hf.co/datasets/tasksource/zero-shot-label-nli) | ✓ |
| 355 | [com2sense](src/tasksource/tasks.py#L1455) | Classification | [tasksource/com2sense](https://hf.co/datasets/tasksource/com2sense) |  |
| 356 | [scone](src/tasksource/tasks.py#L1457) | Classification | [tasksource/scone](https://hf.co/datasets/tasksource/scone) |  |
| 357 | [winodict](src/tasksource/tasks.py#L1459) | MultipleChoice | [tasksource/winodict](https://hf.co/datasets/tasksource/winodict) |  |
| 358 | [fool-me-twice](src/tasksource/tasks.py#L1461) | Classification | [tasksource/fool-me-twice](https://hf.co/datasets/tasksource/fool-me-twice) |  |
| 359 | [monli](src/tasksource/tasks.py#L1465) | Classification | [tasksource/monli](https://hf.co/datasets/tasksource/monli) |  |
| 360 | [corr2cause](src/tasksource/tasks.py#L1467) | Classification | [tasksource/corr2cause](https://hf.co/datasets/tasksource/corr2cause) |  |
| 361 | [lsat_qa/all](src/tasksource/tasks.py#L1469) | MultipleChoice | [lighteval/lsat_qa](https://hf.co/datasets/lighteval/lsat_qa) |  |
| 362 | [apt](src/tasksource/tasks.py#L1471) | Classification | [tasksource/apt](https://hf.co/datasets/tasksource/apt) |  |
| 363 | [twitter-financial-news-sentiment](src/tasksource/tasks.py#L1474) | Classification | [zeroshot/twitter-financial-news-sentiment](https://hf.co/datasets/zeroshot/twitter-financial-news-sentiment) |  |
| 364 | [icl-symbol-tuning-instruct](src/tasksource/tasks.py#L1481) | Classification | [tasksource/icl-symbol-tuning-instruct](https://hf.co/datasets/tasksource/icl-symbol-tuning-instruct) | ✓ |
| 365 | [SpaceNLI](src/tasksource/tasks.py#L1487) | Classification | [tasksource/SpaceNLI](https://hf.co/datasets/tasksource/SpaceNLI) |  |
| 366 | [propsegment/nli](src/tasksource/tasks.py#L1489) | Classification | [sihaochen/propsegment](https://hf.co/datasets/sihaochen/propsegment) (`json` loader) | ✓ |
| 367 | [HatemojiBuild](src/tasksource/tasks.py#L1499) | Classification | [HannahRoseKirk/HatemojiBuild](https://hf.co/datasets/HannahRoseKirk/HatemojiBuild) |  |
| 368 | [regset](src/tasksource/tasks.py#L1502) | Classification | [tasksource/regset](https://hf.co/datasets/tasksource/regset) | ✓ |
| 369 | [esci](src/tasksource/tasks.py#L1508) | Classification | [tasksource/esci](https://hf.co/datasets/tasksource/esci) |  |
| 370 | [chatbot_arena_conversations](src/tasksource/tasks.py#L1527) | MultipleChoice | [lmsys/chatbot_arena_conversations](https://hf.co/datasets/lmsys/chatbot_arena_conversations) | ✓ |
| 371 | [dnd_style_intents](src/tasksource/tasks.py#L1533) | Classification | [neurae/dnd_style_intents](https://hf.co/datasets/neurae/dnd_style_intents) |  |
| 372 | [FLD.v2/default](src/tasksource/tasks.py#L1536) | Classification | [hitachi-nlp/FLD.v2](https://hf.co/datasets/hitachi-nlp/FLD.v2) | ✓ |
| 373 | [FLD.v2/star](src/tasksource/tasks.py#L1540) | Classification | [hitachi-nlp/FLD.v2](https://hf.co/datasets/hitachi-nlp/FLD.v2) | ✓ |
| 374 | [SDOH-NLI](src/tasksource/tasks.py#L1544) | Classification | [tasksource/SDOH-NLI](https://hf.co/datasets/tasksource/SDOH-NLI) |  |
| 375 | [scifact_entailment](src/tasksource/tasks.py#L1547) | Classification | [tasksource/scifact_entailment](https://hf.co/datasets/tasksource/scifact_entailment) |  |
| 376 | [feasibilityQA](src/tasksource/tasks.py#L1551) | Classification | [tasksource/feasibilityQA](https://hf.co/datasets/tasksource/feasibilityQA) |  |
| 377 | [simple_pair](src/tasksource/tasks.py#L1554) | Classification | [tasksource/simple_pair](https://hf.co/datasets/tasksource/simple_pair) |  |
| 378 | [AdjectiveScaleProbe-nli](src/tasksource/tasks.py#L1555) | Classification | [tasksource/AdjectiveScaleProbe-nli](https://hf.co/datasets/tasksource/AdjectiveScaleProbe-nli) |  |
| 379 | [resnli](src/tasksource/tasks.py#L1556) | Classification | [tasksource/resnli](https://hf.co/datasets/tasksource/resnli) |  |
| 380 | [SpaRTUN](src/tasksource/tasks.py#L1558) | MultipleChoice | [tasksource/SpaRTUN](https://hf.co/datasets/tasksource/SpaRTUN) |  |
| 381 | [ReSQ](src/tasksource/tasks.py#L1563) | MultipleChoice | [tasksource/ReSQ](https://hf.co/datasets/tasksource/ReSQ) |  |
| 382 | [semantic_fragments_nli](src/tasksource/tasks.py#L1568) | Classification | [tasksource/semantic_fragments_nli](https://hf.co/datasets/tasksource/semantic_fragments_nli) |  |
| 383 | [dataset_train_nli](src/tasksource/tasks.py#L1571) | Classification | [MoritzLaurer/dataset_train_nli](https://hf.co/datasets/MoritzLaurer/dataset_train_nli) |  |
| 384 | [stepgame](src/tasksource/tasks.py#L1576) | Classification | [tasksource/stepgame](https://hf.co/datasets/tasksource/stepgame) |  |
| 385 | [nlgraph](src/tasksource/tasks.py#L1584) | Classification | [tasksource/nlgraph](https://hf.co/datasets/tasksource/nlgraph) |  |
| 386 | [oasst2_pairwise_rlhf_reward](src/tasksource/tasks.py#L1588) | MultipleChoice | [tasksource/oasst2_pairwise_rlhf_reward](https://hf.co/datasets/tasksource/oasst2_pairwise_rlhf_reward) | ✓ |
| 387 | [hh-rlhf/helpful-online](src/tasksource/tasks.py#L1599) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| 388 | [hh-rlhf/helpful-rejection-sampled](src/tasksource/tasks.py#L1599) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| 389 | [hh-rlhf/helpful-base](src/tasksource/tasks.py#L1599) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| 390 | [hh-rlhf/harmless-base](src/tasksource/tasks.py#L1603) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| 391 | [ruletaker](src/tasksource/tasks.py#L1607) | Classification | [tasksource/ruletaker](https://hf.co/datasets/tasksource/ruletaker) | ✓ |
| 392 | [PARARULE-Plus](src/tasksource/tasks.py#L1611) | Classification | [qbao775/PARARULE-Plus](https://hf.co/datasets/qbao775/PARARULE-Plus) | ✓ |
| 393 | [proofwriter](src/tasksource/tasks.py#L1615) | Classification | [tasksource/proofwriter](https://hf.co/datasets/tasksource/proofwriter) |  |
| 394 | [logical-entailment](src/tasksource/tasks.py#L1618) | Classification | [tasksource/logical-entailment](https://hf.co/datasets/tasksource/logical-entailment) | ✓ |
| 395 | [nope](src/tasksource/tasks.py#L1621) | Classification | [tasksource/nope](https://hf.co/datasets/tasksource/nope) |  |
| 396 | [LogicNLI](src/tasksource/tasks.py#L1625) | Classification | [tasksource/LogicNLI](https://hf.co/datasets/tasksource/LogicNLI) | ✓ |
| 397 | [contract-nli/contractnli_a/seg](src/tasksource/tasks.py#L1628) | Classification | [tasksource/contract-nli](https://hf.co/datasets/tasksource/contract-nli) |  |
| 398 | [contract-nli/contractnli_b/full](src/tasksource/tasks.py#L1630) | Classification | [tasksource/contract-nli](https://hf.co/datasets/tasksource/contract-nli) |  |
| 399 | [nli4ct_semeval2024](src/tasksource/tasks.py#L1632) | Classification | [AshtonIsNotHere/nli4ct_semeval2024](https://hf.co/datasets/AshtonIsNotHere/nli4ct_semeval2024) |  |
| 400 | [lsat-ar](src/tasksource/tasks.py#L1635) | MultipleChoice | [tasksource/lsat-ar](https://hf.co/datasets/tasksource/lsat-ar) |  |
| 401 | [lsat-rc](src/tasksource/tasks.py#L1640) | MultipleChoice | [tasksource/lsat-rc](https://hf.co/datasets/tasksource/lsat-rc) |  |
| 402 | [biosift-nli](src/tasksource/tasks.py#L1645) | Classification | [AshtonIsNotHere/biosift-nli](https://hf.co/datasets/AshtonIsNotHere/biosift-nli) |  |
| 403 | [brainteasers/WP](src/tasksource/tasks.py#L1649) | MultipleChoice | [tasksource/brainteasers](https://hf.co/datasets/tasksource/brainteasers) |  |
| 404 | [brainteasers/SP](src/tasksource/tasks.py#L1649) | MultipleChoice | [tasksource/brainteasers](https://hf.co/datasets/tasksource/brainteasers) |  |
| 405 | [toxigen-data/annotated](src/tasksource/tasks.py#L1655) | Classification | [skg/toxigen-data](https://hf.co/datasets/skg/toxigen-data) |  |
| 406 | [persuasion](src/tasksource/tasks.py#L1666) | Classification | [Anthropic/persuasion](https://hf.co/datasets/Anthropic/persuasion) |  |
| 407 | [AmbigNQ-clarifying-question](src/tasksource/tasks.py#L1672) | Classification | [erbacher/AmbigNQ-clarifying-question](https://hf.co/datasets/erbacher/AmbigNQ-clarifying-question) | ✓ |
| 408 | [SIGA-nli](src/tasksource/tasks.py#L1677) | Classification | [tasksource/SIGA-nli](https://hf.co/datasets/tasksource/SIGA-nli) | ✓ |
| 409 | [FOL-nli](src/tasksource/tasks.py#L1679) | Classification | [unigram/FOL-nli](https://hf.co/datasets/unigram/FOL-nli) |  |
| 410 | [goal-step-wikihow/goal](src/tasksource/tasks.py#L1681) | MultipleChoice | [tasksource/goal-step-wikihow](https://hf.co/datasets/tasksource/goal-step-wikihow) |  |
| 411 | [goal-step-wikihow/step](src/tasksource/tasks.py#L1684) | MultipleChoice | [tasksource/goal-step-wikihow](https://hf.co/datasets/tasksource/goal-step-wikihow) |  |
| 412 | [goal-step-wikihow/order](src/tasksource/tasks.py#L1687) | MultipleChoice | [tasksource/goal-step-wikihow](https://hf.co/datasets/tasksource/goal-step-wikihow) | ✓ |
| 413 | [PARADISE](src/tasksource/tasks.py#L1691) | MultipleChoice | [GGLab/PARADISE](https://hf.co/datasets/GGLab/PARADISE) |  |
| 414 | [doc-nli](src/tasksource/tasks.py#L1694) | Classification | [tasksource/doc-nli](https://hf.co/datasets/tasksource/doc-nli) |  |
| 415 | [mctest-nli](src/tasksource/tasks.py#L1696) | Classification | [tasksource/mctest-nli](https://hf.co/datasets/tasksource/mctest-nli) |  |
| 416 | [patent-phrase-similarity](src/tasksource/tasks.py#L1698) | Classification | [tasksource/patent-phrase-similarity](https://hf.co/datasets/tasksource/patent-phrase-similarity) | ✓ |
| 417 | [natural-language-satisfiability](src/tasksource/tasks.py#L1701) | Classification | [tasksource/natural-language-satisfiability](https://hf.co/datasets/tasksource/natural-language-satisfiability) |  |
| 418 | [idioms-nli](src/tasksource/tasks.py#L1703) | Classification | [tasksource/idioms-nli](https://hf.co/datasets/tasksource/idioms-nli) |  |
| 419 | [lifecycle-entailment](src/tasksource/tasks.py#L1705) | Classification | [tasksource/lifecycle-entailment](https://hf.co/datasets/tasksource/lifecycle-entailment) |  |
| 420 | [safe-guard-prompt-injection](src/tasksource/tasks.py#L1712) | Classification | [xTRam1/safe-guard-prompt-injection](https://hf.co/datasets/xTRam1/safe-guard-prompt-injection) | ✓ |
| 421 | [prompt-injections](src/tasksource/tasks.py#L1718) | Classification | [deepset/prompt-injections](https://hf.co/datasets/deepset/prompt-injections) | ✓ |
| 422 | [prompt-injection-dataset](src/tasksource/tasks.py#L1724) | Classification | [S-Labs/prompt-injection-dataset](https://hf.co/datasets/S-Labs/prompt-injection-dataset) | ✓ |
| 423 | [Prompt-injection-dataset/full](src/tasksource/tasks.py#L1730) | Classification | [neuralchemy/Prompt-injection-dataset](https://hf.co/datasets/neuralchemy/Prompt-injection-dataset) | ✓ |
| 424 | [PromptShield](src/tasksource/tasks.py#L1736) | Classification | [hendzh/PromptShield](https://hf.co/datasets/hendzh/PromptShield) | ✓ |
| 425 | [shell-safety-v2](src/tasksource/tasks.py#L1742) | Classification | [tomngdev/shell-safety-v2](https://hf.co/datasets/tomngdev/shell-safety-v2) | ✓ |
| 426 | [agent_action_safety](src/tasksource/tasks.py#L1747) | Classification | [karanxa/agent-action-safety-dataset](https://hf.co/datasets/karanxa/agent-action-safety-dataset) (`json` loader) | ✓ |
| 427 | [ShellRisk-Bench](src/tasksource/tasks.py#L1762) | Classification | [kontext-security/ShellRisk-Bench](https://hf.co/datasets/kontext-security/ShellRisk-Bench) | ✓ |
| 428 | [wildguardmix-cleaned/prompt_harm](src/tasksource/tasks.py#L1787) | Classification | [bogdanminko/wildguardmix-cleaned](https://hf.co/datasets/bogdanminko/wildguardmix-cleaned) | ✓ |
| 429 | [wildguardmix-cleaned/response_harm](src/tasksource/tasks.py#L1793) | Classification | [bogdanminko/wildguardmix-cleaned](https://hf.co/datasets/bogdanminko/wildguardmix-cleaned) | ✓ |
| 430 | [wildguardmix-cleaned/response_refusal](src/tasksource/tasks.py#L1798) | Classification | [bogdanminko/wildguardmix-cleaned](https://hf.co/datasets/bogdanminko/wildguardmix-cleaned) | ✓ |
| 431 | [BeaverTails](src/tasksource/tasks.py#L1814) | Classification | [PKU-Alignment/BeaverTails](https://hf.co/datasets/PKU-Alignment/BeaverTails) | ✓ |
| 432 | [privacy-200k-Mistral-Large-3](src/tasksource/tasks.py#L1830) | Classification | [gabrielloiseau/privacy-200k-Mistral-Large-3](https://hf.co/datasets/gabrielloiseau/privacy-200k-Mistral-Large-3) | ✓ |
| 433 | [toxic-chat/toxicchat0124/toxicity](src/tasksource/tasks.py#L1839) | Classification | [lmsys/toxic-chat](https://hf.co/datasets/lmsys/toxic-chat) | ✓ |
| 434 | [toxic-chat/toxicchat0124/jailbreaking](src/tasksource/tasks.py#L1844) | Classification | [lmsys/toxic-chat](https://hf.co/datasets/lmsys/toxic-chat) | ✓ |
| 435 | [clinc_oos/plus](src/tasksource/tasks.py#L1850) | Classification | [clinc/clinc_oos](https://hf.co/datasets/clinc/clinc_oos) |  |
| 436 | [IntentGrasp/all](src/tasksource/tasks.py#L1868) | MultipleChoice | [yuweiyin/IntentGrasp](https://hf.co/datasets/yuweiyin/IntentGrasp) |  |
| 437 | [few_rel/default](src/tasksource/tasks.py#L1906) | Classification | [tasksource/few_rel](https://hf.co/datasets/tasksource/few_rel) |  |
| 438 | [docred](src/tasksource/tasks.py#L1984) | Classification | [YufeiHFUT/DocRED_origin](https://hf.co/datasets/YufeiHFUT/DocRED_origin), [thunlp/docred](https://hf.co/datasets/thunlp/docred) (`json` loader) |  |
| 439 | [chemprot/chemprot_full_source](src/tasksource/tasks.py#L2019) | Classification | [bigbio/chemprot](https://hf.co/datasets/bigbio/chemprot) | ✓ |
| 440 | [PKU-SafeRLHF/helpfulness](src/tasksource/tasks.py#L2025) | MultipleChoice | [PKU-Alignment/PKU-SafeRLHF](https://hf.co/datasets/PKU-Alignment/PKU-SafeRLHF) | ✓ |
| 441 | [PKU-SafeRLHF/safety](src/tasksource/tasks.py#L2030) | MultipleChoice | [PKU-Alignment/PKU-SafeRLHF](https://hf.co/datasets/PKU-Alignment/PKU-SafeRLHF) | ✓ |
| 442 | [HelpSteer/helpfulness](src/tasksource/tasks.py#L2052) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 443 | [HelpSteer/correctness](src/tasksource/tasks.py#L2053) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 444 | [HelpSteer/coherence](src/tasksource/tasks.py#L2054) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 445 | [HelpSteer/complexity](src/tasksource/tasks.py#L2055) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 446 | [HelpSteer/verbosity](src/tasksource/tasks.py#L2056) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 447 | [HelpSteer2/helpfulness](src/tasksource/tasks.py#L2058) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 448 | [HelpSteer2/correctness](src/tasksource/tasks.py#L2059) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 449 | [HelpSteer2/coherence](src/tasksource/tasks.py#L2060) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 450 | [HelpSteer2/complexity](src/tasksource/tasks.py#L2061) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 451 | [HelpSteer2/verbosity](src/tasksource/tasks.py#L2062) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 452 | [HelpSteer3/preference](src/tasksource/tasks.py#L2067) | MultipleChoice | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | ✓ |
| 453 | [HelpSteer3/preference_strength](src/tasksource/tasks.py#L2080) | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | ✓ |
| 454 | [HelpSteer3/principle](src/tasksource/tasks.py#L2092) | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) |  |
| 455 | [HelpSteer3/edit_quality](src/tasksource/tasks.py#L2097) | MultipleChoice | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | ✓ |
| 456 | [HelpSteer3/feedback](src/tasksource/tasks.py#L2119) | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | ✓ |
| 457 | [MSciNLI](src/tasksource/tasks.py#L2125) | Classification | [sadat2307/MSciNLI](https://hf.co/datasets/sadat2307/MSciNLI) |  |
| 458 | [UltraFeedback-paired](src/tasksource/tasks.py#L2128) | MultipleChoice | [pushpdeep/UltraFeedback-paired](https://hf.co/datasets/pushpdeep/UltraFeedback-paired) | ✓ |
| 459 | [prm800k_dpo/solution](src/tasksource/tasks.py#L2132) | MultipleChoice | [tasksource/prm800k_dpo](https://hf.co/datasets/tasksource/prm800k_dpo) | ✓ |
| 460 | [prm800k_dpo/step](src/tasksource/tasks.py#L2135) | MultipleChoice | [tasksource/prm800k_dpo](https://hf.co/datasets/tasksource/prm800k_dpo) | ✓ |
| 461 | [AES2-essay-scoring](src/tasksource/tasks.py#L2139) | Classification | [tasksource/AES2-essay-scoring](https://hf.co/datasets/tasksource/AES2-essay-scoring) | ✓ |
| 462 | [argument-feedback](src/tasksource/tasks.py#L2143) | Classification | [tasksource/argument-feedback](https://hf.co/datasets/tasksource/argument-feedback) | ✓ |
| 463 | [english-grading/cohesion](src/tasksource/tasks.py#L2150) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 464 | [english-grading/syntax](src/tasksource/tasks.py#L2151) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 465 | [english-grading/vocabulary](src/tasksource/tasks.py#L2152) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 466 | [english-grading/phraseology](src/tasksource/tasks.py#L2153) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 467 | [english-grading/grammar](src/tasksource/tasks.py#L2154) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 468 | [english-grading/conventions](src/tasksource/tasks.py#L2155) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 469 | [wice](src/tasksource/tasks.py#L2157) | Classification | [tasksource/wice](https://hf.co/datasets/tasksource/wice) |  |
| 470 | [hover](src/tasksource/tasks.py#L2160) | Classification | [Dzeniks/hover](https://hf.co/datasets/Dzeniks/hover) |  |
| 471 | [hover-3way/nli](src/tasksource/tasks.py#L2164) | Classification | [Dzeniks/hover-3way](https://hf.co/datasets/Dzeniks/hover-3way) | ✓ |
| 472 | [tasksource_dpo_pairs](src/tasksource/tasks.py#L2167) | MultipleChoice | [tasksource/tasksource_dpo_pairs](https://hf.co/datasets/tasksource/tasksource_dpo_pairs) | ✓ |
| 473 | [seahorse_summarization_evaluation](src/tasksource/tasks.py#L2170) | Classification | [tasksource/seahorse_summarization_evaluation](https://hf.co/datasets/tasksource/seahorse_summarization_evaluation) |  |
| 474 | [missing-item-prediction/contrastive](src/tasksource/tasks.py#L2182) | Classification | [sileod/missing-item-prediction](https://hf.co/datasets/sileod/missing-item-prediction) | ✓ |
| 475 | [jigsaw_toxicity](src/tasksource/tasks.py#L2188) | Classification | [tasksource/jigsaw_toxicity](https://hf.co/datasets/tasksource/jigsaw_toxicity) |  |
| 476 | [Pol_NLI](src/tasksource/tasks.py#L2191) | Classification | [mlburnham/Pol_NLI](https://hf.co/datasets/mlburnham/Pol_NLI) |  |
| 477 | [synthetic-retrieval-NLI/binary](src/tasksource/tasks.py#L2194) | Classification | [tasksource/synthetic-retrieval-NLI](https://hf.co/datasets/tasksource/synthetic-retrieval-NLI) |  |
| 478 | [synthetic-retrieval-NLI/count](src/tasksource/tasks.py#L2194) | Classification | [tasksource/synthetic-retrieval-NLI](https://hf.co/datasets/tasksource/synthetic-retrieval-NLI) |  |
| 479 | [synthetic-retrieval-NLI/position](src/tasksource/tasks.py#L2194) | Classification | [tasksource/synthetic-retrieval-NLI](https://hf.co/datasets/tasksource/synthetic-retrieval-NLI) |  |
| 480 | [github-issue-similarity](src/tasksource/tasks.py#L2203) | Classification | [WhereIsAI/github-issue-similarity](https://hf.co/datasets/WhereIsAI/github-issue-similarity) |  |
| 481 | [webinstruct/mc](src/tasksource/tasks.py#L2408) | MultipleChoice | [tasksource/webinstruct](https://hf.co/datasets/tasksource/webinstruct) |  |
| 482 | [webinstruct/binary](src/tasksource/tasks.py#L2410) | Classification | [tasksource/webinstruct](https://hf.co/datasets/tasksource/webinstruct) |  |
| 483 | [weblinx/action](src/tasksource/tasks.py#L2474) | Classification | [McGill-NLP/WebLINX](https://hf.co/datasets/McGill-NLP/WebLINX) | ✓ |
| 484 | [weblinx/dom-element](src/tasksource/tasks.py#L2477) | MultipleChoice | [McGill-NLP/WebLINX](https://hf.co/datasets/McGill-NLP/WebLINX) | ✓ |
| 485 | [websrc/yesno](src/tasksource/tasks.py#L2483) | Classification | [tasksource/websrc](https://hf.co/datasets/tasksource/websrc) |  |
| 486 | [websrc/element](src/tasksource/tasks.py#L2485) | MultipleChoice | [tasksource/websrc](https://hf.co/datasets/tasksource/websrc) | ✓ |
| 487 | [mind2web/action](src/tasksource/tasks.py#L2492) | Classification | [tasksource/mind2web-dom](https://hf.co/datasets/tasksource/mind2web-dom) | ✓ |
| 488 | [mind2web/dom-element](src/tasksource/tasks.py#L2495) | MultipleChoice | [tasksource/mind2web-dom](https://hf.co/datasets/tasksource/mind2web-dom) | ✓ |
| 489 | [sentfin](src/tasksource/tasks.py#L2500) | Classification | [tasksource/sentfin](https://hf.co/datasets/tasksource/sentfin) | ✓ |
| 490 | [wands](src/tasksource/tasks.py#L2504) | Classification | [tasksource/wands](https://hf.co/datasets/tasksource/wands) | ✓ |
| 491 | [scirepeval/search](src/tasksource/tasks.py#L2508) | Classification | [tasksource/scirepeval-search](https://hf.co/datasets/tasksource/scirepeval-search) | ✓ |
| 492 | [maud/consideration](src/tasksource/tasks.py#L2560) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 493 | [maud/bringdown-timing](src/tasksource/tasks.py#L2564) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 494 | [maud/ordinary-course-consent](src/tasksource/tasks.py#L2568) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 495 | [maud/ordinary-course-efforts](src/tasksource/tasks.py#L2572) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 496 | [maud/antitrust-efforts](src/tasksource/tasks.py#L2576) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 497 | [maud/specific-performance](src/tasksource/tasks.py#L2580) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 498 | [ddi/relation](src/tasksource/tasks.py#L2585) | Classification | [OpenMed/DDI-Corpus-Processed](https://hf.co/datasets/OpenMed/DDI-Corpus-Processed) | ✓ |
| 499 | [patent-classification/abstract](src/tasksource/tasks.py#L2592) | Classification | [ccdv/patent-classification](https://hf.co/datasets/ccdv/patent-classification) | ✓ |
| 500 | [tabfact](src/tasksource/tasks.py#L2599) | Classification | [wenhu/tab_fact](https://hf.co/datasets/wenhu/tab_fact) | ✓ |
| 501 | [infotabs](src/tasksource/tasks.py#L2607) | Classification | [table-benchmark/infotabs](https://hf.co/datasets/table-benchmark/infotabs) | ✓ |
| 502 | [squad-v2/answerability](src/tasksource/tasks.py#L2619) | Classification | [rajpurkar/squad_v2](https://hf.co/datasets/rajpurkar/squad_v2) | ✓ |
| 503 | [cuad/answerability](src/tasksource/tasks.py#L2624) | Classification | [theatticusproject/cuad-qa](https://hf.co/datasets/theatticusproject/cuad-qa) | ✓ |
| 504 | [hallmarks-of-cancer/yesno](src/tasksource/tasks.py#L2643) | Classification | [bigbio/hallmarks_of_cancer](https://hf.co/datasets/bigbio/hallmarks_of_cancer) | ✓ |
| 505 | [semeval2014-absa/restaurants](src/tasksource/tasks.py#L2665) | Classification | [alexcadillon/SemEval2014Task4](https://hf.co/datasets/alexcadillon/SemEval2014Task4) | ✓ |
| 506 | [semeval2014-absa/laptops](src/tasksource/tasks.py#L2665) | Classification | [alexcadillon/SemEval2014Task4](https://hf.co/datasets/alexcadillon/SemEval2014Task4) | ✓ |
| 507 | [record/candidate](src/tasksource/tasks.py#L2687) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) | ✓ |
| 508 | [multidogo/finance/intent](src/tasksource/tasks.py#L2710) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 509 | [multidogo/fastfood/intent](src/tasksource/tasks.py#L2710) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 510 | [multidogo/airline/intent](src/tasksource/tasks.py#L2710) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 511 | [multidogo/software/intent](src/tasksource/tasks.py#L2710) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 512 | [multidogo/media/intent](src/tasksource/tasks.py#L2710) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 513 | [multidogo/insurance/intent](src/tasksource/tasks.py#L2710) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 514 | [wikisql/aggregation](src/tasksource/tasks.py#L2738) | Classification | [Salesforce/wikisql](https://hf.co/datasets/Salesforce/wikisql) | ✓ |
| 515 | [wikisql/column](src/tasksource/tasks.py#L2743) | MultipleChoice | [Salesforce/wikisql](https://hf.co/datasets/Salesforce/wikisql) | ✓ |
| 516 | [wikisql/operator](src/tasksource/tasks.py#L2748) | Classification | [Salesforce/wikisql](https://hf.co/datasets/Salesforce/wikisql) | ✓ |
| 517 | [acord](src/tasksource/tasks.py#L2753) | Classification | [tasksource/acord](https://hf.co/datasets/tasksource/acord) | ✓ |
| 518 | [edos](src/tasksource/tasks.py#L2757) | Classification | [tasksource/edos](https://hf.co/datasets/tasksource/edos) | ✓ |
| 519 | [evidence-inference](src/tasksource/tasks.py#L2760) | Classification | [tasksource/evidence-inference](https://hf.co/datasets/tasksource/evidence-inference) | ✓ |
| 520 | [quac/answerability](src/tasksource/tasks.py#L2782) | Classification | [allenai/quac](https://hf.co/datasets/allenai/quac) | ✓ |
| 521 | [hatexplain/majority](src/tasksource/tasks.py#L2790) | Classification | [Hate-speech-CNERG/hatexplain](https://hf.co/datasets/Hate-speech-CNERG/hatexplain) | ✓ |
| 522 | [hotpotqa/supporting-fact](src/tasksource/tasks.py#L2807) | Classification | [hotpotqa/hotpot_qa](https://hf.co/datasets/hotpotqa/hotpot_qa) | ✓ |
| 523 | [multiwoz22/dialogue-act](src/tasksource/tasks.py#L2851) | Classification | [pfb30/multi_woz_v22](https://hf.co/datasets/pfb30/multi_woz_v22) | ✓ |
| 524 | [multiwoz22/categorical-slot](src/tasksource/tasks.py#L2854) | Classification | [pfb30/multi_woz_v22](https://hf.co/datasets/pfb30/multi_woz_v22) | ✓ |
| 525 | [schema-guided/intent](src/tasksource/tasks.py#L2857) | Classification | [google-research-datasets/schema_guided_dstc8](https://hf.co/datasets/google-research-datasets/schema_guided_dstc8) | ✓ |
| 526 | [coqa/yesno](src/tasksource/tasks.py#L2874) | Classification | [stanfordnlp/coqa](https://hf.co/datasets/stanfordnlp/coqa) | ✓ |
| 527 | [qasper/yesno](src/tasksource/tasks.py#L2895) | Classification | [allenai/qasper](https://hf.co/datasets/allenai/qasper) | ✓ |
| 528 | [topv2/intent](src/tasksource/tasks.py#L2907) | Classification | [WillHeld/top_v2](https://hf.co/datasets/WillHeld/top_v2) | ✓ |
| 529 | [slurp/intent](src/tasksource/tasks.py#L2915) | Classification | [tasksource/slurp](https://hf.co/datasets/tasksource/slurp) | ✓ |
| 530 | [mcscript2](src/tasksource/tasks.py#L2919) | MultipleChoice | [tasksource/mcscript2](https://hf.co/datasets/tasksource/mcscript2) |  |
| 531 | [fnc1/stance](src/tasksource/tasks.py#L2924) | Classification | [tasksource/fnc1](https://hf.co/datasets/tasksource/fnc1) | ✓ |
| 532 | [perspectrum/stance](src/tasksource/tasks.py#L2929) | Classification | [tasksource/perspectrum](https://hf.co/datasets/tasksource/perspectrum) | ✓ |
| 533 | [ukpconvarg/convincingness](src/tasksource/tasks.py#L2934) | MultipleChoice | [tasksource/ukpconvarg](https://hf.co/datasets/tasksource/ukpconvarg) | ✓ |
| 534 | [discogem/relation](src/tasksource/tasks.py#L2939) | Classification | [tasksource/discogem](https://hf.co/datasets/tasksource/discogem) | ✓ |
| 535 | [pubmed-rct/role](src/tasksource/tasks.py#L2947) | Classification | [tasksource/pubmed-rct](https://hf.co/datasets/tasksource/pubmed-rct) | ✓ |
| 536 | [maven-arg/role](src/tasksource/tasks.py#L2975) | Classification | [mneb/maven-arg](https://hf.co/datasets/mneb/maven-arg) | ✓ |
| 537 | [maven/event](src/tasksource/tasks.py#L3006) | Classification | [mneb/maven](https://hf.co/datasets/mneb/maven) | ✓ |
| 538 | [empathetic-dialogues/emotion](src/tasksource/tasks.py#L3010) | Classification | [facebook/empathetic_dialogues](https://hf.co/datasets/facebook/empathetic_dialogues) | ✓ |
| 539 | [olid/offensive](src/tasksource/tasks.py#L3018) | Classification | [christophsonntag/OLID](https://hf.co/datasets/christophsonntag/OLID) | ✓ |
| 540 | [isarcasm/intended](src/tasksource/tasks.py#L3024) | Classification | [viethq1906/isarcasm_2022_taskA_En](https://hf.co/datasets/viethq1906/isarcasm_2022_taskA_En) | ✓ |
| 541 | [magpie/idiomatic](src/tasksource/tasks.py#L3049) | Classification | [idioms-collaboration/magpie](https://hf.co/datasets/idioms-collaboration/magpie) | ✓ |
| 542 | [preco/coreference](src/tasksource/tasks.py#L3085) | Classification | [coref-data/preco_raw](https://hf.co/datasets/coref-data/preco_raw) | ✓ |
| 543 | [redial/preference](src/tasksource/tasks.py#L3105) | Classification | [community-datasets/re_dial](https://hf.co/datasets/community-datasets/re_dial) | ✓ |
| 544 | [sst5](src/tasksource/tasks.py#L3110) | Classification | [SetFit/sst5](https://hf.co/datasets/SetFit/sst5) | ✓ |
| 545 | [spider/column](src/tasksource/tasks.py#L3118) | MultipleChoice | [tasksource/spider](https://hf.co/datasets/tasksource/spider) | ✓ |
| 546 | [semcor](src/tasksource/tasks.py#L3122) | MultipleChoice | [tasksource/semcor](https://hf.co/datasets/tasksource/semcor) | ✓ |
| 547 | [streusle/noun](src/tasksource/tasks.py#L3125) | Classification | [tasksource/streusle](https://hf.co/datasets/tasksource/streusle) | ✓ |
| 548 | [streusle/verb](src/tasksource/tasks.py#L3129) | Classification | [tasksource/streusle](https://hf.co/datasets/tasksource/streusle) | ✓ |
| 549 | [maven-ere/causal](src/tasksource/tasks.py#L3172) | Classification | [Nofing/Maven-ERE-span](https://hf.co/datasets/Nofing/Maven-ERE-span) | ✓ |
| 550 | [biored/relation](src/tasksource/tasks.py#L3194) | Classification | [gyorilab/BioRED](https://hf.co/datasets/gyorilab/BioRED) | ✓ |
| 551 | [matres/temporal](src/tasksource/tasks.py#L3201) | Classification | [tasksource/matres](https://hf.co/datasets/tasksource/matres) | ✓ |
| 552 | [causal-news](src/tasksource/tasks.py#L3210) | Classification | [thagen/CNCv2](https://hf.co/datasets/thagen/CNCv2) | ✓ |
| 553 | [slurp/slot-type](src/tasksource/tasks.py#L3257) | Classification | [tasksource/slurp](https://hf.co/datasets/tasksource/slurp) | ✓ |
| 554 | [topv2/slot-type](src/tasksource/tasks.py#L3262) | Classification | [WillHeld/top_v2](https://hf.co/datasets/WillHeld/top_v2) | ✓ |
| 555 | [wikiqa](src/tasksource/tasks.py#L3275) | Classification | [microsoft/wiki_qa](https://hf.co/datasets/microsoft/wiki_qa) | ✓ |
| 556 | [nevir](src/tasksource/tasks.py#L3290) | MultipleChoice | [orionweller/NevIR](https://hf.co/datasets/orionweller/NevIR) | ✓ |
| 557 | [antique](src/tasksource/tasks.py#L3298) | Classification | [tasksource/antique](https://hf.co/datasets/tasksource/antique) | ✓ |
| 558 | [causal-reasoning/transitivity](src/tasksource/tasks.py#L3324) | Classification | [ludwigw/causal-reasoning-benchmarks](https://hf.co/datasets/ludwigw/causal-reasoning-benchmarks) | ✓ |
| 559 | [cebab/rating](src/tasksource/tasks.py#L3341) | Classification | [EleutherAI/CEBaB](https://hf.co/datasets/EleutherAI/CEBaB) | ✓ |
| 560 | [cebab/food](src/tasksource/tasks.py#L3345) | Classification | [EleutherAI/CEBaB](https://hf.co/datasets/EleutherAI/CEBaB) | ✓ |
| 561 | [cebab/ambiance](src/tasksource/tasks.py#L3348) | Classification | [EleutherAI/CEBaB](https://hf.co/datasets/EleutherAI/CEBaB) | ✓ |
| 562 | [cebab/service](src/tasksource/tasks.py#L3351) | Classification | [EleutherAI/CEBaB](https://hf.co/datasets/EleutherAI/CEBaB) | ✓ |
| 563 | [cebab/noise](src/tasksource/tasks.py#L3354) | Classification | [EleutherAI/CEBaB](https://hf.co/datasets/EleutherAI/CEBaB) | ✓ |
| 564 | [fabsa/sentiment](src/tasksource/tasks.py#L3367) | Classification | [jordiclive/FABSA](https://hf.co/datasets/jordiclive/FABSA) | ✓ |
| 565 | [tab/identifier](src/tasksource/tasks.py#L3384) | Classification | [doshimit3015/text-anonymization-benchmark](https://hf.co/datasets/doshimit3015/text-anonymization-benchmark) | ✓ |
| 566 | [newsmtsc](src/tasksource/tasks.py#L3399) | Classification | [fhamborg/news_sentiment_newsmtsc](https://hf.co/datasets/fhamborg/news_sentiment_newsmtsc) | ✓ |
| 567 | [sentihood/sentiment](src/tasksource/tasks.py#L3427) | Classification | [bhavnicksm/sentihood](https://hf.co/datasets/bhavnicksm/sentihood) | ✓ |
| 568 | [mams/atsa](src/tasksource/tasks.py#L3444) | Classification | [NEUDM/mams](https://hf.co/datasets/NEUDM/mams) | ✓ |
| 569 | [quality](src/tasksource/tasks.py#L3452) | MultipleChoice | [emozilla/quality](https://hf.co/datasets/emozilla/quality) |  |
| 570 | [clear/readability](src/tasksource/tasks.py#L3461) | Classification | [casey-martin/CommonLit-Ease-of-Readability](https://hf.co/datasets/casey-martin/CommonLit-Ease-of-Readability) | ✓ |
| 571 | [esconv/strategy](src/tasksource/tasks.py#L3474) | Classification | [tasksource/esconv](https://hf.co/datasets/tasksource/esconv) | ✓ |
| 572 | [prosocial-dialog/safety](src/tasksource/tasks.py#L3495) | Classification | [allenai/prosocial-dialog](https://hf.co/datasets/allenai/prosocial-dialog) | ✓ |
| 573 | [gap](src/tasksource/tasks.py#L3515) | Classification | [coref-data/gap_raw](https://hf.co/datasets/coref-data/gap_raw) | ✓ |
| 574 | [trip/cloze](src/tasksource/tasks.py#L3523) | MultipleChoice | [sled-umich/TRIP](https://hf.co/datasets/sled-umich/TRIP) | ✓ |
| 575 | [trip/order](src/tasksource/tasks.py#L3532) | MultipleChoice | [sled-umich/TRIP](https://hf.co/datasets/sled-umich/TRIP) | ✓ |
| 576 | [sbic/offensiveness](src/tasksource/tasks.py#L3542) | Classification | [tasksource/sbic](https://hf.co/datasets/tasksource/sbic) | ✓ |
| 577 | [sbic/intent](src/tasksource/tasks.py#L3549) | Classification | [tasksource/sbic](https://hf.co/datasets/tasksource/sbic) | ✓ |
| 578 | [faithdial/grounding](src/tasksource/tasks.py#L3571) | Classification | [McGill-NLP/FaithDial](https://hf.co/datasets/McGill-NLP/FaithDial) | ✓ |
| 579 | [ragtruth/grounding](src/tasksource/tasks.py#L3588) | Classification | [wandb/RAGTruth-processed](https://hf.co/datasets/wandb/RAGTruth-processed) | ✓ |
| 580 | [comve](src/tasksource/tasks.py#L3600) | MultipleChoice | [tasksource/comve](https://hf.co/datasets/tasksource/comve) |  |

## Soft labels

Annotations whose label is a distribution (annotator votes, rater shares, survey counts), loaded with `load_task(id, soft=True)`. Those with a hard view are also listed above, by their majority label; the others have soft labels only. `votes` are shares of annotators, `mean` a numeric value interpolated over score anchors (see each task for its meaning); annotators is the typical count per item (vote shares from fewer than five are coarse, and the Jev build leaves them out).

| id | kind | aggregation | annotators | default view | dataset |
|---|---|---|--:|---|---|
| [glue/stsb](src/tasksource/tasks.py#L49) | score | mean |  | regression | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |
| [sick/relatedness](src/tasksource/tasks.py#L88) | score | mean | 10 | regression | [tasksource/sick](https://hf.co/datasets/tasksource/sick) |
| [google_wellformed_query](src/tasksource/tasks.py#L847) | noul | votes | 5 | Classification | [tasksource/google_wellformed_query](https://hf.co/datasets/tasksource/google_wellformed_query) |
| [sts-companion](src/tasksource/tasks.py#L1041) | score | mean |  | regression | [tasksource/sts-companion](https://hf.co/datasets/tasksource/sts-companion) |
| [acceptability-prediction/rating_votes](src/tasksource/tasks.py#L1272) | score | votes | 15 |  | [tasksource/acceptability-prediction](https://hf.co/datasets/tasksource/acceptability-prediction) |
| [acceptability-prediction/binary_votes](src/tasksource/tasks.py#L1277) | noul | votes | 15 |  | [tasksource/acceptability-prediction](https://hf.co/datasets/tasksource/acceptability-prediction) |
| [HelpSteer3/individual_preferences](src/tasksource/tasks.py#L2086) | score | votes | 3 |  | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) |
| [HelpSteer3/feedback](src/tasksource/tasks.py#L2119) | score | votes | 3 | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) |
| [proto_qa/proto_qa](src/tasksource/tasks.py#L2213) | choice | votes | 100 |  | [community-datasets/proto_qa](https://hf.co/datasets/community-datasets/proto_qa) |
| [UNLI](src/tasksource/tasks.py#L2227) | noul | mean | 2 |  | [Zhengping/UNLI](https://hf.co/datasets/Zhengping/UNLI) |
| [chaos-mnli-ambiguity/votes](src/tasksource/tasks.py#L2231) | choice | votes | 100 |  | [tasksource/chaos-mnli-ambiguity](https://hf.co/datasets/tasksource/chaos-mnli-ambiguity) |
| [hate_speech_offensive/votes](src/tasksource/tasks.py#L2235) | choice | votes | 3 |  | [tdavidson/hate_speech_offensive](https://hf.co/datasets/tdavidson/hate_speech_offensive) |
| [scruples/verdict_votes](src/tasksource/tasks.py#L2243) | choice | votes | 8 |  | [tasksource/scruples](https://hf.co/datasets/tasksource/scruples) |
| [BeaverTails/unsafe_votes](src/tasksource/tasks.py#L2257) | noul | votes | 3 |  | [PKU-Alignment/BeaverTails](https://hf.co/datasets/PKU-Alignment/BeaverTails) |
| [dynasent/r1_votes](src/tasksource/tasks.py#L2268) | choice | votes | 5 |  | [dynabench/dynasent](https://hf.co/datasets/dynabench/dynasent) |
| [dynasent/r2_votes](src/tasksource/tasks.py#L2272) | choice | votes | 5 |  | [dynabench/dynasent](https://hf.co/datasets/dynabench/dynasent) |
| [hatexplain/votes](src/tasksource/tasks.py#L2277) | choice | votes | 3 |  | [Hate-speech-CNERG/hatexplain](https://hf.co/datasets/Hate-speech-CNERG/hatexplain) |
| [measuring-hate-speech/sentiment](src/tasksource/tasks.py#L2292) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/respect](src/tasksource/tasks.py#L2294) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/insult](src/tasksource/tasks.py#L2296) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/humiliate](src/tasksource/tasks.py#L2297) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/status](src/tasksource/tasks.py#L2298) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/dehumanize](src/tasksource/tasks.py#L2300) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/violence](src/tasksource/tasks.py#L2302) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/genocide](src/tasksource/tasks.py#L2303) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/attack_defend](src/tasksource/tasks.py#L2305) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/hatespeech](src/tasksource/tasks.py#L2307) | choice | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [wikipedia-detox/attack](src/tasksource/tasks.py#L2319) | choice | votes | 10 |  | [tasksource/wikipedia-detox-votes](https://hf.co/datasets/tasksource/wikipedia-detox-votes) |
| [wikipedia-detox/aggression](src/tasksource/tasks.py#L2321) | choice | votes | 10 |  | [tasksource/wikipedia-detox-votes](https://hf.co/datasets/tasksource/wikipedia-detox-votes) |
| [wikipedia-detox/aggression_score](src/tasksource/tasks.py#L2323) | score | votes | 10 |  | [tasksource/wikipedia-detox-votes](https://hf.co/datasets/tasksource/wikipedia-detox-votes) |
| [wikipedia-detox/toxicity](src/tasksource/tasks.py#L2327) | choice | votes | 10 |  | [tasksource/wikipedia-detox-votes](https://hf.co/datasets/tasksource/wikipedia-detox-votes) |
| [wikipedia-detox/toxicity_score](src/tasksource/tasks.py#L2329) | score | votes | 10 |  | [tasksource/wikipedia-detox-votes](https://hf.co/datasets/tasksource/wikipedia-detox-votes) |
| [lewidi/md_agreement](src/tasksource/tasks.py#L2344) | noul | votes | 5 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/hs_brexit](src/tasksource/tasks.py#L2345) | noul | votes | 6 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/armis](src/tasksource/tasks.py#L2347) | noul | votes | 3 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/conv_abuse](src/tasksource/tasks.py#L2349) | noul | votes | 3 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/mp](src/tasksource/tasks.py#L2351) | noul | votes | 5 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/csc](src/tasksource/tasks.py#L2352) | score | votes | 4 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/varierrnli/entailment](src/tasksource/tasks.py#L2360) | noul | votes | 4 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/varierrnli/neutral](src/tasksource/tasks.py#L2361) | noul | votes | 4 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/varierrnli/contradiction](src/tasksource/tasks.py#L2362) | noul | votes | 4 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [civil_comments/toxicity_share](src/tasksource/tasks.py#L2370) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/severe_toxicity_share](src/tasksource/tasks.py#L2371) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/obscene_share](src/tasksource/tasks.py#L2372) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/threat_share](src/tasksource/tasks.py#L2373) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/insult_share](src/tasksource/tasks.py#L2374) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/identity_attack_share](src/tasksource/tasks.py#L2375) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/sexual_explicit_share](src/tasksource/tasks.py#L2376) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [oasst2/quality](src/tasksource/tasks.py#L2393) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/helpfulness](src/tasksource/tasks.py#L2394) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/creativity](src/tasksource/tasks.py#L2395) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/humor](src/tasksource/tasks.py#L2396) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/toxicity](src/tasksource/tasks.py#L2397) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/violence](src/tasksource/tasks.py#L2398) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/spam](src/tasksource/tasks.py#L2399) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/fails_task](src/tasksource/tasks.py#L2400) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/not_appropriate](src/tasksource/tasks.py#L2401) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/hate_speech](src/tasksource/tasks.py#L2402) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/sexual_content](src/tasksource/tasks.py#L2403) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/pii](src/tasksource/tasks.py#L2404) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/lang_mismatch](src/tasksource/tasks.py#L2405) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [scirepeval/search](src/tasksource/tasks.py#L2508) | score | mean |  | regression | [tasksource/scirepeval-search](https://hf.co/datasets/tasksource/scirepeval-search) |
| [discogem/relation](src/tasksource/tasks.py#L2939) | choice | votes | 10 | Classification | [tasksource/discogem](https://hf.co/datasets/tasksource/discogem) |
| [clear/readability](src/tasksource/tasks.py#L3461) | score | mean |  | regression | [casey-martin/CommonLit-Ease-of-Readability](https://hf.co/datasets/casey-martin/CommonLit-Ease-of-Readability) |
