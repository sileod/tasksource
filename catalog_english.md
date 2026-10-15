576 English tasks. Load one with `load_task(id)`; the annotations are in [tasks.py](src/tasksource/tasks.py). Evaluation benchmarks are in [eval_only.py](src/tasksource/eval_only.py); other excluded annotations are in [parked.py](src/tasksource/parked.py).

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
| 17 | [babi_nli/two-supporting-facts](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 18 | [babi_nli/basic-deduction](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 19 | [babi_nli/basic-coreference](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 20 | [babi_nli/yes-no-questions](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 21 | [babi_nli/counting](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 22 | [babi_nli/indefinite-knowledge](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 23 | [babi_nli/lists-sets](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 24 | [babi_nli/path-finding](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 25 | [babi_nli/compound-coreference](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 26 | [babi_nli/positional-reasoning](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 27 | [babi_nli/simple-negation](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 28 | [babi_nli/single-supporting-fact](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 29 | [babi_nli/size-reasoning](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 30 | [babi_nli/three-arg-relations](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 31 | [babi_nli/three-supporting-facts](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 32 | [babi_nli/time-reasoning](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 33 | [babi_nli/two-arg-relations](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 34 | [babi_nli/conjunction](src/tasksource/tasks.py#L76) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) |  |
| 35 | [babi_nli/basic-induction](src/tasksource/tasks.py#L81) | Classification | [tasksource/babi_nli](https://hf.co/datasets/tasksource/babi_nli) | ✓ |
| 36 | [sick/label](src/tasksource/tasks.py#L86) | Classification | [tasksource/sick](https://hf.co/datasets/tasksource/sick) |  |
| 37 | [sick/relatedness](src/tasksource/tasks.py#L88) | Classification | [tasksource/sick](https://hf.co/datasets/tasksource/sick) | ✓ |
| 38 | [snli](src/tasksource/tasks.py#L146) | Classification | [stanfordnlp/snli](https://hf.co/datasets/stanfordnlp/snli) |  |
| 39 | [scitail/snli_format](src/tasksource/tasks.py#L149) | Classification | [allenai/scitail](https://hf.co/datasets/allenai/scitail) |  |
| 40 | [hans](src/tasksource/tasks.py#L151) | Classification | [tasksource/hans](https://hf.co/datasets/tasksource/hans) |  |
| 41 | [WANLI](src/tasksource/tasks.py#L154) | Classification | [alisawuffles/WANLI](https://hf.co/datasets/alisawuffles/WANLI) |  |
| 42 | [recast/recast_verbnet](src/tasksource/tasks.py#L156) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 43 | [recast/recast_factuality](src/tasksource/tasks.py#L156) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 44 | [recast/recast_megaveridicality](src/tasksource/tasks.py#L156) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 45 | [recast/recast_puns](src/tasksource/tasks.py#L156) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 46 | [recast/recast_ner](src/tasksource/tasks.py#L156) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 47 | [recast/recast_verbcorner](src/tasksource/tasks.py#L156) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
| 48 | [recast/recast_sentiment](src/tasksource/tasks.py#L156) | Classification | [tasksource/recast](https://hf.co/datasets/tasksource/recast) |  |
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
| 75 | [paws/labeled_final](src/tasksource/tasks.py#L240) | Classification | [google-research-datasets/paws](https://hf.co/datasets/google-research-datasets/paws) |  |
| 76 | [paws/labeled_swap](src/tasksource/tasks.py#L241) | Classification | [google-research-datasets/paws](https://hf.co/datasets/google-research-datasets/paws) |  |
| 77 | [medical_questions_pairs](src/tasksource/tasks.py#L243) | Classification | [curaihealth/medical_questions_pairs](https://hf.co/datasets/curaihealth/medical_questions_pairs) | ✓ |
| 78 | [conll2003/pos_tags](src/tasksource/tasks.py#L249) | TokenClassification | [tomaarsen/conll2003](https://hf.co/datasets/tomaarsen/conll2003) | ✓ |
| 79 | [conll2003/chunk_tags](src/tasksource/tasks.py#L251) | TokenClassification | [tomaarsen/conll2003](https://hf.co/datasets/tomaarsen/conll2003) | ✓ |
| 80 | [conll2003/ner_tags](src/tasksource/tasks.py#L253) | TokenClassification | [tomaarsen/conll2003](https://hf.co/datasets/tomaarsen/conll2003) |  |
| 81 | [fig-qa](src/tasksource/tasks.py#L259) | MultipleChoice | [nightingal3/fig-qa](https://hf.co/datasets/nightingal3/fig-qa) |  |
| 82 | [cos_e/v1.0](src/tasksource/tasks.py#L268) | MultipleChoice | [Salesforce/cos_e](https://hf.co/datasets/Salesforce/cos_e) |  |
| 83 | [cosmos_qa](src/tasksource/tasks.py#L273) | MultipleChoice | [Samsoup/cosmos_qa](https://hf.co/datasets/Samsoup/cosmos_qa) |  |
| 84 | [dream](src/tasksource/tasks.py#L276) | MultipleChoice | [dataset-org/dream](https://hf.co/datasets/dataset-org/dream) |  |
| 85 | [openbookqa](src/tasksource/tasks.py#L283) | MultipleChoice | [allenai/openbookqa](https://hf.co/datasets/allenai/openbookqa) |  |
| 86 | [qasc](src/tasksource/tasks.py#L289) | MultipleChoice | [allenai/qasc](https://hf.co/datasets/allenai/qasc) |  |
| 87 | [quartz](src/tasksource/tasks.py#L297) | MultipleChoice | [allenai/quartz](https://hf.co/datasets/allenai/quartz) |  |
| 88 | [quail](src/tasksource/tasks.py#L302) | MultipleChoice | [textmachinelab/quail](https://hf.co/datasets/textmachinelab/quail) |  |
| 89 | [head_qa/en](src/tasksource/tasks.py#L308) | MultipleChoice | [EleutherAI/headqa](https://hf.co/datasets/EleutherAI/headqa) |  |
| 90 | [sciq](src/tasksource/tasks.py#L316) | MultipleChoice | [allenai/sciq](https://hf.co/datasets/allenai/sciq) |  |
| 91 | [social_i_qa](src/tasksource/tasks.py#L321) | MultipleChoice | [tasksource/social_i_qa](https://hf.co/datasets/tasksource/social_i_qa) |  |
| 92 | [wiki_hop/original](src/tasksource/tasks.py#L327) | MultipleChoice | [MoE-UNC/wikihop](https://hf.co/datasets/MoE-UNC/wikihop) |  |
| 93 | [wiqa](src/tasksource/tasks.py#L340) | MultipleChoice | [allenai/wiqa](https://hf.co/datasets/allenai/wiqa) (`parquet` loader) |  |
| 94 | [piqa](src/tasksource/tasks.py#L347) | MultipleChoice | [baber/piqa](https://hf.co/datasets/baber/piqa) |  |
| 95 | [hellaswag](src/tasksource/tasks.py#L355) | MultipleChoice | [Rowan/hellaswag](https://hf.co/datasets/Rowan/hellaswag) |  |
| 96 | [super_glue/copa](src/tasksource/tasks.py#L364) | MultipleChoice | [aps/super_glue](https://hf.co/datasets/aps/super_glue) |  |
| 97 | [balanced-copa](src/tasksource/tasks.py#L366) | MultipleChoice | [pkavumba/balanced-copa](https://hf.co/datasets/pkavumba/balanced-copa) |  |
| 98 | [e-CARE](src/tasksource/tasks.py#L369) | MultipleChoice | [12ml/e-CARE](https://hf.co/datasets/12ml/e-CARE) |  |
| 99 | [art](src/tasksource/tasks.py#L372) | MultipleChoice | [allenai/art](https://hf.co/datasets/allenai/art) | ✓ |
| 100 | [winogrande/winogrande_xl](src/tasksource/tasks.py#L380) | MultipleChoice | [allenai/winogrande](https://hf.co/datasets/allenai/winogrande) |  |
| 101 | [codah/codah](src/tasksource/tasks.py#L383) | MultipleChoice | [jaredfern/codah](https://hf.co/datasets/jaredfern/codah) |  |
| 102 | [ai2_arc/ARC-Easy/challenge](src/tasksource/tasks.py#L385) | MultipleChoice | [allenai/ai2_arc](https://hf.co/datasets/allenai/ai2_arc) |  |
| 103 | [ai2_arc/ARC-Challenge/challenge](src/tasksource/tasks.py#L385) | MultipleChoice | [allenai/ai2_arc](https://hf.co/datasets/allenai/ai2_arc) |  |
| 104 | [definite_pronoun_resolution](src/tasksource/tasks.py#L390) | MultipleChoice | [community-datasets/definite_pronoun_resolution](https://hf.co/datasets/community-datasets/definite_pronoun_resolution) |  |
| 105 | [swag/regular](src/tasksource/tasks.py#L396) | MultipleChoice | [allenai/swag](https://hf.co/datasets/allenai/swag) |  |
| 106 | [math_qa](src/tasksource/tasks.py#L402) | MultipleChoice | [tasksource/math_qa](https://hf.co/datasets/tasksource/math_qa) |  |
| 107 | [glue/cola](src/tasksource/tasks.py#L411) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 108 | [glue/sst2](src/tasksource/tasks.py#L412) | Classification | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |  |
| 109 | [utilitarianism](src/tasksource/tasks.py#L426) | Classification | [hendrycks/ethics](https://hf.co/datasets/hendrycks/ethics) (`csv` loader) |  |
| 110 | [amazon_counterfactual/en](src/tasksource/tasks.py#L434) | Classification | [mteb/amazon_counterfactual](https://hf.co/datasets/mteb/amazon_counterfactual) |  |
| 111 | [insincere-questions](src/tasksource/tasks.py#L439) | Classification | [SetFit/insincere-questions](https://hf.co/datasets/SetFit/insincere-questions) |  |
| 112 | [toxic_conversations](src/tasksource/tasks.py#L443) | Classification | [SetFit/toxic_conversations](https://hf.co/datasets/SetFit/toxic_conversations) |  |
| 113 | [TuringBench](src/tasksource/tasks.py#L447) | Classification | [jana4/turingbench-humanized](https://hf.co/datasets/jana4/turingbench-humanized), [turingbench/TuringBench](https://hf.co/datasets/turingbench/TuringBench) (`csv` loader) | ✓ |
| 114 | [trec](src/tasksource/tasks.py#L457) | Classification | [tasksource/trec](https://hf.co/datasets/tasksource/trec) |  |
| 115 | [vitaminc](src/tasksource/tasks.py#L460) | Classification | [tals/vitaminc](https://hf.co/datasets/tals/vitaminc) |  |
| 116 | [hope_edi/english](src/tasksource/tasks.py#L462) | Classification | [dravidianlangtech/hope_edi](https://hf.co/datasets/dravidianlangtech/hope_edi) (`csv` loader) |  |
| 117 | [rumoureval_2019/RumourEval2019](src/tasksource/tasks.py#L476) | Classification | [strombergnlp/rumoureval_2019](https://hf.co/datasets/strombergnlp/rumoureval_2019) (`csv` loader) |  |
| 118 | [ethos/binary](src/tasksource/tasks.py#L490) | Classification | [SetFit/ethos_binary](https://hf.co/datasets/SetFit/ethos_binary) |  |
| 119 | [ethos/multilabel](src/tasksource/tasks.py#L514) | Classification | [tasksource/ethos](https://hf.co/datasets/tasksource/ethos) |  |
| 120 | [tweet_eval/sentiment](src/tasksource/tasks.py#L517) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 121 | [tweet_eval/offensive](src/tasksource/tasks.py#L517) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 122 | [tweet_eval/emotion](src/tasksource/tasks.py#L517) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 123 | [tweet_eval/emoji](src/tasksource/tasks.py#L517) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 124 | [tweet_eval/hate](src/tasksource/tasks.py#L517) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 125 | [tweet_eval/irony](src/tasksource/tasks.py#L517) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) |  |
| 126 | [tweet_eval/stance_abortion](src/tasksource/tasks.py#L532) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 127 | [tweet_eval/stance_atheism](src/tasksource/tasks.py#L533) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 128 | [tweet_eval/stance_climate](src/tasksource/tasks.py#L534) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 129 | [tweet_eval/stance_feminist](src/tasksource/tasks.py#L535) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 130 | [tweet_eval/stance_hillary](src/tasksource/tasks.py#L536) | Classification | [cardiffnlp/tweet_eval](https://hf.co/datasets/cardiffnlp/tweet_eval) | ✓ |
| 131 | [discovery/discovery](src/tasksource/tasks.py#L539) | Classification | [sileod/discovery](https://hf.co/datasets/sileod/discovery) | ✓ |
| 132 | [pragmeval/verifiability](src/tasksource/tasks.py#L542) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 133 | [pragmeval/switchboard](src/tasksource/tasks.py#L542) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 134 | [pragmeval/gum](src/tasksource/tasks.py#L546) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 135 | [pragmeval/pdtb](src/tasksource/tasks.py#L546) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 136 | [pragmeval/sarcasm](src/tasksource/tasks.py#L546) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 137 | [pragmeval/stac](src/tasksource/tasks.py#L546) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 138 | [pragmeval/persuasiveness-claimtype](src/tasksource/tasks.py#L546) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 139 | [pragmeval/persuasiveness-premisetype](src/tasksource/tasks.py#L546) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 140 | [pragmeval/emergent](src/tasksource/tasks.py#L546) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) | ✓ |
| 141 | [pragmeval/emobank-dominance](src/tasksource/tasks.py#L561) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 142 | [pragmeval/squinky-formality](src/tasksource/tasks.py#L562) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 143 | [pragmeval/squinky-implicature](src/tasksource/tasks.py#L563) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 144 | [pragmeval/squinky-informativeness](src/tasksource/tasks.py#L564) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 145 | [pragmeval/persuasiveness-eloquence](src/tasksource/tasks.py#L565) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 146 | [pragmeval/persuasiveness-relevance](src/tasksource/tasks.py#L566) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 147 | [pragmeval/persuasiveness-specificity](src/tasksource/tasks.py#L567) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 148 | [pragmeval/persuasiveness-strength](src/tasksource/tasks.py#L568) | Classification | [sileod/pragmeval](https://hf.co/datasets/sileod/pragmeval) |  |
| 149 | [silicone/sem](src/tasksource/tasks.py#L570) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | ✓ |
| 150 | [silicone/maptask](src/tasksource/tasks.py#L570) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | ✓ |
| 151 | [silicone/oasis](src/tasksource/tasks.py#L570) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | ✓ |
| 152 | [silicone/meld_e](src/tasksource/tasks.py#L570) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | ✓ |
| 153 | [silicone/dyda_da](src/tasksource/tasks.py#L570) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | ✓ |
| 154 | [silicone/dyda_e](src/tasksource/tasks.py#L570) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | ✓ |
| 155 | [silicone/meld_s](src/tasksource/tasks.py#L570) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | ✓ |
| 156 | [silicone/iemocap](src/tasksource/tasks.py#L580) | Classification | [tasksource/silicone](https://hf.co/datasets/tasksource/silicone) | ✓ |
| 157 | [lex_glue/eurlex](src/tasksource/tasks.py#L590) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) | ✓ |
| 158 | [lex_glue/scotus](src/tasksource/tasks.py#L594) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) |  |
| 159 | [lex_glue/ledgar](src/tasksource/tasks.py#L597) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) |  |
| 160 | [lex_glue/unfair_tos](src/tasksource/tasks.py#L599) | Classification | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) | ✓ |
| 161 | [lex_glue/case_hold](src/tasksource/tasks.py#L602) | MultipleChoice | [coastalcph/lex_glue](https://hf.co/datasets/coastalcph/lex_glue) |  |
| 162 | [language-identification](src/tasksource/tasks.py#L610) | Classification | [papluca/language-identification](https://hf.co/datasets/papluca/language-identification) | ✓ |
| 163 | [imdb](src/tasksource/tasks.py#L615) | Classification | [stanfordnlp/imdb](https://hf.co/datasets/stanfordnlp/imdb) |  |
| 164 | [rotten_tomatoes](src/tasksource/tasks.py#L617) | Classification | [cornell-movie-review-data/rotten_tomatoes](https://hf.co/datasets/cornell-movie-review-data/rotten_tomatoes) |  |
| 165 | [ag_news](src/tasksource/tasks.py#L619) | Classification | [fancyzhx/ag_news](https://hf.co/datasets/fancyzhx/ag_news) |  |
| 166 | [yelp_review_full/yelp_review_full](src/tasksource/tasks.py#L621) | Classification | [Yelp/yelp_review_full](https://hf.co/datasets/Yelp/yelp_review_full) | ✓ |
| 167 | [financial_phrasebank/sentences_allagree](src/tasksource/tasks.py#L626) | Classification | [ghbacct/financial-phrasebank-all-agree-classification](https://hf.co/datasets/ghbacct/financial-phrasebank-all-agree-classification) |  |
| 168 | [poem_sentiment](src/tasksource/tasks.py#L631) | Classification | [google-research-datasets/poem_sentiment](https://hf.co/datasets/google-research-datasets/poem_sentiment) |  |
| 169 | [dbpedia_14/dbpedia_14](src/tasksource/tasks.py#L633) | Classification | [fancyzhx/dbpedia_14](https://hf.co/datasets/fancyzhx/dbpedia_14) |  |
| 170 | [amazon_polarity/amazon_polarity](src/tasksource/tasks.py#L635) | Classification | [fancyzhx/amazon_polarity](https://hf.co/datasets/fancyzhx/amazon_polarity) |  |
| 171 | [hate_speech18](src/tasksource/tasks.py#L639) | Classification | [tasksource/hate_speech18](https://hf.co/datasets/tasksource/hate_speech18) |  |
| 172 | [sms_spam](src/tasksource/tasks.py#L645) | Classification | [ucirvine/sms_spam](https://hf.co/datasets/ucirvine/sms_spam) |  |
| 173 | [humicroedit/subtask-1](src/tasksource/tasks.py#L648) | Classification | [tasksource/humicroedit](https://hf.co/datasets/tasksource/humicroedit) | ✓ |
| 174 | [humicroedit/subtask-2](src/tasksource/tasks.py#L654) | Classification | [tasksource/humicroedit](https://hf.co/datasets/tasksource/humicroedit) | ✓ |
| 175 | [snips_built_in_intents](src/tasksource/tasks.py#L659) | Classification | [sonos-nlu-benchmark/snips_built_in_intents](https://hf.co/datasets/sonos-nlu-benchmark/snips_built_in_intents) |  |
| 176 | [hate_speech_offensive](src/tasksource/tasks.py#L663) | Classification | [tdavidson/hate_speech_offensive](https://hf.co/datasets/tdavidson/hate_speech_offensive) |  |
| 177 | [yahoo_answers_topics](src/tasksource/tasks.py#L665) | Classification | [community-datasets/yahoo_answers_topics](https://hf.co/datasets/community-datasets/yahoo_answers_topics) |  |
| 178 | [stackoverflow-questions](src/tasksource/tasks.py#L669) | Classification | [pacovaldez/stackoverflow-questions](https://hf.co/datasets/pacovaldez/stackoverflow-questions) | ✓ |
| 179 | [hyperpartisan_news](src/tasksource/tasks.py#L675) | Classification | [zapsdcn/hyperpartisan_news](https://hf.co/datasets/zapsdcn/hyperpartisan_news) |  |
| 180 | [sciie](src/tasksource/tasks.py#L680) | Classification | [zapsdcn/sciie](https://hf.co/datasets/zapsdcn/sciie) |  |
| 181 | [citation_intent](src/tasksource/tasks.py#L681) | Classification | [zapsdcn/citation_intent](https://hf.co/datasets/zapsdcn/citation_intent) |  |
| 182 | [go_emotions/simplified](src/tasksource/tasks.py#L683) | Classification | [google-research-datasets/go_emotions](https://hf.co/datasets/google-research-datasets/go_emotions) |  |
| 183 | [scicite](src/tasksource/tasks.py#L687) | Classification | [tasksource/scicite](https://hf.co/datasets/tasksource/scicite) |  |
| 184 | [liar](src/tasksource/tasks.py#L689) | Classification | [tasksource/liar](https://hf.co/datasets/tasksource/liar) | ✓ |
| 185 | [lexical_relation_classification/ROOT09](src/tasksource/tasks.py#L700) | Classification | [relbert/lexical_relation_classification](https://hf.co/datasets/relbert/lexical_relation_classification) (`json` loader) | ✓ |
| 186 | [lexical_relation_classification/K&H+N](src/tasksource/tasks.py#L700) | Classification | [relbert/lexical_relation_classification](https://hf.co/datasets/relbert/lexical_relation_classification) (`json` loader) | ✓ |
| 187 | [lexical_relation_classification/BLESS](src/tasksource/tasks.py#L700) | Classification | [relbert/lexical_relation_classification](https://hf.co/datasets/relbert/lexical_relation_classification) (`json` loader) | ✓ |
| 188 | [lexical_relation_classification/CogALexV](src/tasksource/tasks.py#L726) | Classification | [relbert/lexical_relation_classification](https://hf.co/datasets/relbert/lexical_relation_classification) (`json` loader) | ✓ |
| 189 | [linguisticprobing/subj_number](src/tasksource/tasks.py#L753) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 190 | [linguisticprobing/obj_number](src/tasksource/tasks.py#L754) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 191 | [linguisticprobing/past_present](src/tasksource/tasks.py#L755) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) |  |
| 192 | [linguisticprobing/top_constituents](src/tasksource/tasks.py#L756) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | ✓ |
| 193 | [linguisticprobing/tree_depth](src/tasksource/tasks.py#L758) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | ✓ |
| 194 | [linguisticprobing/coordination_inversion](src/tasksource/tasks.py#L759) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | ✓ |
| 195 | [linguisticprobing/odd_man_out](src/tasksource/tasks.py#L761) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | ✓ |
| 196 | [linguisticprobing/bigram_shift](src/tasksource/tasks.py#L762) | Classification | [tasksource/linguisticprobing](https://hf.co/datasets/tasksource/linguisticprobing) | ✓ |
| 197 | [crowdflower/political-media-message](src/tasksource/tasks.py#L764) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 198 | [crowdflower/economic-news](src/tasksource/tasks.py#L764) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 199 | [crowdflower/corporate-messaging](src/tasksource/tasks.py#L764) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 200 | [crowdflower/airline-sentiment](src/tasksource/tasks.py#L764) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 201 | [crowdflower/political-media-audience](src/tasksource/tasks.py#L764) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 202 | [crowdflower/tweet_global_warming](src/tasksource/tasks.py#L764) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 203 | [crowdflower/sentiment_nuclear_power](src/tasksource/tasks.py#L764) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 204 | [crowdflower/political-media-bias](src/tasksource/tasks.py#L764) | Classification | [tasksource/crowdflower](https://hf.co/datasets/tasksource/crowdflower) | ✓ |
| 205 | [ethics/commonsense](src/tasksource/tasks.py#L798) | Classification | [hendrycks/ethics](https://hf.co/datasets/hendrycks/ethics) (`csv` loader) |  |
| 206 | [ethics/deontology](src/tasksource/tasks.py#L806) | Classification | [hendrycks/ethics](https://hf.co/datasets/hendrycks/ethics) (`csv` loader) | ✓ |
| 207 | [ethics/justice](src/tasksource/tasks.py#L815) | Classification | [hendrycks/ethics](https://hf.co/datasets/hendrycks/ethics) (`csv` loader) |  |
| 208 | [ethics/virtue](src/tasksource/tasks.py#L823) | Classification | [hendrycks/ethics](https://hf.co/datasets/hendrycks/ethics) |  |
| 209 | [emo/emo2019](src/tasksource/tasks.py#L832) | Classification | [oneonlee/cleansed_emocontext](https://hf.co/datasets/oneonlee/cleansed_emocontext) |  |
| 210 | [google_wellformed_query](src/tasksource/tasks.py#L838) | Classification | [tasksource/google_wellformed_query](https://hf.co/datasets/tasksource/google_wellformed_query) | ✓ |
| 211 | [tweets_hate_speech_detection](src/tasksource/tasks.py#L844) | Classification | [tweets-hate-speech-detection/tweets_hate_speech_detection](https://hf.co/datasets/tweets-hate-speech-detection/tweets_hate_speech_detection) |  |
| 212 | [wnut_17/wnut_17](src/tasksource/tasks.py#L848) | TokenClassification | [flaitenberger/wnut_17](https://hf.co/datasets/flaitenberger/wnut_17) |  |
| 213 | [ncbi_disease/ncbi_disease](src/tasksource/tasks.py#L851) | TokenClassification | [ncbi/ncbi_disease](https://hf.co/datasets/ncbi/ncbi_disease) | ✓ |
| 214 | [acronym_identification](src/tasksource/tasks.py#L856) | TokenClassification | [amirveyseh/acronym_identification](https://hf.co/datasets/amirveyseh/acronym_identification) | ✓ |
| 215 | [jnlpba/jnlpba](src/tasksource/tasks.py#L860) | TokenClassification | [jnlpba/jnlpba](https://hf.co/datasets/jnlpba/jnlpba) | ✓ |
| 216 | [ontonotes_english/SpeedOfMagic--ontonotes_english](src/tasksource/tasks.py#L869) | TokenClassification | [SpeedOfMagic/ontonotes_english](https://hf.co/datasets/SpeedOfMagic/ontonotes_english) | ✓ |
| 217 | [blog_authorship_corpus/gender](src/tasksource/tasks.py#L875) | Classification | [tasksource/blog_authorship_corpus](https://hf.co/datasets/tasksource/blog_authorship_corpus) | ✓ |
| 218 | [blog_authorship_corpus/age](src/tasksource/tasks.py#L877) | Classification | [tasksource/blog_authorship_corpus](https://hf.co/datasets/tasksource/blog_authorship_corpus) | ✓ |
| 219 | [open_question_type](src/tasksource/tasks.py#L888) | Classification | [Korea-MES/open_question_type](https://hf.co/datasets/Korea-MES/open_question_type) |  |
| 220 | [health_fact](src/tasksource/tasks.py#L890) | Classification | [marcov/health_fact_promptsource](https://hf.co/datasets/marcov/health_fact_promptsource) |  |
| 221 | [commonsense_qa](src/tasksource/tasks.py#L894) | MultipleChoice | [tau/commonsense_qa](https://hf.co/datasets/tau/commonsense_qa) |  |
| 222 | [mc_taco](src/tasksource/tasks.py#L900) | Classification | [marcov/mc_taco_promptsource](https://hf.co/datasets/marcov/mc_taco_promptsource) | ✓ |
| 223 | [ade_corpus_v2/Ade_corpus_v2_classification](src/tasksource/tasks.py#L907) | Classification | [ade-benchmark-corpus/ade_corpus_v2](https://hf.co/datasets/ade-benchmark-corpus/ade_corpus_v2) | ✓ |
| 224 | [discosense](src/tasksource/tasks.py#L910) | MultipleChoice | [tasksource/discosense](https://hf.co/datasets/tasksource/discosense) | ✓ |
| 225 | [circa](src/tasksource/tasks.py#L917) | Classification | [google-research-datasets/circa](https://hf.co/datasets/google-research-datasets/circa) |  |
| 226 | [code_x_glue_cc_defect_detection](src/tasksource/tasks.py#L922) | Classification | [google/code_x_glue_cc_defect_detection](https://hf.co/datasets/google/code_x_glue_cc_defect_detection) | ✓ |
| 227 | [phrase_similarity](src/tasksource/tasks.py#L927) | Classification | [Deehan1866/processed_phrase_similarity](https://hf.co/datasets/Deehan1866/processed_phrase_similarity) |  |
| 228 | [scientific-exaggeration-detection](src/tasksource/tasks.py#L935) | Classification | [copenlu/scientific-exaggeration-detection](https://hf.co/datasets/copenlu/scientific-exaggeration-detection) |  |
| 229 | [quarel](src/tasksource/tasks.py#L949) | MultipleChoice | [community-datasets/quarel](https://hf.co/datasets/community-datasets/quarel) | ✓ |
| 230 | [fever-evidence-related](src/tasksource/tasks.py#L956) | Classification | [mwong/fever-evidence-related](https://hf.co/datasets/mwong/fever-evidence-related) |  |
| 231 | [numer_sense](src/tasksource/tasks.py#L959) | Classification | [tasksource/numer_sense](https://hf.co/datasets/tasksource/numer_sense) |  |
| 232 | [dynasent/dynabench.dynasent.r1.all/r1](src/tasksource/tasks.py#L966) | Classification | [tasksource/dynasent](https://hf.co/datasets/tasksource/dynasent) | ✓ |
| 233 | [dynasent/dynabench.dynasent.r2.all/r2](src/tasksource/tasks.py#L970) | Classification | [tasksource/dynasent](https://hf.co/datasets/tasksource/dynasent) | ✓ |
| 234 | [Sarcasm_News_Headline](src/tasksource/tasks.py#L975) | Classification | [raquiba/Sarcasm_News_Headline](https://hf.co/datasets/raquiba/Sarcasm_News_Headline) |  |
| 235 | [sem_eval_2010_task_8](src/tasksource/tasks.py#L978) | Classification | [SemEvalWorkshop/sem_eval_2010_task_8](https://hf.co/datasets/SemEvalWorkshop/sem_eval_2010_task_8) |  |
| 236 | [auditor_review](src/tasksource/tasks.py#L980) | Classification | [demo-org/auditor_review](https://hf.co/datasets/demo-org/auditor_review) |  |
| 237 | [medmcqa](src/tasksource/tasks.py#L984) | MultipleChoice | [openlifescienceai/medmcqa](https://hf.co/datasets/openlifescienceai/medmcqa) |  |
| 238 | [Dynasent_Disagreement](src/tasksource/tasks.py#L1001) | Classification | [RuyuanWan/Dynasent_Disagreement](https://hf.co/datasets/RuyuanWan/Dynasent_Disagreement) | ✓ |
| 239 | [Politeness_Disagreement](src/tasksource/tasks.py#L1003) | Classification | [RuyuanWan/Politeness_Disagreement](https://hf.co/datasets/RuyuanWan/Politeness_Disagreement) | ✓ |
| 240 | [SBIC_Disagreement](src/tasksource/tasks.py#L1005) | Classification | [RuyuanWan/SBIC_Disagreement](https://hf.co/datasets/RuyuanWan/SBIC_Disagreement) | ✓ |
| 241 | [SChem_Disagreement](src/tasksource/tasks.py#L1007) | Classification | [RuyuanWan/SChem_Disagreement](https://hf.co/datasets/RuyuanWan/SChem_Disagreement) | ✓ |
| 242 | [Dilemmas_Disagreement](src/tasksource/tasks.py#L1009) | Classification | [RuyuanWan/Dilemmas_Disagreement](https://hf.co/datasets/RuyuanWan/Dilemmas_Disagreement) | ✓ |
| 243 | [logiqa](src/tasksource/tasks.py#L1012) | MultipleChoice | [fireworks-ai/logiqa](https://hf.co/datasets/fireworks-ai/logiqa) |  |
| 244 | [wiki_qa](src/tasksource/tasks.py#L1021) | Classification | [microsoft/wiki_qa](https://hf.co/datasets/microsoft/wiki_qa) | ✓ |
| 245 | [cycic_classification](src/tasksource/tasks.py#L1023) | Classification | [tasksource/cycic_classification](https://hf.co/datasets/tasksource/cycic_classification) |  |
| 246 | [cycic_multiplechoice](src/tasksource/tasks.py#L1025) | MultipleChoice | [tasksource/cycic_multiplechoice](https://hf.co/datasets/tasksource/cycic_multiplechoice) |  |
| 247 | [sts-companion](src/tasksource/tasks.py#L1029) | Classification | [tasksource/sts-companion](https://hf.co/datasets/tasksource/sts-companion) |  |
| 248 | [commonsense_qa_2.0](src/tasksource/tasks.py#L1032) | Classification | [tasksource/commonsense_qa_2.0](https://hf.co/datasets/tasksource/commonsense_qa_2.0) |  |
| 249 | [lingnli](src/tasksource/tasks.py#L1035) | Classification | [tasksource/lingnli](https://hf.co/datasets/tasksource/lingnli) |  |
| 250 | [monotonicity-entailment](src/tasksource/tasks.py#L1037) | Classification | [tasksource/monotonicity-entailment](https://hf.co/datasets/tasksource/monotonicity-entailment) |  |
| 251 | [arct](src/tasksource/tasks.py#L1040) | MultipleChoice | [tasksource/arct](https://hf.co/datasets/tasksource/arct) |  |
| 252 | [scinli](src/tasksource/tasks.py#L1043) | Classification | [tasksource/scinli](https://hf.co/datasets/tasksource/scinli) |  |
| 253 | [naturallogic](src/tasksource/tasks.py#L1047) | Classification | [tasksource/naturallogic](https://hf.co/datasets/tasksource/naturallogic) |  |
| 254 | [onestop_qa](src/tasksource/tasks.py#L1049) | MultipleChoice | [malmaud/onestop_qa](https://hf.co/datasets/malmaud/onestop_qa) |  |
| 255 | [moral_stories/full](src/tasksource/tasks.py#L1052) | MultipleChoice | [LabHC/moral_stories](https://hf.co/datasets/LabHC/moral_stories) |  |
| 256 | [dynahate](src/tasksource/tasks.py#L1056) | Classification | [tasksource/dynahate](https://hf.co/datasets/tasksource/dynahate) |  |
| 257 | [syntactic-augmentation-nli](src/tasksource/tasks.py#L1058) | Classification | [tasksource/syntactic-augmentation-nli](https://hf.co/datasets/tasksource/syntactic-augmentation-nli) |  |
| 258 | [autotnli](src/tasksource/tasks.py#L1060) | Classification | [tasksource/autotnli](https://hf.co/datasets/tasksource/autotnli) |  |
| 259 | [CONDAQA](src/tasksource/tasks.py#L1062) | Classification | [lasha-nlp/CONDAQA](https://hf.co/datasets/lasha-nlp/CONDAQA) |  |
| 260 | [webgpt_comparisons](src/tasksource/tasks.py#L1072) | MultipleChoice | [heegyu/webgpt_comparisons_ko](https://hf.co/datasets/heegyu/webgpt_comparisons_ko) | ✓ |
| 261 | [synthetic-instruct-gptj-pairwise](src/tasksource/tasks.py#L1080) | MultipleChoice | [Dahoas/synthetic-instruct-gptj-pairwise](https://hf.co/datasets/Dahoas/synthetic-instruct-gptj-pairwise) | ✓ |
| 262 | [scruples](src/tasksource/tasks.py#L1083) | Classification | [tasksource/scruples](https://hf.co/datasets/tasksource/scruples) | ✓ |
| 263 | [defeasible-nli/snli](src/tasksource/tasks.py#L1088) | Classification | [tasksource/defeasible-nli](https://hf.co/datasets/tasksource/defeasible-nli) |  |
| 264 | [defeasible-nli/atomic](src/tasksource/tasks.py#L1088) | Classification | [tasksource/defeasible-nli](https://hf.co/datasets/tasksource/defeasible-nli) |  |
| 265 | [defeasible-nli/social](src/tasksource/tasks.py#L1091) | Classification | [tasksource/defeasible-nli](https://hf.co/datasets/tasksource/defeasible-nli) |  |
| 266 | [help-nli](src/tasksource/tasks.py#L1094) | Classification | [tasksource/help-nli](https://hf.co/datasets/tasksource/help-nli) |  |
| 267 | [nli-veridicality-transitivity](src/tasksource/tasks.py#L1097) | Classification | [tasksource/nli-veridicality-transitivity](https://hf.co/datasets/tasksource/nli-veridicality-transitivity) |  |
| 268 | [lonli](src/tasksource/tasks.py#L1100) | Classification | [tasksource/lonli](https://hf.co/datasets/tasksource/lonli) |  |
| 269 | [dadc-limit-nli](src/tasksource/tasks.py#L1103) | Classification | [tasksource/dadc-limit-nli](https://hf.co/datasets/tasksource/dadc-limit-nli) |  |
| 270 | [FLUTE](src/tasksource/tasks.py#L1106) | Classification | [ColumbiaNLP/FLUTE](https://hf.co/datasets/ColumbiaNLP/FLUTE) |  |
| 271 | [strategy-qa](src/tasksource/tasks.py#L1109) | Classification | [tasksource/strategy-qa](https://hf.co/datasets/tasksource/strategy-qa) |  |
| 272 | [summarize_from_feedback/comparisons](src/tasksource/tasks.py#L1112) | MultipleChoice | [vwxyzjn/summarize_from_feedback_oai_preprocessing](https://hf.co/datasets/vwxyzjn/summarize_from_feedback_oai_preprocessing) | ✓ |
| 273 | [folio](src/tasksource/tasks.py#L1120) | Classification | [tasksource/folio](https://hf.co/datasets/tasksource/folio) |  |
| 274 | [tomi-nli](src/tasksource/tasks.py#L1147) | Classification | [tasksource/tomi-nli](https://hf.co/datasets/tasksource/tomi-nli) |  |
| 275 | [avicenna](src/tasksource/tasks.py#L1150) | Classification | [tasksource/avicenna](https://hf.co/datasets/tasksource/avicenna) | ✓ |
| 276 | [SHP](src/tasksource/tasks.py#L1153) | MultipleChoice | [stanfordnlp/SHP](https://hf.co/datasets/stanfordnlp/SHP) | ✓ |
| 277 | [MedQA-USMLE-4-options-hf](src/tasksource/tasks.py#L1161) | MultipleChoice | [GBaker/MedQA-USMLE-4-options-hf](https://hf.co/datasets/GBaker/MedQA-USMLE-4-options-hf) |  |
| 278 | [wikimedqa/medwiki](src/tasksource/tasks.py#L1164) | MultipleChoice | [sileod/wikimedqa](https://hf.co/datasets/sileod/wikimedqa) |  |
| 279 | [cicero](src/tasksource/tasks.py#L1173) | MultipleChoice | [declare-lab/cicero](https://hf.co/datasets/declare-lab/cicero) |  |
| 280 | [CREAK](src/tasksource/tasks.py#L1177) | Classification | [amydeng2000/CREAK](https://hf.co/datasets/amydeng2000/CREAK) |  |
| 281 | [mutual](src/tasksource/tasks.py#L1180) | MultipleChoice | [tasksource/mutual](https://hf.co/datasets/tasksource/mutual) |  |
| 282 | [implicatures](src/tasksource/tasks.py#L1185) | MultipleChoice | [tasksource/implicatures](https://hf.co/datasets/tasksource/implicatures) |  |
| 283 | [race/high](src/tasksource/tasks.py#L1190) | MultipleChoice | [ehovy/race](https://hf.co/datasets/ehovy/race) |  |
| 284 | [race/middle](src/tasksource/tasks.py#L1190) | MultipleChoice | [ehovy/race](https://hf.co/datasets/ehovy/race) |  |
| 285 | [race-c](src/tasksource/tasks.py#L1194) | MultipleChoice | [tasksource/race-c](https://hf.co/datasets/tasksource/race-c) |  |
| 286 | [spartqa-yn](src/tasksource/tasks.py#L1197) | Classification | [tasksource/spartqa-yn](https://hf.co/datasets/tasksource/spartqa-yn) |  |
| 287 | [spartqa-mchoice](src/tasksource/tasks.py#L1200) | MultipleChoice | [tasksource/spartqa-mchoice](https://hf.co/datasets/tasksource/spartqa-mchoice) |  |
| 288 | [temporal-nli](src/tasksource/tasks.py#L1203) | Classification | [tasksource/temporal-nli](https://hf.co/datasets/tasksource/temporal-nli) |  |
| 289 | [riddle_sense](src/tasksource/tasks.py#L1206) | MultipleChoice | [jeggers/riddle_sense](https://hf.co/datasets/jeggers/riddle_sense) |  |
| 290 | [clcd-english](src/tasksource/tasks.py#L1211) | Classification | [tasksource/clcd-english](https://hf.co/datasets/tasksource/clcd-english) |  |
| 291 | [twentyquestions](src/tasksource/tasks.py#L1223) | Classification | [tasksource/twentyquestions](https://hf.co/datasets/tasksource/twentyquestions) |  |
| 292 | [reclor](src/tasksource/tasks.py#L1228) | MultipleChoice | [tasksource/reclor](https://hf.co/datasets/tasksource/reclor) |  |
| 293 | [counterfactually-augmented-imdb](src/tasksource/tasks.py#L1231) | Classification | [tasksource/counterfactually-augmented-imdb](https://hf.co/datasets/tasksource/counterfactually-augmented-imdb) |  |
| 294 | [counterfactually-augmented-snli](src/tasksource/tasks.py#L1234) | Classification | [tasksource/counterfactually-augmented-snli](https://hf.co/datasets/tasksource/counterfactually-augmented-snli) |  |
| 295 | [cnli](src/tasksource/tasks.py#L1237) | Classification | [tasksource/cnli](https://hf.co/datasets/tasksource/cnli) |  |
| 296 | [boolq-natural-perturbations](src/tasksource/tasks.py#L1240) | Classification | [tasksource/boolq-natural-perturbations](https://hf.co/datasets/tasksource/boolq-natural-perturbations) |  |
| 297 | [acceptability-prediction](src/tasksource/tasks.py#L1245) | Classification | [tasksource/acceptability-prediction](https://hf.co/datasets/tasksource/acceptability-prediction) | ✓ |
| 298 | [equate](src/tasksource/tasks.py#L1271) | Classification | [tasksource/equate](https://hf.co/datasets/tasksource/equate) |  |
| 299 | [ScienceQA_text_only](src/tasksource/tasks.py#L1274) | MultipleChoice | [tasksource/ScienceQA_text_only](https://hf.co/datasets/tasksource/ScienceQA_text_only) |  |
| 300 | [ekar_english](src/tasksource/tasks.py#L1277) | MultipleChoice | [Jiangjie/ekar_english](https://hf.co/datasets/Jiangjie/ekar_english) | ✓ |
| 301 | [implicit-hate-stg1](src/tasksource/tasks.py#L1281) | Classification | [tasksource/implicit-hate-stg1](https://hf.co/datasets/tasksource/implicit-hate-stg1) | ✓ |
| 302 | [chaos-mnli-ambiguity](src/tasksource/tasks.py#L1285) | Classification | [tasksource/chaos-mnli-ambiguity](https://hf.co/datasets/tasksource/chaos-mnli-ambiguity) | ✓ |
| 303 | [headline_cause/en_simple](src/tasksource/tasks.py#L1289) | Classification | [IlyaGusev/headline_cause](https://hf.co/datasets/IlyaGusev/headline_cause) (`json` loader) |  |
| 304 | [logiqa-2.0-nli](src/tasksource/tasks.py#L1294) | Classification | [tasksource/logiqa-2.0-nli](https://hf.co/datasets/tasksource/logiqa-2.0-nli) |  |
| 305 | [oasst2_dense_flat/quality](src/tasksource/tasks.py#L1299) | Classification | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) | ✓ |
| 306 | [oasst2_dense_flat/toxicity](src/tasksource/tasks.py#L1301) | Classification | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) | ✓ |
| 307 | [oasst2_dense_flat/helpfulness](src/tasksource/tasks.py#L1303) | Classification | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) | ✓ |
| 308 | [mindgames](src/tasksource/tasks.py#L1306) | Classification | [sileod/mindgames](https://hf.co/datasets/sileod/mindgames) |  |
| 309 | [universal_dependencies/en_ewt/deprel](src/tasksource/tasks.py#L1320) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) | ✓ |
| 310 | [universal_dependencies/en_partut/deprel](src/tasksource/tasks.py#L1320) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) | ✓ |
| 311 | [universal_dependencies/en_lines/deprel](src/tasksource/tasks.py#L1320) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) | ✓ |
| 312 | [universal_dependencies/en_gum/deprel](src/tasksource/tasks.py#L1320) | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) | ✓ |
| 313 | [ambient](src/tasksource/tasks.py#L1327) | Classification | [tasksource/ambient](https://hf.co/datasets/tasksource/ambient) | ✓ |
| 314 | [path-naturalness-prediction](src/tasksource/tasks.py#L1332) | MultipleChoice | [tasksource/path-naturalness-prediction](https://hf.co/datasets/tasksource/path-naturalness-prediction) | ✓ |
| 315 | [civil_comments/toxicity](src/tasksource/tasks.py#L1342) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 316 | [civil_comments/severe_toxicity](src/tasksource/tasks.py#L1343) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 317 | [civil_comments/obscene](src/tasksource/tasks.py#L1344) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 318 | [civil_comments/threat](src/tasksource/tasks.py#L1345) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 319 | [civil_comments/insult](src/tasksource/tasks.py#L1346) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 320 | [civil_comments/identity_attack](src/tasksource/tasks.py#L1347) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 321 | [civil_comments/sexual_explicit](src/tasksource/tasks.py#L1348) | Classification | [google/civil_comments](https://hf.co/datasets/google/civil_comments) | ✓ |
| 322 | [cloth](src/tasksource/tasks.py#L1350) | MultipleChoice | [AndyChiang/cloth](https://hf.co/datasets/AndyChiang/cloth) |  |
| 323 | [dgen](src/tasksource/tasks.py#L1351) | MultipleChoice | [AndyChiang/dgen](https://hf.co/datasets/AndyChiang/dgen) |  |
| 324 | [I2D2](src/tasksource/tasks.py#L1353) | Classification | [tasksource/I2D2](https://hf.co/datasets/tasksource/I2D2) | ✓ |
| 325 | [args_me](src/tasksource/tasks.py#L1358) | Classification | [webis/args_me](https://hf.co/datasets/webis/args_me) | ✓ |
| 326 | [Touche23-ValueEval](src/tasksource/tasks.py#L1363) | Classification | [webis/Touche23-ValueEval](https://hf.co/datasets/webis/Touche23-ValueEval) (`csv` loader) | ✓ |
| 327 | [starcon](src/tasksource/tasks.py#L1372) | Classification | [tasksource/starcon](https://hf.co/datasets/tasksource/starcon) | ✓ |
| 328 | [banking77](src/tasksource/tasks.py#L1377) | Classification | [legacy-datasets/banking77](https://hf.co/datasets/legacy-datasets/banking77) |  |
| 329 | [it-support-tickets](src/tasksource/tasks.py#L1379) | Classification | [tasksource/it-support-tickets](https://hf.co/datasets/tasksource/it-support-tickets) |  |
| 330 | [ConTRoL-nli](src/tasksource/tasks.py#L1383) | Classification | [tasksource/ConTRoL-nli](https://hf.co/datasets/tasksource/ConTRoL-nli) |  |
| 331 | [tracie](src/tasksource/tasks.py#L1385) | Classification | [tasksource/tracie](https://hf.co/datasets/tasksource/tracie) | ✓ |
| 332 | [sherliic](src/tasksource/tasks.py#L1387) | Classification | [tasksource/sherliic](https://hf.co/datasets/tasksource/sherliic) |  |
| 333 | [sen-making/1](src/tasksource/tasks.py#L1389) | MultipleChoice | [tasksource/sen-making](https://hf.co/datasets/tasksource/sen-making) | ✓ |
| 334 | [sen-making/2](src/tasksource/tasks.py#L1393) | MultipleChoice | [tasksource/sen-making](https://hf.co/datasets/tasksource/sen-making) | ✓ |
| 335 | [winowhy](src/tasksource/tasks.py#L1396) | Classification | [tasksource/winowhy](https://hf.co/datasets/tasksource/winowhy) | ✓ |
| 336 | [robustLR](src/tasksource/tasks.py#L1400) | Classification | [tasksource/robustLR](https://hf.co/datasets/tasksource/robustLR) |  |
| 337 | [clutrr](src/tasksource/tasks.py#L1402) | Classification | [tasksource/clutrr](https://hf.co/datasets/tasksource/clutrr) |  |
| 338 | [logical-fallacy](src/tasksource/tasks.py#L1404) | Classification | [tasksource/logical-fallacy](https://hf.co/datasets/tasksource/logical-fallacy) |  |
| 339 | [parade](src/tasksource/tasks.py#L1407) | Classification | [tasksource/parade](https://hf.co/datasets/tasksource/parade) |  |
| 340 | [subjectivity](src/tasksource/tasks.py#L1410) | Classification | [tasksource/subjectivity](https://hf.co/datasets/tasksource/subjectivity) | ✓ |
| 341 | [MOH](src/tasksource/tasks.py#L1416) | Classification | [tasksource/MOH](https://hf.co/datasets/tasksource/MOH) | ✓ |
| 342 | [VUAC](src/tasksource/tasks.py#L1417) | Classification | [tasksource/VUAC](https://hf.co/datasets/tasksource/VUAC) | ✓ |
| 343 | [TroFi](src/tasksource/tasks.py#L1418) | Classification | [tasksource/TroFi](https://hf.co/datasets/tasksource/TroFi) (`parquet` loader) | ✓ |
| 344 | [sharc](src/tasksource/tasks.py#L1426) | Classification | [tasksource/sharc](https://hf.co/datasets/tasksource/sharc) |  |
| 345 | [conceptrules_v2](src/tasksource/tasks.py#L1430) | Classification | [tasksource/conceptrules_v2](https://hf.co/datasets/tasksource/conceptrules_v2) | ✓ |
| 346 | [disrpt/eng.dep.scidtb.rels](src/tasksource/tasks.py#L1432) | Classification | [multilingual-discourse-hub/disrpt](https://hf.co/datasets/multilingual-discourse-hub/disrpt) | ✓ |
| 347 | [conll2000](src/tasksource/tasks.py#L1435) | TokenClassification | [eriktks/conll2000](https://hf.co/datasets/eriktks/conll2000) |  |
| 348 | [few-nerd/supervised](src/tasksource/tasks.py#L1438) | TokenClassification | [DFKI-SLT/few-nerd](https://hf.co/datasets/DFKI-SLT/few-nerd) |  |
| 349 | [finer-139](src/tasksource/tasks.py#L1439) | TokenClassification | [nlpaueb/finer-139](https://hf.co/datasets/nlpaueb/finer-139) |  |
| 350 | [zero-shot-label-nli](src/tasksource/tasks.py#L1442) | Classification | [tasksource/zero-shot-label-nli](https://hf.co/datasets/tasksource/zero-shot-label-nli) | ✓ |
| 351 | [com2sense](src/tasksource/tasks.py#L1444) | Classification | [tasksource/com2sense](https://hf.co/datasets/tasksource/com2sense) |  |
| 352 | [scone](src/tasksource/tasks.py#L1446) | Classification | [tasksource/scone](https://hf.co/datasets/tasksource/scone) |  |
| 353 | [winodict](src/tasksource/tasks.py#L1448) | MultipleChoice | [tasksource/winodict](https://hf.co/datasets/tasksource/winodict) |  |
| 354 | [fool-me-twice](src/tasksource/tasks.py#L1450) | Classification | [tasksource/fool-me-twice](https://hf.co/datasets/tasksource/fool-me-twice) |  |
| 355 | [monli](src/tasksource/tasks.py#L1454) | Classification | [tasksource/monli](https://hf.co/datasets/tasksource/monli) |  |
| 356 | [corr2cause](src/tasksource/tasks.py#L1456) | Classification | [tasksource/corr2cause](https://hf.co/datasets/tasksource/corr2cause) |  |
| 357 | [lsat_qa/all](src/tasksource/tasks.py#L1458) | MultipleChoice | [lighteval/lsat_qa](https://hf.co/datasets/lighteval/lsat_qa) |  |
| 358 | [apt](src/tasksource/tasks.py#L1460) | Classification | [tasksource/apt](https://hf.co/datasets/tasksource/apt) |  |
| 359 | [twitter-financial-news-sentiment](src/tasksource/tasks.py#L1463) | Classification | [zeroshot/twitter-financial-news-sentiment](https://hf.co/datasets/zeroshot/twitter-financial-news-sentiment) |  |
| 360 | [icl-symbol-tuning-instruct](src/tasksource/tasks.py#L1470) | Classification | [tasksource/icl-symbol-tuning-instruct](https://hf.co/datasets/tasksource/icl-symbol-tuning-instruct) | ✓ |
| 361 | [SpaceNLI](src/tasksource/tasks.py#L1476) | Classification | [tasksource/SpaceNLI](https://hf.co/datasets/tasksource/SpaceNLI) |  |
| 362 | [propsegment/nli](src/tasksource/tasks.py#L1478) | Classification | [sihaochen/propsegment](https://hf.co/datasets/sihaochen/propsegment) (`json` loader) | ✓ |
| 363 | [HatemojiBuild](src/tasksource/tasks.py#L1488) | Classification | [HannahRoseKirk/HatemojiBuild](https://hf.co/datasets/HannahRoseKirk/HatemojiBuild) |  |
| 364 | [regset](src/tasksource/tasks.py#L1491) | Classification | [tasksource/regset](https://hf.co/datasets/tasksource/regset) | ✓ |
| 365 | [esci](src/tasksource/tasks.py#L1497) | Classification | [tasksource/esci](https://hf.co/datasets/tasksource/esci) |  |
| 366 | [chatbot_arena_conversations](src/tasksource/tasks.py#L1516) | MultipleChoice | [lmsys/chatbot_arena_conversations](https://hf.co/datasets/lmsys/chatbot_arena_conversations) | ✓ |
| 367 | [dnd_style_intents](src/tasksource/tasks.py#L1522) | Classification | [neurae/dnd_style_intents](https://hf.co/datasets/neurae/dnd_style_intents) |  |
| 368 | [FLD.v2/default](src/tasksource/tasks.py#L1525) | Classification | [hitachi-nlp/FLD.v2](https://hf.co/datasets/hitachi-nlp/FLD.v2) | ✓ |
| 369 | [FLD.v2/star](src/tasksource/tasks.py#L1529) | Classification | [hitachi-nlp/FLD.v2](https://hf.co/datasets/hitachi-nlp/FLD.v2) | ✓ |
| 370 | [SDOH-NLI](src/tasksource/tasks.py#L1533) | Classification | [tasksource/SDOH-NLI](https://hf.co/datasets/tasksource/SDOH-NLI) |  |
| 371 | [scifact_entailment](src/tasksource/tasks.py#L1536) | Classification | [tasksource/scifact_entailment](https://hf.co/datasets/tasksource/scifact_entailment) |  |
| 372 | [feasibilityQA](src/tasksource/tasks.py#L1540) | Classification | [tasksource/feasibilityQA](https://hf.co/datasets/tasksource/feasibilityQA) |  |
| 373 | [simple_pair](src/tasksource/tasks.py#L1543) | Classification | [tasksource/simple_pair](https://hf.co/datasets/tasksource/simple_pair) |  |
| 374 | [AdjectiveScaleProbe-nli](src/tasksource/tasks.py#L1544) | Classification | [tasksource/AdjectiveScaleProbe-nli](https://hf.co/datasets/tasksource/AdjectiveScaleProbe-nli) |  |
| 375 | [resnli](src/tasksource/tasks.py#L1545) | Classification | [tasksource/resnli](https://hf.co/datasets/tasksource/resnli) |  |
| 376 | [SpaRTUN](src/tasksource/tasks.py#L1547) | MultipleChoice | [tasksource/SpaRTUN](https://hf.co/datasets/tasksource/SpaRTUN) |  |
| 377 | [ReSQ](src/tasksource/tasks.py#L1552) | MultipleChoice | [tasksource/ReSQ](https://hf.co/datasets/tasksource/ReSQ) |  |
| 378 | [semantic_fragments_nli](src/tasksource/tasks.py#L1557) | Classification | [tasksource/semantic_fragments_nli](https://hf.co/datasets/tasksource/semantic_fragments_nli) |  |
| 379 | [dataset_train_nli](src/tasksource/tasks.py#L1560) | Classification | [MoritzLaurer/dataset_train_nli](https://hf.co/datasets/MoritzLaurer/dataset_train_nli) |  |
| 380 | [stepgame](src/tasksource/tasks.py#L1565) | Classification | [tasksource/stepgame](https://hf.co/datasets/tasksource/stepgame) |  |
| 381 | [nlgraph](src/tasksource/tasks.py#L1573) | Classification | [tasksource/nlgraph](https://hf.co/datasets/tasksource/nlgraph) |  |
| 382 | [oasst2_pairwise_rlhf_reward](src/tasksource/tasks.py#L1577) | MultipleChoice | [tasksource/oasst2_pairwise_rlhf_reward](https://hf.co/datasets/tasksource/oasst2_pairwise_rlhf_reward) | ✓ |
| 383 | [hh-rlhf/helpful-base](src/tasksource/tasks.py#L1588) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| 384 | [hh-rlhf/helpful-online](src/tasksource/tasks.py#L1588) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| 385 | [hh-rlhf/helpful-rejection-sampled](src/tasksource/tasks.py#L1588) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| 386 | [hh-rlhf/harmless-base](src/tasksource/tasks.py#L1592) | MultipleChoice | [tasksource/hh-rlhf](https://hf.co/datasets/tasksource/hh-rlhf) | ✓ |
| 387 | [ruletaker](src/tasksource/tasks.py#L1596) | Classification | [tasksource/ruletaker](https://hf.co/datasets/tasksource/ruletaker) | ✓ |
| 388 | [PARARULE-Plus](src/tasksource/tasks.py#L1600) | Classification | [qbao775/PARARULE-Plus](https://hf.co/datasets/qbao775/PARARULE-Plus) | ✓ |
| 389 | [proofwriter](src/tasksource/tasks.py#L1604) | Classification | [tasksource/proofwriter](https://hf.co/datasets/tasksource/proofwriter) |  |
| 390 | [logical-entailment](src/tasksource/tasks.py#L1607) | Classification | [tasksource/logical-entailment](https://hf.co/datasets/tasksource/logical-entailment) | ✓ |
| 391 | [nope](src/tasksource/tasks.py#L1610) | Classification | [tasksource/nope](https://hf.co/datasets/tasksource/nope) |  |
| 392 | [LogicNLI](src/tasksource/tasks.py#L1614) | Classification | [tasksource/LogicNLI](https://hf.co/datasets/tasksource/LogicNLI) | ✓ |
| 393 | [contract-nli/contractnli_a/seg](src/tasksource/tasks.py#L1617) | Classification | [tasksource/contract-nli](https://hf.co/datasets/tasksource/contract-nli) |  |
| 394 | [contract-nli/contractnli_b/full](src/tasksource/tasks.py#L1619) | Classification | [tasksource/contract-nli](https://hf.co/datasets/tasksource/contract-nli) |  |
| 395 | [nli4ct_semeval2024](src/tasksource/tasks.py#L1621) | Classification | [AshtonIsNotHere/nli4ct_semeval2024](https://hf.co/datasets/AshtonIsNotHere/nli4ct_semeval2024) |  |
| 396 | [lsat-ar](src/tasksource/tasks.py#L1624) | MultipleChoice | [tasksource/lsat-ar](https://hf.co/datasets/tasksource/lsat-ar) |  |
| 397 | [lsat-rc](src/tasksource/tasks.py#L1629) | MultipleChoice | [tasksource/lsat-rc](https://hf.co/datasets/tasksource/lsat-rc) |  |
| 398 | [biosift-nli](src/tasksource/tasks.py#L1634) | Classification | [AshtonIsNotHere/biosift-nli](https://hf.co/datasets/AshtonIsNotHere/biosift-nli) |  |
| 399 | [brainteasers/WP](src/tasksource/tasks.py#L1638) | MultipleChoice | [tasksource/brainteasers](https://hf.co/datasets/tasksource/brainteasers) |  |
| 400 | [brainteasers/SP](src/tasksource/tasks.py#L1638) | MultipleChoice | [tasksource/brainteasers](https://hf.co/datasets/tasksource/brainteasers) |  |
| 401 | [toxigen-data/annotated](src/tasksource/tasks.py#L1644) | Classification | [skg/toxigen-data](https://hf.co/datasets/skg/toxigen-data) |  |
| 402 | [persuasion](src/tasksource/tasks.py#L1655) | Classification | [Anthropic/persuasion](https://hf.co/datasets/Anthropic/persuasion) |  |
| 403 | [AmbigNQ-clarifying-question](src/tasksource/tasks.py#L1661) | Classification | [erbacher/AmbigNQ-clarifying-question](https://hf.co/datasets/erbacher/AmbigNQ-clarifying-question) | ✓ |
| 404 | [SIGA-nli](src/tasksource/tasks.py#L1666) | Classification | [tasksource/SIGA-nli](https://hf.co/datasets/tasksource/SIGA-nli) | ✓ |
| 405 | [FOL-nli](src/tasksource/tasks.py#L1668) | Classification | [unigram/FOL-nli](https://hf.co/datasets/unigram/FOL-nli) |  |
| 406 | [goal-step-wikihow/goal](src/tasksource/tasks.py#L1670) | MultipleChoice | [tasksource/goal-step-wikihow](https://hf.co/datasets/tasksource/goal-step-wikihow) |  |
| 407 | [goal-step-wikihow/step](src/tasksource/tasks.py#L1673) | MultipleChoice | [tasksource/goal-step-wikihow](https://hf.co/datasets/tasksource/goal-step-wikihow) |  |
| 408 | [goal-step-wikihow/order](src/tasksource/tasks.py#L1676) | MultipleChoice | [tasksource/goal-step-wikihow](https://hf.co/datasets/tasksource/goal-step-wikihow) | ✓ |
| 409 | [PARADISE](src/tasksource/tasks.py#L1680) | MultipleChoice | [GGLab/PARADISE](https://hf.co/datasets/GGLab/PARADISE) |  |
| 410 | [doc-nli](src/tasksource/tasks.py#L1683) | Classification | [tasksource/doc-nli](https://hf.co/datasets/tasksource/doc-nli) |  |
| 411 | [mctest-nli](src/tasksource/tasks.py#L1685) | Classification | [tasksource/mctest-nli](https://hf.co/datasets/tasksource/mctest-nli) |  |
| 412 | [patent-phrase-similarity](src/tasksource/tasks.py#L1687) | Classification | [tasksource/patent-phrase-similarity](https://hf.co/datasets/tasksource/patent-phrase-similarity) | ✓ |
| 413 | [natural-language-satisfiability](src/tasksource/tasks.py#L1690) | Classification | [tasksource/natural-language-satisfiability](https://hf.co/datasets/tasksource/natural-language-satisfiability) |  |
| 414 | [idioms-nli](src/tasksource/tasks.py#L1692) | Classification | [tasksource/idioms-nli](https://hf.co/datasets/tasksource/idioms-nli) |  |
| 415 | [lifecycle-entailment](src/tasksource/tasks.py#L1694) | Classification | [tasksource/lifecycle-entailment](https://hf.co/datasets/tasksource/lifecycle-entailment) |  |
| 416 | [safe-guard-prompt-injection](src/tasksource/tasks.py#L1701) | Classification | [xTRam1/safe-guard-prompt-injection](https://hf.co/datasets/xTRam1/safe-guard-prompt-injection) | ✓ |
| 417 | [prompt-injections](src/tasksource/tasks.py#L1707) | Classification | [deepset/prompt-injections](https://hf.co/datasets/deepset/prompt-injections) | ✓ |
| 418 | [prompt-injection-dataset](src/tasksource/tasks.py#L1713) | Classification | [S-Labs/prompt-injection-dataset](https://hf.co/datasets/S-Labs/prompt-injection-dataset) | ✓ |
| 419 | [Prompt-injection-dataset/full](src/tasksource/tasks.py#L1719) | Classification | [neuralchemy/Prompt-injection-dataset](https://hf.co/datasets/neuralchemy/Prompt-injection-dataset) | ✓ |
| 420 | [PromptShield](src/tasksource/tasks.py#L1725) | Classification | [hendzh/PromptShield](https://hf.co/datasets/hendzh/PromptShield) | ✓ |
| 421 | [shell-safety-v2](src/tasksource/tasks.py#L1731) | Classification | [tomngdev/shell-safety-v2](https://hf.co/datasets/tomngdev/shell-safety-v2) | ✓ |
| 422 | [agent_action_safety](src/tasksource/tasks.py#L1736) | Classification | [karanxa/agent-action-safety-dataset](https://hf.co/datasets/karanxa/agent-action-safety-dataset) (`json` loader) | ✓ |
| 423 | [ShellRisk-Bench](src/tasksource/tasks.py#L1751) | Classification | [kontext-security/ShellRisk-Bench](https://hf.co/datasets/kontext-security/ShellRisk-Bench) | ✓ |
| 424 | [wildguardmix-cleaned/prompt_harm](src/tasksource/tasks.py#L1776) | Classification | [bogdanminko/wildguardmix-cleaned](https://hf.co/datasets/bogdanminko/wildguardmix-cleaned) | ✓ |
| 425 | [wildguardmix-cleaned/response_harm](src/tasksource/tasks.py#L1782) | Classification | [bogdanminko/wildguardmix-cleaned](https://hf.co/datasets/bogdanminko/wildguardmix-cleaned) | ✓ |
| 426 | [wildguardmix-cleaned/response_refusal](src/tasksource/tasks.py#L1787) | Classification | [bogdanminko/wildguardmix-cleaned](https://hf.co/datasets/bogdanminko/wildguardmix-cleaned) | ✓ |
| 427 | [BeaverTails](src/tasksource/tasks.py#L1803) | Classification | [PKU-Alignment/BeaverTails](https://hf.co/datasets/PKU-Alignment/BeaverTails) | ✓ |
| 428 | [privacy-200k-Mistral-Large-3](src/tasksource/tasks.py#L1819) | Classification | [gabrielloiseau/privacy-200k-Mistral-Large-3](https://hf.co/datasets/gabrielloiseau/privacy-200k-Mistral-Large-3) | ✓ |
| 429 | [toxic-chat/toxicchat0124/toxicity](src/tasksource/tasks.py#L1828) | Classification | [lmsys/toxic-chat](https://hf.co/datasets/lmsys/toxic-chat) | ✓ |
| 430 | [toxic-chat/toxicchat0124/jailbreaking](src/tasksource/tasks.py#L1833) | Classification | [lmsys/toxic-chat](https://hf.co/datasets/lmsys/toxic-chat) | ✓ |
| 431 | [clinc_oos/plus](src/tasksource/tasks.py#L1839) | Classification | [clinc/clinc_oos](https://hf.co/datasets/clinc/clinc_oos) |  |
| 432 | [IntentGrasp/all](src/tasksource/tasks.py#L1857) | MultipleChoice | [yuweiyin/IntentGrasp](https://hf.co/datasets/yuweiyin/IntentGrasp) |  |
| 433 | [few_rel/default](src/tasksource/tasks.py#L1895) | Classification | [tasksource/few_rel](https://hf.co/datasets/tasksource/few_rel) |  |
| 434 | [docred](src/tasksource/tasks.py#L1973) | Classification | [YufeiHFUT/DocRED_origin](https://hf.co/datasets/YufeiHFUT/DocRED_origin), [thunlp/docred](https://hf.co/datasets/thunlp/docred) (`json` loader) |  |
| 435 | [chemprot/chemprot_full_source](src/tasksource/tasks.py#L2008) | Classification | [bigbio/chemprot](https://hf.co/datasets/bigbio/chemprot) | ✓ |
| 436 | [PKU-SafeRLHF/helpfulness](src/tasksource/tasks.py#L2014) | MultipleChoice | [PKU-Alignment/PKU-SafeRLHF](https://hf.co/datasets/PKU-Alignment/PKU-SafeRLHF) | ✓ |
| 437 | [PKU-SafeRLHF/safety](src/tasksource/tasks.py#L2019) | MultipleChoice | [PKU-Alignment/PKU-SafeRLHF](https://hf.co/datasets/PKU-Alignment/PKU-SafeRLHF) | ✓ |
| 438 | [HelpSteer/helpfulness](src/tasksource/tasks.py#L2041) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 439 | [HelpSteer/correctness](src/tasksource/tasks.py#L2042) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 440 | [HelpSteer/coherence](src/tasksource/tasks.py#L2043) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 441 | [HelpSteer/complexity](src/tasksource/tasks.py#L2044) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 442 | [HelpSteer/verbosity](src/tasksource/tasks.py#L2045) | Classification | [nvidia/HelpSteer](https://hf.co/datasets/nvidia/HelpSteer) | ✓ |
| 443 | [HelpSteer2/helpfulness](src/tasksource/tasks.py#L2047) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 444 | [HelpSteer2/correctness](src/tasksource/tasks.py#L2048) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 445 | [HelpSteer2/coherence](src/tasksource/tasks.py#L2049) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 446 | [HelpSteer2/complexity](src/tasksource/tasks.py#L2050) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 447 | [HelpSteer2/verbosity](src/tasksource/tasks.py#L2051) | Classification | [nvidia/HelpSteer2](https://hf.co/datasets/nvidia/HelpSteer2) | ✓ |
| 448 | [HelpSteer3/preference](src/tasksource/tasks.py#L2056) | MultipleChoice | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | ✓ |
| 449 | [HelpSteer3/preference_strength](src/tasksource/tasks.py#L2069) | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | ✓ |
| 450 | [HelpSteer3/principle](src/tasksource/tasks.py#L2081) | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) |  |
| 451 | [HelpSteer3/edit_quality](src/tasksource/tasks.py#L2086) | MultipleChoice | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | ✓ |
| 452 | [HelpSteer3/feedback](src/tasksource/tasks.py#L2108) | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) | ✓ |
| 453 | [MSciNLI](src/tasksource/tasks.py#L2114) | Classification | [sadat2307/MSciNLI](https://hf.co/datasets/sadat2307/MSciNLI) |  |
| 454 | [UltraFeedback-paired](src/tasksource/tasks.py#L2117) | MultipleChoice | [pushpdeep/UltraFeedback-paired](https://hf.co/datasets/pushpdeep/UltraFeedback-paired) | ✓ |
| 455 | [prm800k_dpo/solution](src/tasksource/tasks.py#L2121) | MultipleChoice | [tasksource/prm800k_dpo](https://hf.co/datasets/tasksource/prm800k_dpo) | ✓ |
| 456 | [prm800k_dpo/step](src/tasksource/tasks.py#L2124) | MultipleChoice | [tasksource/prm800k_dpo](https://hf.co/datasets/tasksource/prm800k_dpo) | ✓ |
| 457 | [AES2-essay-scoring](src/tasksource/tasks.py#L2128) | Classification | [tasksource/AES2-essay-scoring](https://hf.co/datasets/tasksource/AES2-essay-scoring) | ✓ |
| 458 | [argument-feedback](src/tasksource/tasks.py#L2132) | Classification | [tasksource/argument-feedback](https://hf.co/datasets/tasksource/argument-feedback) | ✓ |
| 459 | [english-grading/cohesion](src/tasksource/tasks.py#L2139) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 460 | [english-grading/syntax](src/tasksource/tasks.py#L2140) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 461 | [english-grading/vocabulary](src/tasksource/tasks.py#L2141) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 462 | [english-grading/phraseology](src/tasksource/tasks.py#L2142) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 463 | [english-grading/grammar](src/tasksource/tasks.py#L2143) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 464 | [english-grading/conventions](src/tasksource/tasks.py#L2144) | Classification | [tasksource/english-grading](https://hf.co/datasets/tasksource/english-grading) | ✓ |
| 465 | [wice](src/tasksource/tasks.py#L2146) | Classification | [tasksource/wice](https://hf.co/datasets/tasksource/wice) |  |
| 466 | [hover](src/tasksource/tasks.py#L2149) | Classification | [Dzeniks/hover](https://hf.co/datasets/Dzeniks/hover) |  |
| 467 | [hover-3way/nli](src/tasksource/tasks.py#L2153) | Classification | [Dzeniks/hover-3way](https://hf.co/datasets/Dzeniks/hover-3way) | ✓ |
| 468 | [tasksource_dpo_pairs](src/tasksource/tasks.py#L2156) | MultipleChoice | [tasksource/tasksource_dpo_pairs](https://hf.co/datasets/tasksource/tasksource_dpo_pairs) | ✓ |
| 469 | [seahorse_summarization_evaluation](src/tasksource/tasks.py#L2159) | Classification | [tasksource/seahorse_summarization_evaluation](https://hf.co/datasets/tasksource/seahorse_summarization_evaluation) |  |
| 470 | [missing-item-prediction/contrastive](src/tasksource/tasks.py#L2171) | Classification | [sileod/missing-item-prediction](https://hf.co/datasets/sileod/missing-item-prediction) | ✓ |
| 471 | [jigsaw_toxicity](src/tasksource/tasks.py#L2177) | Classification | [tasksource/jigsaw_toxicity](https://hf.co/datasets/tasksource/jigsaw_toxicity) |  |
| 472 | [Pol_NLI](src/tasksource/tasks.py#L2180) | Classification | [mlburnham/Pol_NLI](https://hf.co/datasets/mlburnham/Pol_NLI) |  |
| 473 | [synthetic-retrieval-NLI/binary](src/tasksource/tasks.py#L2183) | Classification | [tasksource/synthetic-retrieval-NLI](https://hf.co/datasets/tasksource/synthetic-retrieval-NLI) |  |
| 474 | [synthetic-retrieval-NLI/count](src/tasksource/tasks.py#L2183) | Classification | [tasksource/synthetic-retrieval-NLI](https://hf.co/datasets/tasksource/synthetic-retrieval-NLI) |  |
| 475 | [synthetic-retrieval-NLI/position](src/tasksource/tasks.py#L2183) | Classification | [tasksource/synthetic-retrieval-NLI](https://hf.co/datasets/tasksource/synthetic-retrieval-NLI) |  |
| 476 | [github-issue-similarity](src/tasksource/tasks.py#L2192) | Classification | [WhereIsAI/github-issue-similarity](https://hf.co/datasets/WhereIsAI/github-issue-similarity) |  |
| 477 | [webinstruct/mc](src/tasksource/tasks.py#L2397) | MultipleChoice | [tasksource/webinstruct](https://hf.co/datasets/tasksource/webinstruct) |  |
| 478 | [webinstruct/binary](src/tasksource/tasks.py#L2399) | Classification | [tasksource/webinstruct](https://hf.co/datasets/tasksource/webinstruct) |  |
| 479 | [weblinx/action](src/tasksource/tasks.py#L2463) | Classification | [McGill-NLP/WebLINX](https://hf.co/datasets/McGill-NLP/WebLINX) | ✓ |
| 480 | [weblinx/dom-element](src/tasksource/tasks.py#L2466) | MultipleChoice | [McGill-NLP/WebLINX](https://hf.co/datasets/McGill-NLP/WebLINX) | ✓ |
| 481 | [websrc/yesno](src/tasksource/tasks.py#L2472) | Classification | [tasksource/websrc](https://hf.co/datasets/tasksource/websrc) |  |
| 482 | [websrc/element](src/tasksource/tasks.py#L2474) | MultipleChoice | [tasksource/websrc](https://hf.co/datasets/tasksource/websrc) | ✓ |
| 483 | [mind2web/action](src/tasksource/tasks.py#L2481) | Classification | [tasksource/mind2web-dom](https://hf.co/datasets/tasksource/mind2web-dom) | ✓ |
| 484 | [mind2web/dom-element](src/tasksource/tasks.py#L2484) | MultipleChoice | [tasksource/mind2web-dom](https://hf.co/datasets/tasksource/mind2web-dom) | ✓ |
| 485 | [sentfin](src/tasksource/tasks.py#L2489) | Classification | [tasksource/sentfin](https://hf.co/datasets/tasksource/sentfin) | ✓ |
| 486 | [wands](src/tasksource/tasks.py#L2493) | Classification | [tasksource/wands](https://hf.co/datasets/tasksource/wands) | ✓ |
| 487 | [scirepeval/search](src/tasksource/tasks.py#L2497) | Classification | [tasksource/scirepeval-search](https://hf.co/datasets/tasksource/scirepeval-search) | ✓ |
| 488 | [maud/consideration](src/tasksource/tasks.py#L2549) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 489 | [maud/bringdown-timing](src/tasksource/tasks.py#L2553) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 490 | [maud/ordinary-course-consent](src/tasksource/tasks.py#L2557) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 491 | [maud/ordinary-course-efforts](src/tasksource/tasks.py#L2561) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 492 | [maud/antitrust-efforts](src/tasksource/tasks.py#L2565) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 493 | [maud/specific-performance](src/tasksource/tasks.py#L2569) | Classification | [theatticusproject/maud](https://hf.co/datasets/theatticusproject/maud) | ✓ |
| 494 | [ddi/relation](src/tasksource/tasks.py#L2574) | Classification | [OpenMed/DDI-Corpus-Processed](https://hf.co/datasets/OpenMed/DDI-Corpus-Processed) | ✓ |
| 495 | [patent-classification/abstract](src/tasksource/tasks.py#L2581) | Classification | [ccdv/patent-classification](https://hf.co/datasets/ccdv/patent-classification) | ✓ |
| 496 | [tabfact](src/tasksource/tasks.py#L2588) | Classification | [wenhu/tab_fact](https://hf.co/datasets/wenhu/tab_fact) | ✓ |
| 497 | [infotabs](src/tasksource/tasks.py#L2596) | Classification | [table-benchmark/infotabs](https://hf.co/datasets/table-benchmark/infotabs) | ✓ |
| 498 | [squad-v2/answerability](src/tasksource/tasks.py#L2608) | Classification | [rajpurkar/squad_v2](https://hf.co/datasets/rajpurkar/squad_v2) | ✓ |
| 499 | [cuad/answerability](src/tasksource/tasks.py#L2613) | Classification | [theatticusproject/cuad-qa](https://hf.co/datasets/theatticusproject/cuad-qa) | ✓ |
| 500 | [hallmarks-of-cancer/yesno](src/tasksource/tasks.py#L2632) | Classification | [bigbio/hallmarks_of_cancer](https://hf.co/datasets/bigbio/hallmarks_of_cancer) | ✓ |
| 501 | [semeval2014-absa/restaurants](src/tasksource/tasks.py#L2654) | Classification | [alexcadillon/SemEval2014Task4](https://hf.co/datasets/alexcadillon/SemEval2014Task4) | ✓ |
| 502 | [semeval2014-absa/laptops](src/tasksource/tasks.py#L2654) | Classification | [alexcadillon/SemEval2014Task4](https://hf.co/datasets/alexcadillon/SemEval2014Task4) | ✓ |
| 503 | [record/candidate](src/tasksource/tasks.py#L2676) | Classification | [aps/super_glue](https://hf.co/datasets/aps/super_glue) | ✓ |
| 504 | [multidogo/software/intent](src/tasksource/tasks.py#L2699) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 505 | [multidogo/media/intent](src/tasksource/tasks.py#L2699) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 506 | [multidogo/airline/intent](src/tasksource/tasks.py#L2699) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 507 | [multidogo/fastfood/intent](src/tasksource/tasks.py#L2699) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 508 | [multidogo/finance/intent](src/tasksource/tasks.py#L2699) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 509 | [multidogo/insurance/intent](src/tasksource/tasks.py#L2699) | Classification | [jpcorb20/multidogo](https://hf.co/datasets/jpcorb20/multidogo) | ✓ |
| 510 | [wikisql/aggregation](src/tasksource/tasks.py#L2727) | Classification | [Salesforce/wikisql](https://hf.co/datasets/Salesforce/wikisql) | ✓ |
| 511 | [wikisql/column](src/tasksource/tasks.py#L2732) | MultipleChoice | [Salesforce/wikisql](https://hf.co/datasets/Salesforce/wikisql) | ✓ |
| 512 | [wikisql/operator](src/tasksource/tasks.py#L2737) | Classification | [Salesforce/wikisql](https://hf.co/datasets/Salesforce/wikisql) | ✓ |
| 513 | [acord](src/tasksource/tasks.py#L2742) | Classification | [tasksource/acord](https://hf.co/datasets/tasksource/acord) | ✓ |
| 514 | [edos](src/tasksource/tasks.py#L2746) | Classification | [tasksource/edos](https://hf.co/datasets/tasksource/edos) | ✓ |
| 515 | [evidence-inference](src/tasksource/tasks.py#L2749) | Classification | [tasksource/evidence-inference](https://hf.co/datasets/tasksource/evidence-inference) | ✓ |
| 516 | [quac/answerability](src/tasksource/tasks.py#L2771) | Classification | [allenai/quac](https://hf.co/datasets/allenai/quac) | ✓ |
| 517 | [hatexplain/majority](src/tasksource/tasks.py#L2779) | Classification | [Hate-speech-CNERG/hatexplain](https://hf.co/datasets/Hate-speech-CNERG/hatexplain) | ✓ |
| 518 | [hotpotqa/supporting-fact](src/tasksource/tasks.py#L2796) | Classification | [hotpotqa/hotpot_qa](https://hf.co/datasets/hotpotqa/hotpot_qa) | ✓ |
| 519 | [multiwoz22/dialogue-act](src/tasksource/tasks.py#L2840) | Classification | [pfb30/multi_woz_v22](https://hf.co/datasets/pfb30/multi_woz_v22) | ✓ |
| 520 | [multiwoz22/categorical-slot](src/tasksource/tasks.py#L2843) | Classification | [pfb30/multi_woz_v22](https://hf.co/datasets/pfb30/multi_woz_v22) | ✓ |
| 521 | [schema-guided/intent](src/tasksource/tasks.py#L2846) | Classification | [google-research-datasets/schema_guided_dstc8](https://hf.co/datasets/google-research-datasets/schema_guided_dstc8) | ✓ |
| 522 | [coqa/yesno](src/tasksource/tasks.py#L2863) | Classification | [stanfordnlp/coqa](https://hf.co/datasets/stanfordnlp/coqa) | ✓ |
| 523 | [qasper/yesno](src/tasksource/tasks.py#L2884) | Classification | [allenai/qasper](https://hf.co/datasets/allenai/qasper) | ✓ |
| 524 | [topv2/intent](src/tasksource/tasks.py#L2896) | Classification | [WillHeld/top_v2](https://hf.co/datasets/WillHeld/top_v2) | ✓ |
| 525 | [slurp/intent](src/tasksource/tasks.py#L2904) | Classification | [tasksource/slurp](https://hf.co/datasets/tasksource/slurp) | ✓ |
| 526 | [mcscript2](src/tasksource/tasks.py#L2908) | MultipleChoice | [tasksource/mcscript2](https://hf.co/datasets/tasksource/mcscript2) |  |
| 527 | [fnc1/stance](src/tasksource/tasks.py#L2913) | Classification | [tasksource/fnc1](https://hf.co/datasets/tasksource/fnc1) | ✓ |
| 528 | [perspectrum/stance](src/tasksource/tasks.py#L2918) | Classification | [tasksource/perspectrum](https://hf.co/datasets/tasksource/perspectrum) | ✓ |
| 529 | [ukpconvarg/convincingness](src/tasksource/tasks.py#L2923) | MultipleChoice | [tasksource/ukpconvarg](https://hf.co/datasets/tasksource/ukpconvarg) | ✓ |
| 530 | [discogem/relation](src/tasksource/tasks.py#L2928) | Classification | [tasksource/discogem](https://hf.co/datasets/tasksource/discogem) | ✓ |
| 531 | [pubmed-rct/role](src/tasksource/tasks.py#L2936) | Classification | [tasksource/pubmed-rct](https://hf.co/datasets/tasksource/pubmed-rct) | ✓ |
| 532 | [maven-arg/role](src/tasksource/tasks.py#L2964) | Classification | [mneb/maven-arg](https://hf.co/datasets/mneb/maven-arg) | ✓ |
| 533 | [maven/event](src/tasksource/tasks.py#L2995) | Classification | [mneb/maven](https://hf.co/datasets/mneb/maven) | ✓ |
| 534 | [empathetic-dialogues/emotion](src/tasksource/tasks.py#L2999) | Classification | [facebook/empathetic_dialogues](https://hf.co/datasets/facebook/empathetic_dialogues) | ✓ |
| 535 | [olid/offensive](src/tasksource/tasks.py#L3007) | Classification | [christophsonntag/OLID](https://hf.co/datasets/christophsonntag/OLID) | ✓ |
| 536 | [isarcasm/intended](src/tasksource/tasks.py#L3013) | Classification | [viethq1906/isarcasm_2022_taskA_En](https://hf.co/datasets/viethq1906/isarcasm_2022_taskA_En) | ✓ |
| 537 | [magpie/idiomatic](src/tasksource/tasks.py#L3038) | Classification | [idioms-collaboration/magpie](https://hf.co/datasets/idioms-collaboration/magpie) | ✓ |
| 538 | [preco/coreference](src/tasksource/tasks.py#L3074) | Classification | [coref-data/preco_raw](https://hf.co/datasets/coref-data/preco_raw) | ✓ |
| 539 | [redial/preference](src/tasksource/tasks.py#L3094) | Classification | [community-datasets/re_dial](https://hf.co/datasets/community-datasets/re_dial) | ✓ |
| 540 | [sst5](src/tasksource/tasks.py#L3099) | Classification | [SetFit/sst5](https://hf.co/datasets/SetFit/sst5) | ✓ |
| 541 | [spider/column](src/tasksource/tasks.py#L3107) | MultipleChoice | [tasksource/spider](https://hf.co/datasets/tasksource/spider) | ✓ |
| 542 | [semcor](src/tasksource/tasks.py#L3111) | MultipleChoice | [tasksource/semcor](https://hf.co/datasets/tasksource/semcor) | ✓ |
| 543 | [streusle/noun](src/tasksource/tasks.py#L3114) | Classification | [tasksource/streusle](https://hf.co/datasets/tasksource/streusle) | ✓ |
| 544 | [streusle/verb](src/tasksource/tasks.py#L3118) | Classification | [tasksource/streusle](https://hf.co/datasets/tasksource/streusle) | ✓ |
| 545 | [maven-ere/causal](src/tasksource/tasks.py#L3161) | Classification | [Nofing/Maven-ERE-span](https://hf.co/datasets/Nofing/Maven-ERE-span) | ✓ |
| 546 | [biored/relation](src/tasksource/tasks.py#L3183) | Classification | [gyorilab/BioRED](https://hf.co/datasets/gyorilab/BioRED) | ✓ |
| 547 | [matres/temporal](src/tasksource/tasks.py#L3190) | Classification | [tasksource/matres](https://hf.co/datasets/tasksource/matres) | ✓ |
| 548 | [causal-news](src/tasksource/tasks.py#L3199) | Classification | [thagen/CNCv2](https://hf.co/datasets/thagen/CNCv2) | ✓ |
| 549 | [slurp/slot-type](src/tasksource/tasks.py#L3246) | Classification | [tasksource/slurp](https://hf.co/datasets/tasksource/slurp) | ✓ |
| 550 | [topv2/slot-type](src/tasksource/tasks.py#L3251) | Classification | [WillHeld/top_v2](https://hf.co/datasets/WillHeld/top_v2) | ✓ |
| 551 | [wikiqa](src/tasksource/tasks.py#L3264) | Classification | [microsoft/wiki_qa](https://hf.co/datasets/microsoft/wiki_qa) | ✓ |
| 552 | [nevir](src/tasksource/tasks.py#L3279) | MultipleChoice | [orionweller/NevIR](https://hf.co/datasets/orionweller/NevIR) | ✓ |
| 553 | [antique](src/tasksource/tasks.py#L3287) | Classification | [tasksource/antique](https://hf.co/datasets/tasksource/antique) | ✓ |
| 554 | [causal-reasoning/transitivity](src/tasksource/tasks.py#L3313) | Classification | [ludwigw/causal-reasoning-benchmarks](https://hf.co/datasets/ludwigw/causal-reasoning-benchmarks) | ✓ |
| 555 | [cebab/rating](src/tasksource/tasks.py#L3330) | Classification | [EleutherAI/CEBaB](https://hf.co/datasets/EleutherAI/CEBaB) | ✓ |
| 556 | [cebab/food](src/tasksource/tasks.py#L3334) | Classification | [EleutherAI/CEBaB](https://hf.co/datasets/EleutherAI/CEBaB) | ✓ |
| 557 | [cebab/ambiance](src/tasksource/tasks.py#L3337) | Classification | [EleutherAI/CEBaB](https://hf.co/datasets/EleutherAI/CEBaB) | ✓ |
| 558 | [cebab/service](src/tasksource/tasks.py#L3340) | Classification | [EleutherAI/CEBaB](https://hf.co/datasets/EleutherAI/CEBaB) | ✓ |
| 559 | [cebab/noise](src/tasksource/tasks.py#L3343) | Classification | [EleutherAI/CEBaB](https://hf.co/datasets/EleutherAI/CEBaB) | ✓ |
| 560 | [fabsa/sentiment](src/tasksource/tasks.py#L3356) | Classification | [jordiclive/FABSA](https://hf.co/datasets/jordiclive/FABSA) | ✓ |
| 561 | [tab/identifier](src/tasksource/tasks.py#L3373) | Classification | [doshimit3015/text-anonymization-benchmark](https://hf.co/datasets/doshimit3015/text-anonymization-benchmark) | ✓ |
| 562 | [newsmtsc](src/tasksource/tasks.py#L3388) | Classification | [fhamborg/news_sentiment_newsmtsc](https://hf.co/datasets/fhamborg/news_sentiment_newsmtsc) | ✓ |
| 563 | [sentihood/sentiment](src/tasksource/tasks.py#L3416) | Classification | [bhavnicksm/sentihood](https://hf.co/datasets/bhavnicksm/sentihood) | ✓ |
| 564 | [mams/atsa](src/tasksource/tasks.py#L3433) | Classification | [NEUDM/mams](https://hf.co/datasets/NEUDM/mams) | ✓ |
| 565 | [quality](src/tasksource/tasks.py#L3441) | MultipleChoice | [emozilla/quality](https://hf.co/datasets/emozilla/quality) |  |
| 566 | [clear/readability](src/tasksource/tasks.py#L3450) | Classification | [casey-martin/CommonLit-Ease-of-Readability](https://hf.co/datasets/casey-martin/CommonLit-Ease-of-Readability) | ✓ |
| 567 | [esconv/strategy](src/tasksource/tasks.py#L3463) | Classification | [tasksource/esconv](https://hf.co/datasets/tasksource/esconv) | ✓ |
| 568 | [prosocial-dialog/safety](src/tasksource/tasks.py#L3484) | Classification | [allenai/prosocial-dialog](https://hf.co/datasets/allenai/prosocial-dialog) | ✓ |
| 569 | [gap](src/tasksource/tasks.py#L3504) | Classification | [coref-data/gap_raw](https://hf.co/datasets/coref-data/gap_raw) | ✓ |
| 570 | [trip/cloze](src/tasksource/tasks.py#L3512) | MultipleChoice | [sled-umich/TRIP](https://hf.co/datasets/sled-umich/TRIP) | ✓ |
| 571 | [trip/order](src/tasksource/tasks.py#L3521) | MultipleChoice | [sled-umich/TRIP](https://hf.co/datasets/sled-umich/TRIP) | ✓ |
| 572 | [sbic/offensiveness](src/tasksource/tasks.py#L3531) | Classification | [tasksource/sbic](https://hf.co/datasets/tasksource/sbic) | ✓ |
| 573 | [sbic/intent](src/tasksource/tasks.py#L3538) | Classification | [tasksource/sbic](https://hf.co/datasets/tasksource/sbic) | ✓ |
| 574 | [faithdial/grounding](src/tasksource/tasks.py#L3560) | Classification | [McGill-NLP/FaithDial](https://hf.co/datasets/McGill-NLP/FaithDial) | ✓ |
| 575 | [ragtruth/grounding](src/tasksource/tasks.py#L3577) | Classification | [wandb/RAGTruth-processed](https://hf.co/datasets/wandb/RAGTruth-processed) | ✓ |
| 576 | [comve](src/tasksource/tasks.py#L3589) | MultipleChoice | [tasksource/comve](https://hf.co/datasets/tasksource/comve) |  |

## Soft labels

Annotations whose label is a distribution (annotator votes, rater shares, survey counts), loaded with `load_task(id, soft=True)`. Those with a hard view are also listed above, by their majority label; the others have soft labels only. `votes` are shares of annotators, `mean` a numeric value interpolated over score anchors (see each task for its meaning); annotators is the typical count per item (vote shares from fewer than five are coarse, and the Jev build leaves them out).

| id | kind | aggregation | annotators | default view | dataset |
|---|---|---|--:|---|---|
| [glue/stsb](src/tasksource/tasks.py#L49) | score | mean |  | regression | [nyu-mll/glue](https://hf.co/datasets/nyu-mll/glue) |
| [sick/relatedness](src/tasksource/tasks.py#L88) | score | mean | 10 | regression | [tasksource/sick](https://hf.co/datasets/tasksource/sick) |
| [google_wellformed_query](src/tasksource/tasks.py#L838) | noul | votes | 5 | Classification | [tasksource/google_wellformed_query](https://hf.co/datasets/tasksource/google_wellformed_query) |
| [sts-companion](src/tasksource/tasks.py#L1029) | score | mean |  | regression | [tasksource/sts-companion](https://hf.co/datasets/tasksource/sts-companion) |
| [acceptability-prediction/rating_votes](src/tasksource/tasks.py#L1260) | score | votes | 15 |  | [tasksource/acceptability-prediction](https://hf.co/datasets/tasksource/acceptability-prediction) |
| [acceptability-prediction/binary_votes](src/tasksource/tasks.py#L1265) | noul | votes | 15 |  | [tasksource/acceptability-prediction](https://hf.co/datasets/tasksource/acceptability-prediction) |
| [HelpSteer3/individual_preferences](src/tasksource/tasks.py#L2075) | score | votes | 3 |  | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) |
| [HelpSteer3/feedback](src/tasksource/tasks.py#L2108) | score | votes | 3 | Classification | [nvidia/HelpSteer3](https://hf.co/datasets/nvidia/HelpSteer3) |
| [proto_qa/proto_qa](src/tasksource/tasks.py#L2202) | choice | votes | 100 |  | [community-datasets/proto_qa](https://hf.co/datasets/community-datasets/proto_qa) |
| [UNLI](src/tasksource/tasks.py#L2216) | noul | mean | 2 |  | [Zhengping/UNLI](https://hf.co/datasets/Zhengping/UNLI) |
| [chaos-mnli-ambiguity/votes](src/tasksource/tasks.py#L2220) | choice | votes | 100 |  | [tasksource/chaos-mnli-ambiguity](https://hf.co/datasets/tasksource/chaos-mnli-ambiguity) |
| [hate_speech_offensive/votes](src/tasksource/tasks.py#L2224) | choice | votes | 3 |  | [tdavidson/hate_speech_offensive](https://hf.co/datasets/tdavidson/hate_speech_offensive) |
| [scruples/verdict_votes](src/tasksource/tasks.py#L2232) | choice | votes | 8 |  | [tasksource/scruples](https://hf.co/datasets/tasksource/scruples) |
| [BeaverTails/unsafe_votes](src/tasksource/tasks.py#L2246) | noul | votes | 3 |  | [PKU-Alignment/BeaverTails](https://hf.co/datasets/PKU-Alignment/BeaverTails) |
| [dynasent/r1_votes](src/tasksource/tasks.py#L2257) | choice | votes | 5 |  | [dynabench/dynasent](https://hf.co/datasets/dynabench/dynasent) |
| [dynasent/r2_votes](src/tasksource/tasks.py#L2261) | choice | votes | 5 |  | [dynabench/dynasent](https://hf.co/datasets/dynabench/dynasent) |
| [hatexplain/votes](src/tasksource/tasks.py#L2266) | choice | votes | 3 |  | [Hate-speech-CNERG/hatexplain](https://hf.co/datasets/Hate-speech-CNERG/hatexplain) |
| [measuring-hate-speech/sentiment](src/tasksource/tasks.py#L2281) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/respect](src/tasksource/tasks.py#L2283) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/insult](src/tasksource/tasks.py#L2285) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/humiliate](src/tasksource/tasks.py#L2286) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/status](src/tasksource/tasks.py#L2287) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/dehumanize](src/tasksource/tasks.py#L2289) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/violence](src/tasksource/tasks.py#L2291) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/genocide](src/tasksource/tasks.py#L2292) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/attack_defend](src/tasksource/tasks.py#L2294) | score | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [measuring-hate-speech/hatespeech](src/tasksource/tasks.py#L2296) | choice | votes | 3 |  | [tasksource/measuring-hate-speech-votes](https://hf.co/datasets/tasksource/measuring-hate-speech-votes) |
| [wikipedia-detox/attack](src/tasksource/tasks.py#L2308) | choice | votes | 10 |  | [tasksource/wikipedia-detox-votes](https://hf.co/datasets/tasksource/wikipedia-detox-votes) |
| [wikipedia-detox/aggression](src/tasksource/tasks.py#L2310) | choice | votes | 10 |  | [tasksource/wikipedia-detox-votes](https://hf.co/datasets/tasksource/wikipedia-detox-votes) |
| [wikipedia-detox/aggression_score](src/tasksource/tasks.py#L2312) | score | votes | 10 |  | [tasksource/wikipedia-detox-votes](https://hf.co/datasets/tasksource/wikipedia-detox-votes) |
| [wikipedia-detox/toxicity](src/tasksource/tasks.py#L2316) | choice | votes | 10 |  | [tasksource/wikipedia-detox-votes](https://hf.co/datasets/tasksource/wikipedia-detox-votes) |
| [wikipedia-detox/toxicity_score](src/tasksource/tasks.py#L2318) | score | votes | 10 |  | [tasksource/wikipedia-detox-votes](https://hf.co/datasets/tasksource/wikipedia-detox-votes) |
| [lewidi/md_agreement](src/tasksource/tasks.py#L2333) | noul | votes | 5 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/hs_brexit](src/tasksource/tasks.py#L2334) | noul | votes | 6 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/armis](src/tasksource/tasks.py#L2336) | noul | votes | 3 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/conv_abuse](src/tasksource/tasks.py#L2338) | noul | votes | 3 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/mp](src/tasksource/tasks.py#L2340) | noul | votes | 5 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/csc](src/tasksource/tasks.py#L2341) | score | votes | 4 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/varierrnli/entailment](src/tasksource/tasks.py#L2349) | noul | votes | 4 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/varierrnli/neutral](src/tasksource/tasks.py#L2350) | noul | votes | 4 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [lewidi/varierrnli/contradiction](src/tasksource/tasks.py#L2351) | noul | votes | 4 |  | [tasksource/lewidi](https://hf.co/datasets/tasksource/lewidi) |
| [civil_comments/toxicity_share](src/tasksource/tasks.py#L2359) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/severe_toxicity_share](src/tasksource/tasks.py#L2360) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/obscene_share](src/tasksource/tasks.py#L2361) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/threat_share](src/tasksource/tasks.py#L2362) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/insult_share](src/tasksource/tasks.py#L2363) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/identity_attack_share](src/tasksource/tasks.py#L2364) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [civil_comments/sexual_explicit_share](src/tasksource/tasks.py#L2365) | noul | votes | 6 |  | [google/civil_comments](https://hf.co/datasets/google/civil_comments) |
| [oasst2/quality](src/tasksource/tasks.py#L2382) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/helpfulness](src/tasksource/tasks.py#L2383) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/creativity](src/tasksource/tasks.py#L2384) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/humor](src/tasksource/tasks.py#L2385) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/toxicity](src/tasksource/tasks.py#L2386) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/violence](src/tasksource/tasks.py#L2387) | score | mean | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/spam](src/tasksource/tasks.py#L2388) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/fails_task](src/tasksource/tasks.py#L2389) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/not_appropriate](src/tasksource/tasks.py#L2390) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/hate_speech](src/tasksource/tasks.py#L2391) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/sexual_content](src/tasksource/tasks.py#L2392) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/pii](src/tasksource/tasks.py#L2393) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [oasst2/lang_mismatch](src/tasksource/tasks.py#L2394) | noul | votes | 3 |  | [tasksource/oasst2_dense_flat](https://hf.co/datasets/tasksource/oasst2_dense_flat) |
| [scirepeval/search](src/tasksource/tasks.py#L2497) | score | mean |  | regression | [tasksource/scirepeval-search](https://hf.co/datasets/tasksource/scirepeval-search) |
| [discogem/relation](src/tasksource/tasks.py#L2928) | choice | votes | 10 | Classification | [tasksource/discogem](https://hf.co/datasets/tasksource/discogem) |
| [clear/readability](src/tasksource/tasks.py#L3450) | score | mean |  | regression | [casey-martin/CommonLit-Ease-of-Readability](https://hf.co/datasets/casey-martin/CommonLit-Ease-of-Readability) |
