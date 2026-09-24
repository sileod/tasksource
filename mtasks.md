550 multilingual tasks. Load one with `load_task(id, multilingual=True)`; the annotations are in [multilingual_tasks.py](src/tasksource/multilingual_tasks.py), and tasks kept out on purpose are in [parked.py](src/tasksource/parked.py).

| id | type | dataset | question |
|---|---|---|---|
| multilingual-NLI-26lang-2mil7 | Classification | [MoritzLaurer/multilingual-NLI-26lang-2mil7](https://hf.co/datasets/MoritzLaurer/multilingual-NLI-26lang-2mil7) |  |
| xnli | Classification | [facebook/xnli](https://hf.co/datasets/facebook/xnli) |  |
| americas_nli/all_languages | Classification | [nala-cub/americas_nli](https://hf.co/datasets/nala-cub/americas_nli) |  |
| stsb_multi_mt/es | Classification | [PhilipMay/stsb_multi_mt](https://hf.co/datasets/PhilipMay/stsb_multi_mt) | How similar are the two sentences, from 0 (unrelated) to 1 (equivalent)? |
| stsb_multi_mt/de | Classification | [PhilipMay/stsb_multi_mt](https://hf.co/datasets/PhilipMay/stsb_multi_mt) | How similar are the two sentences, from 0 (unrelated) to 1 (equivalent)? |
| stsb_multi_mt/it | Classification | [PhilipMay/stsb_multi_mt](https://hf.co/datasets/PhilipMay/stsb_multi_mt) | How similar are the two sentences, from 0 (unrelated) to 1 (equivalent)? |
| stsb_multi_mt/fr | Classification | [PhilipMay/stsb_multi_mt](https://hf.co/datasets/PhilipMay/stsb_multi_mt) | How similar are the two sentences, from 0 (unrelated) to 1 (equivalent)? |
| stsb_multi_mt/pl | Classification | [PhilipMay/stsb_multi_mt](https://hf.co/datasets/PhilipMay/stsb_multi_mt) | How similar are the two sentences, from 0 (unrelated) to 1 (equivalent)? |
| stsb_multi_mt/pt | Classification | [PhilipMay/stsb_multi_mt](https://hf.co/datasets/PhilipMay/stsb_multi_mt) | How similar are the two sentences, from 0 (unrelated) to 1 (equivalent)? |
| stsb_multi_mt/ru | Classification | [PhilipMay/stsb_multi_mt](https://hf.co/datasets/PhilipMay/stsb_multi_mt) | How similar are the two sentences, from 0 (unrelated) to 1 (equivalent)? |
| stsb_multi_mt/zh | Classification | [PhilipMay/stsb_multi_mt](https://hf.co/datasets/PhilipMay/stsb_multi_mt) | How similar are the two sentences, from 0 (unrelated) to 1 (equivalent)? |
| stsb_multi_mt/nl | Classification | [PhilipMay/stsb_multi_mt](https://hf.co/datasets/PhilipMay/stsb_multi_mt) | How similar are the two sentences, from 0 (unrelated) to 1 (equivalent)? |
| stsb_multi_mt/en | Classification | [PhilipMay/stsb_multi_mt](https://hf.co/datasets/PhilipMay/stsb_multi_mt) | How similar are the two sentences, from 0 (unrelated) to 1 (equivalent)? |
| paws-x/fr | Classification | [google-research-datasets/paws-x](https://hf.co/datasets/google-research-datasets/paws-x) |  |
| paws-x/ja | Classification | [google-research-datasets/paws-x](https://hf.co/datasets/google-research-datasets/paws-x) |  |
| paws-x/ko | Classification | [google-research-datasets/paws-x](https://hf.co/datasets/google-research-datasets/paws-x) |  |
| paws-x/zh | Classification | [google-research-datasets/paws-x](https://hf.co/datasets/google-research-datasets/paws-x) |  |
| paws-x/es | Classification | [google-research-datasets/paws-x](https://hf.co/datasets/google-research-datasets/paws-x) |  |
| paws-x/de | Classification | [google-research-datasets/paws-x](https://hf.co/datasets/google-research-datasets/paws-x) |  |
| paws-x/en | Classification | [google-research-datasets/paws-x](https://hf.co/datasets/google-research-datasets/paws-x) |  |
| miam | Classification | csv |  |
| x-stance | Classification | [michiel/xstance](https://hf.co/datasets/michiel/xstance) |  |
| offenseval_2020/ar | Classification | [khalidalt/offenseval_2020_ar](https://hf.co/datasets/khalidalt/offenseval_2020_ar) |  |
| offenseval_2020/da | Classification | [tasksource/offenseval_2020](https://hf.co/datasets/tasksource/offenseval_2020) |  |
| offenseval_2020/gr | Classification | [tasksource/offenseval_2020](https://hf.co/datasets/tasksource/offenseval_2020) |  |
| offenseval_2020/tr | Classification | [tasksource/offenseval_2020](https://hf.co/datasets/tasksource/offenseval_2020) |  |
| offenseval_dravidian/tamil | Classification | [community-datasets/offenseval_dravidian](https://hf.co/datasets/community-datasets/offenseval_dravidian) |  |
| offenseval_dravidian/malayalam | Classification | [community-datasets/offenseval_dravidian](https://hf.co/datasets/community-datasets/offenseval_dravidian) |  |
| offenseval_dravidian/kannada | Classification | [community-datasets/offenseval_dravidian](https://hf.co/datasets/community-datasets/offenseval_dravidian) |  |
| MLMA_hate_speech | Classification | [nedjmaou/MLMA_hate_speech](https://hf.co/datasets/nedjmaou/MLMA_hate_speech) |  |
| x-fact | Classification | [tasksource/x-fact](https://hf.co/datasets/tasksource/x-fact) |  |
| xglue/nc | Classification | [SetFit/xglue_nc](https://hf.co/datasets/SetFit/xglue_nc) |  |
| xglue/qadsm | Classification | [tasksource/xglue](https://hf.co/datasets/tasksource/xglue) | Is the ad relevant to the query? |
| xglue/qam | Classification | [tasksource/xglue](https://hf.co/datasets/tasksource/xglue) | Does the passage answer the query? |
| xglue/wpr | Classification | [tasksource/xglue](https://hf.co/datasets/tasksource/xglue) | How relevant is the web page to the query? |
| xlwic/xlwic_fr_fr | Classification | [tasksource/xlwic](https://hf.co/datasets/tasksource/xlwic) |  |
| xlwic/xlwic_en_ko | Classification | [tasksource/xlwic](https://hf.co/datasets/tasksource/xlwic) |  |
| xlwic/xlwic_it_it | Classification | [tasksource/xlwic](https://hf.co/datasets/tasksource/xlwic) |  |
| xlwic/xlwic_de_de | Classification | [tasksource/xlwic](https://hf.co/datasets/tasksource/xlwic) |  |
| oasst1_dense_flat/quality | Classification | [tasksource/oasst1_dense_flat](https://hf.co/datasets/tasksource/oasst1_dense_flat) | How good is the reply, from 0 (low quality) to 1 (high quality)? |
| oasst1_dense_flat/toxicity | Classification | [tasksource/oasst1_dense_flat](https://hf.co/datasets/tasksource/oasst1_dense_flat) | How toxic is the reply, from 0 (not toxic) to 1 (very toxic)? |
| oasst1_dense_flat/helpfulness | Classification | [tasksource/oasst1_dense_flat](https://hf.co/datasets/tasksource/oasst1_dense_flat) | How helpful is the reply, from 0 (unhelpful) to 1 (helpful)? |
| language-identification | Classification | [papluca/language-identification](https://hf.co/datasets/papluca/language-identification) | What language is this text in? |
| wili_2018 | Classification | [MartinThoma/wili_2018](https://hf.co/datasets/MartinThoma/wili_2018) | What language is this text in? |
| exams/multilingual | MultipleChoice | [mhardalov/exams](https://hf.co/datasets/mhardalov/exams) |  |
| xcsr/X-CSQA-jap | MultipleChoice | [INK-USC/xcsr](https://hf.co/datasets/INK-USC/xcsr) |  |
| xcsr/X-CSQA-it | MultipleChoice | [INK-USC/xcsr](https://hf.co/datasets/INK-USC/xcsr) |  |
| xcsr/X-CSQA-hi | MultipleChoice | [INK-USC/xcsr](https://hf.co/datasets/INK-USC/xcsr) |  |
| xcsr/X-CSQA-de | MultipleChoice | [INK-USC/xcsr](https://hf.co/datasets/INK-USC/xcsr) |  |
| xcsr/X-CSQA-es | MultipleChoice | [INK-USC/xcsr](https://hf.co/datasets/INK-USC/xcsr) |  |
| xcsr/X-CSQA-nl | MultipleChoice | [INK-USC/xcsr](https://hf.co/datasets/INK-USC/xcsr) |  |
| xcsr/X-CSQA-ar | MultipleChoice | [INK-USC/xcsr](https://hf.co/datasets/INK-USC/xcsr) |  |
| xcsr/X-CSQA-fr | MultipleChoice | [INK-USC/xcsr](https://hf.co/datasets/INK-USC/xcsr) |  |
| xcsr/X-CSQA-pl | MultipleChoice | [INK-USC/xcsr](https://hf.co/datasets/INK-USC/xcsr) |  |
| xcsr/X-CSQA-en | MultipleChoice | [INK-USC/xcsr](https://hf.co/datasets/INK-USC/xcsr) |  |
| xcsr/X-CSQA-ru | MultipleChoice | [INK-USC/xcsr](https://hf.co/datasets/INK-USC/xcsr) |  |
| xcsr/X-CSQA-pt | MultipleChoice | [INK-USC/xcsr](https://hf.co/datasets/INK-USC/xcsr) |  |
| xcsr/X-CSQA-vi | MultipleChoice | [INK-USC/xcsr](https://hf.co/datasets/INK-USC/xcsr) |  |
| xcsr/X-CSQA-zh | MultipleChoice | [INK-USC/xcsr](https://hf.co/datasets/INK-USC/xcsr) |  |
| xcsr/X-CSQA-ur | MultipleChoice | [INK-USC/xcsr](https://hf.co/datasets/INK-USC/xcsr) |  |
| xcsr/X-CSQA-sw | MultipleChoice | [INK-USC/xcsr](https://hf.co/datasets/INK-USC/xcsr) |  |
| xcsr/X-CODAH-ar | MultipleChoice | [INK-USC/xcsr](https://hf.co/datasets/INK-USC/xcsr) | Which sentence is most plausible? |
| xcsr/X-CODAH-pl | MultipleChoice | [INK-USC/xcsr](https://hf.co/datasets/INK-USC/xcsr) | Which sentence is most plausible? |
| xcsr/X-CODAH-pt | MultipleChoice | [INK-USC/xcsr](https://hf.co/datasets/INK-USC/xcsr) | Which sentence is most plausible? |
| xcsr/X-CODAH-ru | MultipleChoice | [INK-USC/xcsr](https://hf.co/datasets/INK-USC/xcsr) | Which sentence is most plausible? |
| xcsr/X-CODAH-sw | MultipleChoice | [INK-USC/xcsr](https://hf.co/datasets/INK-USC/xcsr) | Which sentence is most plausible? |
| xcsr/X-CODAH-vi | MultipleChoice | [INK-USC/xcsr](https://hf.co/datasets/INK-USC/xcsr) | Which sentence is most plausible? |
| xcsr/X-CODAH-zh | MultipleChoice | [INK-USC/xcsr](https://hf.co/datasets/INK-USC/xcsr) | Which sentence is most plausible? |
| xcsr/X-CODAH-ur | MultipleChoice | [INK-USC/xcsr](https://hf.co/datasets/INK-USC/xcsr) | Which sentence is most plausible? |
| xcsr/X-CODAH-jap | MultipleChoice | [INK-USC/xcsr](https://hf.co/datasets/INK-USC/xcsr) | Which sentence is most plausible? |
| xcsr/X-CODAH-it | MultipleChoice | [INK-USC/xcsr](https://hf.co/datasets/INK-USC/xcsr) | Which sentence is most plausible? |
| xcsr/X-CODAH-hi | MultipleChoice | [INK-USC/xcsr](https://hf.co/datasets/INK-USC/xcsr) | Which sentence is most plausible? |
| xcsr/X-CODAH-fr | MultipleChoice | [INK-USC/xcsr](https://hf.co/datasets/INK-USC/xcsr) | Which sentence is most plausible? |
| xcsr/X-CODAH-es | MultipleChoice | [INK-USC/xcsr](https://hf.co/datasets/INK-USC/xcsr) | Which sentence is most plausible? |
| xcsr/X-CODAH-en | MultipleChoice | [INK-USC/xcsr](https://hf.co/datasets/INK-USC/xcsr) | Which sentence is most plausible? |
| xcsr/X-CODAH-de | MultipleChoice | [INK-USC/xcsr](https://hf.co/datasets/INK-USC/xcsr) | Which sentence is most plausible? |
| xcsr/X-CODAH-nl | MultipleChoice | [INK-USC/xcsr](https://hf.co/datasets/INK-USC/xcsr) | Which sentence is most plausible? |
| xcopa/qu | MultipleChoice | [cambridgeltl/xcopa](https://hf.co/datasets/cambridgeltl/xcopa) |  |
| xcopa/et | MultipleChoice | [cambridgeltl/xcopa](https://hf.co/datasets/cambridgeltl/xcopa) |  |
| xcopa/ht | MultipleChoice | [cambridgeltl/xcopa](https://hf.co/datasets/cambridgeltl/xcopa) |  |
| xcopa/id | MultipleChoice | [cambridgeltl/xcopa](https://hf.co/datasets/cambridgeltl/xcopa) |  |
| xcopa/it | MultipleChoice | [cambridgeltl/xcopa](https://hf.co/datasets/cambridgeltl/xcopa) |  |
| xcopa/sw | MultipleChoice | [cambridgeltl/xcopa](https://hf.co/datasets/cambridgeltl/xcopa) |  |
| xcopa/translation-ht | MultipleChoice | [cambridgeltl/xcopa](https://hf.co/datasets/cambridgeltl/xcopa) |  |
| xcopa/th | MultipleChoice | [cambridgeltl/xcopa](https://hf.co/datasets/cambridgeltl/xcopa) |  |
| xcopa/tr | MultipleChoice | [cambridgeltl/xcopa](https://hf.co/datasets/cambridgeltl/xcopa) |  |
| xcopa/translation-et | MultipleChoice | [cambridgeltl/xcopa](https://hf.co/datasets/cambridgeltl/xcopa) |  |
| xcopa/translation-id | MultipleChoice | [cambridgeltl/xcopa](https://hf.co/datasets/cambridgeltl/xcopa) |  |
| xcopa/translation-sw | MultipleChoice | [cambridgeltl/xcopa](https://hf.co/datasets/cambridgeltl/xcopa) |  |
| xcopa/translation-ta | MultipleChoice | [cambridgeltl/xcopa](https://hf.co/datasets/cambridgeltl/xcopa) |  |
| xcopa/translation-th | MultipleChoice | [cambridgeltl/xcopa](https://hf.co/datasets/cambridgeltl/xcopa) |  |
| xcopa/translation-tr | MultipleChoice | [cambridgeltl/xcopa](https://hf.co/datasets/cambridgeltl/xcopa) |  |
| xcopa/translation-vi | MultipleChoice | [cambridgeltl/xcopa](https://hf.co/datasets/cambridgeltl/xcopa) |  |
| xcopa/translation-zh | MultipleChoice | [cambridgeltl/xcopa](https://hf.co/datasets/cambridgeltl/xcopa) |  |
| xcopa/vi | MultipleChoice | [cambridgeltl/xcopa](https://hf.co/datasets/cambridgeltl/xcopa) |  |
| xcopa/zh | MultipleChoice | [cambridgeltl/xcopa](https://hf.co/datasets/cambridgeltl/xcopa) |  |
| xcopa/ta | MultipleChoice | [cambridgeltl/xcopa](https://hf.co/datasets/cambridgeltl/xcopa) |  |
| xcopa/translation-it | MultipleChoice | [cambridgeltl/xcopa](https://hf.co/datasets/cambridgeltl/xcopa) |  |
| xstory_cloze/eu | MultipleChoice | [juletxara/xstory_cloze](https://hf.co/datasets/juletxara/xstory_cloze) | Which ending continues the story? |
| xstory_cloze/my | MultipleChoice | [juletxara/xstory_cloze](https://hf.co/datasets/juletxara/xstory_cloze) | Which ending continues the story? |
| xstory_cloze/hi | MultipleChoice | [juletxara/xstory_cloze](https://hf.co/datasets/juletxara/xstory_cloze) | Which ending continues the story? |
| xstory_cloze/te | MultipleChoice | [juletxara/xstory_cloze](https://hf.co/datasets/juletxara/xstory_cloze) | Which ending continues the story? |
| xstory_cloze/sw | MultipleChoice | [juletxara/xstory_cloze](https://hf.co/datasets/juletxara/xstory_cloze) | Which ending continues the story? |
| xstory_cloze/id | MultipleChoice | [juletxara/xstory_cloze](https://hf.co/datasets/juletxara/xstory_cloze) | Which ending continues the story? |
| xstory_cloze/ar | MultipleChoice | [juletxara/xstory_cloze](https://hf.co/datasets/juletxara/xstory_cloze) | Which ending continues the story? |
| xstory_cloze/es | MultipleChoice | [juletxara/xstory_cloze](https://hf.co/datasets/juletxara/xstory_cloze) | Which ending continues the story? |
| xstory_cloze/zh | MultipleChoice | [juletxara/xstory_cloze](https://hf.co/datasets/juletxara/xstory_cloze) | Which ending continues the story? |
| xstory_cloze/ru | MultipleChoice | [juletxara/xstory_cloze](https://hf.co/datasets/juletxara/xstory_cloze) | Which ending continues the story? |
| xstory_cloze/en | MultipleChoice | [juletxara/xstory_cloze](https://hf.co/datasets/juletxara/xstory_cloze) | Which ending continues the story? |
| disrpt/eus.rst.ert.rels | Classification | [multilingual-discourse-hub/disrpt](https://hf.co/datasets/multilingual-discourse-hub/disrpt) |  |
| disrpt/deu.rst.pcc.rels | Classification | [multilingual-discourse-hub/disrpt](https://hf.co/datasets/multilingual-discourse-hub/disrpt) |  |
| disrpt/fas.rst.prstc.rels | Classification | [multilingual-discourse-hub/disrpt](https://hf.co/datasets/multilingual-discourse-hub/disrpt) |  |
| disrpt/fra.sdrt.annodis.rels | Classification | [multilingual-discourse-hub/disrpt](https://hf.co/datasets/multilingual-discourse-hub/disrpt) |  |
| disrpt/nld.rst.nldt.rels | Classification | [multilingual-discourse-hub/disrpt](https://hf.co/datasets/multilingual-discourse-hub/disrpt) |  |
| disrpt/por.rst.cstn.rels | Classification | [multilingual-discourse-hub/disrpt](https://hf.co/datasets/multilingual-discourse-hub/disrpt) |  |
| disrpt/rus.rst.rrt.rels | Classification | [multilingual-discourse-hub/disrpt](https://hf.co/datasets/multilingual-discourse-hub/disrpt) |  |
| disrpt/spa.rst.rststb.rels | Classification | [multilingual-discourse-hub/disrpt](https://hf.co/datasets/multilingual-discourse-hub/disrpt) |  |
| disrpt/tha.pdtb.tdtb.rels | Classification | [multilingual-discourse-hub/disrpt](https://hf.co/datasets/multilingual-discourse-hub/disrpt) |  |
| disrpt/zho.rst.gcdt.rels | Classification | [multilingual-discourse-hub/disrpt](https://hf.co/datasets/multilingual-discourse-hub/disrpt) |  |
| universal_dependencies/sga_dipsgg/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/orv_torot/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ang_cairo/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/fro_altm/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/oge_glc/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/sga_dipwbg/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/orv_ruthenian/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/fro_profiterole/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/orv_rnc/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/kpv_ikdp/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/cu_proiel/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/or_odtb/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/oc_ttb/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/no_nynorsk/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/no_bokmaal/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/pro_corag/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/gya_autogramm/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/kmr_kurmanji/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/sme_giella/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/orv_birchbark/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/otk_clausal/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/pay_chibergis/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ota_dudu/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/pt_porttinari/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/pt_petrogold/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/pt_gsd/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/pt_dantestocks/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/pt_bosque/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/qpm_philotis/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/pl_pud/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/pl_pdb/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ota_boun/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/pl_mpdt/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/xpg_kul/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/yrl_complin/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/fa_seraji/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/fa_perdt/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/pad_tuecl/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ps_sikaram/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ps_prince/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ota_tuecl/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/pl_lfg/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ne_bk/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/gun_thomas/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/nap_rb/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/lij_glt/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/lv_lvtb/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/lv_cairo/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/la_udante/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/la_proiel/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/la_perseus/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/la_llct/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/la_ittb/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/lt_alksnis/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/la_circse/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ky_tuecl/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ky_ktmu/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ko_pud/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ko_littleprince/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ko_ksl/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ko_kaist/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ko_gsd/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/pt_pud/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ltg_cairo/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/yrk_tundra/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/lt_hse/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/nds_lsdc/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/pcm_nsc/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/nmf_suansu/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/myu_tudet/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/mdf_jr/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/frm_profiterole/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/frm_altm/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/axm_armtdp/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/kpv_lattice/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/olo_kkpp/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/mr_ufal/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/gv_cadhan/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/mt_mudt/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ml_ufal/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/mpu_tudet/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/qaf_arabizi/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/jaa_jarawara/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/mk_mtb/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/lb_luxbank/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/mr_cmupan/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/pa_cs/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/tr_pud/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ro_art/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/uk_parlamint/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/uk_iu/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/qtd_sagt/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/qti_butr/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/tr_tuecl/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/tr_tourism/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/koi_uh/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/tr_penn/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/xum_ikuvina/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/tr_kenet/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/tr_gb/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/tr_framenet/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/tr_boun/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/tr_atis/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/tpn_tudet/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/tn_popapolelo/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/th_tud/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/th_pud/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/tr_imst/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/hsb_ufal/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ur_udtb/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ug_udt/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/zza_zsd/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/say_autogramm/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ess_sli/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/yo_ytb/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/yi_yitb/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/sah_yktdt/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/sjo_xdt/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/xav_xdt/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/wo_wtb/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/nhi_mesotree/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/hyw_armtdp/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/cy_ccg/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/wbp_ufal/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/vi_vtb/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/vi_tuecl/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/vep_vwt/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/uz_uzudt/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/uz_ut/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/uz_tuecl/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/qte_tect/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/pa_rang/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/te_mtg/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/tt_nmctt/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/sd_isra/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/scn_stb/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/wuu_shud/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/sr_set/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/gd_arcosg/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/sa_vedic/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/sa_ufal/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ruc_rdt/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/si_appuwa/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ru_taiga/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ru_pud/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ru_poetry/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ru_gsd/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ro_tuecl/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ro_simonero/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ro_rrt/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ro_nonstandard/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ro_moldoro/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ru_syntagrus/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/si_stb/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/sms_giellagas/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/sk_snk/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ta_ttb/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ta_mwtt/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/tl_ugnayan/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/tl_trg/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/swl_sslc/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/sv_talbanken/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/sv_swell/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/sv_pud/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/sv_old/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/sv_lines/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ssp_lse/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/es_pud/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/es_gsd/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/es_coser/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/es_ancora/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/sdh_garrusi/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ajp_madar/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/sl_sst/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/sl_ssj/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/eme_tudet/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/quc_iu/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/it_kiparlaforest/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/kk_ktb/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/cs_pdtc/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/cs_fictree/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/cs_cltt/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/cs_cac/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/hr_set/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/cop_scriptorium/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/cop_bohairic/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/lzh_tuecl/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/lzh_kyoto/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/xcl_caval/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ckt_hse/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ctn_ctntb/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/zh_pud/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/zh_patentchar/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/zh_hk/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/zh_gsdsimp/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/naq_kdt/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/cs_poetry/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/cs_pud/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/da_ddt/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/nl_alpino/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/eo_prago/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/eo_cairo/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/myv_jr/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/en_pud/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/en_pronouns/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/en_partut/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/en_littleprince/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/en_lines/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/zh_cfl/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/en_gumreddit/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/en_gentle/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/en_ewt/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/en_eslspok/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/en_ctetex/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/en_childes/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/en_atis/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/egy_pc/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/nl_lassysmall/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/en_gum/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/zh_beginner/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ckb_mukri/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ceb_gja/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ar_pud/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ar_padt/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/apu_ufpa/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/hbo_ptnk/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/grc_ptnk/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/grc_proiel/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/grc_perseus/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/am_att/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/hy_armtdp/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/gsw_uzh/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/sq_tsa/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/sq_staf/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/aqz_tudet/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/akk_riao/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/akk_pisandub/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/af_afribooms/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ab_abnc/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/abq_atb/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/gsw_divital/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/et_edt/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/hy_bsut/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/aii_as/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ca_ancora/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/cpg_tuecl/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/cpg_amgic/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/yue_hk/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/bxr_bdt/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/bg_btb/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/br_keb/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/brh_kholum/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/as_aiw/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/bor_bdt/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/bho_bhtb/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/bn_bru/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/be_hse/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/bej_autogramm/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/bar_maibaam/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/eu_bdt/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/bm_crb/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/az_tuecl/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/sab_chibergis/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/et_ewt/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/zh_gsd/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/fo_oft/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ga_idt/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ga_cadhan/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/id_pud/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/id_gsd/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/id_csui/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/arh_chibergis/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/is_pud/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/is_modern/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/is_icepahc/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/is_gc/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/hu_szeged/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/hit_hittb/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/hi_pud/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/hi_hdtb/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/azz_itml/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/he_postrab/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/he_iahltwiki/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ga_twittirish/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/it_isdt/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/it_markit/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/it_old/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/arr_tudet/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/fo_farpahc/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/xnr_kdtb/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/kbc_unicamp/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/urb_tudet/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/jv_csui/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ja_pudluw/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ja_pud/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/he_iahltknesset/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ja_gsdluw/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ja_bccwjluw/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/it_vit/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/it_valico/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/it_twittiro/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/it_pud/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/it_postwita/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/it_partut/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/it_parlamint/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ja_gsd/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/he_htb/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/krl_kkpp/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ha_southernautogramm/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/gl_treegal/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/gl_pud/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/gl_ctg/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/qfn_fame/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/fr_sequoia/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/fr_rhapsodie/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/fr_pud/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/fr_poitevindivital/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ka_glc/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/fr_partut/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/fr_gsd/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/fr_fqb/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/fr_alts/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/fi_tdt/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/fi_pud/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/fi_ood/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/fi_ftb/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ha_westernautogramm/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/fr_parisstories/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/de_gsd/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ka_gnc/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/el_messinian/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ht_autogramm/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ht_adolphe/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/gwi_tuecl/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/gu_gujtb/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/gn_oldtudet/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/gub_tudet/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/de_hdt/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/el_lesbian/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/el_gud/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/el_glcii/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/el_gdt/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/el_cretan/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/got_proiel/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/gor_bungololombi/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/aln_gps/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/de_pud/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/de_lit/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ha_easternautogramm/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| universal_dependencies/ha_northernautogramm/pos | TokenClassification | [universal-dependencies/universal_dependencies](https://hf.co/datasets/universal-dependencies/universal_dependencies) |  |
| oasst1_pairwise_rlhf_reward | MultipleChoice | [tasksource/oasst1_pairwise_rlhf_reward](https://hf.co/datasets/tasksource/oasst1_pairwise_rlhf_reward) | Which reply is better? |
| multilingual-sentiments/all | Classification | [tasksource/multilingual-sentiments](https://hf.co/datasets/tasksource/multilingual-sentiments) |  |
| tweet_sentiment_multilingual | Classification | json |  |
| amazon_reviews_multi/all_languages | Classification | [goosmanlei/amazon_reviews_multi](https://hf.co/datasets/goosmanlei/amazon_reviews_multi) |  |
| universal-joy | Classification | [tasksource/universal-joy](https://hf.co/datasets/tasksource/universal-joy) |  |
| mms | Classification | parquet |  |
| mapa/coarse_grained | TokenClassification | [joelito/mapa](https://hf.co/datasets/joelito/mapa) |  |
| mapa/fine_grained | TokenClassification | [joelito/mapa](https://hf.co/datasets/joelito/mapa) |  |
| massive | Classification | [mteb/MassiveIntentClassification](https://hf.co/datasets/mteb/MassiveIntentClassification) |  |
| masakhanews/yor | Classification | [masakhane/masakhanews](https://hf.co/datasets/masakhane/masakhanews) |  |
| masakhanews/xho | Classification | [masakhane/masakhanews](https://hf.co/datasets/masakhane/masakhanews) |  |
| masakhanews/tir | Classification | [masakhane/masakhanews](https://hf.co/datasets/masakhane/masakhanews) |  |
| masakhanews/swa | Classification | [masakhane/masakhanews](https://hf.co/datasets/masakhane/masakhanews) |  |
| masakhanews/som | Classification | [masakhane/masakhanews](https://hf.co/datasets/masakhane/masakhanews) |  |
| masakhanews/sna | Classification | [masakhane/masakhanews](https://hf.co/datasets/masakhane/masakhanews) |  |
| masakhanews/pcm | Classification | [masakhane/masakhanews](https://hf.co/datasets/masakhane/masakhanews) |  |
| masakhanews/orm | Classification | [masakhane/masakhanews](https://hf.co/datasets/masakhane/masakhanews) |  |
| masakhanews/lug | Classification | [masakhane/masakhanews](https://hf.co/datasets/masakhane/masakhanews) |  |
| masakhanews/lin | Classification | [masakhane/masakhanews](https://hf.co/datasets/masakhane/masakhanews) |  |
| masakhanews/ibo | Classification | [masakhane/masakhanews](https://hf.co/datasets/masakhane/masakhanews) |  |
| masakhanews/hau | Classification | [masakhane/masakhanews](https://hf.co/datasets/masakhane/masakhanews) |  |
| masakhanews/fra | Classification | [masakhane/masakhanews](https://hf.co/datasets/masakhane/masakhanews) |  |
| masakhanews/eng | Classification | [masakhane/masakhanews](https://hf.co/datasets/masakhane/masakhanews) |  |
| masakhanews/run | Classification | [masakhane/masakhanews](https://hf.co/datasets/masakhane/masakhanews) |  |
| masakhanews/amh | Classification | [masakhane/masakhanews](https://hf.co/datasets/masakhane/masakhanews) |  |
| NusaX-senti/sun | Classification | [mteb/NusaX-senti](https://hf.co/datasets/mteb/NusaX-senti) |  |
| NusaX-senti/ban | Classification | [mteb/NusaX-senti](https://hf.co/datasets/mteb/NusaX-senti) |  |
| NusaX-senti/nij | Classification | [mteb/NusaX-senti](https://hf.co/datasets/mteb/NusaX-senti) |  |
| NusaX-senti/min | Classification | [mteb/NusaX-senti](https://hf.co/datasets/mteb/NusaX-senti) |  |
| NusaX-senti/mad | Classification | [mteb/NusaX-senti](https://hf.co/datasets/mteb/NusaX-senti) |  |
| NusaX-senti/jav | Classification | [mteb/NusaX-senti](https://hf.co/datasets/mteb/NusaX-senti) |  |
| NusaX-senti/ind | Classification | [mteb/NusaX-senti](https://hf.co/datasets/mteb/NusaX-senti) |  |
| NusaX-senti/eng | Classification | [mteb/NusaX-senti](https://hf.co/datasets/mteb/NusaX-senti) |  |
| NusaX-senti/bug | Classification | [mteb/NusaX-senti](https://hf.co/datasets/mteb/NusaX-senti) |  |
| NusaX-senti/bjn | Classification | [mteb/NusaX-senti](https://hf.co/datasets/mteb/NusaX-senti) |  |
| NusaX-senti/ace | Classification | [mteb/NusaX-senti](https://hf.co/datasets/mteb/NusaX-senti) |  |
| NusaX-senti/bbc | Classification | [mteb/NusaX-senti](https://hf.co/datasets/mteb/NusaX-senti) |  |
| AfriSenti-twitter-sentiment/yor | Classification | [mteb/AfriSentiClassification](https://hf.co/datasets/mteb/AfriSentiClassification) |  |
| AfriSenti-twitter-sentiment/ary | Classification | [mteb/AfriSentiClassification](https://hf.co/datasets/mteb/AfriSentiClassification) |  |
| AfriSenti-twitter-sentiment/hau | Classification | [mteb/AfriSentiClassification](https://hf.co/datasets/mteb/AfriSentiClassification) |  |
| AfriSenti-twitter-sentiment/ibo | Classification | [mteb/AfriSentiClassification](https://hf.co/datasets/mteb/AfriSentiClassification) |  |
| AfriSenti-twitter-sentiment/arq | Classification | [mteb/AfriSentiClassification](https://hf.co/datasets/mteb/AfriSentiClassification) |  |
| AfriSenti-twitter-sentiment/kin | Classification | [mteb/AfriSentiClassification](https://hf.co/datasets/mteb/AfriSentiClassification) |  |
| AfriSenti-twitter-sentiment/pcm | Classification | [mteb/AfriSentiClassification](https://hf.co/datasets/mteb/AfriSentiClassification) |  |
| AfriSenti-twitter-sentiment/por | Classification | [mteb/AfriSentiClassification](https://hf.co/datasets/mteb/AfriSentiClassification) |  |
| AfriSenti-twitter-sentiment/swa | Classification | [mteb/AfriSentiClassification](https://hf.co/datasets/mteb/AfriSentiClassification) |  |
| AfriSenti-twitter-sentiment/tso | Classification | [mteb/AfriSentiClassification](https://hf.co/datasets/mteb/AfriSentiClassification) |  |
| AfriSenti-twitter-sentiment/twi | Classification | [mteb/AfriSentiClassification](https://hf.co/datasets/mteb/AfriSentiClassification) |  |
| AfriSenti-twitter-sentiment/amh | Classification | [mteb/AfriSentiClassification](https://hf.co/datasets/mteb/AfriSentiClassification) |  |
| clue/ocnli | Classification | [clue/clue](https://hf.co/datasets/clue/clue) |  |
| clue/afqmc | Classification | [clue/clue](https://hf.co/datasets/clue/clue) |  |
| clue/tnews | Classification | [clue/clue](https://hf.co/datasets/clue/clue) |  |
| klue/nli | Classification | [klue/klue](https://hf.co/datasets/klue/klue) |  |
| klue/ynat | Classification | [klue/klue](https://hf.co/datasets/klue/klue) |  |
| klue/sts | Classification | [klue/klue](https://hf.co/datasets/klue/klue) |  |
| indic_glue/iitp-mr.hi/sentiment | Classification | [ai4bharat/indic_glue](https://hf.co/datasets/ai4bharat/indic_glue) |  |
| indic_glue/actsa-sc.te/sentiment | Classification | [ai4bharat/indic_glue](https://hf.co/datasets/ai4bharat/indic_glue) |  |
| indic_glue/iitp-pr.hi/sentiment | Classification | [ai4bharat/indic_glue](https://hf.co/datasets/ai4bharat/indic_glue) |  |
| indic_glue/inltkh.te/sentiment | Classification | [ai4bharat/indic_glue](https://hf.co/datasets/ai4bharat/indic_glue) |  |
| indic_glue/sna.bn/news | Classification | [ai4bharat/indic_glue](https://hf.co/datasets/ai4bharat/indic_glue) |  |
| indic_glue/bbca.hi/news | Classification | [ai4bharat/indic_glue](https://hf.co/datasets/ai4bharat/indic_glue) |  |
| indic_glue/inltkh | Classification | parquet |  |
| indic_glue/md.hi/discourse_mode | Classification | [ai4bharat/indic_glue](https://hf.co/datasets/ai4bharat/indic_glue) |  |
| indic_glue/wstp | MultipleChoice | parquet | Which title fits this section? |
| tydi-as2-balanced | Classification | [tasksource/tydi-as2-balanced](https://hf.co/datasets/tasksource/tydi-as2-balanced) |  |
| conll2002/nl | TokenClassification | parquet |  |
| conll2002/es | TokenClassification | parquet |  |
| multiconer_v2/Italian (IT) | TokenClassification | parquet |  |
| multiconer_v2/Swedish (SV) | TokenClassification | parquet |  |
| multiconer_v2/Spanish (ES) | TokenClassification | parquet |  |
| multiconer_v2/Ukrainian (UK) | TokenClassification | parquet |  |
| multiconer_v2/Hindi (HI) | TokenClassification | parquet |  |
| multiconer_v2/Bangla (BN) | TokenClassification | parquet |  |
| multiconer_v2/French (FR) | TokenClassification | parquet |  |
| multiconer_v2/Farsi (FA) | TokenClassification | parquet |  |
| multiconer_v2/English (EN) | TokenClassification | parquet |  |
| multiconer_v2/Chinese (ZH) | TokenClassification | parquet |  |
| multiconer_v2/German (DE) | TokenClassification | parquet |  |
| multiconer_v2/Portuguese (PT) | TokenClassification | parquet |  |
| mtop | Classification | [tasksource/mtop](https://hf.co/datasets/tasksource/mtop) |  |
| multilingual-zero-shot-label-nli | Classification | [tasksource/multilingual-zero-shot-label-nli](https://hf.co/datasets/tasksource/multilingual-zero-shot-label-nli) |  |
