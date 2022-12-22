import pandas as pd

dfh=pd.read_csv('https://raw.githubusercontent.com/alexwarstadt/blimp/master/raw_results/summary/human_validation_summary.csv')
dfh['linguistic_term']=dfh['Condition']
dfm=pd.read_json('https://raw.githubusercontent.com/alexwarstadt/blimp/master/raw_results/summary/models_summary.jsonl',lines=True)
df=dfm.join(dfh)
df['diff']=df.total_mean - df.gpt2
blimp_hard = set(df[df['diff']>0.1].UID)
del dfh, dfm, df

blimp_groups = {
 "syntax": [
  "adjunct_island",
  "animate_subject_passive",
  "animate_subject_trans",
  "causative",
  "complex_NP_island",
  "coordinate_structure_constraint_complex_left_branch",
  "coordinate_structure_constraint_object_extraction",
  "drop_argument",
  "ellipsis_n_bar_1",
  "ellipsis_n_bar_2",
  "inchoative",
  "intransitive",
  "left_branch_island_echo_question",
  "left_branch_island_simple_question",
  "passive_1",
  "passive_2",
  "sentential_subject_island",
  "transitive",
  "wh_island",
  "wh_questions_object_gap",
  "wh_questions_subject_gap",
  "wh_questions_subject_gap_long_distance",
  "wh_vs_that_no_gap",
  "wh_vs_that_no_gap_long_distance",
  "wh_vs_that_with_gap",
  "wh_vs_that_with_gap_long_distance"
 ],
 "morphology": [
  "anaphor_gender_agreement",
  "anaphor_number_agreement",
  "determiner_noun_agreement_1",
  "determiner_noun_agreement_2",
  "determiner_noun_agreement_irregular_1",
  "determiner_noun_agreement_irregular_2",
  "determiner_noun_agreement_with_adj_2",
  "determiner_noun_agreement_with_adj_irregular_1",
  "determiner_noun_agreement_with_adj_irregular_2",
  "determiner_noun_agreement_with_adjective_1",
  "distractor_agreement_relational_noun",
  "distractor_agreement_relative_clause",
  "irregular_past_participle_adjectives",
  "irregular_past_participle_verbs",
  "irregular_plural_subject_verb_agreement_1",
  "irregular_plural_subject_verb_agreement_2",
  "regular_plural_subject_verb_agreement_1",
  "regular_plural_subject_verb_agreement_2"
 ],
 "syntax_semantics": [
  "existential_there_object_raising",
  "existential_there_subject_raising",
  "expletive_it_object_raising",
  "only_npi_scope",
  "principle_A_c_command",
  "principle_A_case_1",
  "principle_A_domain_1",
  "principle_A_domain_2",
  "principle_A_domain_3",
  "principle_A_reconstruction",
  "sentential_negation_npi_scope",
  "tough_vs_raising_1",
  "tough_vs_raising_2"
 ],
 "semantics": [
  "existential_there_quantifiers_1",
  "existential_there_quantifiers_2",
  "matrix_question_npi_licensor_present",
  "npi_present_1",
  "npi_present_2",
  "only_npi_licensor_present",
  "sentential_negation_npi_licensor_present",
  "superlative_quantifiers_1",
  "superlative_quantifiers_2"
 ],
 "syntax/semantics": [
  "principle_A_case_2"
 ]
}
