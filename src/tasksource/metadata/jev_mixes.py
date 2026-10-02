"""Mixes of tasksource-jev-typed-decisions: buckets of sources (regex on `source`, first match wins) with a
target share each. Inside a bucket, sources share its rows by their score in jev_source_scores.csv
(correctness gate x zone of proximal development x interestingness, see scripts/jev_mix_scores.py)
times the square root of their size, so large sources do not crowd out small ones."""

import csv
import math
import re
from pathlib import Path

SCORES = Path(__file__).with_name("jev_source_scores.csv")

# Shares (2026-10-02) weigh three things: sources picked by reading them (PICKED), use cases (agent
# guardrails, routing, long documents), and the audits (jev_source_scores.csv: zone, interest, binary
# share). Procedural is fixed at 12%. Logic and NLI hold most picks and sit in the zone; knowledge and
# token labels are clean and many-option; safety and toxicity are near-solved binary decisions;
# multilingual stays small for an English target; templated probes carry little per row.
BUCKETS = {
    "procedural": (r"^procedural-typed-decisions/", 0.12),
    "multilingual": (r"^multilingual/", 0.03),
    "logic_synthetic": (r"FOL-nli|LogicNLI|FLD|proofwriter|ruletaker|PARARULE|robustLR|folio|logiqa|reclor|lsat|clutrr|"
                        r"babi_nli|stepgame|SpaRTUN|spartqa|ReSQ|SpaceNLI|tomi-nli|mindgames|nlgraph|corr2cause|cladder|"
                        r"puzzte|brainteasers|math_qa|prm800k|satisfiability|temporal-nli|tracie|conceptrules|regset|"
                        r"logical-|monotonicity|strategy-qa|riddle_sense|winodict|missing-item", 0.15),
    "knowledge_mcqa": (r"medmcqa|MedQA|wikimedqa|head_qa|ScienceQA|sciq|qasc|openbookqa|ai2_arc|^race|quail|cosmos_qa|"
                       r"dream|mutual|wiki_hop|numer_sense|commonsense_qa|mctest|onestop|ekar|quartz|quarel|prost|"
                       r"feasibilityQA|CREAK|com2sense|codah|sen-making|twentyquestions|CONDAQA|boolq|mc_taco", 0.1),
    "long_doc_factcheck": (r"doc-nli|contract-nli|lex_glue|hover|vitaminc|wice|fever|health_fact|scifact|ConTRoL|sharc|"
                           r"liar|x-fact|fool-me-twice|synthetic-retrieval-NLI|seahorse|AmbigNQ|SDOH|nli4ct|biosift|"
                           r"wiki_qa|tydi", 0.09),
    "intent_routing": (r"clinc|banking77|IntentGrasp|dnd_style|trec|ag_news|yahoo|dbpedia|stackoverflow|"
                       r"open_question_type|snips|it-support|esci|github-issue|silicone|miam|blog_authorship|patent|"
                       r"citation_intent|scicite|code_x_glue", 0.08),
    "safety_agentic": (r"PromptShield|[Pp]rompt-injection|safe-guard|wildguard|shell-safety|ShellRisk|"
                       r"agent_action_safety|toxic-chat|BeaverTails|PKU-SafeRLHF/safety|privacy-200k", 0.04),
    "preference_judge": (r"oasst|dpo_pairs|summarize_from_feedback|hh-rlhf|HelpSteer|UltraFeedback|chatbot_arena|SHP|"
                         r"webgpt|PKU-SafeRLHF|synthetic-instruct|argument-feedback|AES2|english-grading|TuringBench", 0.06),
    "graded_calibration": (r"civil_comments|dynasent/.*votes|UNLI|lewidi|Disagreement|chaos|sts-companion|"
                           r"acceptability|proto_qa|wouldyourather|probability_words|scruples|crowdflower|persuasion|"
                           r"emobank", 0.065),
    "nli_general": (r"anli|WANLI|dataset_train_nli|^glue|super_glue|^snli|lingnli|MSciNLI|scinli|scitail|defeasible|"
                    r"cnli|help-nli|joci|mpe|add_one_rte|breaking_nli|dialogue_nli|nli_fever|lonli|resnli|idioms-nli|"
                    r"Pol_NLI|SIGA|avicenna|dadc|fracas|ambient|nan-nli|sick", 0.1),
    "sentiment_stance": (r"tweet_eval/(sent|stance|irony|emo)|[Ss]arcasm|starcon|args_me|Touche|rumoureval|financial|"
                         r"imdb|rotten|yelp|amazon_polarity|app_reviews|auditor_review|emotion|emo/|go_emotions|"
                         r"poem_sentiment|subjectivity|hyperpartisan|hlgd|headline_cause|humicroedit|FLUTE|MOH|TroFi|"
                         r"VUAC|PARADISE|exaggeration|amazon_counterfactual|insincere|arct", 0.045),
    "commonsense": (r"hellaswag|swag|piqa|social_i_qa|winogrande|winowhy|^art$|cicero|wiqa|e-CARE|cos_e|goal-step|"
                    r"path-naturalness|discosense|cycic|moral_stories|ethics|utilitarianism|fig-qa|I2D2|balanced-copa|"
                    r"implicatures|circa|cloth|dgen|definite_pronoun", 0.04),
    "toxicity": (r"hate|toxi|jigsaw|ethos|offens|Hatemoji|hope_edi|sms_spam", 0.02),
    "token_labels": (r"conll2003|wnut|docred|few_rel|chemprot|sem_eval_2010|sciie|ade_corpus|propsegment", 0.03),
    "paraphrase_prag": (r"paws|parade|apt|phrase_similarity|medical_questions_pairs|simple_pair|pragmeval|discovery|"
                        r"disrpt|google_wellformed|clcd", 0.015),
    "templated_probes": (r".", 0.015),  # catch-all: robust_nli, gen_debiased, recast, hans, linguisticprobing...
}


# Sources picked by reading them (2026-10-02): rows the scores alone undervalue, e.g. clean binary sets
PICKED = r"^WANLI$|^anli/|^ai2_arc/|^glue/cola$|dynasent|Dynasent|^IntentGrasp/all$|^MSciNLI$|^ConTRoL-nli$|" \
         r"^winogrande/|^contract-nli/|^folio$|^dynahate$|^head_qa/en$|^sciq$|^art$|^dadc-limit-nli$|^FOL-nli$|^doc-nli$"
PICKED_BOOST = 2.0


def length_factor(chars):
    """Long inputs are rare and worth reading: x1.25 at 2k characters, x1.5 from 4k."""
    return min(1.5, max(1.0, 1 + 0.25 * math.log2(max(chars, 1) / 1000)))


def bucket(source):
    return next(name for name, (pattern, _) in BUCKETS.items() if re.search(pattern, source))


def source_scores(path=SCORES):
    with open(path, newline="") as f:
        return {row["source"]: float(row["score"]) for row in csv.DictReader(f)}


def waterfill(total, weights, limits):
    """Split ``total`` in proportion to ``weights``, no key above its limit; what a key cannot take
    goes to the others in proportion."""
    out, weights = {}, {k: w for k, w in weights.items() if w > 0 and limits[k] > 0}
    while weights and total > 1e-9:
        scale = total / sum(weights.values())
        full = {k for k, w in weights.items() if w * scale >= limits[k]}
        if not full:
            out.update({k: w * scale for k, w in weights.items()})
            break
        for k in full:
            out[k] = limits[k]
            total -= limits[k]
            del weights[k]
    return out


def source_quotas(sizes, total_rows, scores=None):
    """Rows per source: buckets get ``total_rows`` by their share, sources split their bucket's rows by
    score x sqrt(size), never above their size; a bucket short of rows passes the rest to the others."""
    scores = source_scores() if scores is None else scores
    members = {}
    for source, size in sizes.items():
        if scores.get(source, 0) > 0 and size:
            # procedural generators keep equal shares: their difficulty is set by their levels
            weight = 1.0 if bucket(source) == "procedural" else scores[source] * math.sqrt(size)
            members.setdefault(bucket(source), {})[source] = weight
    capacity = {name: sum(sizes[s] for s in sources) for name, sources in members.items()}
    rows = waterfill(total_rows, {name: BUCKETS[name][1] for name in members}, capacity)
    quotas = {}
    for name, sources in members.items():
        quotas.update(waterfill(rows.get(name, 0), sources, {s: sizes[s] for s in sources}))
    return {s: int(q) for s, q in quotas.items() if int(q) > 0}
