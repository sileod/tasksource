import random
from datasets import ClassLabel, DatasetDict, Dataset, List, Sequence
from sorcery import dict_of
import string
from collections import OrderedDict

from .jev_augmentations import stable_fraction
from .jev_token_labels import (
    MAX_JEV_TOKENS_PER_SEQUENCE,
    normalize_token_label,
    readable_token_labels,
)

improper_labels =['recast/recast_kg_relations','linguisticprobing',"lex_glue/scotus",'lexical_relation_classification/ROOT09',"pragmeval/squinky","pragmeval/emobank",'pragmeval/persuasiveness']
improper_labels += ['glue/stsb', 'sick/relatedness', 'joci', 'utilitarianism', 'amazon_counterfactual/en', 'toxic_conversations', 'ethos/multilabel', 'lex_glue/eurlex', 'lex_glue/unfair_tos', 'app_reviews', 'humicroedit/subtask-1', 'stackoverflow-questions', 'go_emotions/simplified', 'google_wellformed_query', 'has_part', 'blog_authorship_corpus/age', 'promptCoherence', 'Sarcasm_News_Headline', 'auditor_review/demo-org--auditor_review', 'Dynasent_Disagreement', 'Politeness_Disagreement', 'SBIC_Disagreement', 'SChem_Disagreement', 'Dilemmas_Disagreement', 'sts-companion', 'acceptability-prediction', 'chaos-mnli-ambiguity', 'headline_cause/en_simple', 'oasst1_dense_flat', 'civil_comments']

improper_labels += ['stsb_multi_mt','MLMA_hate_speech','icl-symbol-tuning-instruct','zero-shot-label-nli']

improper_labels += ['essay-scoring','english-grading','HelpSteer','oasst2']

def render_options(options):
    options = [f'"{x}"' for x in options]
    return f"{', '.join(options[:-1])} or {options[-1]}"

def render_classification(text,options,answer):
    example = 'text_A→text_B' if text.startswith('text_A:') else 'the following'
    inputs = f'With no explanation, label {example} with either {render_options(options)}.\n{text}'
    targets = f"{answer}."
    return dict_of(inputs,targets)

def render_token_classification(tokens,options,labels):
    prefix = f'With no explanation, label each line with {render_options(options)} preceded by ":".\n'
    inputs = prefix+"\n".join(tokens)
    targets = "\n".join([':'.join(x) for x in zip(tokens,labels)])
    return dict_of(inputs,targets)

def render_multiple_choice(prompt, options, labels):
    inputs=(prompt+'\n' if prompt else '')
    letters = string.ascii_uppercase[:len(options)]
    inputs=f'With no explanation, chose the best option from {render_options(letters)}. {inputs}'    
    for letter, option in zip(letters, options):
        inputs+=f'\n{letter}: {option}'
    targets = f'{letters[labels]}.'
    return dict_of(inputs, targets) 

def negative_sample_options(y, labels,N=4):
    if len(labels)<N:
        return labels
    else:
        return [y]+random.sample([x for x in labels if x!=y], N-1)

def shuffle_choices(x):
    choices = sorted([k for k in x if 'choice' in k])
    choices_texts = [x[c] for c in choices]
    correct_choice =choices_texts[x['labels']]
    random.shuffle(choices_texts)
    for c, ct in zip(choices, choices_texts):
        x[c]=ct
    x["labels"]=choices_texts.index(correct_choice)
    return x

def recast_dataset_classification_to_mc(dataset,sep="[SEP]",N=4):

    def recast_split(d,N=N):
        labels = d.features['labels']
        df=d.to_pandas()
        df['inputs'] = df.sentence1
        if "sentence2" in df:
            df['inputs'] +=sep + df.sentence2

        N=min(N, len(labels.names))
        df['choices']=df.apply(lambda x:negative_sample_options(labels.int2str(x['labels']), labels.names,N),axis=1)     
        df['labels']=df.apply(lambda x:x['choices'].index(labels.int2str(x['labels'])),axis=1)

        for i in range(N):
            df[f'choice{i}']= "This example is " + df.choices.map(lambda x:x[i])

        choices = [f'choice{i}' for i in range(N)]
        return Dataset.from_pandas(df[['inputs',*choices,'labels']],preserve_index=False)

    return DatasetDict({k: recast_split(v) for k,v in dataset.items()})


def recast_instruct(dataset):
    features = dataset['train'].features
    labels = features['labels']

    if "sentence1" in features:
        task_type='Classification'
    if "choice0" in features:
        task_type = "MultipleChoice"
    if "tokens" in features:
        task_type = "TokenClassification"

    def recast_MultipleChoice(x):
        x=shuffle_choices(x)
        choices = sorted([k for k in x if 'choice' in k])
        if all([x[c] in x['inputs'] for c in choices]):
            return {"inputs":x['inputs'], 'targets': x[f"choice{x['labels']}"].strip()+"."}
        else:
            return render_multiple_choice(x['inputs'],[x[c] for c in choices],x['labels'])

    def recast_TokenClassification(x):
        distractors = list(labels.feature.names)
        x_labels = [labels.feature.int2str(y) for y in x['labels']]
        labels_set= list({labels.feature.int2str(y) for y in x['labels']})
        options=list(dict.fromkeys(labels_set+distractors))[:max(len(labels_set),10)]
        return render_token_classification(x['tokens'],options,x_labels)

    def recast_Classification(x):
        if 'sentence2' in x:
            text=f"text_A: {x['sentence1']}\ntext_B: {x['sentence2']}"
        else:
            text=x['sentence1']
            
        answer=labels.int2str(x['labels']).strip()
        options= negative_sample_options(answer, labels._int2str)
        return render_classification(text, options, answer)
        
    dataset = dataset.map(eval(f"recast_{task_type}"))
    dataset = dataset.remove_columns([k for k in features if k not in ['inputs','targets']])
    return dataset


JEV_CLASSIFICATION_INSTRUCTIONS = "Choose the criterion that best describes the state."
JEV_MULTIPLE_CHOICE_INSTRUCTIONS = "Choose the criterion that best answers the question."


def _choice_columns(features):
    """Return choice columns in numeric order (choice2 before choice10)."""
    choices = [name for name in features if name.startswith("choice")]

    def key(name):
        suffix = name[len("choice"):]
        return (0, int(suffix)) if suffix.isdigit() else (1, name)

    return sorted(choices, key=key)


def _jev_task_type(features):
    if "sentence1" in features:
        return "Classification"
    if _choice_columns(features) and "inputs" in features:
        return "MultipleChoice"
    if "tokens" in features and "labels" in features:
        labels = features["labels"]
        if not isinstance(labels, (Sequence, List)) or not isinstance(labels.feature, ClassLabel):
            raise NotImplementedError(
                "TokenClassification labels must be Sequence(ClassLabel) or List(ClassLabel) for JEV"
            )
        if not readable_token_labels(labels.feature.names):
            raise NotImplementedError(
                "TokenClassification labels are not semantically readable for JEV"
            )
        return "TokenClassification"
    raise NotImplementedError(
        "Jev recasting currently supports Classification, MultipleChoice, and "
        "readable TokenClassification tasks"
    )


def _token_question_indices(tokens, labels, names, identifier):
    """Select at most two deterministic token judgments from one sequence."""
    valid = [index for index in range(min(len(tokens), len(labels))) if 0 <= int(labels[index]) < len(names)]
    if not valid:
        return []
    default = names.index("O") if "O" in names else None
    non_default = [index for index in valid if default is None or int(labels[index]) != default]
    ranked_non_default = sorted(
        non_default,
        key=lambda index: stable_fraction(f"{identifier}:{index}", "token-primary"),
    )
    selected = ranked_non_default[:1]
    remaining = [index for index in valid if index not in selected]
    remaining.sort(
        key=lambda index: stable_fraction(f"{identifier}:{index}", "token-secondary")
    )
    selected.extend(remaining[: MAX_JEV_TOKENS_PER_SEQUENCE - len(selected)])
    return selected


def _token_state(tokens, target_index):
    marked = list(tokens)
    marked[target_index] = f"[TARGET: {tokens[target_index]}]"
    sentence = " ".join(tokens)
    return (
        f"Sentence: {sentence}\n"
        f"Target token at position {target_index}: {tokens[target_index]}\n"
        f"Marked sentence: {' '.join(marked)}"
    )


def recast_jev(dataset, task=None):
    """Recast a standardized Tasksource dataset as runtime-defined choices.

    The output is deliberately model- and wire-format-independent. ``criteria``
    contains the choices presented at runtime, ``label`` is their zero-based
    index, and ``answer`` is the corresponding criterion text. No choices are
    shuffled and no random augmentation is performed here.
    """
    if not isinstance(dataset, DatasetDict):
        raise TypeError("recast_jev expects a datasets.DatasetDict")
    if "train" not in dataset:
        raise ValueError("recast_jev expects a train split")

    features = dataset["train"].features
    task_type = _jev_task_type(features)
    labels = features.get("labels")

    if task_type == "Classification":
        if not hasattr(labels, "names"):
            raise TypeError("Classification labels must use datasets.ClassLabel")
        criteria = list(labels.names)

        def convert(example):
            state = example["sentence1"]
            if "sentence2" in example:
                state = f"text_A: {state}\ntext_B: {example['sentence2']}"
            label = int(example["labels"])
            return {
                "state": state,
                "instructions": JEV_CLASSIFICATION_INSTRUCTIONS,
                "criteria": criteria,
                "label": label,
                "answer": criteria[label],
                "task": task or "",
            }

    elif task_type == "MultipleChoice":
        choices = _choice_columns(features)

        def convert(example):
            criteria = [example[name] for name in choices]
            label = int(example["labels"])
            return {
                "state": example["inputs"],
                "instructions": JEV_MULTIPLE_CHOICE_INSTRUCTIONS,
                "criteria": criteria,
                "label": label,
                "answer": criteria[label],
                "task": task or "",
            }

    else:
        raw_names = list(labels.feature.names)
        criteria = [normalize_token_label(name) for name in raw_names]
        if len(criteria) != len(set(criteria)):
            raise NotImplementedError(
                "TokenClassification labels collapse to duplicate readable JEV criteria"
            )

        def convert_batch(batch, indices):
            output = {
                "state": [], "instructions": [], "criteria": [], "label": [],
                "answer": [], "task": [], "shared_state": [],
                "question_id": [], "source_row": [], "target_index": [],
                "target_token": [],
            }
            for tokens, token_labels, source_index in zip(
                batch["tokens"], batch["labels"], indices
            ):
                if len(tokens) != len(token_labels):
                    raise ValueError(
                        f"Token and label lengths differ at source row {source_index}"
                    )
                identifier = f"{task or 'token-task'}:{source_index}"
                for token_index in _token_question_indices(
                    tokens, token_labels, raw_names, identifier
                ):
                    label = int(token_labels[token_index])
                    output["state"].append(_token_state(tokens, token_index))
                    output["instructions"].append(
                        "Choose the criterion that best labels the target token."
                    )
                    output["criteria"].append(criteria)
                    output["label"].append(label)
                    output["answer"].append(criteria[label])
                    output["task"].append(task or "")
                    output["shared_state"].append(f"Sentence: {' '.join(tokens)}")
                    output["question_id"].append(f"token-{token_index}")
                    output["source_row"].append(source_index)
                    output["target_index"].append(token_index)
                    output["target_token"].append(tokens[token_index])
            return output

        converted = dataset.map(
            convert_batch,
            batched=True,
            batch_size=1_000,
            with_indices=True,
            remove_columns=dataset["train"].column_names,
        )
        return converted

    converted = dataset.map(convert)
    keep = {"state", "instructions", "criteria", "label", "answer", "task"}
    remove = [name for name in converted["train"].column_names if name not in keep]
    if remove:
        converted = converted.remove_columns(remove)
    return converted


def render_systemone(example, question_id="decision", model=None):
    """Render one canonical Jev row as a System One choice request."""
    criteria = list(example["criteria"])
    if len(criteria) != len(set(criteria)):
        raise ValueError("System One choice criterion names must be unique")
    request = OrderedDict()
    if model is not None:
        request["model"] = model
    request["state"] = example["state"]
    request["questions"] = {
        question_id: {
            "type": "choice",
            "instructions": example["instructions"],
            "criteria": {criterion: None for criterion in criteria},
        }
    }
    return dict(request)


def render_systemone_group(examples, model=None):
    """Render related decisions over one source state as a multi-question request."""
    examples = list(examples)
    if not examples:
        raise ValueError("At least one decision is required")
    state = examples[0].get("shared_state", examples[0]["state"])
    request = OrderedDict()
    if model is not None:
        request["model"] = model
    request["state"] = state
    questions = {}
    for example in examples:
        if example.get("shared_state", example["state"]) != state:
            raise ValueError("Grouped decisions must share one state")
        question_id = example.get("question_id", "decision")
        if question_id in questions:
            raise ValueError(f"Duplicate question id: {question_id}")
        criteria = list(example["criteria"])
        if len(criteria) != len(set(criteria)):
            raise ValueError("System One choice criterion names must be unique")
        instructions = example["instructions"]
        if "target_index" in example:
            instructions += (
                f" Target token at position {example['target_index']}: "
                f"{example['target_token']}"
            )
        questions[question_id] = {
            "type": "choice", "instructions": instructions,
            "criteria": {criterion: None for criterion in criteria},
        }
    request["questions"] = questions
    return dict(request)
