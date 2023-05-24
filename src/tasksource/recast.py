import random
from datasets import DatasetDict, Dataset
from sorcery import dict_of
import string

improper_labels = ['recast/recast_kg_relations','linguisticprobing',"lexglue/scotus","pragmeval/squinky","pragmeval/emobank",'pragmeval/persuasiveness']
improper_labels += ['glue/stsb', 'sick/relatedness', 'joci', 'utilitarianism', 'amazon_counterfactual/en', 'toxic_conversations', 'ethos/multilabel', 'lex_glue/eurlex', 'lex_glue/unfair_tos', 'app_reviews', 'humicroedit/subtask-1', 'stackoverflow-questions', 'go_emotions/simplified', 'google_wellformed_query', 'has_part', 'blog_authorship_corpus/age', 'promptCoherence', 'Sarcasm_News_Headline', 'auditor_review/demo-org--auditor_review', 'Dynasent_Disagreement', 'Politeness_Disagreement', 'SBIC_Disagreement', 'SChem_Disagreement', 'Dilemmas_Disagreement', 'sts-companion', 'acceptability-prediction', 'chaos-mnli-ambiguity', 'headline_cause/en_simple', 'oasst1_dense_flat', 'civil_comments']


def render_options(options):
    options = [f'"{x}"' for x in options]
    return f"{', '.join(options[:-1])} or {options[-1]}"

def render_classification(text,options,answer):
    example = 'A→B' if text.startswith('A:') else 'the following'
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
    inputs=f'{inputs}With no explanation, chose the best option from {render_options(letters)}.'    
    for letter, option in zip(letters, options):
        inputs+=f'\n{letter}: {option}'
        assert "{letter}" not in inputs
    targets = f'{letters[labels]}.'
    return dict_of(inputs, targets) 

def negative_sample_options(y, labels,N=4):
    if len(labels)<N:
        return labels
    else:
        return [y]+random.sample([x for x in labels if x!=y], N-1)

def recast_dataset_zero_shot_nli(dataset,N=4):

    def recast_split(d,N=N):
        labels = d.features['labels']
        df=d.to_pandas()
        df['inputs'] = df.sentence1
        if "sentence2" in df:
            df['inputs'] +="[SEP]" + df.sentence2

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
        choices = [k for k in x if 'choice' in k]
        if all([x[c] in x['inputs'] for c in choices]):
            return {"inputs":x['inputs'], 'targets': x[f"choice{x['labels']}"]}
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
            text=f"A: {x['sentence1']}\nB: {x['sentence2']}"
        else:
            text=x['sentence1']
            
        answer=labels.int2str(x['labels'])
        options= negative_sample_options(answer, labels._int2str)
        return render_classification(text, options, answer)
        
    dataset = dataset.map(eval(f"recast_{task_type}"))
    dataset = dataset.remove_columns([k for k in features if k not in ['inputs','targets']])
    return dataset
 