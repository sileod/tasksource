import json

import pytest
from scripts.repackage_dataset.sbic import ANNOTATION_FIELDS, aggregate_sbic


def annotation(worker,post='A post.',**values):
    return dict.fromkeys(ANNOTATION_FIELDS,'') | dict(post=post,WorkerId=worker,HITId='hit',
        offensiveYN='1.0',intentYN='1.0',whoTarget='1.0') | values


def aggregate(train,test=(),validation=()):
    return aggregate_sbic(dict(train=train,test=test,validation=validation))


def test_repeat_stereotypes_count_once_and_original_ratings_survive():
    rows=[annotation('a',targetStereotype='First'),annotation('a',targetStereotype='Second'),
          annotation('b'),annotation('c')]
    output,report,excluded=aggregate(rows)
    row=output['train'][0];meta=json.loads(row['metadata'])
    assert row['offensiveness']=='offensive'
    assert meta['aggregation']['offensiveness']['counts']=={'1.0':3}
    assert len(meta['annotations'])==4
    assert meta['annotations'][1]['targetStereotype']=='Second'
    assert report['train']['retained_posts']==1 and not excluded


def test_three_workers_and_strong_agreement_are_required_per_field():
    output,_,_=aggregate([annotation('a'),annotation('b'),annotation('c',offensiveYN='0.0')])
    row=output['train'][0]
    assert row['offensiveness'] is None and row['intent']=='definitely intended to offend'
    assert json.loads(row['metadata'])['aggregation']['offensiveness']['status']=='disagreement'
    output,_,excluded=aggregate([annotation('a')]*3)
    assert not output['train'] and excluded[0]['reason']=='no_reliable_label'


def test_conflicting_worker_abstains_and_blank_group_answers_are_preserved():
    output,_,_=aggregate([annotation('a',whoTarget=''),annotation('b',whoTarget=''),
                         annotation('c',whoTarget=''),annotation('a',intentYN='0.0')])
    row=output['train'][0];stats=json.loads(row['metadata'])['aggregation']
    assert row['intent'] is None
    assert [a['whoTarget'] for a in json.loads(row['metadata'])['annotations']]==['','','','1.0']
    assert stats['intent']['conflicting_workers']==1 and stats['intent']['valid_workers']==2
    assert row['offensiveness']=='offensive'


def test_native_uncertainty_is_a_category_not_a_probability():
    output,_,_=aggregate([annotation(w,offensiveYN='0.5',intentYN='0.66') for w in 'abc'])
    row=output['train'][0]
    assert row['offensiveness']=='uncertain' and row['intent']=='probably intended to offend'
    assert json.loads(row['metadata'])['aggregation']['offensiveness']['counts']=={'0.5':3}


def test_rejected_holdout_still_reserves_post_against_training():
    output,report,excluded=aggregate([annotation(w) for w in 'abc'],test=[annotation('d',post=' A POST. ')])
    assert not output['train'] and not output['test']
    assert report['train']['decisions']['cross_split_duplicate']==1
    assert {r['reason'] for r in excluded}=={'no_reliable_label','cross_split_duplicate'}
    with pytest.raises(ValueError):aggregate_sbic({},min_votes=1)


def test_existing_disagreement_holdouts_do_not_leak_into_new_views():
    from scripts.repackage_dataset.relevance import text_key
    rows=[annotation(w) for w in 'abc']
    dataset=dict(train=rows,validation=[],test=[])
    output,report,excluded=aggregate_sbic(dataset,reserved_splits={'test':{text_key('A post.')}})
    assert not output['train']
    assert report['train']['decisions']['existing_task_split_conflict']==1
    assert excluded[0]['conflicting_splits']==['test']
    output,_,_=aggregate_sbic(dataset,reserved_splits={'train':{text_key('A post.')}})
    assert len(output['train'])==1
