"""Published repairs keep gold supervision while fixing task-specific surfaces."""
import pytest
from scripts.repair_jev_questions import pair, rewrite

Q={'probability_words_nli/reasoning_1hop': 'Given the context, is this conclusion valid?\n{sentence2}',
   'multidogo/finance/intent': 'What intent does the current utterance express? Use earlier dialogue when available.'}


def row(source='probability_words_nli/reasoning_1hop', **changes):
    return dict(source=source, state='text_A: Probably A.\ntext_B: A.',
        question='Given the probabilities stated in text_A, is concluding text_B valid?',
        variant='direct',id='source:train:0',kind='choice',options=['invalid','valid'],target=[0.,1.],**changes)


@pytest.mark.parametrize('state', ['text_A: Probably A.\ntext_B: A.',
    'First text:\nProbably A.\n\nSecond text:\nA.', 'Passage A:\nProbably A.\n\nPassage B:\nA.', 'A: Probably A.\nB: A.'])
def test_native_and_augmented_pairs_keep_the_same_evidence(state):
    original=row();original['state']=state
    result=rewrite(original,Q)
    assert pair(state)=={'sentence1':'Probably A.','sentence2':'A.'}
    assert result['state']=='Probably A.' and result['question'].endswith('\nA.')
    assert {k:v for k,v in result.items() if k not in {'state','question'}} == {k:v for k,v in original.items() if k not in {'state','question'}}


def test_verification_proposal_and_boolean_gold_are_unchanged():
    original=row();original.update(variant='label_verification',kind='noul',options=[],target=[0.],
        question=original['question']+' Is "invalid" the correct answer?')
    result=rewrite(original,Q)
    assert result['question'].endswith('\nA. Is "invalid" the correct answer?')
    assert result['target']==[0.] and result['options']==[]


def test_packed_view_keeps_separate_queries_inside_items():
    original=row();original.update(variant='packed_derived',state='Item A:\n'+original['state'])
    assert rewrite(original,Q)==original
    original['source']='maven-arg/role'
    with pytest.raises(ValueError,match='packed'):
        rewrite(original,{**Q,'maven-arg/role':'Role? {sentence2}'})


def test_format_augmentation_retains_two_named_fields():
    original=row();original.update(variant='paired_text_format')
    result=rewrite(original,Q)
    assert result['state']=='Context:\nProbably A.\n\nConclusion:\nA.'
    assert result['question']=='Given the context, is the conclusion valid?'


def test_empty_history_and_blank_second_field_are_omitted():
    original=row('multidogo/finance/intent');original['state']='text_A: \ntext_B: Check my balance.'
    result=rewrite(original,Q)
    assert result['state']=='Check my balance.' and 'text_A' not in result['question']
    original=row('edos');original.update(state='text_A: A post.\ntext_B: ',question='Is it sexist?')
    assert rewrite(original,Q)['state']=='A post.'


def test_unrelated_rows_are_exact_and_parked_tasks_removed():
    original=row('other')
    assert rewrite(original,Q) is original
    for source in ['wouldyourather','puzzte','linguisticprobing/sentence_length']:
        assert rewrite(row(source),Q) is None
    original=row();original['state']='Missing paired context'
    with pytest.raises(ValueError,match='delimiter'):
        rewrite(original,Q)
