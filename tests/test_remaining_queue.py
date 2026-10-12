import json
import pytest
from tasksource.tasks import _slot_decisions, _top_slots


def test_slots_preserve_context_spans_and_group_without_label_leakage():
    text='send this to Alice tomorrow'
    rows=list(_slot_decisions(text,[dict(type='person',span=[3]),dict(type='date',span=[4])],
                             dict(id='utterance',split_group_id='group')))
    assert rows[0]['sentence1']=='send this to [TARGET]Alice[/TARGET] tomorrow'
    assert rows[0]['labels']=='person'
    assert json.loads(rows[0]['metadata'])['split_group_id']=='group'
    assert rows[0]['metadata']!=rows[1]['metadata']
    assert 'person' not in rows[0]['sentence1']
    with pytest.raises(ValueError,match='outside'):
        list(_slot_decisions(text,[dict(type='date',span=[10])],dict(id='u')))


def test_top_slots_strip_gold_parse_tags_and_keep_only_leaf_targets():
    row=dict(domain='navigation',utterance='Go to my current location',semantic_parse=
             '[IN:GET_DIRECTIONS Go to [SL:DESTINATION [IN:GET_LOCATION my [SL:LOCATION_USER current location ] ] ] ]')
    rows=list(_top_slots(row))
    assert len(rows)==1
    assert rows[0]['sentence1']=='Go to my [TARGET]current location[/TARGET]'
    assert rows[0]['labels']=='location user'
    assert 'IN:' not in rows[0]['sentence1'] and 'SL:' not in rows[0]['sentence1']
    assert list(_top_slots(dict(row,utterance='A different utterance')))==[]
    with pytest.raises(ValueError,match='Unbalanced'):
        list(_top_slots(dict(row,semantic_parse='[IN:GET_DIRECTIONS Go')))


def test_slurp_uses_native_contraction_tokens():
    from tasksource.tasks import _slurp_slots
    row=dict(sentence1="what's happening tonight in Chicago",metadata=json.dumps(dict(
        id='u',split_group_id='g',tokens=['what',"'s",'happening','tonight','in','Chicago'],
        entities=[dict(type='timeofday',span=[3])])))
    result=list(_slurp_slots(row))
    assert result[0]['sentence1']=="what 's happening [TARGET]tonight[/TARGET] in Chicago"
    assert result[0]['labels']=='timeofday'
    assert list(_slurp_slots(dict(row,sentence1='unrelated text')))==[]
