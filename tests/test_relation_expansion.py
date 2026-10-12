import json

from scripts.repackage_dataset.relations import timeml_document
from tasksource.tasks import _biored_records, _maven_ere_records, _MAVEN_ERE_TEMPORAL


def test_timeml_joins_instance_ids_and_preserves_original_offsets():
    text, instances = timeml_document(b'<TimeML><TEXT>He <EVENT eid="e9"> left</EVENT> yesterday.</TEXT>'
        b'<MAKEINSTANCE eventID="e9" eiid="ei2"/></TimeML>')
    assert text == 'He  left yesterday.'
    assert text[slice(*instances['ei2'])] == 'left'
    assert 'e9' not in instances


def test_maven_ere_uses_clean_tokens_and_only_event_event_pairs():
    source = dict(id='document', tokens=['Today', 'he', 'left', 'then', 'returned'],
        mentions=['t0', 'e0', 'e1'], spans=[[0], [2], [4]],
        relations={label: [] for label in _MAVEN_ERE_TEMPORAL}, annots='GOLD ANSWER LEAK')
    source['relations']['BEFORE'] = [[1, 2], [0, 2]]
    rows = list(_maven_ere_records(source, _MAVEN_ERE_TEMPORAL))
    assert len(rows) == 1
    assert rows[0]['sentence1'] == 'Today he [A]left[/A] then [B]returned[/B]'
    assert rows[0]['labels'] == 'BEFORE'
    assert json.loads(rows[0]['metadata'])['split_group_id'] == 'document'
    assert 'GOLD ANSWER' not in rows[0]['sentence1']


def test_biored_marks_every_grounded_mention_and_excludes_conflicts():
    source = dict(document_id='pmid', document_text='A binds B. A.', annotation_texts=['A', 'B', 'A'],
        annotation_offsets=[0, 8, 11], annotation_lengths=[1, 1, 1], annotation_groundings=['x', 'y', 'x'],
        relation_entity1=['x'], relation_entity2=['y'], relation_type=['Bind'])
    rows = list(_biored_records(source))
    assert rows[0]['sentence1'] == '[A]A[/A] binds [B]B[/B]. [A]A[/A].'
    assert rows[0]['labels'] == 'Bind'
    source.update(relation_entity1=['x', 'y'], relation_entity2=['y', 'x'],
                  relation_type=['Bind', 'Association'])
    assert list(_biored_records(source)) == []


def test_native_temporal_label_error_keeps_view_out_of_training():
    from tasksource import list_tasks, parked
    assert 'maven-ere/temporal' not in set(list_tasks().id)
    assert 'maven_ere__temporal' in parked.by_kind('unsound')
    assert 'maven-ere/causal' in set(list_tasks().id)
