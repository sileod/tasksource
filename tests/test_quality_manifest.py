import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tasksource.quality import decision_key, ExclusionIndex
from scripts.extract_jev_exclusions import extract, policy_category


def row(identifier='one',source='task',license_use='commercial'):
    return dict(id=identifier,source=source,state='Some input',question='Which answer?',
                kind='choice',options=['first','second'],target=[1.0,0.0],group_id='group',
                question_id='decision',variant='direct',split='train',license_use=license_use)


def test_identity_matches_reordered_choices_and_internal_recast():
    published=row()
    internal=dict(task='task',state=published['state'],instructions=published['question'],
                  criteria=['second','first'],label=1)
    assert decision_key(published)==decision_key(internal)
    assert decision_key(published)!=decision_key({**published,'target':[0,1]})
    assert decision_key(published)!=decision_key({**published,'question':'Another task?'})
    assert decision_key(published)!=decision_key({**published,'state':'Repaired context'})


def test_scores_and_positional_options_keep_order():
    original={**row(),'options':['first','both of the above']}
    changed={**original,'options':list(reversed(original['options'])),'target':[0,1]}
    assert decision_key(original)!=decision_key(changed)
    original={**row(),'kind':'score'}
    changed={**original,'options':list(reversed(original['options'])),'target':[0,1]}
    assert decision_key(original)!=decision_key(changed)


def test_policy_overlap_uses_benchmark_precedence():
    rule={'blocked_roots':['BLOCKED'],'casefolded_source_prefixes':['multilingual/heldout/']}
    assert policy_category(row(source='blocked/config',license_use='non-commercial'),rule)=='benchmark_policy'
    assert policy_category(row(source='multilingual/heldout/foo'),rule)=='benchmark_policy'


def test_inferred_candidates_never_become_approved_errors(tmp_path):
    rows=[row('kept'),row('license',license_use='non-commercial'),row('holdout',source='blocked/config'),
          row('unknown'),{**row('derived'),'variant':'instruction_paraphrase','question':'Choose differently'}]
    source=tmp_path/'source.parquet';selected=tmp_path/'selected.parquet'
    pq.write_table(pa.Table.from_pylist(rows),source)
    pq.write_table(pa.Table.from_pylist(rows[:1]),selected)
    report=dict(retained_rows=1,source_rows=5,source_revision='frozen',
                benchmark_source_holdout_rule={'blocked_roots':['blocked'],'casefolded_source_prefixes':[]},
                prefilter_exclusions={'license_not_commercial':1,'benchmark_source_holdout':1},
                llm_rejected_rows=1,unresolved_api_error_rows_excluded=0)
    output=tmp_path/'output';manifest=extract([source],[selected],report,output,validate_hashes=False)
    assert manifest['categories']==dict(license_policy=1,benchmark_policy=1,unattributed_exclusion=2)
    assert manifest['approved_removals']==[]
    assert manifest['identified_error_ids_available'] is False
    candidates=pq.read_table(output/'candidates.parquet').to_pylist()
    assert {candidate['id'] for candidate in candidates}=={'unknown','derived'}
    index=ExclusionIndex.from_parquet(output/'candidates.parquet')
    assert {match['id'] for match in index.match(row())}=={'unknown'}
    assert not index.match({**row(),'target':[0,1]})
    assert not index.match({**row(),'question':'An unrelated decision in the same group'})


def test_extract_rejects_missing_retained_source_ids(tmp_path):
    source=tmp_path/'source.parquet';selected=tmp_path/'selected.parquet'
    pq.write_table(pa.Table.from_pylist([row('source')]),source)
    pq.write_table(pa.Table.from_pylist([row('missing')]),selected)
    report=dict(retained_rows=1,source_rows=1,source_revision='frozen',
        benchmark_source_holdout_rule={'blocked_roots':[],'casefolded_source_prefixes':[]})
    with pytest.raises(ValueError,match='absent from frozen source'):
        extract([source],[selected],report,tmp_path/'output',validate_hashes=False)
