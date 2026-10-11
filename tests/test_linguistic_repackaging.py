from collections import Counter

from scripts.repackage_dataset.linguistic import spider_rows, streusle_rows


def test_spider_column_and_aggregation_match_native_sql_without_exposing_it():
    schema=dict(table_names_original=['players'],column_names_original=[[-1,'*'],[0,'name'],[0,'score']],column_types=['text','text','number'])
    row=dict(db_id='league',question='What is the highest score?',query='SELECT MAX(score) FROM players',
        sql=dict(select=[False,[[1,[0,[0,2,False],None]]]]))
    excluded=Counter();out=list(spider_rows(row,schema,'aggregation',excluded))[0]
    assert out['labels']=='maximum' and row['query'] not in out['sentence1']
    assert not list(spider_rows(row,schema,'column',excluded))
    row['sql']['select'][1][0][0]=0
    out=list(spider_rows(row,schema,'column',excluded))[0]
    assert out['choices_list'][out['labels']]=='players.score'


def test_streusle_marks_discontinuous_expression_and_keeps_review_identity():
    import json
    row=dict(sent_id='reviews-123-0002',toks=[dict(word=w) for w in ['They','took','it','off','.']],
        swes={},smwes={'1':dict(ss='v.motion',toknums=[2,4],lexlemma='take off')})
    out=list(streusle_rows(row))[0]
    assert out['sentence1']=='They [TARGET] took [/TARGET] it [TARGET] off [/TARGET] .'
    assert out['labels']=='v.motion'
    assert json.loads(out['metadata'])['split_group_id']=='reviews-123'


def test_spider_does_not_make_equivalent_join_keys_compete():
    schema=dict(table_names_original=['people','visits'],column_names_original=[[-1,'*'],[0,'id'],[1,'person_id']],
                column_types=['text','number','number'],foreign_keys=[[2,1]])
    row=dict(db_id='db',question='Which people visited?',query='SELECT people.id FROM people JOIN visits',
        sql=dict(select=[False,[[0,[0,[0,1,False],None]]]]))
    excluded=Counter()
    assert not list(spider_rows(row,schema,'column',excluded))
    assert excluded['foreign_key_equivalent_column']==1
