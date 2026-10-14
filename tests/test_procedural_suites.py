import pytest

from scripts.repackage_dataset.procedural.suites import nearby_choices, sphinx_options, unique_maze_move


def test_nearby_counts_are_close_unique_and_verified():
    choices,label,metadata=nearby_choices(12,lambda c:int(c=='12'),42,upper=20)
    assert len(choices)==len(set(choices))==4 and choices[label]=='12'
    assert all(abs(int(c)-12)<=3 for c in choices)
    assert metadata['verifier_scores']==[int(i==label) for i in range(4)]
    with pytest.raises(ValueError,match='No verifier-rejected'):
        nearby_choices(0,lambda c:int(c=='0'),42,upper=0)


def test_native_panel_labels_and_text_choices_remain_distinct():
    meta={'labels':['(a)','(b)','(c)','(d)'],'answer':'(c)','question':'Which panel completes the pattern?'}
    question,choices,label,_=sphinx_options(meta,42)
    assert choices[label]=='Image panel C' and 'printed in the image' in question
    meta={'options':{'a':'V,L,T','b':'L,V,T'},'answer':'b',
          'question':'Order categories (a)–(d).\n\n(a) V,L,T\n(b) L,V,T\n\nAnswer with one letter.'}
    question,choices,label,_=sphinx_options(meta,42)
    assert choices[label]=='L,V,T' and '\n' not in question and 'one letter' not in question
    assert '(a)' not in question


def test_maze_target_and_unique_next_move_are_checked():
    grid=[[1,1,1,1,1],[1,0,0,0,1],[1,0,1,0,1],[1,0,0,0,1],[1,1,1,1,1]]
    with pytest.raises(ValueError,match='Multiple optimal'):
        unique_maze_move(grid,(1,1),(3,3))
    grid[2][1]=1
    label,scores=unique_maze_move(grid,(1,1),(3,3))
    assert label==0 and scores==[1,0,0,0]
    with pytest.raises(ValueError,match='Solved, unreachable or trivial'):
        unique_maze_move(grid,(3,3),(3,3))


def test_one_match_requires_one_removed_and_one_added_segment():
    from scripts.repackage_dataset.procedural.visgym import one_match_reachable
    segments={'6':set('abcdef'),'8':set('abcdefg'),'+':set('hv'),'-':set('h'),'4':set('bcdg'),'=':set('xy'),'2':set('abdeg')}
    assert one_match_reachable('6+4=4','8-4=4',segments)
    assert not one_match_reachable('6+4=4','6-2=4',segments)
    assert not one_match_reachable('6+4=4','6+4=4',segments)
    assert not one_match_reachable('6+4=4','8-44=4',segments)


def test_block_legality_rejects_collision_and_border_but_allows_self_overlap():
    from scripts.repackage_dataset.procedural.visgym import block_move_legal
    board=[[1,1,0],[0,0,2]]
    blocks={1:dict(position=(0,0),size=(1,2)),2:dict(position=(1,2),size=(1,1))}
    assert block_move_legal(board,blocks,1,1)
    assert block_move_legal(board,blocks,1,2)
    assert not block_move_legal(board,blocks,1,0)
    assert not block_move_legal(board,blocks,2,1)
    board[0][2]=2
    assert not block_move_legal(board,blocks,1,1)
