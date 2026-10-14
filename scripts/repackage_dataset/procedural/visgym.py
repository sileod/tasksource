"""Other static VisGym decisions, with every distractor checked on its instance."""
from collections import deque
import copy
import random
import sys

from .common import checkout, verified_choices
from .suites import VISGYM, _export


def one_match_reachable(before, after, segments):
    if len(before)!=len(after):return False
    changes=[(set(segments[a])-set(segments[b]),set(segments[b])-set(segments[a]))
             for a,b in zip(before,after) if a!=b]
    return len(changes)==2 and sorted((len(a),len(b)) for a,b in changes)==[(0,1),(1,0)]


def equation_candidates(env, correct, rng, max_states=300):
    start=tuple(env.state.broken);first=env._neighbors_one_move(start)
    reachable={n for n,_ in first};gold=sorted(n for n in reachable if correct(''.join(n)))
    if correct(''.join(start)) or not gold:raise ValueError('No broken one-match puzzle')
    # Every displayed equation is arithmetically true: arithmetic alone cannot
    # reveal the answer. Distractors require more than one native match transfer.
    queue=deque([(start,0)]);seen={start};donors=set();visited=0
    while queue and len(donors)<3 and visited<max_states:
        state,depth=queue.popleft();visited+=1
        if depth>=4:continue
        neighbors=env._neighbors_one_move(state);rng.shuffle(neighbors)
        for nxt,_ in neighbors:
            if nxt in seen:continue
            seen.add(nxt);queue.append((nxt,depth+1))
            if nxt not in reachable and correct(''.join(nxt)):donors.add(nxt)
    return ''.join(rng.choice(gold)),[(''.join(n),'native_2_to_4_moves') for n in sorted(donors)],start


def _native():
    root,_=checkout(*VISGYM);sys.path.insert(0,str(root))
    from PIL import Image
    return Image


def visgym_equation(max_rows=1000,max_rows_eval=100,seed=42):
    """VisGym one-match equation repair, with mathematically correct distractors.

    Each candidate is a true equation. Exactly one can be reached from the shown
    equation by moving one match between different symbols; others require two
    to four native moves. Native transitions and independent segment accounting
    verify reachability. Two to four choices, depending on available distractors.
    No model annotations, published puzzles, or external image assets are used.
    """
    Image=_native()
    from gymnasium.envs.matchstick_equation.matchstick_equation import MatchstickEquationEnv
    from gymnasium.envs.matchstick_equation import matchstick_puzzles as puzzle
    env=MatchstickEquationEnv(image_width=900,image_height=180,break_moves=1)
    def generate(s,index):
        image,_=env.reset(seed=s)
        gold,donors,start=equation_candidates(env,puzzle.is_equation_correct,random.Random(s))
        def verify(c):
            independent=one_match_reachable(start,c,puzzle.SYMBOL_TO_SEGMENTS)
            native=tuple(c) in {n for n,_ in env._neighbors_one_move(start)}
            if native!=independent:raise ValueError('Native and independent match reachability disagree')
            return int(native and puzzle.is_equation_correct(c))
        choices,label,alignment=verified_choices(gold,donors,verify,s)
        return Image.fromarray(image),'Which equation can be obtained from the picture by moving exactly one matchstick from one symbol to a different symbol? All choices are arithmetically true. Use standard arithmetic precedence; * and / denote multiplication and division.',choices,label,dict(task='equation_repair',equation=''.join(start),
            verification='native_transitions_and_independent_segment_accounting',**alignment)
    return _export('visgym-equation-mcq',VISGYM,'apache-2.0',generate,max_rows,max_rows_eval,seed,
                   dict(scope='one_match_equation_repair',max_search_states=300,max_distractor_depth=4,image_size=[900,180]),builder=visgym_equation)


def visgym_rotation(max_rows=1000,max_rows_eval=100,seed=42):
    """VisGym stick-orientation decisions with nearby, verifier-rejected angles.

    Uses the native stick renderer and angle reward modulo 180 degrees. Translation
    is explicitly ignored: no hidden movement scale is needed. Both sticks are
    fully visible and separated. Choices differ by 15 or 30 degrees; every option
    is tested with the native reward after eliminating positional error.
    """
    Image=_native()
    from gymnasium.envs.matchstick_rotation.matchstick_rotation import MatchstickRotationEnv
    env=MatchstickRotationEnv(scale_range=(1,1),stick_length=30,pos_tolerance=0.1,ang_tolerance=1)
    def generate(s,index):
        rng=random.Random(s);blue=[rng.uniform(20,80),rng.uniform(20,80)]
        red=[rng.uniform(20,80),rng.uniform(20,80)]
        while sum((a-b)**2 for a,b in zip(blue,red))<35**2:
            red=[rng.uniform(20,80),rng.uniform(20,80)]
        angle=rng.randrange(180);gold=15*rng.randrange(12)
        geometry=dict(scale=1.0,target_mid=red,initial_mid=blue,target_ang=(angle+gold)%180,initial_ang=angle)
        image,_=env.reset(seed=s,init_state=geometry)
        original=copy.deepcopy(env.state)
        def verify(c):
            try:
                env.state=copy.deepcopy(original);env.state['mid']=env.target['mid'].copy()
                env._move([0,0,int(c)])
                return int(env._compute_reward())
            finally:env.state=copy.deepcopy(original)
        donors=[(str((gold+d)%180),f'angle_offset:{d}') for d in (-30,-15,15,30)]
        raw,label,alignment=verified_choices(str(gold),donors,verify,s)
        return Image.fromarray(image),'Which counterclockwise rotation makes the blue stick parallel to the red dashed stick? Ignore their positions. Zero degrees means no rotation.',[c+' degrees counterclockwise' for c in raw],label,dict(task='stick_rotation',geometry=geometry,
            verifier_answers=raw,verification='native_modulo_180_orientation_reward',**alignment)
    return _export('visgym-rotation-mcq',VISGYM,'apache-2.0',generate,max_rows,max_rows_eval,seed,
                   dict(scope='stick_orientation',angle_step=15,minimum_separation=35,translation_ignored=True),builder=visgym_rotation)


def block_move_legal(board, blocks, block_id, direction):
    block=blocks[block_id];r,c=block['position'];h,w=block['size']
    dr,dc=((-1,0),(0,1),(1,0),(0,-1))[direction];r+=dr;c+=dc
    return r>=0 and c>=0 and r+h<=len(board) and c+w<=len(board[0]) and all(
        board[y][x] in (0,block_id) for y in range(r,r+h) for x in range(c,c+w))


def visgym_blocks(max_rows=1000,max_rows_eval=100,seed=42):
    """VisGym sliding-block move legality from fresh native legal random walks.

    Select one collision-free one-cell move among four native block/direction
    candidates. Distractors stay within board bounds but collide with another
    block. Native placement rules and independent occupancy checks must agree.
    Only the current board is shown; the native target panel is cropped away.
    This task predicts legality, not an arbitrarily chosen optimal solution.
    """
    Image=_native()
    from gymnasium.envs.sliding_block.sliding_block import SlidingBlockEnv
    env=SlidingBlockEnv(num_shuffle_moves=0,seed=0)
    directions=('up','right','down','left')
    def generate(s,index):
        rng=random.Random(s);env.reset(seed=s%(2**32));previous=None;walk=rng.randrange(128,513)
        for _ in range(walk):
            moves=env._get_valid_moves();allowed=[a for a in moves if a!=previous] or moves
            block,direction=rng.choice(allowed);env._move_block(block,direction);previous=(block,(direction+2)%4)
        candidates=[(b,d) for b in env.blocks for d in range(4)]
        valid=[];blocked=[]
        for b,d in candidates:
            native=env._can_move(b,d)
            if native!=block_move_legal(env.board,env.blocks,b,d):raise ValueError('Native and independent block legality disagree')
            if native:valid.append((b,d))
            else:
                block=env.blocks[b];r,c=block['position'];h,w=block['size'];dr,dc=((-1,0),(0,1),(1,0),(0,-1))[d]
                if 0<=r+dr<=env.rows-h and 0<=c+dc<=env.cols-w:blocked.append((b,d))
        if not valid or len(blocked)<3:raise ValueError('Insufficient collision distractors')
        gold=rng.choice(valid);rng.shuffle(blocked);chosen=[gold]+blocked[:3];rng.shuffle(chosen);label=chosen.index(gold)
        full=env.render();crop=[220,40,420,290];image=Image.fromarray(full).crop(crop)
        choices=[f'Move block {b} one cell {directions[d]}' for b,d in chosen]
        return image,'Which move is legal in the displayed board? Move exactly one numbered block by one cell, without rotating it, crossing the border, or overlapping any other block.',choices,label,dict(task='block_legality',board=env.board.tolist(),
            blocks=copy.deepcopy(env.blocks),candidate_moves=chosen,crop_xyxy=crop,random_walk_steps=walk,
            verifier_scores=[int(env._can_move(b,d)) for b,d in chosen],verification='native_placement_and_independent_occupancy')
    return _export('visgym-block-mcq',VISGYM,'apache-2.0',generate,max_rows,max_rows_eval,seed,
                   dict(scope='sliding_block_legality',random_walk_steps=[128,512],crop='current_board_only'),builder=visgym_blocks)
