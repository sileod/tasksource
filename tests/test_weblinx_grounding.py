import json
from types import SimpleNamespace

from scripts.repackage_dataset.weblinx_grounding import native_query, visual_query
from tasksource.grounding import coordinate_text


def query():
    return native_query(dict(demo='demo',turn=3,action='click(uid="a", text="FUTURE_SECRET")',
        action_history='Opened the page',utterances='Choose Search',clean_html='unused',viewport='50h x 100w',
        candidates='(uid = a) [[bbox]] x=10 y=20 width=30 height=20 [[text]] Search\n'
                   '(uid = b) [[bbox]] x=70 y=10 width=20 height=10 [[text]] Cancel'),[])


def test_zoomed_native_target_and_no_future_action_leakage():
    native=query();turn=SimpleNamespace(viewport_width=100,viewport_height=50,
        element={'bbox':dict(left=20,top=40,right=80,bottom=80)})
    excluded=[];qa=visual_query(native,turn,[200,100],'screen.png',excluded)
    info=json.loads(qa['metadata'])
    assert info['target_bbox']==[.1,.4,.4,.8]
    assert qa['choices_list'][qa['labels']]==coordinate_text(info['target_bbox'])
    assert 'FUTURE_SECRET' not in qa['inputs'] and 'DOM:' not in qa['inputs']
    assert 'Opened the page' in qa['inputs'] and not excluded


def test_mismatched_screenshot_and_target_frame_are_rejected():
    native=query();turn=SimpleNamespace(viewport_width=100,viewport_height=50,
        element={'bbox':dict(left=140,top=20,right=180,bottom=40)})
    excluded=[]
    assert visual_query(native,turn,[200,120],'screen.png',excluded) is None
    assert excluded[-1]['reason']=='screenshot_viewport_mismatch'
    assert visual_query(native,turn,[200,100],'screen.png',excluded) is None
    assert excluded[-1]['reason']=='target_coordinate_frame_mismatch'
