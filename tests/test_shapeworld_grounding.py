from scripts.repackage_dataset.procedural.shapeworld_grounding import scene_queries


def entity(color,box):
    return dict(color=dict(name=color),shape=dict(name='circle'),bounding_box=dict(
        topleft=dict(zip(('x','y'),box[:2])),bottomright=dict(zip(('x','y'),box[2:]))))


def test_queries_require_unique_referents_in_the_native_scene():
    objects=[entity('red',[.1,.1,.2,.2]),entity('red',[.5,.1,.6,.2]),entity('blue',[.3,.5,.4,.6])]
    boxes,queries=scene_queries(objects)
    assert ('Locate the red circle.',boxes[0],'attribute') not in queries
    assert ('Locate the blue circle.',boxes[2],'attribute') in queries
    assert ('Locate the red circle left of the blue circle.',boxes[0],'spatial') in queries
    assert ('Locate the red circle right of the blue circle.',boxes[1],'spatial') in queries
    assert not any(text=='Locate the red circle above the blue circle.' for text,_,_ in queries)

    assert not any(kind=='spatial' and text.startswith('Locate the blue circle') for text,_,kind in queries)
