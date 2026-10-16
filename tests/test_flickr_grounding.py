from scripts.repackage_dataset.flickr_grounding import PHRASE, marked_caption, phrase_boxes


def test_native_pixel_edges_and_multiple_boxes():
    xml='''<annotation><size><width>20</width><height>10</height></size>
    <object><name>1</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>20</xmax><ymax>10</ymax></bndbox></object>
    <object><name>1</name><bndbox><xmin>2</xmin><ymin>2</ymin><xmax>4</xmax><ymax>5</ymax></bndbox></object>
    <object><name>2</name><nobndbox>1</nobndbox><scene>0</scene></object></annotation>'''
    size,boxes=phrase_boxes(xml)
    assert size==[20,10] and boxes['1']==[[0,0,1,1],[.05,.1,.2,.5]]
    assert '2' not in boxes


def test_repeated_phrase_keeps_queried_occurrence_and_context():
    source='[/EN#1/people A man] watches [/EN#2/people a man] .'
    phrases=list(PHRASE.finditer(source))
    assert marked_caption(source,phrases[1])=='A man watches [PHRASE]a man[/PHRASE] .'
    assert [p[1] for p in phrases]==['1','2']
