"""WebSRC and SWDE span extraction over visible page text (scripts/repackage_dataset/web_extraction.py)."""
import json

from scripts.repackage_dataset.web_extraction import (VisibleText, swde_record, swde_split, swde_value,
                                                      visible_text, websrc_page)
from tasksource.extractive_tasks import _swde_page, _websrc_page
from tasksource.jev.extraction import qa_rows


def test_visible_text_blocks_cells_entities_and_hidden_content():
    html = ('<html><head><title>T &amp; U</title><script>var x = "<b>no</b>";</script><style>p{}</style></head>'
            '<body><!-- note --><div>a<b>b</b>  c&nbsp;d</div><table><tr><td>1</td><td> 2 </td></tr>'
            '<tr><th>3</th></tr></table>x<br>y</body></html>')
    assert visible_text(html) == 'T & U\nab c d\n1 | 2\n3\nx\ny'


def test_visible_text_tid_ranges_and_unbalanced_markup():
    page = VisibleText('<div tid="0"><p tid="1">Price: <b tid="2">$5</b></p><p tid="3">Size</p></div>')
    assert page.text == 'Price: $5\nSize'
    start, end = page.ranges['2']
    assert page.text[start:end].strip() == '$5'  # a range may open with the separator before it
    assert 'Size' in page.text[slice(*page.ranges['0'])]
    assert visible_text('<div><p>open <i>never closed</div></span>tail') == 'open never closed\ntail'


def _qa(identity, element, start, answer):
    return {'id': identity, 'question': 'q?', 'element_id': element, 'answer_start': start, 'answer': answer}


def test_websrc_page_relocates_the_kth_occurrence_and_drops_non_spans():
    html = ('<html tid="0"><body tid="1"><ul tid="2"><li tid="3">Age: 25 / 25</li>'
            '<li tid="4">Model <br tid="5"/>X</li></ul></body></html>')
    context, questions, dropped = websrc_page(html, [
        _qa('a', 3, 10, '25'), _qa('b', 3, 5, '25'), _qa('c', -1, 1, 'yes'), _qa('d', 4, 0, 'Model X'),
        _qa('e', 3, 0, 'Nope')])
    assert context == 'Age: 25 / 25\nModel\nX'
    starts = {q['id']: q['answers']['answer_start'][0] for q in questions}
    assert starts == {'a': 10, 'b': 5}
    assert dict(dropped) == {'c': 'yesno', 'd': 'answer_cut_by_break', 'e': 'span_mismatch'}


def test_websrc_windows_never_cut_an_answer():
    context = ''.join(f'Line {i} of the page.\n' for i in range(400))
    answer = 'Line 350 of the page.'
    start = context.index(answer)
    row = {'id': 'p', 'context': context, 'questions': [
        {'id': 'q1', 'question': 'Which line?', 'answers': {'text': [answer], 'answer_start': [start]}},
        {'id': 'q2', 'question': 'First?', 'answers': {'text': ['Line 0'], 'answer_start': [0]}}]}
    records = list(_websrc_page(row))
    assert len(records) == 2 and len({r['context'] for r in records}) == 2
    for r in records:
        (text,), (s,) = r['answers']['text'], r['answers']['answer_start']
        assert r['context'][s:s + len(text)] == text and len(r['context']) <= 4000


def test_swde_value_trims_captions_naming_the_attribute_only():
    assert swde_value("Height: 6'10'&nbsp;", 'height') == "6'10'"
    assert swde_value('Official telephone: (902) 585-2201', 'phone') == '(902) 585-2201'
    assert swde_value('Date Posted: 05/20/2011', 'date_posted') == '05/20/2011'
    assert swde_value('$61,200 |', 'price') == '$61,200'
    assert swde_value('&gt; Online Store Manager', 'title') == 'Online Store Manager'
    assert swde_value("Pawns of Yalta: Soviet Refugees", 'title') == 'Pawns of Yalta: Soviet Refugees'
    assert swde_value('St. Martin&#039;s  Press', 'publisher') == "St. Martin's Press"


def test_swde_record_window_nulls_and_unlocated_attributes():
    text = 'Americana Express\n' + 'filler line\n' * 800 + 'Title: Avatar\nGenre: Drama | Action\n' + 'x\n' * 900
    labels = {'title': ['Avatar'], 'genre': ['Drama', 'Action'], 'cuisine': ['American'], 'rating': []}
    context, fields, reasons = swde_record(text, labels, 'k', window=500)
    assert len(context) <= 500 and context.startswith('filler') and reasons['unlocated'] == 1
    assert 'cuisine' not in fields and fields['rating'] == []
    for spans in fields.values():
        for found in spans:
            for start, value in found:
                assert context[start:start + len(value)] == value
    far = 'Avatar\n' + 'filler\n' * 200 + 'Drama Action'
    assert swde_record(far, {'title': ['Avatar'], 'genre': ['Drama']}, 'k', window=100)[0] is None


def test_swde_split_is_site_disjoint_and_deterministic():
    sites = [f's{i}' for i in range(10)]
    split = swde_split('book', sites)
    assert sorted(split.values()).count('train') == 8 and split == swde_split('book', sites[::-1])
    assert list(split.values()).count('test') == list(split.values()).count('validation') == 1


def test_swde_records_and_named_extraction_fields():
    context = 'Avatar (2009)\nAvatar\nGenres: Drama, Action, Drama'
    fields = {'title': [[[0, 'Avatar (2009)']], [[0, 'Avatar'], [14, 'Avatar']]],
              'genre': [[[29, 'Drama'], [44, 'Drama']], [[36, 'Action']]], 'mpaa_rating': []}
    rows = list(_swde_page('movie')({'id': 'movie/x/1', 'context': context, 'fields': json.dumps(fields)}))
    by_field = {json.loads(r['metadata'])['field']: r for r in rows}
    assert set(by_field) == {'title', 'genre', 'mpaa_rating'}  # director: not located, not asked
    assert sorted(zip(by_field['title']['answers']['answer_start'], by_field['title']['answers']['text'])) == \
        [(0, 'Avatar'), (0, 'Avatar (2009)'), (14, 'Avatar')]
    assert by_field['genre']['answers'] == {'text': ['Drama', 'Action'], 'answer_start': [29, 36]}
    assert by_field['mpaa_rating']['answers'] == {'text': [], 'answer_start': []}
    [row] = qa_rows(rows, 'swde/movie', 'train')
    assert [(f['name'], f['type']) for f in row['fields']] == [('title', 'text'), ('genre', 'list'),
                                                               ('mpaa_rating', 'text')]
    assert row['answers'][2]['null'] and not row['answers'][1]['null']
