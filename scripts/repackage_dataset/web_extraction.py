"""Span extraction over the visible text of web pages: WebSRC questions and SWDE attribute records.

Both sources ship HTML; models see the deterministic visible text of ``VisibleText`` and every gold span
is a character range of that text (``context[start:start + len(text)] == text``, checked on write).
"""
from collections import Counter
import csv
import hashlib
from html import unescape
from html.parser import HTMLParser
import inspect
import io
import json
from pathlib import Path
import re
import zipfile

from datasets import Dataset, DatasetDict, Features, List, Value
from huggingface_hub import hf_hub_download

from .browser import WEB_SRC, WEB_SRC_REVISION, NativeDOM


# Line-breaking elements; table cells of a row are joined by ' | '.
BLOCK = {'address', 'article', 'aside', 'blockquote', 'body', 'br', 'caption', 'center', 'dd', 'div', 'dl',
         'dt', 'fieldset', 'figcaption', 'figure', 'footer', 'form', 'frame', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
         'head', 'header', 'hr', 'html', 'iframe', 'legend', 'li', 'main', 'nav', 'noscript', 'ol', 'option', 'p',
         'pre', 'section', 'select', 'table', 'tbody', 'textarea', 'tfoot', 'thead', 'title', 'tr', 'ul'}
CELL = {'td', 'th'}
SKIP = {'script', 'style', 'template'}
VOID = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}
RANK = {None: 0, ' ': 1, ' | ': 2, '\n': 3}


class VisibleText(HTMLParser):
    """Deterministic HTML to visible text, tolerant of unbalanced markup.

    Character references are decoded and whitespace runs (including no-break spaces) collapse to one
    space. Block elements start a new line, table cells of a row are separated by ``' | '``, inline
    elements add nothing. Script, style and template content, comments and the head (except its
    ``title``) are dropped. ``ranges`` maps each element's ``tid`` attribute (WebSRC) to its
    ``[start, end)`` range in ``text``.
    """
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.parts, self.length, self.pending = [], 0, None
        self.skip, self.head, self.title = None, False, False
        self.ranges, self.open = {}, []
        self.feed(html)
        self.close()
        self.text = ''.join(self.parts)

    def _break(self, separator):
        if RANK[separator] > RANK[self.pending]:
            self.pending = separator

    def handle_starttag(self, tag, attrs):
        if self.skip:
            return
        if tag in SKIP:
            self.skip = tag
            return
        if tag == 'head':
            self.head = True
        elif tag == 'body':
            self.head = False
        elif tag == 'title':
            self.title = True
        if tag in BLOCK:
            self._break('\n')
        elif tag in CELL:
            self._break(' | ')
        identity = dict(attrs).get('tid')
        if identity is not None and tag not in VOID:
            self.open.append((tag, identity, self.length))
        elif identity is not None:
            self.ranges[identity] = (self.length, self.length)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID and self.open and self.open[-1][0] == tag:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if self.skip:
            if tag == self.skip:
                self.skip = None
            return
        if tag == 'head':
            self.head = False
        elif tag == 'title':
            self.title = False
        if tag in BLOCK:
            self._break('\n')
        elif tag in CELL:
            self._break(' | ')
        for i in range(len(self.open) - 1, -1, -1):
            if self.open[i][0] == tag:
                for _, identity, start in self.open[i:]:
                    self.ranges[identity] = (start, self.length)
                del self.open[i:]
                break

    def handle_data(self, data):
        if self.skip or (self.head and not self.title):
            return
        text = re.sub(r'\s+', ' ', data)
        if text[:1] == ' ':
            self._break(' ')
        text = text.strip()
        if not text:
            return
        if self.parts and self.pending:
            self.parts.append(self.pending)
            self.length += len(self.pending)
        self.parts.append(text)
        self.length += len(text)
        self.pending = ' ' if data[-1:].isspace() else None


def visible_text(html):
    return VisibleText(html).text


def occurrences(text, value, start=0, end=None):
    """Start offsets of every (possibly overlapping) occurrence of ``value`` in ``text[start:end]``."""
    end = len(text) if end is None else end
    found, at = [], text.find(value, start, end)
    while at >= 0:
        found.append(at)
        at = text.find(value, at + 1, end)
    return found


# WebSRC ---------------------------------------------------------------------------------------------

def websrc_page(html, qas):
    """``(context, questions, dropped)`` of one WebSRC page; ``dropped`` lists ``(id, reason)``.

    A span answer is given as the deepest element (``element_id``) holding it and a character offset
    into that element's whitespace-collapsed text. It is located in the element's range of the visible
    text: the answer is the k-th occurrence in the element text, so it is the k-th occurrence in the
    range when both hold the same number of occurrences. Anything else (an answer cut by a line or
    cell break, a count mismatch, an unparsable page) is dropped with its reason. Yes/no answers
    (``element_id == -1``) are not spans: ``yesno``.
    """
    page = VisibleText(html)
    try:
        nodes = NativeDOM(html).nodes
    except ValueError:
        nodes = None
    questions, dropped = [], []
    for qa in qas:
        target, answer, offset = str(qa['element_id']), qa['answer'], int(qa['answer_start'])
        if target == '-1':
            dropped.append((qa['id'], 'yesno'))
            continue
        if nodes is None or target not in nodes or target not in page.ranges:
            dropped.append((qa['id'], 'invalid_dom_target'))
            continue
        element = nodes[target]['text']
        if not answer.strip() or offset < 0 or element[offset:offset + len(answer)] != answer:
            dropped.append((qa['id'], 'span_mismatch'))
            continue
        k = len([s for s in occurrences(element, answer) if s < offset])
        start, end = page.ranges[target]
        found = occurrences(page.text, answer, start, end)
        if not found:
            dropped.append((qa['id'], 'answer_cut_by_break'))
            continue
        if len(found) != len(occurrences(element, answer)):
            dropped.append((qa['id'], 'ambiguous_occurrence'))
            continue
        position = found[k]
        assert page.text[position:position + len(answer)] == answer
        questions.append({'id': qa['id'], 'question': qa['question'],
                          'answers': {'text': [answer], 'answer_start': [position]}})
    return page.text, questions, dropped


WEBSRC_FEATURES = Features({
    'id': Value('string'), 'context': Value('string'),
    'questions': List({'id': Value('string'), 'question': Value('string'),
                       'answers': {'text': List(Value('string')), 'answer_start': List(Value('int32'))}}),
    'metadata': Value('string')})


def _websrc_pages(archive, root, split, website, domain, kind, answers=None):
    """Yield ``(page_id, html, qas)`` of one WebSRC website, in page order."""
    prefix = f'{root}/{domain}/{int(website):02d}'
    if answers is None:
        rows = csv.DictReader(io.StringIO(archive.read(prefix + '/dataset.csv').decode('utf-8-sig')))
        pages = {}
        for qa in rows:
            pages.setdefault(qa['id'][2:9], []).append(qa)
    else:
        pages = answers
    for page in sorted(pages):
        html = archive.read(f'{prefix}/processed_data/{page}.html').decode('utf-8-sig')
        yield page, prefix, html, pages[page]


def websrc_spans():
    """WebSRC v1.0 span questions over the visible text of each web page (CC BY 4.0).

    WebSRC (Chen et al., EMNLP 2021) has crowd-written questions over 6.4k web page segments; span
    answers are the annotated DOM element plus a character offset into its text. Each page's HTML is
    converted to visible text (``VisibleText``: decoded entities, collapsed whitespace, one line per
    block element, ``' | '`` between table cells) and every span answer is relocated in the answer
    element's range of that text, verified verbatim. Yes/no questions are not spans and are left out
    (the ``tasksource/websrc`` yes/no config holds them); unlocatable answers are dropped with a
    reason, never turned into nulls. One row per page groups its questions. Native website-disjoint
    splits: train and dev (``validation``) from the train+dev archive, ``test`` from the test archive
    with the answers released in ``WebSRC_v1.0_test_answers.json``.
    """
    archive_path = hf_hub_download(WEB_SRC, 'WebSRC_v1.0_train+dev.zip', repo_type='dataset',
        revision=WEB_SRC_REVISION, local_dir='build/browser-research/websrc-native')
    test_path = hf_hub_download(WEB_SRC, 'WebSRC_v1.0_test.zip', repo_type='dataset',
        revision=WEB_SRC_REVISION, local_dir='build/browser-research/websrc-test')
    answers_path = hf_hub_download(WEB_SRC, 'WebSRC_v1.0_test_answers.json', repo_type='dataset',
        revision=WEB_SRC_REVISION, local_dir='build/browser-research/websrc-test')
    splits, counts, excluded = {'train': [], 'validation': [], 'test': []}, Counter(), []

    def add(split, website, domain, kind, pages, archive, root):
        for page, prefix, html, qas in _websrc_pages(archive, root, split, website, domain, kind, pages):
            context, questions, dropped = websrc_page(html, qas)
            for identity, reason in dropped:
                excluded.append({'id': identity, 'split': split, 'reason': reason})
                counts[f'{split}/excluded/{reason}'] += 1
            counts[f'{split}/questions'] += len(questions)
            if not questions:
                counts[f'{split}/pages_without_span_questions'] += 1
                continue
            splits[split].append({'id': f'{prefix.split("/", 1)[1]}/{page}', 'context': context,
                'questions': questions, 'metadata': json.dumps({'domain': domain, 'website': int(website),
                'page': page, 'type': kind, 'source_dataset': WEB_SRC, 'source_revision': WEB_SRC_REVISION},
                sort_keys=True)})

    with zipfile.ZipFile(archive_path) as archive:
        sites = list(csv.DictReader(io.StringIO(archive.read('release/dataset_split.csv').decode())))
        for site in sites:
            split = {'train': 'train', 'dev': 'validation'}[site['split']]
            add(split, site['website'], site['domain'], site['type'], None, archive, 'release')
    with zipfile.ZipFile(test_path) as archive:
        released = json.loads(Path(answers_path).read_text())['data']
        for domain in released:
            for website in domain['websites']:
                first = website['qas'][0]['id']
                pages = {}
                for qa in website['qas']:
                    (answer,) = qa['answers']
                    pages.setdefault(qa['id'][2:9], []).append({'id': qa['id'], 'question': qa['question'],
                        'answer': answer['text'], 'element_id': answer['element_id'],
                        'answer_start': answer['answer_start']})
                add('test', first[2:4], domain['domain'], None, pages, archive, 'release_testset')
    data = DatasetDict({s: Dataset.from_list(rows, features=WEBSRC_FEATURES) for s, rows in splits.items()})
    for rows in data.values():
        for row in rows:
            for q in row['questions']:
                (text,), (start,) = q['answers']['text'], q['answers']['answer_start']
                assert row['context'][start:start + len(text)] == text
    report = {'source': f'https://huggingface.co/datasets/{WEB_SRC}', 'source_revision': WEB_SRC_REVISION,
              'source_files_sha256': {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                      for p in (archive_path, test_path, answers_path)},
              'license': 'cc-by-4.0', 'counts': dict(sorted(counts.items())),
              'visible_text_sha256': hashlib.sha256(inspect.getsource(VisibleText).encode()).hexdigest()}
    return _write_mirror('websrc-spans', data, excluded, report)


def _write_mirror(name, data, excluded, report):
    directory = Path('build') / (name + '-release')
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split, rows in data.items():
        rows.to_parquet(str(directory / f'{split}.parquet'))
        hashes[f'{split}.parquet'] = hashlib.sha256((directory / f'{split}.parquet').read_bytes()).hexdigest()
    report = dict(report, mirror=f'tasksource/{name}', parquet_sha256=hashes,
                  splits={split: {'rows': len(rows)} for split, rows in data.items()},
                  excluded_count=len(excluded),
                  conversion_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (directory / 'provenance.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    (directory / 'excluded-questions.jsonl').write_text(''.join(json.dumps(e, sort_keys=True) + '\n' for e in excluded))
    return data


# SWDE -----------------------------------------------------------------------------------------------

# The original CodePlex release (Hao et al., SIGIR 2011), as archived by the Wayback Machine.
SWDE_URL = ('http://web.archive.org/web/20210630013015id_/'
            'https://codeplexarchive.blob.core.windows.net/archive/projects/swde/swde.zip')
SWDE_SHA256 = '9ca9090b6fa45fb6c66e3f129481a40d7a99d7fddf0fea3d686e4fb41eb293da'
SWDE_VERTICALS = ('auto', 'book', 'camera', 'job', 'movie', 'nbaplayer', 'restaurant', 'university')
SWDE_WINDOW = 6000
SWDE_PAGES_PER_SITE = 250


def swde_split(vertical, sites):
    """Website-disjoint split of one vertical's ten sites: in sha256 order of ``swde:vertical/site``, the
    first site is ``test``, the second ``validation`` and the other eight ``train``."""
    ranked = sorted(sites, key=lambda s: hashlib.sha256(f'swde:{vertical}/{s}'.encode()).hexdigest())
    return {site: 'test' if i == 0 else 'validation' if i == 1 else 'train' for i, site in enumerate(ranked)}


def swde_value(raw, attribute):
    """A ground-truth value as it reads in the visible text: the labels keep raw character references
    (``St. Martin&#039;s Press``), which are decoded, and whitespace runs collapse to one space.

    Labels are whole DOM text nodes, so some sites' values carry the field caption or a separator
    (``Height: 6'10'``, ``Official telephone: (902) 585-2201``, ``$61,200 |``, ``> Online Store Manager``).
    A leading ``caption:`` whose caption names the attribute (a word of three or more letters of the
    attribute name, e.g. ``phone`` in ``telephone``) and surrounding ``|``, ``>`` or ``:`` are trimmed;
    the result is a substring of the labelled node, so it is still verbatim page text."""
    value = re.sub(r'\s+', ' ', unescape(raw)).strip()
    caption = re.match(r'([^:]{1,30}):\s*(?=\S)', value)
    words = [w for w in attribute.split('_') if len(w) >= 3]
    if caption and any(w in caption.group(1).lower() for w in words):
        value = value[caption.end():]
    return value.strip(' |>:')


def _bounded(text, start, value):
    """An occurrence that is not part of a longer word (``American`` inside ``Americana``)."""
    before = text[start - 1] if start else ' '
    after = text[start + len(value)] if start + len(value) < len(text) else ' '
    return not ((value[0].isalnum() and before.isalnum()) or (value[-1].isalnum() and after.isalnum()))


def _cover(spans):
    """Smallest ``(start, end)`` holding one span of every key; ``spans`` maps a key to its ``(start, end)`` list."""
    events = sorted((s, e, key) for key, found in spans.items() for s, e in found)
    need, best, inside, left = len(spans), None, Counter(), 0
    for right, (_, end, value) in enumerate(events):
        inside[value] += 1
        while len(inside) == need:
            start = events[left][0]
            stop = max(e for _, e, _ in events[left:right + 1])
            if best is None or stop - start < best[1] - best[0]:
                best = (start, stop)
            inside[events[left][2]] -= 1
            if not inside[events[left][2]]:
                del inside[events[left][2]]
            left += 1
    return best


def swde_record(text, labels, key, window=SWDE_WINDOW):
    """``(context, fields, reasons)`` of one page: ``labels`` maps each attribute to its distinct
    ground-truth values (``[]`` when the page has none). Every value of an attribute is located verbatim
    and word-bounded in the visible text, or the attribute is left out (``unlocated``): a page keeps
    only fully located attributes and labelled nulls. The context is the whole text, or a window of at
    most ``window`` characters on line boundaries holding an occurrence of every located value, placed
    deterministically from ``key``; pages whose values lie further apart are dropped (``spread``).
    Spans are every occurrence of a value inside the context."""
    reasons, found, nulls = Counter(), {}, []
    for attribute, values in labels.items():
        trimmed = sum(swde_value(v, attribute) != re.sub(r'\s+', ' ', unescape(v)).strip() for v in values)
        if trimmed:
            reasons['trimmed'] += trimmed
        values = [v for v in dict.fromkeys(swde_value(v, attribute) for v in values) if v]
        if not values:
            nulls.append(attribute)
            continue
        hits = {v: [s for s in occurrences(text, v) if _bounded(text, s, v)] for v in values}
        if all(hits.values()):
            found[attribute] = hits
        else:
            reasons['unlocated'] += 1
    if not found:
        return None, None, reasons + Counter(no_located_value=1)
    start, end = 0, len(text)
    if len(text) > window:
        a, b = _cover({(attribute, v): [(s, s + len(v)) for s in starts]
                       for attribute, hits in found.items() for v, starts in hits.items()})
        if b - a > window:
            return None, None, reasons + Counter(spread=1)
        slack = window - (b - a)
        shift = int.from_bytes(hashlib.sha256(f'swde-window:{key}'.encode()).digest()[:8], 'big') % (slack + 1)
        start = max(0, min(a - shift, len(text) - window))
        end = start + window
        # Snap inwards to line boundaries, never past the covered values.
        if start > 0:
            line = text.find('\n', start - 1, a)
            start = line + 1 if line >= 0 else start
        if end < len(text):
            line = text.rfind('\n', b, end + 1)
            end = line if line >= 0 else end
    context = text[start:end]
    fields = {}
    for attribute, hits in found.items():
        spans = []
        for value, starts in hits.items():
            spans.append([(s - start, value) for s in starts if start <= s and s + len(value) <= end])
        assert all(spans)
        fields[attribute] = spans
    for attribute in nulls:
        fields[attribute] = []
    return context, fields, reasons


class _Selected:
    """py7zr writer factory that keeps only the selected members, decoded, one at a time."""
    def __init__(self, wanted):
        from py7zr.io import NullIOFactory
        self.wanted, self.pages, self.null = wanted, {}, NullIOFactory()

    def create(self, filename):
        from py7zr.io import Py7zBytesIO
        if filename not in self.wanted:
            return self.null.create(filename)
        writer = Py7zBytesIO(filename, 1 << 30)
        self.pages[filename] = writer
        return writer


def _swde_ground_truth(archive):
    """``{vertical: {site: {page: {attribute: [values]}}}}`` from ``groundtruth.7z``."""
    import py7zr
    from py7zr.io import BytesIOFactory
    factory = BytesIOFactory(1 << 30)
    with py7zr.SevenZipFile(io.BytesIO(archive), 'r') as z:
        z.extractall(factory=factory)
    truth = {}
    for name, product in sorted(factory.products.items()):
        if not name.endswith('.txt'):
            continue
        product.seek(0)
        lines = product.read().decode('utf-8-sig').split('\n')
        vertical, site, attribute = lines[0].rstrip('\r').split('\t')
        for line in lines[2:]:
            line = line.rstrip('\r')
            if not line.strip():
                continue
            page, count, *values = line.split('\t')
            values = [] if values == ['<NULL>'] else values
            assert len(values) == int(count) or not values, (name, page)
            truth.setdefault(vertical, {}).setdefault(site, {}).setdefault(page, {})[attribute] = values
    return truth


def swde(pages_per_site=SWDE_PAGES_PER_SITE, window=SWDE_WINDOW):
    """SWDE (Hao et al., SIGIR 2011) attribute records over the visible text of real web pages.

    SWDE has 124,291 product, book, job, movie, player, restaurant, car and university pages from 80
    websites (8 verticals, 10 sites each), each page one entity, with page-level values of 3 to 5
    attributes per vertical labelled by per-site handcrafted regular expressions over DOM text nodes
    (procedural labels; ``<NULL>`` when the page has no value). Each page's HTML is converted to
    visible text (``VisibleText``) and every value is located verbatim, word-bounded, after decoding
    character references and collapsing whitespace; attributes with an unlocated value are left out,
    never turned into nulls. Pages longer than ``window`` characters are cut to one window holding
    every located value. One config per vertical; one row per page with ``fields`` =
    ``{attribute: [[(start, value), ...] per distinct value]}`` (JSON, empty for a labelled null).
    SWDE has no official split: each vertical's ten sites are split by hash (8 train, 1 validation,
    1 test; ``swde_split``), so evaluation websites are unseen. Pages are template-generated, so each
    site contributes its ``pages_per_site`` lowest-hash eligible pages.
    """
    import urllib.request
    import py7zr
    directory = Path('build/swde-native')
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'swde.zip'
    if not path.exists():
        urllib.request.urlretrieve(SWDE_URL, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SWDE_SHA256:
        raise ValueError(f'SWDE archive hash {digest} differs from the pinned {SWDE_SHA256}')
    with zipfile.ZipFile(path) as outer:
        inner = zipfile.ZipFile(io.BytesIO(outer.read('sourceCode\\sourceCode.zip')))
    members = {name: hashlib.sha256(inner.read(name)).hexdigest()
               for name in [f'{v}.7z' for v in SWDE_VERTICALS] + ['groundtruth.7z', 'readme.txt', 'website_list.txt']}
    truth = _swde_ground_truth(inner.read('groundtruth.7z'))
    features = Features({'id': Value('string'), 'context': Value('string'), 'fields': Value('string'),
                         'metadata': Value('string')})
    result, counts, excluded, splits_by_site, unlabelled_by_vertical = {}, Counter(), [], {}, {}
    for vertical in SWDE_VERTICALS:
        sites = truth[vertical]
        assignment = swde_split(vertical, sorted(sites))
        attributes = sorted({a for pages in sites.values() for page in pages.values() for a in page})
        unlabelled = {site: {a for a in attributes
                             if not any(page.get(a) for page in pages.values())} for site, pages in sites.items()}
        for site, missing in unlabelled.items():
            counts[f'{vertical}/{assignment[site]}/site_attributes_unlabelled'] += len(missing)
        unlabelled_by_vertical[vertical] = {s: sorted(m) for s, m in unlabelled.items() if m}

        splits_by_site[vertical] = assignment
        with py7zr.SevenZipFile(io.BytesIO(inner.read(f'{vertical}.7z')), 'r') as z:
            names = [n for n in z.getnames() if n.endswith('.htm')]
        folders = {}
        for name in names:
            folder, page = name.split('/')[-2], name.split('/')[-1][:-4]
            site = folder[len(vertical) + 1:folder.index('(')]
            folders.setdefault(site, {})[page] = name
        assert set(folders) == set(sites), (vertical, set(folders) ^ set(sites))
        # Rank each site's pages; convert a margin beyond the cap for pages that turn out ineligible.
        ranked = {site: sorted(pages, key=lambda p: hashlib.sha256(f'swde:{vertical}/{site}/{p}'.encode()).hexdigest())
                  for site, pages in folders.items()}
        wanted = {folders[s][p] for s, pages in ranked.items() for p in pages[:pages_per_site * 2]}
        factory = _Selected(wanted)
        with py7zr.SevenZipFile(io.BytesIO(inner.read(f'{vertical}.7z')), 'r') as z:
            z.extractall(factory=factory)
        rows = {'train': [], 'validation': [], 'test': []}
        for site in sorted(sites):
            split, kept = assignment[site], 0
            for page in ranked[site][:pages_per_site * 2]:
                if kept == pages_per_site:
                    break
                writer = factory.pages[folders[site][page]]
                writer.seek(0)
                html = writer.read().decode('utf-8-sig', errors='replace')
                url = re.match(r'\s*<base href="([^"]*)"', html)
                text = visible_text(html)
                labels = truth[vertical][site].get(page)
                if labels is None:
                    counts[f'{vertical}/{split}/excluded/no_ground_truth'] += 1
                    continue
                # An attribute without a value on any page of the site was not labelled there.
                labels = {a: v for a, v in labels.items() if a not in unlabelled[site]}
                key = f'{vertical}/{site}/{page}'
                context, fields, reasons = swde_record(text, labels, key, window)
                for reason, n in reasons.items():
                    counts[f'{vertical}/{split}/{"pages" if reason in ("spread", "no_located_value") else "values" if reason == "trimmed" else "attributes"}_{reason}'] += n
                if context is None:
                    excluded.append({'id': key, 'split': split, 'reason': sorted(set(reasons) - {'unlocated'})[0]})
                    continue
                kept += 1
                rows[split].append({'id': key, 'context': context, 'fields': json.dumps(fields, ensure_ascii=False),
                    'metadata': json.dumps({'vertical': vertical, 'site': site, 'page': page,
                        'url': url.group(1) if url else None, 'page_text_length': len(text)}, sort_keys=True)})
            counts[f'{vertical}/{split}/pages'] += kept
        result[vertical] = DatasetDict({s: Dataset.from_list(r, features=features) for s, r in rows.items()})
        for split, data in result[vertical].items():
            for row in data:
                for spans in json.loads(row['fields']).values():
                    for occurrences_ in spans:
                        for start, value in occurrences_:
                            assert row['context'][start:start + len(value)] == value
    report = {'source': SWDE_URL, 'source_sha256': SWDE_SHA256, 'source_members_sha256': members,
              'license': 'ms-rl (CodePlex project license)', 'window': window, 'pages_per_site': pages_per_site,
              'site_splits': splits_by_site, 'unlabelled_site_attributes': unlabelled_by_vertical, 'counts': dict(sorted(counts.items())),
              'visible_text_sha256': hashlib.sha256(inspect.getsource(VisibleText).encode()).hexdigest()}
    directory = Path('build/swde-release')
    hashes = {}
    for vertical, data in result.items():
        (directory / vertical).mkdir(parents=True, exist_ok=True)
        for split, rows in data.items():
            name = f'{vertical}/{split}.parquet'
            rows.to_parquet(str(directory / name))
            hashes[name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    report.update(mirror='tasksource/swde', parquet_sha256=hashes, excluded_count=len(excluded),
                  configs={v: {s: len(r) for s, r in d.items()} for v, d in result.items()},
                  conversion_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (directory / 'provenance.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    (directory / 'excluded-questions.jsonl').write_text(''.join(json.dumps(e, sort_keys=True) + '\n' for e in excluded))
    return result
