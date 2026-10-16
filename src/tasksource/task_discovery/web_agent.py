"""Web search agent as a candidate source: a tool-calling chat model with search tools.

    PYTHONPATH=src python -m tasksource.task_discovery.web_agent --output build/discovery/web-gui.jsonl \
        --brief "English GUI grounding or computer-use action datasets with a training split"

Tag listings miss untagged and newly released datasets; an agent searching the web, papers and the
Hub by meaning finds them. Each brief runs one tool-calling session (Exa, Parallel and Firecrawl
search, page reading, Hub search). Submitted Hub repositories become ordinary candidate rows for
``prospect.py --candidates``; other links (GitHub, papers) are kept as leads, since they need a
repackage. The agent only searches and reads: it has no shell or file access.

The model is any OpenAI-compatible endpoint (``tasksource.llm``: ``OPENAI_BASE_URL``, ``OPENAI_API_KEY``,
``OPENAI_MODEL`` or ``--model``). Search keys: ``EXA_API_KEY``, ``PARALLEL_API_KEY``, ``FIRECRAWL_API_KEY``,
from the environment or ``~/.config/tasksource/search_keys.env``. Nothing is sent but the queries and
URLs the agent chooses.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import re
import threading
import time
import urllib.error
import urllib.request

from huggingface_hub import HfApi

from .. import llm
from .discover_task_prospects import write_rows
from .hf_hub import hub_record, hub_row

KEY_FILE = Path.home() / '.config/tasksource/search_keys.env'
REPO = re.compile(r'huggingface\.co/datasets/([\w.-]+/[\w.-]+)')

SYSTEM = '''You find datasets that could become supervised training tasks for Tasksource, a collection of \
classification, multiple-choice and grounding decision tasks. A useful dataset has a genuine training split, \
human, procedural or otherwise verified labels (not unverified model outputs), and a decision target: a class, \
a choice among options, or for interfaces a coordinate/element/action. Search broadly (papers, benchmark \
pages, GitHub, the Hugging Face Hub), read pages to confirm, and prefer the original upload over mirrors. \
Use several different queries and engines. When done, call submit once with every dataset you found worth \
checking; give the Hugging Face dataset repository when one exists, otherwise the best source URL.'''

TOOLS = [
    {'type': 'function', 'function': {'name': 'web_search', 'description': 'Search the web. Engines differ; vary them.',
        'parameters': {'type': 'object', 'required': ['query'], 'properties': {
            'query': {'type': 'string'}, 'engine': {'type': 'string', 'enum': ['exa', 'parallel', 'firecrawl']}}}}},
    {'type': 'function', 'function': {'name': 'read_page', 'description': 'Read a web page as markdown (truncated).',
        'parameters': {'type': 'object', 'required': ['url'], 'properties': {'url': {'type': 'string'}}}}},
    {'type': 'function', 'function': {'name': 'hub_search', 'description': 'Search Hugging Face dataset ids by substring.',
        'parameters': {'type': 'object', 'required': ['query'], 'properties': {'query': {'type': 'string'}}}}},
    {'type': 'function', 'function': {'name': 'check_known', 'description': 'Which datasets (names, Hub ids or URLs) '
        'Tasksource already has or this search already found. Check before reading further.',
        'parameters': {'type': 'object', 'required': ['datasets'], 'properties': {
            'datasets': {'type': 'array', 'items': {'type': 'string'}}}}}},
    {'type': 'function', 'function': {'name': 'submit', 'description': 'Submit the datasets found; ends the session.',
        'parameters': {'type': 'object', 'required': ['datasets'], 'properties': {'datasets': {'type': 'array', 'items': {
            'type': 'object', 'required': ['url', 'why'], 'properties': {
                'url': {'type': 'string', 'description': 'huggingface.co/datasets/... or the best source URL'},
                'why': {'type': 'string', 'description': 'task, label source, training split'}}}}}}}},
]


def load_keys():
    if KEY_FILE.exists():
        for line in KEY_FILE.read_text().splitlines():
            match = re.match(r'^(?:export\s+)?([A-Z_0-9]+)=["\']?([^"\'\s]+)', line)
            if match and match.group(1) in ('EXA_API_KEY', 'PARALLEL_API_KEY', 'FIRECRAWL_API_KEY'):
                os.environ.setdefault(*match.groups())


def post(url, body, headers, timeout=120):
    request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json', **headers})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


class Chat:
    """Calls are spaced to ``rpm`` across all workers (endpoints rate-limit per key); retry with backoff."""
    lock, last = threading.Lock(), 0.0

    def __init__(self, model, rpm=40):
        self.model, self.interval = model, 60 / rpm

    def chat(self, messages, tool_choice='auto'):
        for attempt in range(5):
            with Chat.lock:
                time.sleep(max(0, self.interval - (time.monotonic() - Chat.last)))
                Chat.last = time.monotonic()
            try:
                # tool_choice is explicit: some models answer in text and never call a tool without it.
                return post(llm.base_url() + '/chat/completions', dict(model=self.model, messages=messages, tools=TOOLS,
                            tool_choice=tool_choice, temperature=0.3), {'Authorization': 'Bearer ' + os.environ[llm.KEY_ENV]},
                            timeout=300)['choices'][0]['message']
            except (urllib.error.URLError, TimeoutError, KeyError) as error:
                if attempt == 4:
                    raise
                time.sleep(10 * 2 ** attempt)


ENGINES = ('exa', 'parallel', 'firecrawl')


def web_search(query, engine=None):
    engine = engine or random.choice(ENGINES)  # engines index different corners of the web
    if engine == 'parallel':
        found = post('https://api.parallel.ai/v1beta/search', {'objective': query, 'search_queries': [query], 'max_results': 8},
                     {'x-api-key': os.environ['PARALLEL_API_KEY']})['results']
        return [dict(url=r.get('url'), title=r.get('title'), text=' '.join(r.get('excerpts') or [])[:500]) for r in found]
    if engine == 'firecrawl':
        found = post('https://api.firecrawl.dev/v1/search', {'query': query, 'limit': 8},
                     {'Authorization': 'Bearer ' + os.environ['FIRECRAWL_API_KEY']})['data']
        return [dict(url=r.get('url'), title=r.get('title'), text=(r.get('description') or '')[:500]) for r in found]
    found = post('https://api.exa.ai/search', {'query': query, 'numResults': 8, 'type': 'auto',
                 'contents': {'text': {'maxCharacters': 500}}}, {'x-api-key': os.environ['EXA_API_KEY']})['results']
    return [dict(url=r.get('url'), title=r.get('title'), text=(r.get('text') or '')[:500]) for r in found]


def read_page(url):
    data = post('https://api.firecrawl.dev/v1/scrape', {'url': url, 'formats': ['markdown'], 'onlyMainContent': True},
                {'Authorization': 'Bearer ' + os.environ['FIRECRAWL_API_KEY']})['data']
    return (data.get('markdown') or '')[:8000]


def hub_search(query):
    return [dict(id=d.id, downloads=d.downloads, likes=d.likes)
            for d in HfApi().list_datasets(search=query, sort='downloads', limit=20)]


class Known:
    """The pile to exclude: the catalog, recorded decisions and earlier finds; grows each round."""

    def __init__(self, paths=()):
        from .identifiers import normalized
        from .match import catalog_index, previous_decisions
        self.normalized = normalized
        names, repos = catalog_index()
        self.entries = {('repo', r): 'in tasksource' for r in repos}
        self.entries.update({('name', n): 'in tasksource' for n in names})
        for record in previous_decisions():
            for value in [record['name'], *record.get('aliases', []), *record.get('huggingface', [])]:
                self.entries.setdefault(('name', normalized(value.rsplit('/', 1)[-1])), 'reviewed and declined or parked')
        for path in paths:
            for line in Path(path).read_text().splitlines():
                self.add(json.loads(line)['name'], 'found earlier')

    def keys(self, value):
        match = REPO.search(value)
        repo = (match.group(1) if match else value.strip().strip('/')).casefold()
        return [('repo', repo), ('name', self.normalized(repo.rsplit('/', 1)[-1]))]

    def add(self, value, status):
        for key in self.keys(value):
            self.entries.setdefault(key, status)

    def check(self, datasets):
        return {value: next((self.entries[k] for k in self.keys(value) if k in self.entries), 'new') for value in datasets}


def session(brief, known, chat, max_steps=40, log=None):
    """One agent session; returns the submitted datasets and the transcript."""
    messages = [dict(role='system', content=SYSTEM), dict(role='user', content=brief)]
    submitted, read = None, []
    for step in range(max_steps):
        last = step == max_steps - 1 and submitted is None
        message = chat.chat(messages, {'type': 'function', 'function': {'name': 'submit'}} if last else 'auto')
        messages.append({k: v for k, v in message.items() if k in ('role', 'content', 'tool_calls')})
        calls = message.get('tool_calls') or []
        if not calls:
            messages.append(dict(role='user', content='Continue searching, then call submit with the datasets found.'))
            continue
        for call in calls:
            name = call['function']['name']
            try:
                arguments = json.loads(call['function']['arguments'] or '{}')
                if name == 'submit':
                    submitted = arguments.get('datasets', [])
                    result = 'submitted'
                else:
                    if name == 'read_page':
                        read.append(arguments.get('url', ''))
                    result = dict(web_search=web_search, read_page=read_page, hub_search=hub_search,
                                  check_known=known.check)[name](**arguments)
            except Exception as error:  # tool errors go back to the agent, which can retry differently
                result = f'error: {type(error).__name__}: {str(error)[:200]}'
            if log:
                log.write(json.dumps(dict(brief=brief[:80], step=step, tool=name, arguments=call['function']['arguments'][:300]),
                                     ensure_ascii=False) + '\n'); log.flush()
            messages.append(dict(role='tool', tool_call_id=call['id'], content=json.dumps(result, ensure_ascii=False)[:12000]))
        if submitted is not None:
            break
        if step == max_steps - 2:
            messages.append(dict(role='user', content='Last step: call submit now with every dataset found.'))
    # Hub pages the agent chose to read are candidates too: submission is a summary, not the only signal.
    known = {REPO.search(d.get('url', '')).group(1) for d in submitted or [] if REPO.search(d.get('url', ''))}
    extra = [dict(url=url, why='read by the agent') for url in dict.fromkeys(read)
             if REPO.search(url) and REPO.search(url).group(1) not in known]
    return (submitted or []) + extra, messages


def candidates(found, brief):
    """Hub repositories as canonical rows; other URLs as leads."""
    api, rows, leads = HfApi(), {}, []
    for item in found:
        match = REPO.search(item.get('url', ''))
        if not match:
            leads.append(dict(item, brief=brief)); continue
        repo = match.group(1).removesuffix('.git')
        try:
            row = hub_row(hub_record(api.dataset_info(repo)))
        except Exception as error:
            leads.append(dict(item, brief=brief, error=str(error)[:200])); continue
        row['sources'] = [dict(source='web_agent', url=item['url'], why=item.get('why', ''), brief=brief)]
        rows[row['name']] = row
    return list(rows.values()), leads


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True, help='Candidate JSONL; leads and transcripts go beside it')
    parser.add_argument('--brief', action='append', required=True, help='What to look for; one session per brief per round')
    parser.add_argument('--rounds', type=int, default=1, help='Rounds; each excludes everything found before')
    parser.add_argument('--known', type=Path, action='append', default=[], help='Earlier candidate JSONL to exclude too')
    parser.add_argument('--max-steps', type=int, default=40)
    parser.add_argument('--model', help='Chat model with tool calling (default: OPENAI_MODEL)')
    parser.add_argument('--rpm', type=float, default=40, help='Request starts per minute, all sessions together')
    args = parser.parse_args()
    load_keys()
    chat = Chat(llm.model(args.model), args.rpm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stem = str(args.output.with_suffix(''))
    known, rows, leads, transcripts = Known(args.known), {}, [], []
    for round_ in range(args.rounds):
        found_names = sorted(rows)[-300:]
        prompt = lambda brief: brief + ('' if not found_names else
            '\n\nAlready found by earlier rounds (do not resubmit; look elsewhere, with different queries and angles): '
            + ', '.join(found_names))
        with open(stem + '.tools.jsonl', 'a') as log, ThreadPoolExecutor(max_workers=4) as pool:  # ≤4: shared rate limit
            results = list(pool.map(lambda brief: (brief, *session(prompt(brief), known, chat, args.max_steps, log)), args.brief))
        new = 0
        for brief, found, messages in results:
            transcripts.append(dict(round=round_, brief=brief, messages=messages))
            found_rows, found_leads = candidates(found, brief)
            for row in found_rows:
                row['sources'][0]['round'] = round_
                if row['name'] in rows:
                    rows[row['name']]['sources'] += row['sources']
                else:
                    rows[row['name']] = row; new += 1
                known.add(row['name'], 'found earlier')
            leads += [dict(lead, round=round_) for lead in found_leads]
        print(f'round {round_}: {new} new Hub candidates, {len(rows)} total', flush=True)
        write_rows(args.output, list(rows.values()))
    write_rows(Path(stem + '.leads.jsonl'), leads)
    write_rows(Path(stem + '.transcripts.jsonl'), transcripts)
    Path(stem + '.manifest.json').write_text(json.dumps(dict(
        model=chat.model, briefs=args.brief, rounds=args.rounds, known=[str(p) for p in args.known],
        created=datetime.now(timezone.utc).isoformat(), candidates=len(rows), leads=len(leads)), indent=2) + '\n')
    print(f'{len(rows)} Hub candidates, {len(leads)} other leads -> {args.output}')


if __name__ == '__main__':
    main()
