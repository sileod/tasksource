"""Workflow-based synthetic Jev data: fixed decision schemas, many states, intended readings.

    python -m tasksource.jev.synthetic.workflows --out DIR --workflows 8 --states 6

Synthetic data covers judgment that cannot be formalized (meaning, intent, tone, plausibility, fuzzy
categorization); rules, thresholds, deadlines and counting belong to the exact procedural generators.

1. design: per domain, an LLM writes a reusable application workflow of typed judgment questions;
   slots fix formats and option ranges (some workflows get a 12-40 option choice), and the designer
   picks the skills a real application would use from twice as many candidates (SKILL_DEFINITIONS
   minus RULE_SKILLS).
2. state: for each workflow, many items are written; one or two focus questions per item get a
   sampled intended reading (a clear option, a borderline one, or a yes/no probability), so answers
   are balanced and uncertainty is deliberate; the other questions are answered by whatever the item says.
3. check: the checker answers blind with distributions and a well-posedness flag; ill-posed
   questions are dropped. Whether a focus answer fits its intended reading is kept as metadata: a
   miss is still a valid question, it only measures how well the writer steers.
4. label: Jev annotates the kept questions; questions where Jev and the checker confidently
   disagree are dropped. Target, checker and Jev distributions are all stored.

Every LLM call is cached by content hash under DIR/cache, so reruns resume.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
from pathlib import Path

from . import providers
from .annotate import annotate_bundle_jev
from .config import AnnotatorConfig, ProviderConfig
from .generate import PROMPTS_DIR, extract_json_object
from .specs import DOMAINS, SKILL_DEFINITIONS

SOURCES = ["customer emails", "live chat transcripts", "support tickets", "product or service reviews",
           "social media posts and replies", "forum or community threads", "internal team chat messages",
           "meeting notes", "call transcripts or call summaries", "field or site reports",
           "incident reports", "free-text survey or feedback responses", "complaints and appeals",
           "requests submitted through a web form", "interview or assessment notes", "messages between partners or vendors",
           "news items and press releases", "case notes written by staff", "proposals and requests for approval",
           "handover or shift notes"]
LENGTHS = ["40-80 words", "80-150 words", "150-300 words", "300-500 words"]
DIFFICULTY = ["Make the relevant cues easy to find.",
              "Spread the relevant cues across the item, among irrelevant details.",
              "Make readers combine several cues, including at least one plausible distractor."]
CHOICE_RANGE, MANY_RANGE, P_MANY = (2, 8), (12, 40), 0.3
SCORE_LEVELS = [3, 4, 5, 5, 7]
RULE_SKILLS = {"sla_breach", "refund_approval", "access_justification", "completeness"}  # procedural ground
UNNATURAL_SKILLS = {"argument_role"}  # yields vague questions inside decision applications
SKILLS = sorted(set(SKILL_DEFINITIONS) - RULE_SKILLS - UNNATURAL_SKILLS)
NOUL_P = [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
NAMES = """Aiko Amara Andrés Anika Arjun Astrid Ayşe Bao Beatriz Bilal Bongani Carmen Chen Chidi Dalia Dmitri
Elif Emeka Esperanza Fatima Femi Freya Giorgos Hamid Hana Ibrahim Ingrid Isabela Jamal Jia Joaquín Kaito Kalani
Kemal Kenji Kwame Lars Layla Leilani Liam Lucía Mahnoor Malik Mariana Mateus Mei Mehmet Mira Nadia Naveen Nia Nikolai
Noa Olumide Omar Oskar Pablo Parisa Pedro Priyanka Quang Rafael Rania Ravi Rosa Ruth Saanvi Salma Sanjay Santiago
Sekou Shira Siddharth Sione Soo-ah Sven Tamar Tariq Thandiwe Tomás Uchenna Valentina Vikram Wanjiru Wei Xavier Yara
Yusuf Zainab Zeynep Zofia Ahmed Brigid Colm Daniela Eitan Farah Gustavo Halima Ines Jonas Keanu Lindiwe Marek
Nkechi Orla Paulo Rashida Stefan Thuy Ulla Viktor Yasmin""".split()
SURNAMES = """Abara Achterberg Adeyemi Alvarez Andersson Banerjee Bauer Bianchi Bose Castillo Chua Costa Dahl Dlamini
Dubois Eze Fernandes Fischer Galloway Guerrero Haddad Hashemi Horvat Ibarra Iwasaki Jankowski Jensen Kamau Kaplan
Khan Kowalski Kuznetsov Lambert Lee Lindqvist Lopes Mbeki Mendoza Moreau Murphy Nakamura Ndiaye Nguyen Novak Obi
Okafor Oliveira Ortiz Park Patel Petrov Quispe Rahman Reyes Rossi Saito Santos Schmidt Shapiro Silva Singh Sokolov
Suzuki Takahashi Tanaka Thompson Tran Vargas Varga Wang Weber Williams Yilmaz Yoon Zhang Zulu""".split()


def prompt(name: str, **fields) -> str:
    text = (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")
    for key, value in fields.items():
        text = text.replace("{{%s}}" % key.upper(), str(value))
    return text


def stable_rng(*parts) -> random.Random:
    return random.Random(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest())


class LLM:
    """Cached, key-rotating chat calls returning parsed JSON."""

    def __init__(self, provider: ProviderConfig, cache: Path, concurrency: int):
        self.provider, self.cache = provider, cache
        self.clients = [providers.make_client(provider, key) for key in providers.available_api_keys(provider)]
        self.sem, self.calls = asyncio.Semaphore(concurrency), 0
        cache.mkdir(parents=True, exist_ok=True)

    async def json(self, text: str, temperature: float, max_tokens: int = 6000) -> dict:
        key = hashlib.sha256(f"{self.provider.model}\n{temperature}\n{text}".encode()).hexdigest()
        path = self.cache / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        async with self.sem:
            self.calls += 1
            client = self.clients[self.calls % len(self.clients)]
            result = await providers.chat_complete(client, self.provider.model, [{"role": "user", "content": text}],
                                                   temperature=temperature, max_tokens=max_tokens)
        parsed = extract_json_object(result["text"])
        path.write_text(json.dumps(parsed, ensure_ascii=False), encoding="utf-8")
        return parsed


def sample_slots(rng: random.Random, n: int, many: bool) -> list[dict]:
    slots = []
    for _ in range(n):
        kind = rng.choices(["choice", "noul", "score"], [0.5, 0.3, 0.2])[0]
        lo = rng.choice(SCORE_LEVELS)
        slots.append({"type": kind, "range": (lo, lo) if kind == "score" else CHOICE_RANGE})
    if many:
        slots[rng.randrange(n)] = {"type": "choice", "range": MANY_RANGE}
    return slots


def describe_slot(slot: dict) -> str:
    lo, hi = slot["range"]
    count = {"choice": f" with {lo} to {hi} options", "score": f" with {lo} levels"}.get(slot["type"], "")
    return f"- {slot['type']}{count}"


def valid_workflow(workflow: dict, slots: list[dict], skills: list[str]) -> bool:
    questions = workflow.get("questions")
    if not isinstance(questions, list) or len(questions) != len(slots) \
            or len({q.get("skill") for q in questions}) != len(questions):
        return False
    for question, slot in zip(questions, slots):
        options = question.get("options") or []
        lo, hi = slot["range"]
        if question.get("type") != slot["type"] or not str(question.get("question", "")).strip() \
                or question.get("skill") not in skills \
                or len(set(options)) != len(options) or slot["type"] != "noul" and not lo <= len(options) <= hi:
            return False
    return True


def sample_target(rng: random.Random, question: dict) -> dict:
    """An intended reading: a yes probability, or an option that is clear or borderline."""
    if question["type"] == "noul":
        p = rng.choice(NOUL_P)
        return {"reading": f"yes with probability about {round(p * 100)}%", "p_yes": p}
    index = rng.randrange(len(question["options"]))
    kind = "borderline" if rng.random() < 0.25 else "clear"
    label = "borderline, mostly" if kind == "borderline" else "clear:"
    return {"reading": f"{label} {question['options'][index]!r}", "kind": kind, "index": index,
            "ordinal": question["type"] == "score"}


def fits(target: dict, probs: list[float]) -> bool:
    """Whether a blind answer agrees with the intended reading."""
    if "p_yes" in target:
        return abs(probs[0] - target["p_yes"]) <= 0.3
    total = sum(probs) or 1.0
    p = probs[target["index"]] / total
    top = max(range(len(probs)), key=probs.__getitem__)
    if target["ordinal"]:  # neighbouring levels are inherently fuzzy
        return abs(top - target["index"]) <= (target["kind"] == "clear") and p >= 0.25
    return top == target["index"] and (p >= 0.5 if target["kind"] == "clear" else p <= 0.85)


def entities(rng: random.Random) -> str:
    people = [f"{rng.choice(NAMES)} {rng.choice(SURNAMES)}" for _ in range(3)]
    ref = f"{rng.choice('ABCDEFGHJKLMNPRSTVWXZ')}{rng.choice('ABCDEFGHJKLMNPRSTVWXZ')}-{rng.randint(1000, 999999)}"
    return f"If the item needs people or reference numbers, use names like {', '.join(people)} and references like {ref}."


async def design(llm: LLM, index: int, seed: int) -> dict | None:
    rng = stable_rng(seed, "workflow", index)
    domain = rng.choice(DOMAINS)
    many, source = rng.random() < P_MANY, rng.choice(SOURCES)
    for attempt in range(3):
        slots = sample_slots(rng, rng.randint(3, 8), many)
        candidates = sorted(rng.sample(SKILLS, min(len(SKILLS), 2 * len(slots))))
        skills = "\n".join(f"- {s}: {SKILL_DEFINITIONS[s]}" for s in candidates)
        text = prompt("workflow_design", domain=domain.replace("_", " "), source=source, n_questions=len(slots),
                      slots="\n".join(map(describe_slot, slots)), skills=skills)
        try:
            workflow = await llm.json(text, temperature=0.9)
        except ValueError:
            continue
        if valid_workflow(workflow, slots, candidates):
            for i, question in enumerate(workflow["questions"]):
                question.update(id=f"q{i}", options=question.get("options") or [])
            return {"workflow_id": f"wf{index:05d}", "domain": domain, "source": source, **workflow}
    return None


async def write_state(llm: LLM, workflow: dict, index: int, seed: int) -> dict | None:
    rng = stable_rng(seed, workflow["workflow_id"], index)
    questions = rng.sample(workflow["questions"], rng.randint(1, len(workflow["questions"])))
    questions.sort(key=lambda q: q["id"])
    focus = rng.sample(questions, min(len(questions), rng.choice([1, 1, 2])))
    targets = {q["id"]: sample_target(rng, q) for q in focus}
    listing = "\n".join(f"- {q['question']}" + (f" Options: {q['options']}" if q["options"] else "")
                        + f"\n  Intended reading: {targets[q['id']]['reading'] if q['id'] in targets else 'whatever fits the item'}"
                        for q in questions)
    text = prompt("workflow_state", application=workflow["application"], item_kind=workflow["item_kind"],
                  length=rng.choice(LENGTHS), entities=entities(rng),
                  targets=listing, difficulty=rng.choice(DIFFICULTY))
    try:
        state = (await llm.json(text, temperature=0.9)).get("state")
    except ValueError:
        return None
    if not isinstance(state, str) or len(state) < 40:
        return None
    return {"state_id": f"{workflow['workflow_id']}_s{index:03d}", "workflow_id": workflow["workflow_id"],
            "domain": workflow["domain"], "state": state,
            "questions": [{**q, "target": targets.get(q["id"])} for q in questions]}


async def check(llm: LLM, item: dict) -> dict:
    listing = [{"id": q["id"], "type": q["type"], "question": q["question"], "options": q["options"]}
               for q in item["questions"]]
    text = prompt("workflow_check", state=item["state"], questions=json.dumps(listing, ensure_ascii=False, indent=1))
    try:
        answers = {a.get("id"): a for a in (await llm.json(text, temperature=0.0)).get("answers", [])}
    except ValueError:
        answers = {}
    for q in item["questions"]:
        answer = answers.get(q["id"], {})
        probs = answer.get("probabilities")
        width = 1 if q["type"] == "noul" else len(q["options"])
        ok = isinstance(probs, list) and len(probs) == width and all(isinstance(p, (int, float)) for p in probs)
        q["check"] = {"probabilities": probs if ok else None, "well_posed": answer.get("well_posed") is True,
                      "issue": answer.get("issue", "")}
        q["check"]["fits_target"] = None if q["target"] is None or not ok else fits(q["target"], probs)
        q["kept"] = ok and q["check"]["well_posed"]
    return item


def label(item: dict, annotator: AnnotatorConfig, api_key: str, cache: Path) -> dict:
    kept = [q for q in item["questions"] if q["kept"]]
    if not kept:
        return item
    bundle = {"state_id": item["state_id"], "state": item["state"],
              "questions": [{"question_id": q["id"], "format": q["type"], "question": q["question"],
                             "options": q["options"]} for q in kept]}
    labelled = annotate_bundle_jev(bundle, annotator, api_key, cache)
    for q, annotation in zip(kept, labelled["annotations"]):
        q["jev"] = annotation["probabilities"]
        q["kept"] = not confident_disagreement(q["type"], q["check"]["probabilities"], q["jev"])
    return item


def confident_disagreement(kind: str, a: list[float], b: list[float], confidence: float = 0.8) -> bool:
    if kind == "noul":
        a, b = [a[0], 1 - a[0]], [b[0], 1 - b[0]]
    top_a, top_b = (max(range(len(p)), key=p.__getitem__) for p in (a, b))
    far = abs(top_a - top_b) > 1 if kind == "score" else top_a != top_b
    return far and a[top_a] / (sum(a) or 1) >= confidence and b[top_b] >= confidence


async def run(args) -> None:
    out = Path(args.out)
    provider = ProviderConfig(name="albert", api_key_env="ALBERT_API_KEY", optional_api_key_envs=["ALBERT_API_KEY_2"],
                              base_url="https://albert.api.etalab.gouv.fr/v1", model=args.model)
    llm = LLM(provider, out / "cache" / "llm", args.concurrency)
    annotator = AnnotatorConfig(name="jev", version=args.jev, base_url="https://openrouter.ai",
                                api_key_env=args.jev_key_env, model=args.jev)
    api_key = os.environ[args.jev_key_env] if args.jev else ""
    workflows = [w for w in await asyncio.gather(*[design(llm, i, args.seed) for i in range(args.workflows)]) if w]
    (out / "workflows.jsonl").write_text("".join(json.dumps(w, ensure_ascii=False) + "\n" for w in workflows))
    print(f"workflows: {len(workflows)}/{args.workflows}", flush=True)

    in_flight = asyncio.Semaphore(2 * args.concurrency)  # finish states steadily instead of writing all first

    async def one(workflow, index):
        async with in_flight:
            return await one_state(workflow, index)

    async def one_state(workflow, index):
        try:
            item = await write_state(llm, workflow, index, args.seed)
            if item is None:
                return None
            item = await check(llm, item)
            if args.jev:
                item = await asyncio.to_thread(label, item, annotator, api_key, out / "cache" / "jev")
            return item
        except Exception as exc:  # one failed state must not stop a long run; reruns retry it
            print(f"{workflow['workflow_id']}_s{index:03d} failed: {str(exc)[:200]}", flush=True)
            return None

    path = out / "items.jsonl"
    done = {json.loads(line)["state_id"] for line in path.open()} if path.exists() else set()
    jobs = [one(w, s) for w in workflows for s in range(args.states)
            if f"{w['workflow_id']}_s{s:03d}" not in done]
    kept = total = 0
    with path.open("a") as handle:
        for n, job in enumerate(asyncio.as_completed(jobs), 1):
            item = await job
            if item is not None:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                handle.flush()
                total += len(item["questions"])
                kept += sum(q["kept"] for q in item["questions"])
            if n % 50 == 0 or n == len(jobs):
                print(f"states {n + len(done)}/{len(done) + len(jobs)}, questions kept {kept}/{total} this session", flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--workflows", type=int, default=8)
    parser.add_argument("--states", type=int, default=6, help="states per workflow")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="deepseek-v4-flash-0731")
    parser.add_argument("--jev", default="typesafe/jev-1.13", help="empty to skip Jev labels")
    parser.add_argument("--jev-key-env", default="JEV_OPENROUTER_API_KEY")
    parser.add_argument("--concurrency", type=int, default=16)
    asyncio.run(run(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
