"""Generation run: design schemas, write documents, verify fields (see the package docstring)."""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import unicodedata
from pathlib import Path

from ..synthetic.config import ProviderConfig
from ..synthetic.specs import DOMAINS
from ..synthetic.workflows import LLM, NAMES, SOURCES, SURNAMES, stable_rng

PROMPTS = Path(__file__).parent / "prompts"
KEY_ENVS = ["OPENAI_API_KEY_2", "OPENAI_API_KEY_3", "OPENAI_API_KEY_4"]
LENGTHS = ["80-150 words", "150-300 words", "300-500 words", "500-800 words"]
PLANS = {"present": 0.35, "near miss": 0.3, "absent": 0.35}  # absent fields carry a near miss: earned nulls
PLAN_TEXT = {"present": "state one value that satisfies the instruction",
             "near miss": "state one value that satisfies the instruction, and elsewhere a similar value that does not",
             "absent": "do NOT state any value that satisfies the instruction anywhere in the document; only mention a "
                       "similar value that clearly does not qualify, so the answer is null"}
# reading skills suggested to the schema designer, a few per schema so schemas do not converge on the same fields
SKILLS = ["the value in force after a correction or a later message", "who committed to what",
          "the item a complaint is about", "every action assigned to a given role", "the reason given for a decision",
          "the condition attached to an approval", "the requested (not offered) option", "the current (not former) holder",
          "every item that was rejected", "the deadline that applies to a given party", "the quantity actually delivered",
          "the person a pronoun or title refers to", "every place visited in order", "the cause named for a failure",
          "the option finally chosen among those discussed", "every party that signed or agreed",
          "the amount owed after a discount or credit", "the version or edition being referred to",
          "every symptom or defect reported", "the date an event actually happened (not when it was planned)",
          "the recipient of a transfer or handover", "every identifier of an affected item",
          "the source a claim is attributed to", "the unresolved question at the end"]


def prompt(template: str, **fields) -> str:
    text = (PROMPTS / f"{template}.txt").read_text(encoding="utf-8")
    for key, value in fields.items():
        text = text.replace("{{%s}}" % key.upper(), str(value))
    return text


def normalized(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold().strip().strip(".,;:!?\"'()[]")
    return re.sub(r"^(the|a|an) ", "", re.sub(r"\s+", " ", text))


def as_set(value) -> frozenset:
    values = [value] if isinstance(value, str) else value if isinstance(value, list) else []
    return frozenset(normalized(v) for v in values if isinstance(v, str) and v.strip())


def same(a, b) -> bool:
    """Equal answers up to case, edge punctuation, spacing and a leading article; a list as a set; null equals an
    empty list."""
    return as_set(a) == as_set(b)


def locate(document: str, field: dict, value) -> list[dict] | None:
    """Character spans of a verbatim answer (list items in order of appearance), or None if one is not found."""
    if value is None or value == []:
        return []
    items = [value] if field["type"] == "text" else value
    if field["type"] == "text" and not isinstance(value, str) or not isinstance(items, list):
        return None
    spans, pos = [], 0
    for item in items:
        if not isinstance(item, str) or not item.strip():
            return None
        start = document.find(item, pos)
        start = document.find(item) if start < 0 else start
        if start < 0:
            return None
        spans.append({"text": item, "start": start, "end": start + len(item)})
        pos = start + len(item)
    return spans


def valid_schema(schema: dict, n_fields: int) -> bool:
    fields = schema.get("fields")
    if not isinstance(fields, list) or not 3 <= len(fields) <= n_fields + 1:
        return False
    names = [f.get("name") for f in fields]
    return len(set(names)) == len(names) and all(
        isinstance(f.get("name"), str) and re.fullmatch(r"[a-z][a-z0-9_]{1,40}", f["name"])
        and f.get("type") in ("text", "list") and len(str(f.get("instruction", ""))) >= 15 for f in fields)


async def design(llm: LLM, index: int, seed: int) -> dict | None:
    rng = stable_rng(seed, "schema", index)
    domain, source, n_fields = rng.choice(DOMAINS), rng.choice(SOURCES), rng.randint(3, 7)
    skills = "; ".join(rng.sample(SKILLS, 3))
    for attempt in range(3):
        text = prompt("design", domain=domain.replace("_", " "), source=source, n_fields=n_fields, skills=skills)
        try:
            schema = await llm.json(text + "\n" * attempt, temperature=0.9)
        except ValueError:
            continue
        if valid_schema(schema, n_fields):
            fields = [{k: str(f[k]) for k in ("name", "type", "instruction")} for f in schema["fields"]]
            return {"schema_id": f"sx{index:05d}", "domain": domain, "source": source,
                    "application": str(schema.get("application", "")), "document_kind":
                    str(schema.get("document_kind", source)), "fields": fields}
    return None


async def write(llm: LLM, schema: dict, index: int, seed: int) -> dict | None:
    rng = stable_rng(seed, schema["schema_id"], index)
    plans = {f["name"]: rng.choices(list(PLANS), list(PLANS.values()))[0] for f in schema["fields"]}
    plan = "\n".join(f"- {f['name']} ({f['type']}): {f['instruction']}\n  Plan: {PLAN_TEXT[plans[f['name']]]}"
                     for f in schema["fields"])
    names = ", ".join(f"{rng.choice(NAMES)} {rng.choice(SURNAMES)}" for _ in range(4))
    text = prompt("write", document_kind=schema["document_kind"], application=schema["application"],
                  length=rng.choice(LENGTHS), names=names, plan=plan)
    try:
        out = await llm.json(text, temperature=0.9)
    except ValueError:
        return None
    document, answers = out.get("document"), out.get("answers")
    if not isinstance(document, str) or len(document) < 80 or not isinstance(answers, dict):
        return None
    return {"document_id": f"{schema['schema_id']}_d{index:03d}", "schema_id": schema["schema_id"],
            "domain": schema["domain"], "document": document, "plans": plans,
            "fields": [{**f, "gold": answers.get(f["name"])} for f in schema["fields"]]}


async def read(llm: LLM, item: dict) -> dict:
    fields = "\n".join(f"- {f['name']} ({f['type']}): {f['instruction']}" for f in item["fields"])
    try:
        return await llm.json(prompt("check", document=item["document"], fields=fields), temperature=0.0)
    except ValueError:
        return {}


async def judge(llm: LLM, item: dict, field: dict, other) -> dict:
    """Blind to which answer is the writer's: the order is fixed by the document and field."""
    flip = stable_rng(item["document_id"], field["name"], "judge").random() < 0.5
    answers = (other, field["gold"]) if flip else (field["gold"], other)
    text = prompt("judge", document=item["document"], name=field["name"], type=field["type"],
                  instruction=field["instruction"], answer_1=json.dumps(answers[0], ensure_ascii=False),
                  answer_2=json.dumps(answers[1], ensure_ascii=False))
    try:
        out = await llm.json(text, temperature=0.0)
    except ValueError:
        return {"verdict": None}
    verdict = {"1": "2", "2": "1"}.get(out.get("verdict"), out.get("verdict")) if flip else out.get("verdict")
    return {"verdict": {"1": "writer", "2": "reader"}.get(verdict, verdict), "ambiguous": out.get("ambiguous") is True,
            "reason": str(out.get("reason", ""))[:300]}


async def verify(item: dict, readers: dict, judge_llm: LLM) -> dict:
    readings = dict(zip(readers, await asyncio.gather(*[read(llm, item) for llm in readers.values()])))
    for field in item["fields"]:
        field["gold_from"] = "writer"
        spans = locate(item["document"], field, field["gold"])
        field["readers"] = {name: r.get(field["name"]) for name, r in readings.items()}
        disagree = [v for v in field["readers"].values() if not same(v, field["gold"])]
        field["judge"] = None
        if spans is None:
            field["kept"], field["drop"] = False, "not verbatim"
        elif not disagree:
            field["kept"], field["drop"] = True, None
        elif len(disagree) == 1:
            field["judge"] = await judge(judge_llm, item, field, disagree[0])
            field["kept"] = field["judge"]["verdict"] == "writer" and not field["judge"]["ambiguous"]
            field["drop"] = None if field["kept"] else "judge"
        else:
            field["kept"], field["drop"] = False, "both readers disagree"
            first, second = field["readers"].values()
            reader_spans = locate(item["document"], field, first) if same(first, second) else None
            if reader_spans:  # the writer missed a value both readers copied: a blind judge settles it
                field["judge"] = await judge(judge_llm, item, field, first)
                if field["judge"]["verdict"] == "reader" and not field["judge"]["ambiguous"]:
                    field["gold"], spans, field["kept"], field["drop"] = first, reader_spans, True, None
                    field["gold_from"] = "readers"
        field["spans"] = spans if field["kept"] else None
    return item


def provider(model: str) -> ProviderConfig:
    return ProviderConfig(model=model, optional_api_key_envs=KEY_ENVS)


async def run(args) -> None:
    out = Path(args.out)
    cache = out / "cache"
    writer = LLM(provider(args.writer), cache / "writer", args.concurrency)
    readers = {m: LLM(provider(m), cache / "readers", args.concurrency) for m in args.readers}
    judge_llm = LLM(provider(args.judge or args.writer), cache / "judge", args.concurrency)
    path = out / "schemas.jsonl"
    schemas = [s for s in await asyncio.gather(*[design(writer, i, args.seed) for i in range(args.schemas)]) if s]
    path.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in schemas))
    print(f"schemas: {len(schemas)}/{args.schemas}", flush=True)

    in_flight = asyncio.Semaphore(4 * args.concurrency)

    async def one(schema, index):
        async with in_flight:
            try:
                item = await write(writer, schema, index, args.seed)
                return item and await verify(item, readers, judge_llm)
            except Exception as exc:  # one failed document must not stop a long run; reruns retry it
                print(f"{schema['schema_id']}_d{index:03d} failed: {str(exc)[:200]}", flush=True)
                return None

    items = out / "items.jsonl"
    done = {json.loads(line)["document_id"] for line in items.open()} if items.exists() else set()
    jobs = [one(s, d) for s in schemas for d in range(args.documents) if f"{s['schema_id']}_d{d:03d}" not in done]
    kept = total = 0
    with items.open("a") as handle:
        for n, job in enumerate(asyncio.as_completed(jobs), 1):
            item = await job
            if item is not None:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                handle.flush()
                total += len(item["fields"])
                kept += sum(f["kept"] for f in item["fields"])
            if n % 50 == 0 or n == len(jobs):
                print(f"documents {n + len(done)}/{len(done) + len(jobs)}, fields kept {kept}/{total} this session",
                      flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--schemas", type=int, default=40)
    parser.add_argument("--documents", type=int, default=25, help="documents per schema")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--writer", default="deepseek-v4-flash-0731")
    parser.add_argument("--readers", nargs="+", default=["gpt-oss-120b", "gemma-4-31b-it"])
    parser.add_argument("--judge", help="default: the writer model")
    parser.add_argument("--concurrency", type=int, default=12)
    asyncio.run(run(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
