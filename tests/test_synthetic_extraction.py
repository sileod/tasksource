import asyncio
import json

from tasksource.jev.synthetic_extraction import __main__ as gen
from tasksource.jev.synthetic_extraction import export

DOC = "Delivery moved to 12 March (not 10 March). Assigned: Ana Ruiz, Tom Lee. The pallet of cartons was damaged."
FIELDS = [{"name": "date", "type": "text", "instruction": "The corrected delivery date."},
          {"name": "people", "type": "list", "instruction": "Every person assigned an action."},
          {"name": "item", "type": "text", "instruction": "The item the complaint is about."},
          {"name": "refund", "type": "text", "instruction": "The refund amount granted, if any."}]


class Fake:
    def __init__(self, answer):
        self.answer = answer

    async def json(self, text, temperature=0.0):
        return self.answer


def item():
    gold = {"date": "12 March", "people": ["Ana Ruiz", "Tom Lee"], "item": "pallet of cartons", "refund": None}
    return {"document_id": "sx00000_d000", "schema_id": "sx00000", "domain": "logistics", "document": DOC,
            "plans": {f["name"]: "present" for f in FIELDS}, "fields": [{**f, "gold": gold[f["name"]]} for f in FIELDS]}


def test_same_and_locate():
    assert gen.same("The pallet of cartons.", "pallet of cartons") and gen.same(None, [])
    assert gen.same(["Tom Lee", "ana ruiz"], ["Ana Ruiz", "Tom Lee"]) and not gen.same("10 March", "12 March")
    spans = gen.locate(DOC, FIELDS[1], ["Ana Ruiz", "Tom Lee"])
    assert [DOC[s["start"]:s["end"]] for s in spans] == ["Ana Ruiz", "Tom Lee"]
    assert gen.locate(DOC, FIELDS[0], "13 March") is None and gen.locate(DOC, FIELDS[0], None) == []


def test_verify_rules():
    agree = {"date": "12 March", "people": ["Tom Lee", "Ana Ruiz"], "item": "the pallet of cartons", "refund": None}
    other = {**agree, "date": "10 March", "item": "cartons"}
    readers = {"a": Fake(agree), "b": Fake(other)}
    out = asyncio.run(gen.verify(item(), readers, Fake({"verdict": None})))
    kept = {f["name"]: (f["kept"], f["drop"]) for f in out["fields"]}
    assert kept == {"date": (False, "judge"), "people": (True, None), "item": (False, "judge"), "refund": (True, None)}
    readers = {"a": Fake(other), "b": Fake(other)}
    out = asyncio.run(gen.verify(item(), readers, Fake({})))
    assert out["fields"][0]["drop"] == "both readers disagree"


def test_readers_recover_a_value_the_writer_missed():
    found = {"date": "12 March", "people": ["Ana Ruiz", "Tom Lee"], "item": "pallet of cartons", "refund": "10 March"}
    readers = {"a": Fake(found), "b": Fake(found)}
    flipped = gen.stable_rng("sx00000_d000", "refund", "judge").random() < 0.5
    answer = "1" if flipped else "2"  # the readers' span, in the judge's presentation order
    out = asyncio.run(gen.verify(item(), readers, Fake({"verdict": answer, "ambiguous": False})))
    refund = out["fields"][3]
    assert refund["kept"] and refund["gold"] == "10 March" and refund["gold_from"] == "readers"
    assert DOC[refund["spans"][0]["start"]:refund["spans"][0]["end"]] == "10 March"
    out = asyncio.run(gen.verify(item(), readers, Fake({"verdict": answer, "ambiguous": True})))
    assert not out["fields"][3]["kept"]


def test_judge_is_blind_to_order():
    it, field = item(), {**item()["fields"][0]}
    flip = gen.stable_rng(it["document_id"], field["name"], "judge").random() < 0.5
    verdict = asyncio.run(gen.judge(Fake({"verdict": "2" if flip else "1", "ambiguous": False}), it, field, "x"))
    assert verdict["verdict"] == "writer" and not verdict["ambiguous"]


def test_export_rows(tmp_path):
    readers = {"a": Fake({"date": "12 March", "people": ["Ana Ruiz", "Tom Lee"], "item": "x", "refund": None})}
    verified = asyncio.run(gen.verify(item(), {**readers, "b": readers["a"]}, Fake({})))
    schema = {"schema_id": "sx00000", "source": "emails", "document_kind": "thread", "application": "claims"}
    row = export.to_row(verified, schema)
    assert [f["name"] for f in row["fields"]] == ["date", "people", "refund"]
    assert row["answers"][2] == {"null": True, "values": []}
    assert json.loads(row["metadata"])["dropped"] == {"item": "both readers disagree"}
    assert export.split_of("sx00000") in ("train", "validation", "test")
    splits = [export.split_of(f"sx{i:05d}") for i in range(1000)]
    assert 0.7 < splits.count("train") / 1000 < 0.9
