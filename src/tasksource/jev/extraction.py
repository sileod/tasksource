"""Extraction rows for the ``extraction`` config of tasksource-jev-typed-decisions.

One row is one text and several typed fields to fill from it, answered in one pass (the request shape
of extraction models such as Cloudflare's Clairic):

- ``state``: the text.
- ``fields``: ``[{"name", "type", "instruction"}]``. ``type`` is ``text`` (one span: extractive question
  answering, the instruction is the question) or ``list`` (every span of a kind: named entities).
- ``answers``: one per field, in order: ``{"null": bool, "values": [{"text", "start", "end"}]}``.
  Offsets are character positions in ``state`` (``state[start:end] == text``). ``null`` means the text
  does not contain the answer: an unanswerable question, or an entity type absent from the text.
  Several values of a ``text`` field are alternative gold spans; of a ``list`` field, every item.

Question answering rows group the questions asked about the same context (SQuAD-style datasets ask
several per paragraph); entity rows ask for every entity type of the dataset, or, for large ontologies,
the types present plus absent ones up to ``MAX_FIELDS``.
"""
import json
import random
import re
from collections import OrderedDict

from datasets import Features, List, Value

from .token_labels import TOKEN_LABEL_NAMES

MAX_FIELDS = 8

FIELD = {"name": Value("string"), "type": Value("string"), "instruction": Value("string")}
SPAN = {"text": Value("string"), "start": Value("int32"), "end": Value("int32")}
FEATURES = Features({
    "id": Value("string"), "source": Value("string"), "split": Value("string"), "state": Value("string"),
    "fields": List(FIELD), "answers": List({"null": Value("bool"), "values": List(SPAN)}),
    "metadata": Value("string"),
})


def check(row):
    """Raise unless every answer span matches the state and fields and answers align."""
    if len(row["fields"]) != len(row["answers"]) or not row["fields"]:
        raise ValueError(f"{row['id']}: fields and answers do not align")
    if len({f["name"] for f in row["fields"]}) != len(row["fields"]):
        raise ValueError(f"{row['id']}: duplicate field names")
    for field, answer in zip(row["fields"], row["answers"]):
        if answer["null"] != (not answer["values"]):
            raise ValueError(f"{row['id']}: null must mean no values ({field['name']})")
        for span in answer["values"]:
            if row["state"][span["start"]:span["end"]] != span["text"] or not span["text"].strip():
                raise ValueError(f"{row['id']}: span {span} does not match the state ({field['name']})")
    return row


def _metadata(value):
    if value is None:
        return {}
    return json.loads(value) if isinstance(value, str) else dict(value)


def qa_rows(rows, source, split, max_fields=MAX_FIELDS, kind="text"):
    """SQuAD-format rows (``context``, ``question``, ``answers``) grouped by context into fields of type
    ``kind``: ``text`` when several gold spans are alternatives (SQuAD), ``list`` when a question's
    answer is all of them (Quoref). Sources with fixed slots name their fields: a question's metadata ``field``
    (an attribute or entity type) and ``type`` override ``answer_<i>`` and ``kind``, and ``group`` splits a
    context's questions into rows (TAB, WDC PAVE, SWDE)."""
    groups = OrderedDict()
    for row in rows:
        groups.setdefault((row["context"], _metadata(row.get("metadata")).get("group")), []).append(row)
    for (context, _), questions in groups.items():
        # Chunks of at most max_fields; a named field repeated (two records with the same text) opens a new one.
        parts, names = [[]], [set()]
        for q in questions:
            name = _metadata(q.get("metadata")).get("field")
            if len(parts[-1]) == max_fields or (name and name in names[-1]):
                parts.append([]); names.append(set())
            parts[-1].append(q); names[-1].add(name)
        for part in parts:
            fields, answers, ids = [], [], []
            for i, q in enumerate(part, 1):
                meta = _metadata(q.get("metadata"))
                spans = {(s, t) for t, s in zip(q["answers"]["text"], q["answers"]["answer_start"])}
                values = [{"text": t, "start": s, "end": s + len(t)} for s, t in sorted(spans)]
                fields.append({"name": meta.get("field") or f"answer_{i}", "type": meta.get("type") or kind,
                               "instruction": q["question"].strip()})
                answers.append({"null": not values, "values": values})
                ids.append(meta.get("id"))
            yield check({"id": "", "source": source, "split": split, "state": context, "fields": fields,
                         "answers": answers, "metadata": json.dumps({"question_ids": ids}, ensure_ascii=False)})


def entity_type(label):
    """``B-PER`` -> ``PER``; IO labels (``art-film``) are their own type; ``O`` is None."""
    if label == "O":
        return None
    match = re.match(r"^([BIESLU])-(.+)$", label)
    return match.group(2) if match else label


# Entity types whose label alone is opaque (OntoNotes, acronyms) or too short to read well.
NAMES = {
    "NP": "noun phrase", "VP": "verb phrase", "PP": "prepositional phrase",
    "ADJP": "adjective phrase", "ADVP": "adverb phrase", "CONJP": "conjunction phrase",
    "INTJ": "interjection", "LST": "list marker", "PRT": "particle",
    "SBAR": "subordinate clause", "UCP": "unlike coordinated phrase",
    "ChemicalEntity": "chemical entity", "DiseaseOrPhenotypicFeature": "disease or phenotypic feature",
    "OrganismTaxon": "organism or species",
    "VEH": "vehicle", "Generic": "generic scientific term", "OtherScientificTerm": "other scientific term",
    'academicjournal': 'academic journal',
    'astronomicalobject': 'astronomical object',
    'chemicalcompound': 'chemical compound',
    'chemicalelement': 'chemical element',
    'literarygenre': 'literary genre',
    'musicalartist': 'musical artist',
    'musicalinstrument': 'musical instrument',
    'musicgenre': 'music genre',
    'politicalparty': 'political party',
    'programlang': 'programming language',
    'MATE': 'material',
    'MANP': 'manufacturing process',
    'MACEQ': 'machine or equipment',
    'APPL': 'application',
    'FEAT': 'feature',
    'PRO': 'mechanical property',
    'CHAR': 'characterization',
    'PARA': 'parameter',
    'ENAT': 'enabling technology',
    'CONPRI': 'concept or principle',
    'MANS': 'manufacturing standard',
    'BIOP': 'biomedical entity',
    'GeneOrGeneProduct': 'gene or gene product',
    'SequenceVariant': 'sequence variant',
    'CellLine': 'cell line',
    'T005': 'virus',
    'T007': 'bacterium',
    'T017': 'anatomical structure',
    'T022': 'body system',
    'T031': 'body substance',
    'T033': 'finding',
    'T037': 'injury or poisoning',
    'T038': 'biologic function',
    'T058': 'health care activity',
    'T062': 'research activity',
    'T074': 'medical device',
    'T082': 'spatial concept',
    'T091': 'biomedical occupation or discipline',
    'T092': 'organization',
    'T097': 'professional or occupational group',
    'T098': 'population group',
    'T103': 'chemical',
    'T168': 'food',
    'T170': 'intellectual product',
    'T201': 'clinical attribute',
    'T204': 'eukaryote',
    'Parties.BUYING_COMPANY': 'buying company',
    'Parties.SELLING_COMPANY': 'selling company',
    'Parties.ACQUIRED_COMPANY': 'acquired company',
    'Advisors.LEGAL_CONSULTING_COMPANY': 'legal consulting company',
    'Advisors.GENERIC_CONSULTING_COMPANY': 'consulting company',
    'Generic_Info.ANNUAL_REVENUES': 'annual revenues',

    "Peop": "person", "Other": "other named entity", "creative_work": "creative work",
    "Character_Name": "character name", "Restaurant_Name": "restaurant name",
    "TARGET": "opinion target", "ANATOMY": "anatomical entity",
    "Language": "programming language", "Class": "programming class", "Library": "software library",
    "Keyboard_IP": "keyboard input", "HTML_XML_Tag": "HTML or XML tag", "Tag": "HTML tag",
    "TextGarbage": "malformed text", "EntityOfFutureInterest": "other astronomy term of interest",
    "abstract": "abstract concept",
    "VIOLATED BY": "party committing the violation", "VIOLATED ON": "party affected by the violation",
    "NORP": "nationality, religious or political group", "FAC": "facility", "WORK_OF_ART": "work of art",
    "LAW": "law", "LANGUAGE": "language", "PRODUCT": "product", "EVENT": "event", "QUANTITY": "quantity",
    "CARDINAL": "cardinal number", "ORDINAL": "ordinal number", "short": "acronym", "long": "acronym expansion",
    "DNA": "DNA sequence or gene", "RNA": "RNA molecule", "corporation": "corporation", "creative-work": "creative work",
    "MISC": "other named entity (event, nationality, title)", "misc": "other named entity (event, nationality, title)",
}
FEW_NERD = {"art", "building", "event", "location", "organization", "other", "person", "product"}
FEW_NERD_FINE = {
    "broadcastprogram": "broadcast program", "writtenart": "written work", "sportsfacility": "sports facility",
    "attack/battle/war/militaryconflict": "attack, battle, war or military conflict", "sportsevent": "sports event",
    "GPE": "country, city or state", "bodiesofwater": "body of water", "road/railway/highway/transit": "road, railway, highway or transit line",
    "government/governmentagency": "government or government agency", "media/newspaper": "media outlet or newspaper",
    "politicalparty": "political party", "showorganization": "show organization", "sportsleague": "sports league",
    "sportsteam": "sports team", "astronomything": "astronomical object", "biologything": "biological entity",
    "chemicalthing": "chemical", "educationaldegree": "educational degree", "livingthing": "living thing",
    "medical": "medical term", "artist/author": "artist or author", "education": "educational institution",
    "religion": "religious organization", "law": "law", "god": "god",
}


def readable(kind):
    """``PER`` -> person; ``art-film`` -> film (art); ``AmortizationOfIntangibleAssets`` -> amortization of intangible assets."""
    if kind in NAMES:
        return NAMES[kind]
    if kind in TOKEN_LABEL_NAMES or kind.upper() in TOKEN_LABEL_NAMES:
        return TOKEN_LABEL_NAMES.get(kind, TOKEN_LABEL_NAMES.get(kind.upper()))
    if "-" in kind and kind.split("-", 1)[0] in FEW_NERD:  # Few-NERD coarse-fine
        coarse, fine = kind.split("-", 1)
        fine = FEW_NERD_FINE.get(fine, fine.replace("/", " or "))
        if fine == "other":
            return f"other {coarse}"
        return fine if coarse == "other" else f"{fine} ({coarse})"
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", kind).replace("_", " ").replace("-", " ")
    return words.lower()


def entity_spans(tokens, labels, names):
    """Detokenized text (tokens joined by spaces) and ``{type: [(start, end)]}``. BIO and IO schemes:
    ``B-`` opens a span, a change of type closes it."""
    if len(tokens) != len(labels):
        raise ValueError("Token and label counts differ")
    text, offsets, position = [], [], 0
    for token in tokens:
        offsets.append((position, position + len(token)))
        text.append(token); position += len(token) + 1
    spans, current = {}, None
    for (start, end), label in zip(offsets, labels):
        name = names[label] if isinstance(label, int) else label
        kind = entity_type(name)
        if kind is None:
            current = None
            continue
        if current and current[0] == kind and not name.startswith("B-"):
            spans[kind][-1] = (spans[kind][-1][0], end)
        else:
            spans.setdefault(kind, []).append((start, end))
        current = (kind,)
    return " ".join(text), spans


def ner_rows(rows, source, split, names, instruction="Every {name} mentioned in the text, as written.",
             max_fields=MAX_FIELDS, seed=0):
    """Token classification rows (``tokens``, ``labels``) as one list field per entity type."""
    kinds = list(dict.fromkeys(k for k in map(entity_type, names) if k))
    for index, row in enumerate(rows):
        state, spans = entity_spans(row["tokens"], row["labels"], names)
        if not state.strip():
            continue
        asked = kinds
        if len(kinds) > max_fields:
            # Large ontologies: the types present, and as many absent ones (at least two), so that
            # nulls do not swamp the row; texts with no entity at all say nothing about the ontology.
            present = [k for k in kinds if k in spans][:max_fields]
            if not present:
                continue
            rng = random.Random(f"{seed}:{source}:{split}:{index}")
            absent = [k for k in kinds if k not in spans]
            asked = present + rng.sample(absent, min(len(absent), max_fields - len(present), max(2, len(present))))
            asked.sort(key=kinds.index)
        fields, answers = [], []
        for kind in asked:
            values = [{"text": state[s:e], "start": s, "end": e} for s, e in spans.get(kind, [])]
            fields.append({"name": re.sub(r"\W+", "_", readable(kind)).strip("_"), "type": "list",
                           "instruction": instruction.format(name=readable(kind))})
            answers.append({"null": not values, "values": values})
        yield check({"id": "", "source": source, "split": split, "state": state, "fields": fields,
                     "answers": answers, "metadata": json.dumps({"entity_types": asked}, ensure_ascii=False)})
