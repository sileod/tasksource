"""Synthetic extraction requests written by an LLM and kept only where independent blind readers confirm them.

    python -m tasksource.jev.synthetic_extraction --out DIR --schemas 40 --documents 25

Procedural generators cover what a solver can decide (records, dates in force, filters, trees, markup); this
covers natural documents whose fields need reading (corrections in a thread, commitments, the subject of a
complaint), in the style of extraction applications such as Cloudflare's Clairic.

1. design: per domain and document source, the writer model drafts an application's extraction schema: a document
   kind and typed fields (`text`: one span, `list`: every matching span) with precise instructions.
2. write: per schema, documents are written with an intended answer per field; each field is planned as present,
   present with a near miss (a similar value that does not qualify), or absent with a near miss (an earned null).
3. verify: answers must be verbatim spans of the document. Two blind readers from other model families extract
   every field. A field is kept when both agree with the writer; when one disagrees, a judge sees both readings and
   the field is kept only for an unambiguous verdict for the writer. When both disagree it is dropped, unless they
   copy the same span and the judge, choosing between that span and the writer's answer, rules for the readers:
   the writer missed a value it wrote (typically a planned null). Documents keep their verified fields (at least
   two).

Every call is cached by content hash under DIR/cache, so reruns resume and an export is reproducible from the
cache. Models are any OpenAI-compatible endpoint (OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_API_KEY_2...).
``python -m tasksource.jev.synthetic_extraction.export --run DIR`` writes the dataset rows, card and provenance.
"""
