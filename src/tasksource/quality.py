"""Portable decision identities for quality evidence, independent of rendering.

Matching is evidence lookup, not an instruction to discard. Inferred exclusions
and model flags are distinct from approved defects. Carry this key into other
recasts before changing the question, context or label vocabulary.
"""
import hashlib
import json
import math

from tasksource.jev.options import refers_to_other_options


def decision_key(row):
    """Hash exact input/task/gold; neutral choice order can differ across recasts.

    Accept published typed decisions or canonical ``recast='jev'`` rows. Question
    wording, context, gold, score order and positional options remain significant.
    Source row indices and group IDs never establish a match by themselves.
    """
    source = row.get('source', row.get('task'))
    state = row['state']
    question = row.get('question', row.get('instructions'))
    options = list(row.get('options', row.get('criteria', [])))
    kind = row.get('kind', 'choice')
    if 'target' in row:
        target = [float(value) for value in row['target']]
    else:
        label = row['label']
        if type(label) is not int or not 0 <= label < len(options):
            raise ValueError('Invalid gold index')
        target = [float(index == label) for index in range(len(options))]
    if not all(isinstance(text,str) for text in [source,state,question,*options]):
        raise ValueError('Source, state, question and options must be strings')
    if not all(math.isfinite(value) for value in target):
        raise ValueError('Targets must be finite')
    if kind == 'noul':
        if options or len(target) != 1:
            raise ValueError('noul requires one target and no options')
        answers = target
    else:
        if len(options) != len(target) or kind not in {'choice','score'}:
            raise ValueError('Invalid choice/score target')
        answers = list(zip(options,target))
        if kind == 'choice' and not any(refers_to_other_options(option) for option in options):
            answers.sort()
    payload = ['tasksource-decision-v1',source,state,question,kind,answers]
    return hashlib.sha256(json.dumps(payload,ensure_ascii=False,separators=(',',':'),
                                     allow_nan=False).encode()).hexdigest()


class ExclusionIndex:
    """Look up observed exclusions by canonical content, without applying them."""
    def __init__(self, records):
        self.by_key = {}
        for record in records:
            if record.get('decision_key'):
                self.by_key.setdefault(record['decision_key'],[]).append(record)

    @classmethod
    def from_parquet(cls, path):
        import pyarrow.parquet as pq
        columns = ['id','source','group_id','variant','split','category','decision_key']
        return cls(pq.read_table(path,columns=columns).to_pylist())

    def match(self, row):
        """Return matching evidence; no group-wide or fuzzy propagation."""
        return list(self.by_key.get(decision_key(row),[]))

    def lookup(self, key):
        """Use a canonical key carried through an unchanged recast/rendering."""
        return list(self.by_key.get(key,[]))
