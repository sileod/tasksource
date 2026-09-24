"""Load every task with prompted=True and record what a model would see.

    PYTHONPATH=src python scripts/audit_prompted.py build/prompt_audit.jsonl [task ids...]

Each line holds the task question, label names and two prompted examples, or the load error.
"""
import json, sys, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings("ignore")
OUT = sys.argv[1]

def audit(task_id, multilingual):
    import datasets
    datasets.disable_progress_bars(); datasets.logging.set_verbosity_error()
    from tasksource import load_task
    try:
        d = load_task(task_id, multilingual=multilingual, max_rows=200, max_rows_eval=10, prompted=True)
        tr = d["train"]; f = tr.features["labels"]
        names = getattr(f, "names", None) or getattr(getattr(f, "feature", None), "names", None)
        rows = []
        for ex in tr.select(range(min(2, len(tr)))):
            if "inputs" in ex:
                choices = [str(ex[c])[:70] for c in sorted(k for k in ex if k.startswith("choice")) if ex[c] is not None]
                rows.append(dict(inputs=str(ex["inputs"])[-250:], choices=choices[:4], gold=ex["labels"]))
            elif "tokens" in ex:
                rows.append(dict(tokens=" ".join(ex["tokens"])[:120]))
            else:
                rows.append(dict(s1=str(ex["sentence1"])[:220], s2=str(ex.get("sentence2", ""))[:160],
                                 gold=names[ex["labels"]] if names else ex["labels"]))
        return dict(task=task_id, ml=multilingual, type=d.task_type, question=d.question,
                    names=[str(n)[:40] for n in names[:15]] if names else None, n_names=len(names) if names else None,
                    rows=rows)
    except Exception as e:
        return dict(task=task_id, ml=multilingual, error=f"{type(e).__name__}: {str(e)[:200]}")

if __name__ == "__main__":
    from tasksource import list_tasks
    todo = [(i, False) for i in list_tasks().id] + [(i, True) for i in list_tasks(multilingual=True).id]
    if sys.argv[2:]:
        todo = [t for t in todo if t[0] in sys.argv[2:]]
    print(len(todo), "tasks", file=sys.stderr, flush=True)
    with open(OUT, "w") as out, ProcessPoolExecutor(12) as pool:
        for f in as_completed([pool.submit(audit, *t) for t in todo]):
            out.write(json.dumps(f.result(), ensure_ascii=False) + "\n"); out.flush()
