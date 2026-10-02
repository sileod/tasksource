"""Per-source scores for the steered mixes (src/tasksource/metadata/jev_mixes.py).

score = trust x zone x interest x clean x options x length x picked, from audits of sampled decisions:
- trust: 0 for sources excluded from Jev builds (their labels failed review);
- zone (proximal development): how much probability decision models (Jev, Solar, D1 by default) put
  on the gold answer. The judges are stronger than the models trained on the mix, so a source they
  solve loses little (x0.75 at most) while one they miss outright, often unanswerable, loses more (x0.5);
- interest: Jev's judgment that an example exercises a transferable skill and is not trivial;
- clean: 1.3 minus how often Jev finds examples malformed, so curated data counts;
- options: fewer rows for yes/no and two-option sources, more for many-option ones;
- length: more rows for long inputs, which few sources have (x1.25 at 2k characters, x1.5 from 4k);
- picked: a boost for sources picked by reading them (PICKED in jev_mixes.py).

    python scripts/jev_mix_scores.py --models build/jev-acc-jev build/jev-acc-d1 build/jev-acc-solar \\
        --meta build/jev-meta-audit-v2/per-source.csv
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from scripts.build_jev_dataset import JEV_EXCLUDED_SOURCES, PUBLISH_EXCLUDED_PREFIXES
from tasksource.metadata.jev_mixes import PICKED, PICKED_BOOST, length_factor

OUT = Path(__file__).resolve().parent.parent / "src" / "tasksource" / "metadata" / "jev_source_scores.csv"


def gold_probability(row):
    p = row["jev_probabilities"]
    if row["kind"] == "noul":
        return p[0] if row["gold"] == 1 else 1 - p[0]
    return p[int(row["gold"])] / (sum(p) or 1)


def model_signals(runs):
    """Mean gold probability per source over models, and how often the models split."""
    frames = []
    for run in runs:
        rows = [json.loads(line) for line in (Path(run) / "decisions.jsonl").open()]
        frame = pd.DataFrame(rows).set_index("id")
        frame["pg"] = [gold_probability(r) for r in rows]
        frames.append(frame[["source", "jev", "pg"]].rename(columns={"jev": f"top{len(frames)}", "pg": f"pg{len(frames)}"}))
    joined = frames[0].join([f.drop(columns="source") for f in frames[1:]], how="inner")
    pg = joined[[c for c in joined if c.startswith("pg")]]
    tops = joined[[c for c in joined if c.startswith("top")]]
    joined["pg"] = pg.mean(axis=1)
    joined["split"] = tops.nunique(axis=1) > 1
    return joined.groupby("source").agg(n=("pg", "size"), pg=("pg", "mean"), split=("split", "mean"))


def zone(pg, low=0.35, high=0.8):
    """1 inside [low, high]; down to 0.75 for sources the judges always solve, 0.5 for ones they always miss."""
    if pg > high:
        return 1 - 0.25 * (pg - high) / (1 - high)
    if pg < low:
        return 1 - 0.5 * (low - pg) / low
    return 1.0


def options_factor(n_options):
    return 0.6 if n_options <= 2 else 0.9 if n_options == 3 else 1.0 if n_options <= 5 else 1.2


def shard_stats(shards):
    """Median option count (yes/no counts as 2) and state length per source, from train shards."""
    import glob
    import numpy as np
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    counts, lengths = {}, {}
    for path in glob.glob(str(Path(shards) / "train-*.parquet")):
        table = pq.read_table(path, columns=["source", "options", "state"])
        if table.num_rows:
            options = pc.list_value_length(table.column("options")).fill_null(0).to_numpy(zero_copy_only=False)
            source = table.column("source")[0].as_py()
            counts[source] = float(np.median(np.maximum(options, 2)))
            lengths[source] = float(np.median(pc.utf8_length(table.column("state")).to_numpy(zero_copy_only=False)))
    return pd.DataFrame({"n_options": counts, "state_chars": lengths})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["build/jev-acc-jev", "build/jev-acc-d1", "build/jev-acc-solar"])
    parser.add_argument("--meta", nargs="+", default=["build/jev-meta-audit-v2/per-source.csv",
                                                      "build/jev-meta-audit-long/per-source.csv"],
                        help="later files override earlier ones (e.g. a rerun of long sources without truncation)")
    parser.add_argument("--shards", default="build/tasksource-jev-typed-decisions-v11/data")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    runs = [run for run in args.models if (Path(run) / "decisions.jsonl").exists()]
    meta = pd.concat([pd.read_csv(path) for path in args.meta if Path(path).exists()])
    meta = meta.drop_duplicates("source", keep="last").set_index("source")[["transferable", "trivial", "malformed"]]
    scores = model_signals(runs).join(meta, how="left").join(shard_stats(args.shards), how="left")
    scores = scores.fillna(scores.median(numeric_only=True))
    excluded = scores.index.isin(JEV_EXCLUDED_SOURCES) | scores.index.str.startswith(PUBLISH_EXCLUDED_PREFIXES)
    scores["trust"] = (~excluded).astype(float)
    scores["zone"] = scores.pg.map(zone)
    scores["interest"] = (0.5 + scores.transferable - 0.5 * scores.trivial).clip(0.3, 1.5)
    scores["clean"] = 1.3 - scores.malformed
    scores["options"] = scores.n_options.map(options_factor)
    scores["length"] = scores.state_chars.map(length_factor)
    scores["picked"] = [PICKED_BOOST if picked else 1.0 for picked in scores.index.str.contains(PICKED)]
    scores["score"] = scores.trust * scores.zone * scores.interest * scores.clean * scores.options * scores.length * scores.picked
    scores.round(3).sort_values("score", ascending=False).to_csv(args.out, index_label="source")
    print(f"{len(scores)} sources from {len(runs)} model runs -> {args.out}")


if __name__ == "__main__":
    main()
