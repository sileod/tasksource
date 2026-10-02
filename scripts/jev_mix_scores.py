"""Per-source scores for the steered mixes (src/tasksource/metadata/jev_mixes.py).

score = trust x zone x interest, from audits of sampled decisions:
- trust: 0 for sources excluded from Jev builds (their labels failed review), lowered for sources
  whose examples Jev finds malformed;
- zone (proximal development): how much probability decision models (Jev, Solar, D1 by default) put
  on the gold answer; sources they solve or miss outright weigh less than sources they half solve;
- interest: Jev's judgment that an example exercises a transferable skill and is not trivial.

    python scripts/jev_mix_scores.py --models build/jev-acc-jev build/jev-acc-d1 build/jev-acc-solar \\
        --meta build/jev-meta-audit-v2/per-source.csv
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from scripts.build_jev_dataset import JEV_EXCLUDED_SOURCES, PUBLISH_EXCLUDED_PREFIXES

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


def zone(pg, center=0.55, width=0.3):
    return 0.4 + 0.6 * math.exp(-((pg - center) / width) ** 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["build/jev-acc-jev", "build/jev-acc-d1", "build/jev-acc-solar"])
    parser.add_argument("--meta", default="build/jev-meta-audit-v2/per-source.csv")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    runs = [run for run in args.models if (Path(run) / "decisions.jsonl").exists()]
    scores = model_signals(runs).join(pd.read_csv(args.meta).set_index("source")
                                      [["transferable", "trivial", "malformed"]], how="left")
    scores = scores.fillna(scores.median(numeric_only=True))
    excluded = scores.index.isin(JEV_EXCLUDED_SOURCES) | scores.index.str.startswith(PUBLISH_EXCLUDED_PREFIXES)
    scores["trust"] = (~excluded) * (1 - (scores.malformed - 0.3).clip(lower=0))
    scores["zone"] = scores.pg.map(zone)
    scores["interest"] = (0.5 + scores.transferable - 0.5 * scores.trivial).clip(0.3, 1.5)
    scores["score"] = scores.trust * scores.zone * scores.interest
    scores.round(3).sort_values("score", ascending=False).to_csv(args.out, index_label="source")
    print(f"{len(scores)} sources from {len(runs)} model runs -> {args.out}")


if __name__ == "__main__":
    main()
