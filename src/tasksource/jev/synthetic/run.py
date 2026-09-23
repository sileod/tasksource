"""CLI entrypoint: restartable staged pipeline.

    python -m tasksource.jev.synthetic.run --config <config.yaml> [--stage all|specs|generate|validate|critic|dedup|annotate|select|split|export]

The very first action of the generate stage is the API-key preflight;
no network call or generated artifact happens before that check.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from pathlib import Path

import pandas as pd

from . import annotate as annot_mod
from . import critic as critic_mod
from . import dedup as dedup_mod
from . import manifest as manifest_mod
from . import providers, select as select_mod
from . import specs as specs_mod
from . import split as split_mod
from . import validate as validate_mod
from .config import load_config
from .generate import PROMPTS_DIR, generate_bundles, generation_cache_key, prompt_hash
from .schemas import bundle_to_flat_rows, flat_to_training_row

STAGES = ("specs", "generate", "validate", "critic", "dedup",
          "annotate", "select", "split", "export")


def run_dir_for(cfg) -> Path:
    return Path(cfg.output_dir) / cfg.run_name


def _write_bundles(path: Path, bundles: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for bundle in bundles:
            handle.write(json.dumps(bundle, ensure_ascii=False) + "\n")


def _read_bundles(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _to_frame(bundles: list[dict]) -> pd.DataFrame:
    rows = []
    for bundle in bundles:
        row = dict(bundle)
        for key in ("questions", "annotations", "annotation_stats"):
            if key in row and row[key] is not None:
                row[key] = json.dumps(row[key], ensure_ascii=False)
        rows.append(row)
    return pd.DataFrame(rows)


def _from_frame(frame: pd.DataFrame) -> list[dict]:
    bundles = []
    for record in frame.to_dict(orient="records"):
        for key in ("questions", "annotations", "annotation_stats"):
            if key in record and isinstance(record[key], str):
                try:
                    record[key] = json.loads(record[key])
                except json.JSONDecodeError:
                    pass
        bundles.append(record)
    return bundles


def stage_specs(cfg, run_dir: Path, n_states: int | None = None) -> list[dict]:
    specs = specs_mod.sample_specs(cfg.sampler, n_states or cfg.sampler.n_states)
    frame = pd.DataFrame([{**s, "questions": json.dumps(s["questions"])} for s in specs])
    frame.to_parquet(run_dir / "specs.parquet", index=False)
    (run_dir / "specs.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in specs), encoding="utf-8")
    return specs


def stage_generate(cfg, run_dir: Path) -> list[dict]:
    # Preflight FIRST: raises before any sampling or artifact writes.
    preflight = asyncio.run(providers.preflight(cfg.provider))
    specs = _read_bundles(run_dir / "specs.jsonl")
    spec_by_id = {s["state_id"]: s for s in specs}
    results = generate_bundles(cfg, specs, run_dir / "raw" / "generation")
    bundles = [r["bundle"] for r in results]
    _write_bundles(run_dir / "candidates.jsonl", bundles)
    _to_frame(bundles).to_parquet(run_dir / "candidates.parquet", index=False)
    manifest_path = run_dir / "manifest.json"
    prompt_hashes = {}
    for name in ("generate_v1", "critic_v1"):
        prompt_file = PROMPTS_DIR / f"{name}.txt"
        if prompt_file.exists():
            prompt_hashes[name] = prompt_hash(prompt_file.read_text(encoding="utf-8"))
    manifest_mod.write_manifest(
        manifest_path, manifest_mod.build_manifest(
            cfg, preflight, {"candidates": len(bundles),
                             "generate_cache_key": generation_cache_key(cfg),
                             "annotator": cfg.annotator.name,
                             "critic_provider": cfg.critic_provider().name,
                             "critic_model": cfg.critic.model},
            prompt_hashes))
    # Keep spec lookup for validation.
    (run_dir / "_spec_by_id.json").write_text(json.dumps(spec_by_id), encoding="utf-8")
    return bundles


def stage_validate(cfg, run_dir: Path) -> list[dict]:
    bundles = _read_bundles(run_dir / "candidates.jsonl")
    spec_by_id = json.loads((run_dir / "_spec_by_id.json").read_text(encoding="utf-8")) \
        if (run_dir / "_spec_by_id.json").exists() else {}
    valid, report = [], []
    for bundle in bundles:
        errors = validate_mod.validate_bundle(bundle, spec_by_id.get(bundle.get("state_id")))
        report.append({"state_id": bundle.get("state_id"), "errors": errors})
        if not errors:
            valid.append(bundle)
    _write_bundles(run_dir / "validated.jsonl", valid)
    _to_frame(valid).to_parquet(run_dir / "validated.parquet", index=False)
    (run_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return valid


def stage_critic(cfg, run_dir: Path) -> list[dict]:
    bundles = _read_bundles(run_dir / "validated.jsonl")
    verdicts = critic_mod.critique_bundles(cfg, bundles, run_dir / "raw" / "critic")
    passing_ids = {v["state_id"] for v in verdicts if v.get("pass", True)}
    kept = [b for b in bundles if b["state_id"] in passing_ids]
    _write_bundles(run_dir / "critic_passed.jsonl", kept)
    return kept


def stage_dedup(cfg, run_dir: Path) -> list[dict]:
    source = run_dir / "critic_passed.jsonl"
    bundles = _read_bundles(source if source.exists() else run_dir / "validated.jsonl")
    kept, dropped = dedup_mod.dedup_bundles(bundles)
    _write_bundles(run_dir / "deduped.jsonl", kept)
    (run_dir / "dedup_report.json").write_text(json.dumps({"dropped": dropped}, indent=2), encoding="utf-8")
    return kept


def stage_annotate(cfg, run_dir: Path) -> list[dict]:
    source = run_dir / "deduped.jsonl"
    bundles = _read_bundles(source if source.exists() else run_dir / "validated.jsonl")
    annotated = annot_mod.annotate_bundles(bundles, cfg.annotator)
    _write_bundles(run_dir / "annotated.jsonl", annotated)
    _to_frame(annotated).to_parquet(run_dir / "annotated.parquet", index=False)
    # Persist raw Jev annotation artifacts.
    jev_dir = run_dir / "raw" / "jev"
    jev_dir.mkdir(parents=True, exist_ok=True)
    for bundle in annotated:
        (jev_dir / f"{bundle['state_id']}.json").write_text(
            json.dumps(bundle.get("annotations", []), ensure_ascii=False, indent=2), encoding="utf-8")
    return annotated


def stage_select(cfg, run_dir: Path, n_target: int | None = None) -> list[dict]:
    annotated = _read_bundles(run_dir / "annotated.jsonl")
    result = select_mod.select_bundles(annotated, cfg.selection, n_target)
    _write_bundles(run_dir / "selected.jsonl", result["selected"])
    (run_dir / "selection_report.json").write_text(
        json.dumps(result["diagnostics"], indent=2), encoding="utf-8")
    return result["selected"]


def stage_split(cfg, run_dir: Path) -> list[dict]:
    selected = _read_bundles(run_dir / "selected.jsonl")
    with_splits = split_mod.assign_splits(selected, cfg.split)
    _write_bundles(run_dir / "final.jsonl", with_splits)
    _to_frame(with_splits).to_parquet(run_dir / "final.parquet", index=False)
    return with_splits


def stage_export(cfg, run_dir: Path) -> dict:
    bundles = _read_bundles(run_dir / "final.jsonl")
    flat_rows: list[dict] = []
    for bundle in bundles:
        annotations = {q["question_id"]: a for q, a in
                       zip(bundle.get("questions", []), bundle.get("annotations", []))}
        for flat in bundle_to_flat_rows(bundle):
            target = annotations.get(flat["question_id"], {}).get("probabilities")
            flat_rows.append({**flat_to_training_row(flat, target, "synthetic/jev",
                                                     bundle.get("split", "train")),
                              "state_id": flat["state_id"], "question_id": flat["question_id"],
                              "bundle_size": flat["bundle_size"], "domain": flat["domain"],
                              "skill": flat.get("skill", "")})
    flat_frame = pd.DataFrame(flat_rows)
    flat_frame.to_parquet(run_dir / "flat.parquet", index=False)
    # HF dataset dir with bundled + flat configs.
    hf_dir = run_dir / "hf_dataset"
    hf_dir.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import Dataset
        for split_name in ("train", "validation", "test", "ood"):
            part = flat_frame[flat_frame["split"] == split_name] if len(flat_frame) else flat_frame
            if len(part):
                Dataset.from_pandas(part, preserve_index=False).to_parquet(
                    hf_dir / f"flat-{split_name}.parquet")
    except Exception:
        pass
    (hf_dir / "README.md").write_text(
        "# jev-synthetic-decisions\n\nConfigs: `bundled` (one row per state, "
        "`final.parquet`) and `flat` (one row per decision, `flat.parquet`).\n",
        encoding="utf-8")
    shutil.copyfile(run_dir / "final.parquet", hf_dir / "bundled.parquet")
    shutil.copyfile(run_dir / "flat.parquet", hf_dir / "flat.parquet")
    # Update manifest counts.
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["counts"] = {**manifest.get("counts", {}),
                          "final_states": len(bundles),
                          "final_decisions": len(flat_rows)}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"states": len(bundles), "decisions": len(flat_rows)}


def run_stage(cfg, run_dir: Path, stage: str, **kwargs):
    if stage == "specs":
        return stage_specs(cfg, run_dir, kwargs.get("n_states"))
    if stage == "generate":
        return stage_generate(cfg, run_dir)
    if stage == "validate":
        return stage_validate(cfg, run_dir)
    if stage == "critic":
        return stage_critic(cfg, run_dir)
    if stage == "dedup":
        return stage_dedup(cfg, run_dir)
    if stage == "annotate":
        return stage_annotate(cfg, run_dir)
    if stage == "select":
        return stage_select(cfg, run_dir, kwargs.get("n_target"))
    if stage == "split":
        return stage_split(cfg, run_dir)
    if stage == "export":
        return stage_export(cfg, run_dir)
    raise ValueError(f"Unknown stage: {stage}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Synthetic Jev dataset pipeline")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--stage", default="all",
                        help="Stage name or 'all'. Restartable; prior stage outputs are reused.")
    parser.add_argument("--n-states", type=int, default=None, help="Override number of specs (pilot)")
    parser.add_argument("--n-target", type=int, default=None, help="Override selection target size")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.n_states:
        cfg.sampler.n_states = args.n_states
        cfg.generation.n_states = args.n_states
    run_dir = run_dir_for(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Persist the exact config used.
    shutil.copyfile(args.config, run_dir / "config.yaml")
    stages = STAGES if args.stage == "all" else (args.stage,)
    for stage in stages:
        print(f"[jev-synthetic] stage: {stage}", flush=True)
        run_stage(cfg, run_dir, stage, n_states=args.n_states, n_target=args.n_target)
    print(f"[jev-synthetic] done -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
