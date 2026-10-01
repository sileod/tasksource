#!/usr/bin/env bash
set -euo pipefail

run_dir=${1:?usage: publish_synthetic_typed_decisions.sh RUN_DIR [REPO_ID]}
repo_id=${2:-tasksource/synthetic-typed-decisions}
hf_dir="$run_dir/hf_dataset"
gate_report="$run_dir/publication_gate.json"

python - "$run_dir" > "$gate_report" <<'PY'
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

run_dir = Path(sys.argv[1])
states = pq.ParquetFile(run_dir / "hf_dataset" / "bundled.parquet").metadata.num_rows
audit = json.loads((run_dir / "teacher_audit_report.json").read_text())
questions = int(audit.get("questions", 0))
answered = int(audit.get("answered_questions", 0))
sampled = int(audit.get("sampled_states", 0))
quality_failures = int(audit.get("quality_failures", 0))
confident_disagreements = int(audit.get("confident_disagreements", 0))
answer_rate = answered / questions if questions else 0.0
quality_failure_rate = quality_failures / questions if questions else 1.0
checks = {
    "at_least_5000_states": states >= 5000,
    "at_least_100_audited_states": sampled >= 100,
    "audit_answer_rate_at_least_90_percent": answer_rate >= 0.90,
    "zero_confident_disagreements": confident_disagreements == 0,
    "quality_failure_rate_at_most_5_percent": quality_failure_rate <= 0.05,
}
report = {
    "states": states,
    "sampled_states": sampled,
    "audit_questions": questions,
    "audit_answer_rate": answer_rate,
    "confident_disagreements": confident_disagreements,
    "quality_failure_rate": quality_failure_rate,
    "checks": checks,
    "pass": all(checks.values()),
}
print(json.dumps(report, indent=2))
if not report["pass"]:
    raise SystemExit(2)
PY

state_count=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["states"])' "$gate_report")
hf repos create "$repo_id" --repo-type dataset --public --exist-ok
hf upload "$repo_id" "$hf_dir" . --repo-type dataset \
  --commit-message "Publish ${state_count} Jev-labeled synthetic typed decisions"
echo "Published ${state_count} states to ${repo_id}."
