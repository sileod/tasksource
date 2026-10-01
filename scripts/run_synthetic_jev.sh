#!/usr/bin/env bash
# Generate synthetic typed decisions and upload every STEP kept decisions, up to TARGET (counting earlier runs).
#   scripts/run_synthetic_jev.sh OUT WORKFLOWS STATES SEED TARGET STEP [earlier run dirs...]
# Keys come from the environment (ALBERT_API_KEY, ALBERT_API_KEY_2, JEV_OPENROUTER_API_KEY).
set -u
out=$1 workflows=$2 states=$3 seed=$4 target=$5 step=$6; shift 6
runs=("$@" "$out")
PYTHONPATH=src python -m tasksource.jev.synthetic.workflows --out "$out" --workflows "$workflows" \
    --states "$states" --seed "$seed" >> "$out.log" 2>&1 &
generator=$!
count() { python -c '
import json, sys, os
print(sum(q["kept"] and "jev" in q for d in sys.argv[1:] if os.path.exists(d + "/items.jsonl")
          for line in open(d + "/items.jsonl") for q in json.loads(line)["questions"]))' "${runs[@]}"; }
next=$step
while [ "$next" -le "$target" ]; do
    while kill -0 $generator 2>/dev/null && [ "$(count)" -lt "$next" ]; do sleep 300; done
    echo "$(date) uploading at $(count) decisions"
    python scripts/build_synthetic_jev.py "${runs[@]}" --upload 2>&1 | grep -E "^(train|validation|test):"
    kill -0 $generator 2>/dev/null || break
    next=$((next + step))
done
wait $generator
