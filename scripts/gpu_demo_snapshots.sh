#!/usr/bin/env sh
# Run the validated demo order against the live app (egress attempt, run, load
# right away) and freeze two dashboard snapshots: under load and after the run.
#
# Usage: scripts/gpu_demo_snapshots.sh <out_root> [base_url]
set -eu
ROOT="$1"
BASE="${2:-http://127.0.0.1:8765}"
curl -s -X POST "$BASE/api/egress-demo" > /dev/null
curl -s -X POST "$BASE/api/run" > /dev/null
curl -s -X POST "$BASE/api/simulate-load" -H 'Content-Type: application/json' \
  -d '{"duration_s": 30, "intensity": 1.0}' > /dev/null
sleep 20
scripts/snapshot_dashboard.sh "$ROOT/load" "$BASE"
i=0
until curl -s "$BASE/api/status" | grep -q '"run_state":"done"'; do
  i=$((i + 1))
  [ "$i" -gt 200 ] && break
  sleep 3
done
scripts/snapshot_dashboard.sh "$ROOT/done" "$BASE"
