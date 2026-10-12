#!/usr/bin/env sh
# Freeze the live dashboard into a static folder (page + API JSON) so it can be
# viewed through any plain file server, e.g. JupyterLab's /files/ on the sandbox
# where app ports are not reachable from the browser.
#
# Usage: scripts/snapshot_dashboard.sh <out_dir> [base_url]
set -eu
OUT="$1"
BASE="${2:-http://127.0.0.1:8765}"
mkdir -p "$OUT/api" "$OUT/static"
cp app/dashboard.html "$OUT/index.html"
cp app/static/chart.umd.js "$OUT/static/"
for e in status metrics alerts calls digest; do
  curl -s "$BASE/api/$e" > "$OUT/api/$e"
done
curl -s "$BASE/api/events?since=0" > "$OUT/api/events"
echo "snapshot written to $OUT"
