#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

REPO="$HOME/downloads/gold18-frontier-benchmark"
OUT="$HOME/storage/downloads/gold18_forecast_report.html"
WORKFLOW="gold18_live_forecast.yml"

cd "$REPO"

echo "Syncing Gold4Cast..."
git pull --ff-only

OLD_ID="$(gh run list --workflow="$WORKFLOW" --branch main --limit 1 \
  --json databaseId --jq '.[0].databaseId // 0' 2>/dev/null || echo 0)"

echo "Starting Gold18 forecast..."
gh workflow run "$WORKFLOW" --ref main

RUN_ID=""
for i in $(seq 1 30); do
  CANDIDATE="$(gh run list --workflow="$WORKFLOW" --branch main --limit 1 \
    --json databaseId --jq '.[0].databaseId // 0' 2>/dev/null || echo 0)"
  if [ "$CANDIDATE" != "0" ] && [ "$CANDIDATE" != "$OLD_ID" ]; then
    RUN_ID="$CANDIDATE"
    break
  fi
  sleep 2
done

if [ -z "$RUN_ID" ]; then
  echo "خطا: Run جدید GitHub پیدا نشد."
  exit 1
fi

echo "Run ID: $RUN_ID"
echo "Waiting for forecast to finish..."
gh run watch "$RUN_ID" --exit-status

echo "Forecast completed. Pulling report directly from Git..."
FOUND=0
for i in $(seq 1 15); do
  git pull --ff-only >/dev/null 2>&1 || true
  if [ -s "$REPO/gold18_forecast_report.html" ]; then
    FOUND=1
    break
  fi
  sleep 2
done

if [ "$FOUND" != "1" ]; then
  echo "خطا: فایل HTML بعد از اجرای موفق در مخزن پیدا نشد."
  echo "Run: https://github.com/bdlykhanbary-crypto/gold18-frontier-benchmark/actions/runs/$RUN_ID"
  exit 1
fi

cp "$REPO/gold18_forecast_report.html" "$OUT"

echo "HTML آماده است:"
echo "$OUT"

termux-open "$OUT"
