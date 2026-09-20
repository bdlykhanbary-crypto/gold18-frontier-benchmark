#!/data/data/com.termux/files/usr/bin/bash
set -e

cd ~/downloads/gold18-frontier-benchmark

echo "Starting Gold18 forecast..."
gh workflow run "Gold18 Live XGBoost Forecast"

sleep 5

RUN_ID=$(gh run list \
  --workflow="gold18_live_forecast.yml" \
  --limit 1 \
  --json databaseId \
  --jq '.[0].databaseId')

echo "Run ID: $RUN_ID"
echo "Waiting for forecast to finish..."

gh run watch "$RUN_ID" --exit-status

OUT="$HOME/downloads/gold18-live-latest"
rm -rf "$OUT"
mkdir -p "$OUT"

gh run download "$RUN_ID" \
  -n gold18-live-forecast \
  -D "$OUT"

cp "$OUT/gold18_forecast_report.html" \
   "$HOME/storage/downloads/gold18_forecast_report.html"

echo
echo "Opening report..."
termux-open "$HOME/storage/downloads/gold18_forecast_report.html"
