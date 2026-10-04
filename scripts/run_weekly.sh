#!/usr/bin/env bash
# The Linux equivalent of scripts/run_weekly.ps1: runs the pipeline in a one-shot
# container, logs it, and keeps 60 days of logs.
#
#   scripts/run_weekly.sh                          # full refresh
#   scripts/run_weekly.sh --days-back 3 --days-ahead 4 --with-odds
set -uo pipefail

project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project"
mkdir -p data/logs
log="data/logs/weekly-$(date +%Y%m%d-%H%M).log"

printf '%s  starting: weekly pipeline %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >>"$log"

docker compose run --rm jobs python -u -m app.pipeline.weekly "$@" >>"$log" 2>&1
code=$?

if [ "$code" -eq 0 ]; then
  printf '%s  finished OK\n' "$(date '+%Y-%m-%d %H:%M:%S')" >>"$log"
else
  printf '%s  FAILED with exit code %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$code" >>"$log"
fi

find data/logs -name 'weekly-*.log' -mtime +60 -delete
exit "$code"
