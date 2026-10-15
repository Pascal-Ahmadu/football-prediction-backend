#!/usr/bin/env bash
# Nightly database backup. Keeps 7 daily copies and one per month thereafter.
#
#   scripts/backup_db.sh
#   scripts/backup_db.sh --restore data/backups/fmep-20261004.dump   # careful
set -uo pipefail

project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project"
mkdir -p data/backups

if [ "${1:-}" = "--restore" ]; then
  file="${2:?usage: backup_db.sh --restore <file>}"
  echo "This REPLACES the current database with $file."
  read -r -p "Type the word restore to continue: " answer
  [ "$answer" = "restore" ] || { echo "cancelled"; exit 1; }
  docker compose exec -T db pg_restore -U postgres -d fmep --clean --if-exists <"$file"
  echo "restored from $file"
  exit 0
fi

stamp="$(date +%Y%m%d)"
file="data/backups/fmep-$stamp.dump"

# Custom format: compressed, and pg_restore can rebuild selectively.
if ! docker compose exec -T db pg_dump -U postgres -d fmep -Fc >"$file"; then
  echo "backup FAILED" >&2
  rm -f "$file"
  exit 1
fi

size="$(du -h "$file" | cut -f1)"
echo "$(date '+%Y-%m-%d %H:%M:%S')  backup written: $file ($size)"

# Keep the last 7 days, plus the first backup of each month, and drop the rest.
find data/backups -name 'fmep-*.dump' -mtime +7 ! -name 'fmep-??????01.dump' -delete
