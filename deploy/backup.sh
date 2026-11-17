#!/usr/bin/env sh
# Nightly Postgres backup (strategies, results, champions, signals). Run from the repo root,
# e.g. from cron. Market data in ./data can always be re-downloaded, so it isn't dumped here.
#   17 3 * * * cd /opt/swarm-trader && ./deploy/backup.sh >> backups/backup.log 2>&1
set -eu
mkdir -p backups
stamp=$(date -u +%Y%m%dT%H%M%SZ)
docker compose -f docker-compose.prod.yml --env-file deploy/.env exec -T postgres \
  pg_dump -U swarm --format=custom swarm > "backups/swarm-$stamp.dump"
# Keep 14 days of database dumps.
find backups -name 'swarm-*.dump' -mtime +14 -delete
echo "$(date -u) backup ok: backups/swarm-$stamp.dump"
