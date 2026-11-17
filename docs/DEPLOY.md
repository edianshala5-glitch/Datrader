# Deployment

Run the whole system on one small machine for free: Postgres, Redis, API (including the
swarm and the live engine), the data updater, the dashboard, and Caddy for HTTPS.

```
Internet ──443──▶ Caddy ─┬─ /api/* ──▶ api:8000 ──▶ postgres, redis, ./data (Parquet)
                         └─ /*     ──▶ web:3000      updater: Dukascopy every 6 h
```

Only Caddy listens publicly. `/metrics` is blocked at the proxy, and API docs are off in prod.

## 1. Pick a machine (free options)

| Option | Good for | Notes |
|---|---|---|
| **A computer you already own** (old laptop, mini PC, home server) | Simplest and most private | Dukascopy throttles home connections far less than cloud IPs. To reach it from your phone without opening ports, use [Tailscale](https://tailscale.com) (free personal plan) and skip the domain + Caddy steps. |
| **Oracle Cloud "Always Free"** Ampere (ARM) VM | Always-on, public HTTPS | The free tier has offered ARM VMs with several cores and plenty of RAM. Sign-up needs a card for verification, free capacity is sometimes unavailable in a region, and Oracle may reclaim free instances it considers idle. Check their current terms. |

Minimum: 2 CPU cores, 4 GB RAM, 20 GB disk. More cores make the swarm faster (`--workers`).
All images build natively on x86-64 and ARM64.

## 2. Domain and firewall (public server only)

1. Get a free subdomain at [DuckDNS](https://www.duckdns.org) and point it at the server's
   public IP.
2. Allow inbound **TCP 80 and 443** (plus UDP 443 for HTTP/3) in the cloud firewall. On Oracle,
   that's the VCN security list. Allow them in the OS firewall too. Oracle's Ubuntu images
   ship with restrictive host rules.
3. Keep SSH key-only, and don't open 5432, 6379, 8000 or 3000.

Caddy fetches and renews a Let's Encrypt certificate automatically on first start.

## 3. Install and configure

```bash
# Docker Engine + compose plugin (official convenience script)
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker

git clone <your repo> swarm-trader && cd swarm-trader
cp deploy/.env.prod.example deploy/.env
docker compose -f docker-compose.prod.yml --env-file deploy/.env build
```

Generate the secrets with the image you just built, and paste them into `deploy/.env`:

```bash
docker run --rm -it swarm-trader/api:local swarm hash-password   # → SWARM_AUTH_PASSWORD_HASH
docker run --rm swarm-trader/api:local swarm gen-secret          # → SWARM_SECRET_KEY
openssl rand -hex 24                                              # → POSTGRES_PASSWORD
```

Set `SWARM_DOMAIN` to your DuckDNS name. The API refuses to start in prod with a missing or
example password hash or secret key. Then start the stack:

```bash
make prod-up        # = docker compose -f docker-compose.prod.yml --env-file deploy/.env up -d --build
make prod-logs
```

Open `https://<your-domain>` and sign in. Database migrations run automatically on every API start.

## 4. Load history

The updater only extends existing data (`--start last`), so do the first backfill once:

```bash
docker compose -f docker-compose.prod.yml --env-file deploy/.env run --rm updater \
  swarm ingest dukascopy --start 2016-01-01 --end now --rps 2
```

It's resumable: run it again if it stops. **If Dukascopy throttles the server** (HTTP 429,
very slow progress), run the same command at home with `uv run swarm ingest ...`, copy
`data/raw/dukascopy/` to the server's `./data/raw/dukascopy/`, and run the command above
again. Every file is then a cache hit, so nothing is downloaded.

## 5. Evolve, validate, promote

Start runs from the dashboard (**Lab → New run**) or the CLI:

```bash
docker compose -f docker-compose.prod.yml --env-file deploy/.env exec api \
  swarm evolve --start 2016-01-01 --end 2025-01-01 --timeframe 15m --islands 4 --workers 3 --meta-top 5
```

Leave one core for the API. Promote strategies from their page (**Promote** runs the one-time
holdout test).

## 6. Live signals (optional, free)

1. Open a free **OANDA practice** account and create an API token (Manage API Access).
2. In `deploy/.env`, set `SWARM_OANDA_TOKEN=...` and `SWARM_LIVE_AUTOSTART=true`.
   Optionally set `SWARM_TELEGRAM_BOT_TOKEN` + `SWARM_TELEGRAM_CHAT_ID` or
   `SWARM_NOTIFY_WEBHOOK_URL` for alerts.
3. Restart with `make prod-up`.

The engine starts with the API and runs under a supervisor. After a crash (feed outage,
network loss) it restarts with exponential backoff, from 5 s up to 5 min.
`GET /api/v1/live/status` and the dashboard header show its state.

## 7. Monitoring

- **Health:** `https://<domain>/api/v1/health` (public; checks DB and event bus). A free uptime
  monitor such as UptimeRobot or Healthchecks.io can ping it.
- **Logs:** one JSON line per request, with `request_id`, route, status and `duration_ms`.
  Every response carries `X-Request-ID`, and a 500 returns the same ID so you can grep for it.
  Docker rotates logs at 5 × 10 MB per service.
  ```bash
  docker compose -f docker-compose.prod.yml --env-file deploy/.env logs api | grep <request-id>
  ```
- **Metrics:** Prometheus format at `api:8000/metrics`, reachable only inside the compose network.
  It requires the bearer token when auth is on. Key series:

  | Metric | Meaning |
  |---|---|
  | `swarm_http_requests_total`, `swarm_http_request_duration_seconds` | traffic and latency per route template |
  | `swarm_live_engine_running`, `swarm_live_last_bar_age_seconds` | live engine up, and data freshness (alert if age > 15 min on a weekday) |
  | `swarm_live_engine_restarts_total` | supervisor restarts |
  | `swarm_events_published_total{type=...}` | generations, signals, job progress |
  | `swarm_login_attempts_total{outcome=...}` | failed and locked logins |

  To check them by hand:
  ```bash
  docker compose -f docker-compose.prod.yml --env-file deploy/.env exec api \
    python -c "import urllib.request as u; print(u.urlopen('http://127.0.0.1:8000/metrics').read().decode())" | grep swarm_
  ```
  (With auth on, add an `Authorization: Bearer` header from a login, or scrape with a Prometheus
  `bearer_token`.)

## 8. Backups and updates

```bash
crontab -e
# 17 3 * * * cd /home/ubuntu/swarm-trader && ./deploy/backup.sh >> backups/backup.log 2>&1
```

`deploy/backup.sh` writes a `pg_dump` (custom format) to `backups/` and keeps 14 days. Copy
`backups/` off the machine now and then. Market data under `./data` can always be re-downloaded.

Restore:

```bash
docker compose -f docker-compose.prod.yml --env-file deploy/.env exec -T postgres \
  pg_restore -U swarm -d swarm --clean < backups/swarm-<stamp>.dump
```

Update: `git pull && make prod-up`. Images rebuild, and migrations apply on API start.

## Security notes

- **Single user.**
  - The password is stored only as a scrypt hash. Sessions are HMAC-signed tokens (7 days by
    default, `SWARM_TOKEN_TTL_HOURS`).
  - Changing the password or `SWARM_SECRET_KEY` signs out every session.
  - After 5 failed logins, that client is locked out for 15 minutes.
- **Where the token travels.**
  - The dashboard keeps the token in the browser's local storage and sends it as
    `Authorization: Bearer`.
  - WebSockets can't send headers, so they pass `?token=`. Logs mask it (`token=***`), and the
    API rejects query-string tokens on normal HTTP routes.
- **Exposure.** Only Caddy is exposed, and it adds HSTS. The API and dashboard add `nosniff`,
  `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer` and `Cache-Control: no-store` on API
  responses.
- **No trading access.** The system emits signals only and holds no broker credentials that can
  trade. The OANDA token is for a practice account's price stream.
- **Local compose.** `docker-compose.yml` (local) binds every port to 127.0.0.1 and runs without
  login (`SWARM_AUTH_DISABLED=true`). Never expose it.

## Troubleshooting

| Symptom | Fix |
|---|---|
| API exits with `prod needs SWARM_AUTH_PASSWORD_HASH ...` | Fill in `deploy/.env` (step 3). |
| Caddy can't get a certificate | The DNS name must resolve to this server, and ports 80/443 must be open in both firewalls. |
| Dashboard keeps returning to the login page | Token expired, or the password or secret key changed. Sign in again. |
| `updater` logs `no stored dukascopy M1 bars` | Do the first backfill (step 4). |
| Live engine shows `restarting_in_s` | Check `make prod-logs` for the feed error (token, network). The supervisor retries on its own. |
