"""Prometheus metrics (``GET /metrics``). Process-local, default registry."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info

from app import __version__

BUILD = Info("swarm_build", "Build information")
BUILD.info({"version": __version__})

HTTP_REQUESTS = Counter("swarm_http_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_LATENCY = Histogram(
    "swarm_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)
WS_CONNECTIONS = Gauge("swarm_ws_connections", "Open dashboard WebSocket streams", ["channel"])
EVENTS = Counter(
    "swarm_events_published_total",
    "Events published on the bus (generations, signals, job progress, ...)",
    ["channel", "type"],
)
LOGIN_ATTEMPTS = Counter("swarm_login_attempts_total", "Login attempts", ["outcome"])

LIVE_RUNNING = Gauge("swarm_live_engine_running", "1 while the live signal engine runs")
LIVE_RESTARTS = Counter("swarm_live_engine_restarts_total", "Supervisor restarts after a crash")
LIVE_BARS = Gauge("swarm_live_bars_processed", "Bars processed by the current live engine")
LIVE_LAST_BAR_AGE = Gauge(
    "swarm_live_last_bar_age_seconds", "Seconds since the live engine's last completed bar"
)
RUNS_ACTIVE = Gauge("swarm_evolution_runs_active", "Evolution runs executing in this process")
