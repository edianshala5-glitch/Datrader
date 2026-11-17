# Run from the repo root. Backend commands execute inside backend/ via uv.
BACKEND := cd backend &&

.PHONY: install dev test test-all lint typecheck migrate seed revision up down

install:            ## install backend deps (incl. dev tools)
	$(BACKEND) uv sync

dev: migrate seed   ## run the API with autoreload on :8000
	$(BACKEND) uv run swarm serve --reload

test:               ## fast suite (SQLite + in-memory bus)
	$(BACKEND) uv run pytest -q

test-all:           ## also against Postgres/Redis from docker compose
	$(BACKEND) SWARM_TEST_POSTGRES_URL=postgresql+asyncpg://swarm:swarm@localhost:5432/swarm_test \
	  SWARM_TEST_REDIS_URL=redis://localhost:6379/0 uv run pytest -q

lint:
	$(BACKEND) uv run ruff check . && uv run ruff format --check .

typecheck:
	$(BACKEND) uv run mypy app

migrate:
	$(BACKEND) uv run swarm migrate

seed:
	$(BACKEND) uv run swarm seed

revision:           ## make revision m="add foo"
	$(BACKEND) uv run alembic revision --autogenerate -m "$(m)"

up:
	docker compose up -d --build

down:
	docker compose down
