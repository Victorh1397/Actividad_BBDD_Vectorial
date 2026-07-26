.PHONY: setup up down doctor test lint format ingest evaluate deliver clean reset

COMPOSE := docker compose -f deploy/qdrant/compose.yaml

setup:
	uv sync --group dev
	@test -f .env || cp .env.example .env

up:
	$(COMPOSE) up -d --wait

down:
	$(COMPOSE) down

doctor:
	uv run aurum doctor

test:
	uv run pytest

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

ingest:
	uv run aurum ingest --profile full

evaluate:
	uv run aurum evaluate --profile full

# Comando único que regenera todos los artefactos de entrega (RF-28).
deliver:
	uv run aurum deliver

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
	find . -type d -name .ruff_cache -prune -exec rm -rf {} +

# Destructivo: exige AURUM_ALLOW_RESET=true y AURUM_CONFIRM_CLEANUP con el
# nombre exacto de la colección. Nunca se ejecuta por accidente.
reset:
	uv run aurum reset
