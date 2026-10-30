.PHONY: install lint typecheck test smoke check clean

install:      ## Create venv and install all dependencies
	uv sync

lint:         ## Ruff lint + format check
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

typecheck:    ## Pyright static type check
	uv run pyright src scripts

test:         ## Run the test suite
	uv run pytest

smoke:        ## Dogfood all tools against this repo (tool-layer smoke test)
	uv run python scripts/smoke.py

check: lint typecheck test  ## Everything CI runs, locally

clean:        ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
