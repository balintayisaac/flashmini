.PHONY: install lint typecheck test check clean

install:      ## Create venv and install all dependencies
	uv sync

lint:         ## Ruff lint + format check
	uv run ruff check src tests
	uv run ruff format --check src tests

typecheck:    ## Pyright static type check
	uv run pyright src

test:         ## Run the test suite
	uv run pytest

check: lint typecheck test  ## Everything CI runs, locally

clean:        ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
