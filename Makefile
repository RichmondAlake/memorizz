.PHONY: help install lint format check test clean docs-serve docs-build

help:
	@echo "Memorizz Development Commands:"
	@echo ""
	@echo "  make install        Install package in editable mode with dev dependencies"
	@echo "  make lint           Run linting (flake8, check syntax)"
	@echo "  make format         Format code with black and isort"
	@echo "  make check          Run lint + format check (pre-commit)"
	@echo "  make test           Run tests"
	@echo "  make docs-serve     Launch mkdocs with hot reload"
	@echo "  make docs-build     Build the static documentation site"
	@echo "  make clean          Clean up generated files"
	@echo ""

install:
	pip install -e ".[dev]"
	pip install pre-commit black flake8 isort
	pre-commit install

lint:
	@echo "Running syntax check..."
	@find src/memorizz -name "*.py" -exec python -m py_compile {} \;
	@echo "✓ Syntax check passed"
	@echo ""
	@echo "Running flake8 (Python errors are enforced; style is advisory)..."
	@flake8 src/memorizz --select=E9,F
	@flake8 src/memorizz --max-line-length=120 --extend-ignore=E203,E501 || true
	@echo ""

format:
	@echo "Formatting with black..."
	@black src/memorizz
	@echo ""
	@echo "Sorting imports with isort..."
	@isort src/memorizz --profile black
	@echo ""
	@echo "✓ Code formatted"

check:
	@echo "Running pre-commit checks..."
	@pre-commit run --all-files || true

test:
	pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

docs-serve:
	mkdocs serve

docs-build:
	mkdocs build --strict
