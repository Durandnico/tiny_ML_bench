# ============================================
# Makefile - Common Development Commands
# ============================================
.PHONY: help install install-dev test lint format clean build upload docs

help:
	@echo "Available commands:"
	@echo "  make install       Install the package"
	@echo "  make install-dev   Install package with dev dependencies"
	@echo "  make test          Run tests with pytest"
	@echo "  make lint          Run linting checks"
	@echo "  make format        Format code with black and isort"
	@echo "  make clean         Clean build artifacts"
	@echo "  make build         Build the package"
	@echo "  make docs          Build documentation"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=mlbench --cov-report=html --cov-report=term

test-fast:
	pytest tests/ -v -x

lint:
	flake8 mlbench tests
	mypy mlbench
	pylint mlbench

format:
	black mlbench tests tutorials
	isort mlbench tests tutorials

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build htmlcov

upload: build
	twine upload dist/*

docs:
	cd docs && make html

.DEFAULT_GOAL := help
