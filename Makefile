# ================================================================================
# GenAI-RAG-EEG Makefile
# ================================================================================
# One-command workflows for common tasks.
#
# Usage:
#   make install      - Install dependencies
#   make test         - Run all tests
#   make test-smoke   - Run smoke tests only
#   make validate     - Validate setup
#   make demo         - Run demo with sample data
#   make train        - Train on synthetic data
#   make clean        - Clean generated files
#
# Author: GenAI-RAG-EEG Team
# Version: 3.0.0
# ================================================================================

.PHONY: all install test test-smoke test-shapes test-config test-repro validate demo train sample-data clean help

# Default Python
PYTHON ?= python3
PIP ?= pip

# Directories
PROJECT_ROOT := $(shell pwd)
TESTS_DIR := $(PROJECT_ROOT)/tests
RESULTS_DIR := $(PROJECT_ROOT)/results
OUTPUTS_DIR := $(PROJECT_ROOT)/outputs

# ================================================================================
# MAIN TARGETS
# ================================================================================

all: help

help:
	@echo "GenAI-RAG-EEG Makefile"
	@echo "====================="
	@echo ""
	@echo "Setup:"
	@echo "  make install        - Install all dependencies"
	@echo "  make install-dev    - Install with dev dependencies"
	@echo "  make validate       - Validate environment setup"
	@echo ""
	@echo "Testing:"
	@echo "  make test           - Run all tests"
	@echo "  make test-smoke     - Run smoke tests (fast)"
	@echo "  make test-shapes    - Run model shape tests"
	@echo "  make test-config    - Run config tests"
	@echo "  make test-repro     - Run reproducibility tests"
	@echo "  make test-coverage  - Run tests with coverage"
	@echo ""
	@echo "Running:"
	@echo "  make demo           - Run demo with sample data"
	@echo "  make train          - Train on synthetic data"
	@echo "  make sample-data    - Generate sample data"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean          - Remove generated files"
	@echo "  make clean-outputs  - Remove output directories"
	@echo "  make clean-cache    - Remove Python cache files"
	@echo ""

# ================================================================================
# INSTALLATION
# ================================================================================

install:
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt
	@echo "Done. Run 'make validate' to verify setup."

install-dev:
	@echo "Installing dependencies with dev tools..."
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov black isort flake8
	@echo "Done."

# ================================================================================
# VALIDATION
# ================================================================================

validate:
	@echo "Validating environment setup..."
	$(PYTHON) scripts/validate_setup.py
	@echo ""
	@echo "Running quick smoke test..."
	$(PYTHON) -m pytest tests/test_smoke.py::TestImports -v --tb=short
	@echo ""
	@echo "Validation complete!"

check-deps:
	@echo "Checking dependencies..."
	$(PYTHON) -c "import torch; print(f'PyTorch: {torch.__version__}')"
	$(PYTHON) -c "import numpy; print(f'NumPy: {numpy.__version__}')"
	$(PYTHON) -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"
	$(PYTHON) -c "from src.config import Config; print('Config: OK')"
	$(PYTHON) -c "from src.models.genai_rag_eeg import GenAIRAGEEG; print('Model: OK')"

# ================================================================================
# TESTING
# ================================================================================

test:
	@echo "Running all tests..."
	$(PYTHON) -m pytest $(TESTS_DIR) -v --tb=short

test-smoke:
	@echo "Running smoke tests..."
	$(PYTHON) -m pytest $(TESTS_DIR)/test_smoke.py -v --tb=short

test-shapes:
	@echo "Running model shape tests..."
	$(PYTHON) -m pytest $(TESTS_DIR)/test_model_shapes.py -v --tb=short

test-config:
	@echo "Running config tests..."
	$(PYTHON) -m pytest $(TESTS_DIR)/test_config.py -v --tb=short

test-repro:
	@echo "Running reproducibility tests..."
	$(PYTHON) -m pytest $(TESTS_DIR)/test_reproducibility.py -v --tb=short

test-coverage:
	@echo "Running tests with coverage..."
	$(PYTHON) -m pytest $(TESTS_DIR) -v --cov=src --cov-report=term-missing --cov-report=html

test-quick:
	@echo "Running quick tests (imports only)..."
	$(PYTHON) -m pytest $(TESTS_DIR)/test_smoke.py::TestImports -v --tb=short

# ================================================================================
# RUNNING
# ================================================================================

demo:
	@echo "Running demo with sample data..."
	$(PYTHON) main.py --mode demo --verbose

train:
	@echo "Training on synthetic data..."
	$(PYTHON) main.py --mode train --synthetic --verbose

train-sam40:
	@echo "Training on SAM-40 dataset..."
	$(PYTHON) main.py --mode train --dataset sam40 --verbose

train-eegmat:
	@echo "Training on EEGMAT dataset..."
	$(PYTHON) main.py --mode train --dataset eegmat --verbose

train-eegmat:
	@echo "Training on EEGMAT dataset..."
	$(PYTHON) main.py --mode train --dataset eegmat --verbose

sample-data:
	@echo "Generating sample data..."
	$(PYTHON) scripts/generate_sample_data.py

# ================================================================================
# MAINTENANCE
# ================================================================================

clean: clean-cache clean-outputs
	@echo "Clean complete."

clean-cache:
	@echo "Removing Python cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

clean-outputs:
	@echo "Removing output directories..."
	rm -rf $(OUTPUTS_DIR)/* 2>/dev/null || true
	rm -rf .coverage htmlcov 2>/dev/null || true

clean-results:
	@echo "WARNING: This will remove all results!"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] && rm -rf $(RESULTS_DIR)/*

# ================================================================================
# CODE QUALITY
# ================================================================================

lint:
	@echo "Running linters..."
	$(PYTHON) -m flake8 src/ tests/ --max-line-length=120 --ignore=E501,W503

format:
	@echo "Formatting code..."
	$(PYTHON) -m black src/ tests/ --line-length=120
	$(PYTHON) -m isort src/ tests/

format-check:
	@echo "Checking code format..."
	$(PYTHON) -m black src/ tests/ --line-length=120 --check
	$(PYTHON) -m isort src/ tests/ --check-only

# ================================================================================
# DOCKER (if needed)
# ================================================================================

docker-build:
	@echo "Building Docker image..."
	docker build -t genai-rag-eeg:latest .

docker-run:
	@echo "Running in Docker..."
	docker run --rm -it genai-rag-eeg:latest

# ================================================================================
# DEVELOPMENT
# ================================================================================

dev-setup: install-dev
	@echo "Setting up development environment..."
	pre-commit install 2>/dev/null || echo "pre-commit not available"

watch-tests:
	@echo "Watching for changes and running tests..."
	$(PYTHON) -m pytest_watch -- -v --tb=short

# ================================================================================
# PAPER/DOCS
# ================================================================================

compile-paper:
	@echo "Compiling LaTeX paper..."
	cd paper && pdflatex genai_rag_eeg_v4.tex && bibtex genai_rag_eeg_v4 && pdflatex genai_rag_eeg_v4.tex && pdflatex genai_rag_eeg_v4.tex
