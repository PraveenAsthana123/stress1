# Contributing to GenAI-RAG-EEG

Thank you for your interest in contributing to GenAI-RAG-EEG!

## How to Contribute

### Reporting Bugs

1. Check existing issues first
2. Use the bug report template
3. Include:
   - Python version
   - PyTorch version
   - OS (Windows/Linux/Mac)
   - Full error traceback
   - Steps to reproduce

### Suggesting Features

1. Open an issue with `[Feature]` prefix
2. Describe the use case
3. Provide examples if possible

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Run linting: `black src/ && flake8 src/`
6. Commit with descriptive message
7. Push and create PR

### Code Style

- Use [Black](https://black.readthedocs.io/) for formatting
- Use [Flake8](https://flake8.pycqa.org/) for linting
- Type hints encouraged
- Docstrings for public functions (Google style)

### Testing

- Add tests for new features
- Maintain >80% coverage
- Run full test suite before PR

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_models.py -v
```

### Commit Messages

Format: `<type>: <description>`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance

Example: `feat: add EEGMAT dataset support`

## Development Setup

```bash
# Clone
git clone https://github.com/PraveenAsthana123/stress.git
cd stress

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dev dependencies
pip install -r requirements.txt
pip install black flake8 pytest pytest-cov

# Run tests
pytest tests/ -v
```

## Questions?

Open an issue or contact the maintainers.
