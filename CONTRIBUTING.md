# Contributing to LARSA

Thank you for your interest in contributing. Please follow the guidelines below.

---

## Prerequisites

- Python 3.10, 3.11, or 3.12
- pip >= 23
- git

---

## Setup

```bash
git clone https://github.com/mattbusel/FinRL_DeepSeek_Crypto_Trading.git
cd FinRL_DeepSeek_Crypto_Trading

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and fill in your `DEEPSEEK_API_KEY` before running
any signal-extraction code.

---

## Running Tests

Run the full test suite with coverage:

```bash
pytest tests/ --cov=. --cov-report=term-missing --cov-fail-under=60 -v
```

Run a specific test file:

```bash
pytest tests/test_config.py -v
```

---

## Code Style

This project enforces style via **ruff** and type correctness via **mypy**.

Check and auto-fix style issues:

```bash
ruff check . --fix
ruff format .
```

Run the type checker:

```bash
mypy config.py exceptions.py metrics.py logger.py \
     deepseek_signals.py task1_ensemble.py task1_eval.py
```

All public functions and classes must have Google-style docstrings. New code
must not introduce `# type: ignore` suppressions without an explanatory comment.

---

## Submitting Pull Requests

1. Create a feature branch from `main`: `git checkout -b feat/my-feature`
2. Make your changes; ensure `pytest`, `ruff check`, and `mypy` all pass locally.
3. Keep commits atomic and write clear commit messages (imperative mood).
4. Open a PR against `main` with a description of what changed and why.
5. Address any CI failures or review comments before requesting a re-review.

PRs that break existing tests or lower coverage below 60 % will not be merged.
