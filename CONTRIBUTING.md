# Contributing to Pindai

Thanks for your interest in improving Indonesian STEM OCR and open education benchmarks.

## Local Setup

Pindai uses [uv](https://docs.astral.sh/uv/) for fast, reproducible Python environments:

```bash
git clone https://github.com/nakafaai/pindai.git
cd pindai
uv sync --dev
```

For rendering exam pages locally (requires Playwright Chromium):

```bash
uv sync --dev --extra render
uv run playwright install chromium
```

## Quality Checks

Run all checks before opening a pull request:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Adding New Indonesian STEM Generators

1. Add a generator function in `src/pindai/dataset/templates.py` using SymPy or exact arithmetic so answers and LaTeX are mathematically verified.
2. Register the generator with its `Subject`, `Topic`, and difficulty tags.
3. Add a deterministic unit test in `tests/test_dataset.py` verifying valid Markdown + LaTeX output.
