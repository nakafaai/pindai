# AGENTS.md

Instructions for AI coding agents and contributors working on **Pindai** (`nakafaai/pindai`).

## Toolchain

- Use `uv` exclusively for Python dependency management and execution (`uv sync`, `uv run ...`, `uv add ...`). Never use `pip`, `poetry`, or `conda`.
- Python version is pinned in `.python-version` (`3.12`) to match Google Colab Linux runtimes.
- Lint and format with `ruff`; test with `pytest`.

## Architecture

- `src/pindai/dataset/schema.py`: Typed Pydantic v2 contracts for dataset samples, splits, augmentations, and evaluation reports.
- `src/pindai/dataset/templates.py`: Deterministic, SymPy-backed Indonesian STEM problem generators (Mathematics, Physics, Chemistry). Every generated sample must be 100% original and Apache-2.0 / CC-BY-4.0 clean (never copy proprietary content from `aksara`).
- `src/pindai/dataset/augment.py`: Realistic scan and mobile-camera degradation pipeline (Pillow + NumPy).
- `src/pindai/dataset/renderer.py`: High-fidelity exam/worksheet renderer (HTML + KaTeX via Playwright).
- `src/pindai/eval/metrics.py`: Deterministic OCR metrics (CER, normalized edit similarity, LaTeX extraction, LaTeX structural validity, LaTeX F1, structure F1).
- `src/pindai/eval/runner.py`: GPU VLM evaluator for open-weight models (`Qwen/Qwen3-VL-8B-Instruct`, `PaddlePaddle/PaddleOCR-VL-1.6`, LoRA adapters).
- `src/pindai/train/lora.py`: Reproducible QLoRA/LoRA fine-tuning pipeline for Colab GPUs.
- `scripts/colab/`: Self-contained scripts meant to run via `colab run --gpu L4` / `colab run --gpu A100`.

## Validation Commands

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Agent Experience (AX) Rules

- Every CLI command supports machine-readable `--json` output where applicable and exits non-zero on invalid inputs.
- Keep dataset generation strictly deterministic for a given `--seed`.
- Split assignment (`train`, `val`, `test`) must isolate template families or hash deterministically without leakage.
- Never commit generated images, checkpoints, or large datasets to Git; publish datasets and model weights to Hugging Face instead.
