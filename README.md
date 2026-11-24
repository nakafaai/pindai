# Pindai

[![CI](https://github.com/nakafaai/pindai/actions/workflows/ci.yml/badge.svg)](https://github.com/nakafaai/pindai/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)](.python-version)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://docs.astral.sh/uv/)

**Open-source Indonesian STEM OCR dataset generator, VLM benchmark, and LoRA fine-tuning toolkit by [Nakafa](https://nakafa.com).**

Most open-source math OCR datasets focus on isolated English formulas. Real Indonesian exam sheets, worksheets, and textbooks mix Bahasa Indonesia prose, Kurikulum Merdeka / UTBK-SNBT multiple-choice layouts (`A`-`E`), Markdown tables, SI units, chemical reactions, and mobile-camera scan noise.

**Pindai** closes that gap with a fully reproducible pipeline:

1. **Synthetic-realistic dataset (`IndoSTEM-OCR`):** Deterministic, SymPy-verified Indonesian Mathematics, Physics, and Chemistry problems rendered with KaTeX + realistic scan/mobile-photo degradations (`clean`, `scanned_pdf`, `mobile_photo`, `low_dpi`).
2. **Objective OCR benchmark:** Evaluates open-weight Vision-Language Models (`Qwen/Qwen3-VL-8B-Instruct`, `PaddlePaddle/PaddleOCR-VL-1.6`, and LoRA adapters) on CER, normalized similarity, LaTeX F1, structural LaTeX validity, and Markdown structure F1.
3. **1-command Colab GPU execution:** Reproducible benchmark and QLoRA fine-tuning scripts built for `google-colab-cli` (`colab run --gpu L4` / `--gpu A100`).

---

## Quick Start

Pindai uses [uv](https://docs.astral.sh/uv/) for fast, lockfile-pinned Python environments.

```bash
git clone https://github.com/nakafaai/pindai.git
cd pindai
uv sync --dev
```

Inspect registered generators and generate a deterministic dataset:

```bash
uv run pindai info
uv run pindai generate --count 300 --seed 42 -o data/generated
```

To also render augmented PNG exam sheets locally (requires Playwright Chromium):

```bash
uv sync --dev --extra render
uv run playwright install chromium
uv run pindai generate --count 100 --render -o data/generated
```

Every command supports `--json` for machine-readable agent workflows:

```bash
uv run pindai info --json
uv run pindai score reference.md predicted.md --json
```

---

## Reproducing Benchmarks & LoRA Training on Google Colab GPU

With [`google-colab-cli`](https://github.com/googlecolab/google-colab-cli) installed, you can run the full GPU benchmark or QLoRA fine-tuning from your terminal:

```bash
# Zero-shot benchmark (Qwen3-VL-8B vs PaddleOCR-VL-1.6) on L4 GPU
colab run --gpu L4 scripts/colab/benchmark.py -- --count 100 --limit 20 --use-4bit

# Full QLoRA fine-tuning + Before/After evaluation on A100 GPU
colab run --gpu A100 scripts/colab/train_lora.py -- --count 500 --epochs 2 --eval-limit 50
```

---

## Evaluation Metrics

Pindai avoids subjective "vibe checks" by scoring every prediction against five deterministic metrics:

| Metric | Direction | What it measures |
| --- | :---: | --- |
| **CER** | ↓ | Character Error Rate over normalized Markdown |
| **Similarity** | ↑ | Bounded `[0, 1]` edit similarity |
| **LaTeX F1** | ↑ | Multiset F1 over canonicalized inline & display LaTeX expressions |
| **LaTeX Valid** | ↑ | Fraction of predicted LaTeX blocks with balanced braces & environments |
| **Structure F1** | ↑ | Preservation of headings, `A`-`E` options, and Markdown tables |

---

## Sample Rendered Sheets (`IndoSTEM-OCR`)

Pindai renders SymPy-verified Indonesian STEM problems via KaTeX across four layout templates and four degradation profiles (`clean`, `scanned_pdf`, `mobile_photo`, `low_dpi`):

| Clean Exam Sheet | Mobile Camera Photo | Scanned Table + Formula |
| :---: | :---: | :---: |
| ![Clean Math](docs/assets/sample-clean-math.png) | ![Mobile Photo](docs/assets/sample-mobile-photo.png) | ![Scanned Table](docs/assets/sample-table-scan.png) |

---

## Benchmark Results (`IndoSTEM-OCR` v0.1, Colab A100 40GB)

Evaluated on the held-out `test` split (`n = 22`, seed `42`, balanced across `matematika`, `fisika`, `kimia` and all four degradation profiles). Saved reports live in [`results/`](results/).

| Model | Mode | CER (↓) | Similarity (↑) | LaTeX F1 (↑) | LaTeX Valid (↑) | Structure F1 (↑) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3-VL-4B-Instruct` | zero-shot | `0.3943` | `0.6718` | `0.4807` | `1.0000` | `0.8210` |
| `Qwen/Qwen3-VL-8B-Instruct` (4-bit) | zero-shot | `0.3114` | `0.7399` | `0.5459` | `1.0000` | `0.8467` |
| **`Qwen/Qwen3-VL-8B-Instruct` (4-bit)** | **`pindai-lora` (2 ep, 176 train)** | **`0.0667`** | **`0.9352`** | **`0.6683`** | **`1.0000`** | **`1.0000`** |

### Breakdown by Subject (`Qwen3-VL-8B` Zero-Shot vs `pindai-lora`)

| Subject | Zero-Shot CER | LoRA CER | Zero-Shot LaTeX F1 | LoRA LaTeX F1 | LoRA Structure F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Matematika** | `0.3538` | **`0.0495`** | `0.6450` | **`0.8315`** | **`1.0000`** |
| **Kimia** | `0.2497` | **`0.0868`** | `0.4095` | **`0.5972`** | **`1.0000`** |
| **Fisika** | `0.2677` | **`0.0878`** | `0.4385` | `0.3894` | **`1.0000`** |

**Key Takeaways:**
- Just **2 epochs of QLoRA on 176 synthetic samples** cut overall CER by **78.6%** (`0.3114` -> `0.0667`), lifted normalized similarity to **93.52%**, raised Mathematics LaTeX F1 to **0.8315**, and achieved **100% Structure F1** across all subjects and scan/photo degradations (including `mobile_photo` CER `0.0601`).
- **Open opportunity (Physics units):** Physics CER dropped sharply (`0.2677` -> `0.0878`), while strict LaTeX F1 stayed lower due to equivalent unit notation variants (e.g. `\text{ m/s}` vs `\mathrm{m/s}` or plain units outside `$...$`). Expanding unit canonicalization and training scale (`1,000+` samples) is the next milestone.

Regenerate this table locally from any `report.json` folder:

```bash
uv run pindai leaderboard results/
```

---

## Project Structure

```text
pindai/
├── src/pindai/
│   ├── cli.py                # Typer CLI with --json AX support
│   ├── dataset/
│   │   ├── schema.py         # Pydantic v2 contracts
│   │   ├── templates.py      # SymPy-verified Indonesian STEM generators
│   │   ├── augment.py        # Scan & mobile-camera degradation profiles
│   │   └── renderer.py       # Playwright + KaTeX exam sheet renderer
│   ├── eval/
│   │   ├── metrics.py        # CER, LaTeX F1, validity, structure F1
│   │   └── runner.py         # GPU VLM benchmark runner
│   └── train/
│       └── lora.py           # Reproducible QLoRA/LoRA fine-tuning
├── scripts/colab/            # 1-command Colab GPU runners
└── tests/                    # Pytest suite
```

---

## License

Code is released under the [Apache 2.0 License](LICENSE). Synthetic datasets generated by Pindai are original, programmatically generated works suitable for CC-BY-4.0 open release.
