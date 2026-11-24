"""Colab A100 GPU runner: fine-tune Qwen3-VL with QLoRA on IndoSTEM-OCR and evaluate Before vs After.

Usage from your terminal:
    colab run --gpu A100 scripts/colab/train_lora.py -- --count 500 --epochs 2
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="https://github.com/nakafaai/pindai.git")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--eval-limit", type=int, default=50)
    args = parser.parse_args()

    workdir = Path("/content/pindai")
    if not workdir.exists():
        run(["git", "clone", "--depth", "1", args.repo, str(workdir)])

    run([sys.executable, "-m", "pip", "install", "-q", "uv"])
    run(
        [
            "uv",
            "pip",
            "install",
            "--system",
            "-q",
            "-e",
            str(workdir),
            "playwright",
            "peft",
            "bitsandbytes",
            "qwen-vl-utils",
            "accelerate",
            "transformers>=4.51.0",
        ]
    )
    run(["playwright", "install", "chromium"])
    run(["playwright", "install-deps", "chromium"])

    data_dir = workdir / "data" / "generated"
    run(["pindai", "generate", "--count", str(args.count), "--render", "-o", str(data_dir)])

    out_root = workdir / "outputs" / "lora-exp"
    ckpt_dir = workdir / "checkpoints" / "pindai-qwen3-vl-8b-lora"

    # 1. Baseline zero-shot evaluation
    run(
        [
            "pindai",
            "eval",
            "--dataset",
            str(data_dir / "dataset.jsonl"),
            "--images",
            str(data_dir / "images"),
            "--model",
            args.model,
            "--4bit",
            "--split",
            "test",
            "--limit",
            str(args.eval_limit),
            "-o",
            str(out_root / "baseline"),
        ]
    )

    # 2. QLoRA fine-tuning
    run(
        [
            "pindai",
            "train",
            "--dataset",
            str(data_dir / "dataset.jsonl"),
            "--images",
            str(data_dir / "images"),
            "--model",
            args.model,
            "--epochs",
            str(args.epochs),
            "-o",
            str(ckpt_dir),
        ]
    )

    # 3. Post-training evaluation on the exact same test split
    run(
        [
            "pindai",
            "eval",
            "--dataset",
            str(data_dir / "dataset.jsonl"),
            "--images",
            str(data_dir / "images"),
            "--model",
            args.model,
            "--adapter",
            str(ckpt_dir),
            "--4bit",
            "--split",
            "test",
            "--limit",
            str(args.eval_limit),
            "-o",
            str(out_root / "lora"),
        ]
    )

    run(["pindai", "leaderboard", str(out_root)])


if __name__ == "__main__":
    main()
