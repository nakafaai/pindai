"""Colab GPU runner: generate IndoSTEM-OCR, render images, and benchmark baseline VLMs.

Usage from your terminal:
    colab run --gpu L4 scripts/colab/benchmark.py -- --count 60 --limit 15
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
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Qwen/Qwen3-VL-8B-Instruct", "PaddlePaddle/PaddleOCR-VL-1.6"],
    )
    parser.add_argument("--use-4bit", action="store_true")
    args = parser.parse_args()

    workdir = Path("/content/pindai")
    if not workdir.exists():
        run(["git", "clone", "--depth", "1", args.repo, str(workdir)])

    run([sys.executable, "-m", "pip", "install", "-q", "uv"])
    run(["uv", "sync", "--project", str(workdir), "--extra", "render", "--extra", "ml"])
    run(["uv", "run", "--project", str(workdir), "playwright", "install", "chromium"])
    run(["uv", "run", "--project", str(workdir), "playwright", "install-deps", "chromium"])

    data_dir = workdir / "data" / "generated"
    run(
        [
            "uv",
            "run",
            "--project",
            str(workdir),
            "pindai",
            "generate",
            "--count",
            str(args.count),
            "--render",
            "-o",
            str(data_dir),
        ]
    )

    out_root = workdir / "outputs" / "benchmark"
    for model_id in args.models:
        slug = model_id.replace("/", "__")
        cmd = [
            "uv",
            "run",
            "--project",
            str(workdir),
            "pindai",
            "eval",
            "--dataset",
            str(data_dir / "dataset.jsonl"),
            "--images",
            str(data_dir / "images"),
            "--model",
            model_id,
            "--split",
            "test",
            "--limit",
            str(args.limit),
            "-o",
            str(out_root / slug),
        ]
        if args.use_4bit:
            cmd.append("--4bit")
        run(cmd)

    run(["uv", "run", "--project", str(workdir), "pindai", "leaderboard", str(out_root)])


if __name__ == "__main__":
    main()
