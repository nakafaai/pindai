"""Typer CLI for Pindai with structured --json output for developer and agent workflows."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from pindai import __version__
from pindai.dataset.renderer import render_samples
from pindai.dataset.schema import DatasetManifest, EvalReport, Split
from pindai.dataset.templates import GENERATOR_REGISTRY, generate_dataset
from pindai.eval.metrics import format_leaderboard_markdown, score_prediction
from pindai.eval.runner import evaluate_vlm, load_samples_jsonl
from pindai.train.lora import TrainConfig, train_lora

app = typer.Typer(
    name="pindai",
    help="Indonesian STEM OCR dataset generator, VLM benchmark, and LoRA toolkit by Nakafa.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def info(
    as_json: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """Show package version, registered generators, and supported models."""
    payload = {
        "name": "pindai",
        "version": __version__,
        "generators": [
            {
                "template_id": g.template_id,
                "subject": g.subject.value,
                "topic": g.topic,
                "format": g.format_kind.value,
            }
            for g in GENERATOR_REGISTRY
        ],
        "recommended_models": [
            "Qwen/Qwen3-VL-8B-Instruct",
            "Qwen/Qwen3-VL-4B-Instruct",
            "PaddlePaddle/PaddleOCR-VL-1.6",
        ],
    }
    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    table = Table(title=f"Pindai v{__version__}: Registered Indonesian STEM Generators")
    table.add_column("Template ID", style="cyan")
    table.add_column("Subject")
    table.add_column("Topic")
    table.add_column("Format")
    for g in GENERATOR_REGISTRY:
        table.add_row(g.template_id, g.subject.value, g.topic, g.format_kind.value)
    console.print(table)


@app.command()
def generate(
    output: Annotated[Path, typer.Option("-o", "--output", help="Output directory.")] = Path(
        "data/generated"
    ),
    count: Annotated[int, typer.Option("-n", "--count", help="Total number of samples.")] = 300,
    seed: Annotated[int, typer.Option("--seed", help="Random seed for determinism.")] = 42,
    render: Annotated[
        bool, typer.Option("--render", help="Render PNG images with Playwright + KaTeX.")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Emit manifest as JSON.")] = False,
) -> None:
    """Generate deterministic IndoSTEM-OCR dataset (JSONL + optional augmented PNG renders)."""
    samples = generate_dataset(count=count, seed=seed)
    output.mkdir(parents=True, exist_ok=True)

    if render:
        samples = render_samples(samples, output_dir=output / "images")

    jsonl_path = output / "dataset.jsonl"
    jsonl_path.write_text(
        "\n".join(s.model_dump_json() for s in samples) + "\n",
        encoding="utf-8",
    )

    manifest = DatasetManifest(
        seed=seed,
        total_samples=len(samples),
        counts_by_split=dict(Counter(s.split.value for s in samples)),
        counts_by_subject=dict(Counter(s.subject.value for s in samples)),
        counts_by_augmentation=dict(Counter(s.augmentation.value for s in samples)),
    )
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(), indent=2) + "\n", encoding="utf-8")

    if as_json:
        typer.echo(
            json.dumps({"jsonl": str(jsonl_path), "manifest": manifest.model_dump()}, indent=2)
        )
        return

    console.print(f"[green]Generated {len(samples)} samples[/green] -> {jsonl_path}")
    console.print(f"Splits: {manifest.counts_by_split} | Subjects: {manifest.counts_by_subject}")


@app.command()
def score(
    reference: Annotated[Path, typer.Argument(help="Reference Markdown file.")],
    predicted: Annotated[Path, typer.Argument(help="Predicted Markdown file.")],
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON metrics.")] = False,
) -> None:
    """Score a single predicted Markdown file against its reference."""
    metrics = score_prediction(
        reference.read_text(encoding="utf-8"),
        predicted.read_text(encoding="utf-8"),
    )
    if as_json:
        typer.echo(json.dumps(metrics, indent=2))
        return
    for key, val in metrics.items():
        console.print(f"{key}: {val:.4f}")


@app.command(name="eval")
def evaluate(
    dataset: Annotated[Path, typer.Option("--dataset", help="Path to dataset.jsonl.")] = Path(
        "data/generated/dataset.jsonl"
    ),
    images: Annotated[
        Path, typer.Option("--images", help="Root directory of rendered images.")
    ] = Path("data/generated/images"),
    model: Annotated[
        str, typer.Option("--model", help="Hugging Face model ID.")
    ] = "Qwen/Qwen3-VL-8B-Instruct",
    adapter: Annotated[
        str | None, typer.Option("--adapter", help="Optional LoRA adapter path or HF ID.")
    ] = None,
    split: Annotated[
        Split, typer.Option("--split", help="Dataset split to evaluate.")
    ] = Split.TEST,
    limit: Annotated[
        int | None, typer.Option("--limit", help="Optional max samples to evaluate.")
    ] = None,
    load_in_4bit: Annotated[
        bool, typer.Option("--4bit", help="Load base model in 4-bit NF4.")
    ] = False,
    output: Annotated[
        Path, typer.Option("-o", "--output", help="Directory for evaluation artifacts.")
    ] = Path("outputs/eval"),
    as_json: Annotated[bool, typer.Option("--json", help="Emit report JSON.")] = False,
) -> None:
    """Evaluate a Vision-Language Model on an IndoSTEM-OCR split (GPU required)."""
    samples = load_samples_jsonl(dataset, split=split)
    if limit is not None:
        samples = samples[:limit]
    report = evaluate_vlm(
        samples=samples,
        image_root=images,
        model_id=model,
        split=split,
        output_dir=output,
        adapter_id=adapter,
        load_in_4bit=load_in_4bit,
    )
    if as_json:
        typer.echo(json.dumps(report.model_dump(), indent=2))
        return
    console.print(format_leaderboard_markdown([report]))


@app.command()
def leaderboard(
    reports_dir: Annotated[Path, typer.Argument(help="Directory containing */report.json files.")],
) -> None:
    """Render a Markdown leaderboard table from saved evaluation report.json files."""
    reports = [
        EvalReport.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(reports_dir.rglob("report.json"))
    ]
    if not reports:
        raise typer.BadParameter(f"No report.json files found under {reports_dir}")
    typer.echo(format_leaderboard_markdown(reports))


@app.command()
def train(
    dataset: Annotated[Path, typer.Option("--dataset", help="Path to dataset.jsonl.")] = Path(
        "data/generated/dataset.jsonl"
    ),
    images: Annotated[
        Path, typer.Option("--images", help="Root directory of rendered images.")
    ] = Path("data/generated/images"),
    model: Annotated[
        str, typer.Option("--model", help="Base Hugging Face VLM ID.")
    ] = "Qwen/Qwen3-VL-8B-Instruct",
    output: Annotated[
        Path, typer.Option("-o", "--output", help="Checkpoint output directory.")
    ] = Path("checkpoints/pindai-qwen3-vl-8b-lora"),
    epochs: Annotated[int, typer.Option("--epochs", help="Number of training epochs.")] = 2,
    lr: Annotated[float, typer.Option("--lr", help="Learning rate.")] = 2e-4,
    limit: Annotated[
        int | None, typer.Option("--limit", help="Optional max training samples.")
    ] = None,
    load_in_4bit: Annotated[
        bool, typer.Option("--4bit/--no-4bit", help="Use QLoRA 4-bit quantization.")
    ] = True,
) -> None:
    """Fine-tune a VLM with LoRA/QLoRA on IndoSTEM-OCR (GPU required)."""
    cfg = TrainConfig(
        model_id=model,
        output_dir=str(output),
        epochs=epochs,
        learning_rate=lr,
        load_in_4bit=load_in_4bit,
    )
    saved_dir = train_lora(
        dataset_jsonl=dataset, image_root=images, config=cfg, max_train_samples=limit
    )
    console.print(f"[green]Saved LoRA adapter[/green] -> {saved_dir}")


if __name__ == "__main__":
    app()
