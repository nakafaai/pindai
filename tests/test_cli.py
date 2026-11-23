"""End-to-end smoke tests for the Pindai CLI."""

import json
from pathlib import Path

from typer.testing import CliRunner

from pindai.cli import app

runner = CliRunner()


def test_cli_info_json() -> None:
    result = runner.invoke(app, ["info", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["name"] == "pindai"
    assert len(payload["generators"]) >= 10


def test_cli_generate_and_score(tmp_path: Path) -> None:
    out_dir = tmp_path / "ds"
    gen = runner.invoke(app, ["generate", "-n", "20", "--seed", "7", "-o", str(out_dir), "--json"])
    assert gen.exit_code == 0
    data = json.loads(gen.stdout)
    assert data["manifest"]["total_samples"] == 20
    assert (out_dir / "dataset.jsonl").exists()
    assert (out_dir / "manifest.json").exists()

    ref = tmp_path / "ref.md"
    pred = tmp_path / "pred.md"
    ref.write_text("### Soal 1\n\nNilai $x = 2$.", encoding="utf-8")
    pred.write_text("### Soal 1\n\nNilai $x = 2$.", encoding="utf-8")

    sc = runner.invoke(app, ["score", str(ref), str(pred), "--json"])
    assert sc.exit_code == 0
    metrics = json.loads(sc.stdout)
    assert metrics["cer"] == 0.0
    assert metrics["latex_f1"] == 1.0
