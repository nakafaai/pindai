"""Tests for OCR evaluation metrics and leaderboard formatting."""

from pindai.dataset.schema import AugmentationProfile, SampleEvalResult, Split, Subject
from pindai.eval.metrics import (
    aggregate_report,
    character_error_rate,
    format_leaderboard_markdown,
    is_valid_latex,
    latex_f1,
    score_prediction,
    structure_f1,
)


def test_exact_match_scores_perfect() -> None:
    md = "### Soal 1\n\nHitung $\\int_0^2 3x^2 \\, dx = 8$.\n\nA. $4$\nB. $8$"
    scores = score_prediction(md, md)
    assert scores["cer"] == 0.0
    assert scores["normalized_similarity"] == 1.0
    assert scores["latex_f1"] == 1.0
    assert scores["latex_valid_rate"] == 1.0
    assert scores["structure_f1"] == 1.0


def test_latex_validity_catches_unbalanced_braces() -> None:
    assert is_valid_latex(r"\frac{x^2 + 1}{x - 1}")
    assert not is_valid_latex(r"\frac{x^2 + 1}{x - 1")
    assert not is_valid_latex(r"\begin{pmatrix} 1 & 2 \end{bmatrix}")


def test_cer_and_latex_f1_penalize_errors() -> None:
    ref = "Nilai $x_1 + x_2 = 5$ untuk $x^2 - 5x + 6 = 0$."
    pred = "Nilai $x_1 + x_2 = 6$ untuk $x^2 - 5x + 6 = 0$."
    assert character_error_rate(ref, pred) > 0.0
    assert 0.4 <= latex_f1(ref, pred) <= 0.6


def test_structure_f1_counts_headings_options_and_tables() -> None:
    ref = "### Soal 1\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n\nA. $1$\nB. $2$"
    pred = "### Soal 1\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n\nA. $1$"
    score = structure_f1(ref, pred)
    assert 0.8 < score < 1.0


def test_aggregate_report_and_leaderboard() -> None:
    row = SampleEvalResult(
        id="indostem_00001",
        subject=Subject.MATHEMATICS,
        augmentation=AugmentationProfile.MOBILE_PHOTO,
        cer=0.05,
        normalized_similarity=0.95,
        latex_f1=0.9,
        latex_valid_rate=1.0,
        structure_f1=1.0,
        latency_ms=420.0,
        reference_markdown="a",
        predicted_markdown="a",
    )
    report = aggregate_report("Qwen/Qwen3-VL-8B-Instruct", Split.TEST, [row])
    md = format_leaderboard_markdown([report])
    assert "Qwen/Qwen3-VL-8B-Instruct" in md
    assert "0.9000" in md
