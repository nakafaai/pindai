"""Tests for deterministic Indonesian STEM dataset generation."""

from pindai.dataset.renderer import build_sample_html
from pindai.dataset.schema import FormatKind, Split, Subject
from pindai.dataset.templates import GENERATOR_REGISTRY, generate_dataset
from pindai.eval.metrics import extract_latex_blocks, is_valid_latex


def test_generate_dataset_is_deterministic() -> None:
    a = generate_dataset(count=44, seed=123)
    b = generate_dataset(count=44, seed=123)
    assert [s.model_dump() for s in a] == [s.model_dump() for s in b]


def test_generate_dataset_covers_all_subjects_and_splits() -> None:
    samples = generate_dataset(count=60, seed=42)
    subjects = {s.subject for s in samples}
    splits = {s.split for s in samples}
    assert subjects == {Subject.MATHEMATICS, Subject.PHYSICS, Subject.CHEMISTRY}
    assert splits == {Split.TRAIN, Split.VAL, Split.TEST}


def test_all_generated_latex_is_structurally_valid() -> None:
    samples = generate_dataset(count=len(GENERATOR_REGISTRY) * 4, seed=7)
    for sample in samples:
        blocks = extract_latex_blocks(sample.markdown)
        assert blocks, f"Expected LaTeX blocks in {sample.template_id}"
        for block in blocks:
            assert is_valid_latex(block), f"Invalid LaTeX in {sample.template_id}: {block}"


def test_renderer_html_includes_body_and_tables_without_external_markdown_js() -> None:
    samples = generate_dataset(count=len(GENERATOR_REGISTRY), seed=42)
    for sample in samples:
        page_html = build_sample_html(sample)
        assert "<h3>" in page_html
        assert "marked.min.js" not in page_html
        if sample.format_kind == FormatKind.TABLE_PROBLEM:
            assert "<table>" in page_html
