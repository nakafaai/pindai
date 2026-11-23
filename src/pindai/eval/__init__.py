"""OCR evaluation metrics and VLM benchmark runner."""

from pindai.eval.metrics import (
    aggregate_report,
    character_error_rate,
    extract_latex_blocks,
    format_leaderboard_markdown,
    is_valid_latex,
    latex_f1,
    normalized_edit_similarity,
    score_prediction,
    structure_f1,
)

__all__ = [
    "aggregate_report",
    "character_error_rate",
    "extract_latex_blocks",
    "format_leaderboard_markdown",
    "is_valid_latex",
    "latex_f1",
    "normalized_edit_similarity",
    "score_prediction",
    "structure_f1",
]
