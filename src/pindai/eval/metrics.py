"""Deterministic OCR evaluation metrics for Indonesian STEM Markdown + LaTeX."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from pindai.dataset.schema import EvalReport, SampleEvalResult, Split

_MATH_PATTERN = re.compile(
    r"\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|\$([^$\n]+?)\$"
)
_BEGIN_PATTERN = re.compile(r"\\begin\{([^}]+)\}")
_END_PATTERN = re.compile(r"\\end\{([^}]+)\}")
_OPTION_PATTERN = re.compile(r"^[A-E]\.\s+", re.MULTILINE)
_HEADING_PATTERN = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_TABLE_ROW_PATTERN = re.compile(r"^\|.+\|$", re.MULTILINE)


def normalize_text(text: str) -> str:
    """Normalize whitespace and line endings for fair OCR comparison."""
    text = text.replace("\r\n", "\n").strip()
    return re.sub(r"[ \t]+", " ", text)


def normalize_latex(expr: str) -> str:
    """Canonicalize minor LaTeX formatting differences without altering math meaning."""
    expr = expr.strip()
    expr = re.sub(r"\\left\s*", "", expr)
    expr = re.sub(r"\\right\s*", "", expr)
    expr = re.sub(r"\\[,;:!]", "", expr)
    expr = re.sub(r"\s+", "", expr)
    return expr


def levenshtein_distance(a: str, b: str) -> int:
    """Compute exact Levenshtein edit distance in O(min(len(a), len(b))) memory."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ch_a in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, ch_b in enumerate(b, start=1):
            cost = 0 if ch_a == ch_b else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def character_error_rate(reference: str, predicted: str) -> float:
    """Character Error Rate (lower is better; 0.0 is an exact match)."""
    ref = normalize_text(reference)
    pred = normalize_text(predicted)
    if not ref:
        return 0.0 if not pred else 1.0
    return levenshtein_distance(ref, pred) / len(ref)


def normalized_edit_similarity(reference: str, predicted: str) -> float:
    """Bounded [0, 1] similarity derived from edit distance (higher is better)."""
    ref = normalize_text(reference)
    pred = normalize_text(predicted)
    denom = max(len(ref), len(pred))
    if denom == 0:
        return 1.0
    return max(0.0, 1.0 - levenshtein_distance(ref, pred) / denom)


def extract_latex_blocks(markdown: str) -> list[str]:
    """Extract all inline and display LaTeX expressions from Markdown."""
    blocks: list[str] = []
    for match in _MATH_PATTERN.finditer(markdown):
        content = next((g for g in match.groups() if g is not None), "")
        stripped = content.strip()
        if stripped:
            blocks.append(stripped)
    return blocks


def is_valid_latex(expr: str) -> bool:
    """Check structural LaTeX validity (balanced braces and begin/end environments)."""
    stripped = expr.strip()
    if not stripped:
        return False

    depth = 0
    escaped = False
    for ch in stripped:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
    if depth != 0 or escaped:
        return False

    begins = _BEGIN_PATTERN.findall(stripped)
    ends = _END_PATTERN.findall(stripped)
    return begins == list(reversed(ends)) or Counter(begins) == Counter(ends)


def latex_f1(reference: str, predicted: str) -> float:
    """Multiset F1 over normalized LaTeX expressions extracted from Markdown."""
    ref_blocks = [normalize_latex(b) for b in extract_latex_blocks(reference)]
    pred_blocks = [normalize_latex(b) for b in extract_latex_blocks(predicted)]
    if not ref_blocks and not pred_blocks:
        return 1.0
    if not ref_blocks or not pred_blocks:
        return 0.0

    ref_counts = Counter(ref_blocks)
    pred_counts = Counter(pred_blocks)
    overlap = sum((ref_counts & pred_counts).values())
    precision = overlap / len(pred_blocks)
    recall = overlap / len(ref_blocks)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def latex_valid_rate(predicted: str) -> float:
    """Fraction of extracted LaTeX expressions in prediction that pass structural validation."""
    blocks = extract_latex_blocks(predicted)
    if not blocks:
        return 1.0
    valid = sum(1 for b in blocks if is_valid_latex(b))
    return valid / len(blocks)


def _structure_tokens(markdown: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    counts["heading"] = len(_HEADING_PATTERN.findall(markdown))
    counts["option"] = len(_OPTION_PATTERN.findall(markdown))
    counts["table_row"] = len(_TABLE_ROW_PATTERN.findall(markdown))
    return counts


def structure_f1(reference: str, predicted: str) -> float:
    """F1 score over structural Markdown elements (headings, A-E options, table rows)."""
    ref = _structure_tokens(reference)
    pred = _structure_tokens(predicted)
    ref_total = sum(ref.values())
    pred_total = sum(pred.values())
    if ref_total == 0 and pred_total == 0:
        return 1.0
    if ref_total == 0 or pred_total == 0:
        return 0.0
    overlap = sum((ref & pred).values())
    precision = overlap / pred_total
    recall = overlap / ref_total
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_prediction(reference: str, predicted: str) -> dict[str, float]:
    """Compute all OCR metrics for a single reference/prediction pair."""
    return {
        "cer": round(character_error_rate(reference, predicted), 4),
        "normalized_similarity": round(normalized_edit_similarity(reference, predicted), 4),
        "latex_f1": round(latex_f1(reference, predicted), 4),
        "latex_valid_rate": round(latex_valid_rate(predicted), 4),
        "structure_f1": round(structure_f1(reference, predicted), 4),
    }


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def aggregate_report(
    model_id: str,
    split: Split,
    results: list[SampleEvalResult],
    adapter_id: str | None = None,
) -> EvalReport:
    """Aggregate per-sample OCR evaluation results into a structured EvalReport."""
    by_subject_rows: dict[str, list[SampleEvalResult]] = defaultdict(list)
    by_aug_rows: dict[str, list[SampleEvalResult]] = defaultdict(list)
    for row in results:
        by_subject_rows[row.subject.value].append(row)
        by_aug_rows[row.augmentation.value].append(row)

    def summarize(rows: list[SampleEvalResult]) -> dict[str, float]:
        return {
            "count": float(len(rows)),
            "cer": _mean([r.cer for r in rows]),
            "similarity": _mean([r.normalized_similarity for r in rows]),
            "latex_f1": _mean([r.latex_f1 for r in rows]),
            "latex_valid_rate": _mean([r.latex_valid_rate for r in rows]),
            "structure_f1": _mean([r.structure_f1 for r in rows]),
        }

    return EvalReport(
        model_id=model_id,
        adapter_id=adapter_id,
        split=split,
        num_samples=len(results),
        mean_cer=_mean([r.cer for r in results]),
        mean_similarity=_mean([r.normalized_similarity for r in results]),
        mean_latex_f1=_mean([r.latex_f1 for r in results]),
        mean_latex_valid_rate=_mean([r.latex_valid_rate for r in results]),
        mean_structure_f1=_mean([r.structure_f1 for r in results]),
        mean_latency_ms=_mean([r.latency_ms for r in results]),
        by_subject={k: summarize(v) for k, v in sorted(by_subject_rows.items())},
        by_augmentation={k: summarize(v) for k, v in sorted(by_aug_rows.items())},
    )


def format_leaderboard_markdown(reports: list[EvalReport]) -> str:
    """Render a sorted Markdown leaderboard table from one or more EvalReports."""
    ordered = sorted(reports, key=lambda r: (-r.mean_latex_f1, r.mean_cer))
    lines = [
        "| Model | Adapter | Samples | CER (↓) | Similarity (↑) | LaTeX F1 (↑) | LaTeX Valid (↑) | Structure F1 (↑) | Latency ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in ordered:
        adapter = r.adapter_id or "zero-shot"
        lines.append(
            f"| `{r.model_id}` | `{adapter}` | {r.num_samples} | "
            f"{r.mean_cer:.4f} | {r.mean_similarity:.4f} | {r.mean_latex_f1:.4f} | "
            f"{r.mean_latex_valid_rate:.4f} | {r.mean_structure_f1:.4f} | {r.mean_latency_ms:.1f} |"
        )
    return "\n".join(lines)
