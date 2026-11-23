"""Typed data contracts for IndoSTEM-OCR dataset samples and evaluation reports."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Subject(StrEnum):
    MATHEMATICS = "matematika"
    PHYSICS = "fisika"
    CHEMISTRY = "kimia"


class FormatKind(StrEnum):
    MULTIPLE_CHOICE = "pilihan_ganda"
    ESSAY = "uraian"
    TABLE_PROBLEM = "tabel_analisis"


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class LayoutStyle(StrEnum):
    EXAM_SHEET = "exam_sheet"
    WORKSHEET_BOXED = "worksheet_boxed"
    TEXTBOOK_CARD = "textbook_card"
    COMPACT_PRINT = "compact_print"


class AugmentationProfile(StrEnum):
    CLEAN = "clean"
    SCANNED_PDF = "scanned_pdf"
    MOBILE_PHOTO = "mobile_photo"
    LOW_DPI = "low_dpi"


class StemSample(BaseModel):
    """Single Indonesian STEM OCR sample pairing ground-truth Markdown+LaTeX with render config."""

    id: str
    split: Split
    subject: Subject
    topic: str
    template_id: str
    format_kind: FormatKind
    layout: LayoutStyle
    augmentation: AugmentationProfile
    markdown: str = Field(min_length=10)
    image_path: str | None = None
    seed: int


class DatasetManifest(BaseModel):
    """Summary metadata written alongside generated dataset JSONL files."""

    name: str = "IndoSTEM-OCR"
    version: str = "0.1.0"
    license: str = "CC-BY-4.0"
    seed: int
    total_samples: int
    counts_by_split: dict[str, int]
    counts_by_subject: dict[str, int]
    counts_by_augmentation: dict[str, int]


class SampleEvalResult(BaseModel):
    """Per-sample OCR evaluation output."""

    id: str
    subject: Subject
    augmentation: AugmentationProfile
    cer: float
    normalized_similarity: float
    latex_f1: float
    latex_valid_rate: float
    structure_f1: float
    latency_ms: float
    reference_markdown: str
    predicted_markdown: str


class EvalReport(BaseModel):
    """Aggregated benchmark report across a dataset split."""

    model_id: str
    adapter_id: str | None = None
    split: Split
    num_samples: int
    mean_cer: float
    mean_similarity: float
    mean_latex_f1: float
    mean_latex_valid_rate: float
    mean_structure_f1: float
    mean_latency_ms: float
    by_subject: dict[str, dict[str, float]] = Field(default_factory=dict)
    by_augmentation: dict[str, dict[str, float]] = Field(default_factory=dict)
