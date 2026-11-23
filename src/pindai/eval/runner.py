"""GPU Vision-Language Model evaluation runner for IndoSTEM-OCR."""

from __future__ import annotations

import json
import time
from pathlib import Path

from PIL import Image

from pindai.dataset.schema import EvalReport, SampleEvalResult, Split, StemSample
from pindai.eval.metrics import aggregate_report, score_prediction

OCR_SYSTEM_PROMPT = (
    "Transkripsikan gambar soal atau materi STEM Bahasa Indonesia berikut ke Markdown bersih. "
    "Gunakan LaTeX standar ($...$ untuk rumus inline dan $$...$$ untuk rumus blok), "
    "pertahankan struktur heading, tabel Markdown, dan opsi pilihan ganda A-E persis seperti gambar. "
    "Keluarkan hanya Markdown hasil transkripsi tanpa penjelasan tambahan."
)


def load_samples_jsonl(path: Path, split: Split | None = None) -> list[StemSample]:
    """Load StemSample records from a JSONL file, optionally filtering by split."""
    samples: list[StemSample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample = StemSample.model_validate_json(line)
        if split is None or sample.split == split:
            samples.append(sample)
    return samples


def evaluate_vlm(
    samples: list[StemSample],
    image_root: Path,
    model_id: str,
    split: Split,
    output_dir: Path,
    adapter_id: str | None = None,
    max_new_tokens: int = 1024,
    load_in_4bit: bool = False,
) -> EvalReport:
    """Run OCR inference on GPU with a Hugging Face VLM (and optional LoRA adapter)."""
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "ML dependencies are required for GPU evaluation. Install with `uv sync --extra ml`."
        ) from exc

    quant_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if load_in_4bit
        else None
    )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quant_config,
        trust_remote_code=True,
    )

    if adapter_id:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_id)

    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[SampleEvalResult] = []

    for sample in samples:
        if not sample.image_path:
            raise ValueError(f"Sample {sample.id} has no image_path; render images first.")
        img_file = image_root / sample.image_path
        image = Image.open(img_file).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": OCR_SYSTEM_PROMPT},
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt").to(
            model.device
        )

        t0 = time.perf_counter()
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        trimmed = generated_ids[:, inputs["input_ids"].shape[1] :]
        predicted = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        metrics = score_prediction(sample.markdown, predicted)

        results.append(
            SampleEvalResult(
                id=sample.id,
                subject=sample.subject,
                augmentation=sample.augmentation,
                cer=metrics["cer"],
                normalized_similarity=metrics["normalized_similarity"],
                latex_f1=metrics["latex_f1"],
                latex_valid_rate=metrics["latex_valid_rate"],
                structure_f1=metrics["structure_f1"],
                latency_ms=round(elapsed_ms, 2),
                reference_markdown=sample.markdown,
                predicted_markdown=predicted,
            )
        )

    report = aggregate_report(
        model_id=model_id, split=split, results=results, adapter_id=adapter_id
    )

    preds_path = output_dir / "predictions.jsonl"
    preds_path.write_text(
        "\n".join(r.model_dump_json() for r in results) + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report.model_dump(), indent=2) + "\n", encoding="utf-8")
    return report
