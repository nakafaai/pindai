"""Reproducible LoRA / QLoRA fine-tuning pipeline for Vision-Language OCR models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from pindai.dataset.schema import Split, StemSample
from pindai.eval.runner import OCR_SYSTEM_PROMPT, load_samples_jsonl


@dataclass(frozen=True)
class TrainConfig:
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    output_dir: str = "checkpoints/pindai-qwen3-vl-8b-lora"
    epochs: int = 2
    learning_rate: float = 2e-4
    batch_size: int = 1
    grad_accum_steps: int = 8
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    load_in_4bit: bool = True
    seed: int = 42


def _build_training_example(sample: StemSample, image_root: Path) -> dict[str, object]:
    if not sample.image_path:
        raise ValueError(f"Sample {sample.id} is missing image_path.")
    return {
        "image_path": str(image_root / sample.image_path),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": OCR_SYSTEM_PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample.markdown}],
            },
        ],
    }


def train_lora(
    dataset_jsonl: Path,
    image_root: Path,
    config: TrainConfig,
    max_train_samples: int | None = None,
) -> Path:
    """Fine-tune a VLM with LoRA/QLoRA on IndoSTEM-OCR and save the adapter."""
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "ML dependencies are required for training. Install with `uv sync --extra ml`."
        ) from exc

    train_samples = load_samples_jsonl(dataset_jsonl, split=Split.TRAIN)
    if max_train_samples is not None:
        train_samples = train_samples[:max_train_samples]
    if not train_samples:
        raise ValueError("No training samples found in dataset.")

    examples = [_build_training_example(s, image_root) for s in train_samples]

    quant_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if config.load_in_4bit
        else None
    )

    processor = AutoProcessor.from_pretrained(config.model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        config.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quant_config,
        trust_remote_code=True,
    )

    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    def collate_fn(batch: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        texts: list[str] = []
        images: list[Image.Image] = []
        for item in batch:
            img = Image.open(str(item["image_path"])).convert("RGB")
            images.append(img)
            texts.append(processor.apply_chat_template(item["messages"], tokenize=False))

        encoded = processor(text=texts, images=images, padding=True, return_tensors="pt")
        labels = encoded["input_ids"].clone()
        pad_id = processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        encoded["labels"] = labels
        return encoded

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum_steps,
        learning_rate=config.learning_rate,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
        seed=config.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=examples,
        data_collator=collate_fn,
    )
    trainer.train()

    model.save_pretrained(str(out_dir))
    processor.save_pretrained(str(out_dir))
    (out_dir / "train_config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n",
        encoding="utf-8",
    )
    return out_dir
