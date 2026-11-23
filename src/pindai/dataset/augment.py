"""Realistic document scan and mobile-camera augmentations for OCR robustness."""

from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from pindai.dataset.schema import AugmentationProfile


def _add_illumination_gradient(img: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32)
    h, w = arr.shape[:2]
    y_grad = np.linspace(rng.uniform(0.86, 1.0), rng.uniform(0.88, 1.04), h, dtype=np.float32)[
        :, None
    ]
    x_grad = np.linspace(rng.uniform(0.88, 1.02), rng.uniform(0.85, 1.0), w, dtype=np.float32)[
        None, :
    ]
    mask = (y_grad * x_grad)[:, :, None]
    shaded = np.clip(arr * mask, 0, 255).astype(np.uint8)
    return Image.fromarray(shaded, mode="RGB")


def _add_paper_noise(img: Image.Image, rng: random.Random, sigma: float) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32)
    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    noise = np_rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode="RGB")


def _jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply_augmentation(
    image: Image.Image,
    profile: AugmentationProfile,
    seed: int,
) -> Image.Image:
    """Apply a deterministic document degradation profile to an RGB image."""
    rgb = image.convert("RGB")
    if profile == AugmentationProfile.CLEAN:
        return rgb

    rng = random.Random(seed)
    w, h = rgb.size

    if profile == AugmentationProfile.SCANNED_PDF:
        angle = rng.uniform(-1.5, 1.5)
        out = rgb.rotate(
            angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(248, 248, 245)
        )
        out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.92, 1.15))
        out = _add_paper_noise(out, rng, sigma=3.0)
        return _jpeg_roundtrip(out, quality=rng.randint(72, 86))

    if profile == AugmentationProfile.MOBILE_PHOTO:
        angle = rng.uniform(-2.8, 2.8)
        out = rgb.rotate(
            angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(236, 234, 228)
        )
        out = _add_illumination_gradient(out, rng)
        out = out.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.8)))
        out = _add_paper_noise(out, rng, sigma=5.5)
        return _jpeg_roundtrip(out, quality=rng.randint(58, 78))

    if profile == AugmentationProfile.LOW_DPI:
        scale = rng.uniform(0.55, 0.72)
        small = rgb.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR
        )
        out = small.resize((w, h), Image.Resampling.BILINEAR)
        out = _add_paper_noise(out, rng, sigma=2.5)
        return _jpeg_roundtrip(out, quality=rng.randint(55, 70))

    return rgb
