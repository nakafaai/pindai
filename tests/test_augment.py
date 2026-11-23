"""Tests for document scan and mobile-camera augmentations."""

from PIL import Image

from pindai.dataset.augment import apply_augmentation
from pindai.dataset.schema import AugmentationProfile


def test_all_augmentation_profiles_preserve_size_and_mode() -> None:
    base = Image.new("RGB", (320, 200), color=(255, 255, 255))
    for profile in AugmentationProfile:
        out = apply_augmentation(base, profile=profile, seed=42)
        assert out.size == base.size
        assert out.mode == "RGB"


def test_augmentations_are_deterministic_for_same_seed() -> None:
    base = Image.new("RGB", (160, 120), color=(240, 240, 240))
    a = apply_augmentation(base, AugmentationProfile.MOBILE_PHOTO, seed=99)
    b = apply_augmentation(base, AugmentationProfile.MOBILE_PHOTO, seed=99)
    assert a.tobytes() == b.tobytes()
