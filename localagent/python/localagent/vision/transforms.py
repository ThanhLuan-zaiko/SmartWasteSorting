from __future__ import annotations

from pathlib import Path
from typing import Any

IMAGENET_NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
IMAGENET_NORMALIZATION_STD = (0.229, 0.224, 0.225)


def load_rgb_image(image_path: Path, *, raw_image_size: int | None = None) -> Any:
    if image_path.suffix.lower() == ".raw":
        import numpy as np

        if raw_image_size is None:
            raise ValueError("raw_image_size is required when loading raw cached images")
        buffer = np.fromfile(image_path, dtype=np.uint8)
        expected_size = raw_image_size * raw_image_size * 3
        if int(buffer.size) != expected_size:
            raise ValueError(
                f"Unexpected raw cache size for {image_path}: {buffer.size} != {expected_size}"
            )
        return buffer.reshape(raw_image_size, raw_image_size, 3)

    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image at {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def normalization_stats(preset: str = "imagenet") -> tuple[tuple[float, ...], tuple[float, ...]]:
    if preset != "imagenet":
        raise ValueError(f"Unsupported normalization preset: {preset}")
    return IMAGENET_NORMALIZATION_MEAN, IMAGENET_NORMALIZATION_STD


def build_training_transforms(
    image_size: int = 224,
    *,
    pre_resized: bool = False,
    normalization_preset: str = "imagenet",
) -> Any:
    from torchvision import transforms

    mean, std = normalization_stats(normalization_preset)
    steps: list[Any] = [transforms.ToPILImage()]
    if not pre_resized:
        steps.append(transforms.Resize((image_size, image_size)))
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transforms.Compose(steps)
