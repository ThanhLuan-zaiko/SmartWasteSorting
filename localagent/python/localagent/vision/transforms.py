from __future__ import annotations

from pathlib import Path
from typing import Any


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


def build_training_transforms(image_size: int = 224, *, pre_resized: bool = False) -> Any:
    from torchvision import transforms

    steps: list[Any] = [transforms.ToPILImage()]
    if not pre_resized:
        steps.append(transforms.Resize((image_size, image_size)))
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transforms.Compose(steps)
