from __future__ import annotations

from pathlib import Path
from typing import Any


def load_bgr_image(image_path: Path) -> Any:
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image at {image_path}")
    return image


def build_training_transforms(image_size: int = 224) -> Any:
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
