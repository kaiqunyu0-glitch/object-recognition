"""Turn a photo into the tensor the network expects.

Training uses strong augmentation so the model cannot memorize one angle,
one background, or one brightness. Validation and test use a fixed resize
and center crop. The test score must not depend on a lucky random crop.
"""

from __future__ import annotations

from torchvision import transforms

# ImageNet statistics. Pretrained backbones expect these numbers.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def eval_transform(image_size: int) -> transforms.Compose:
    """Deterministic transform for validation, test, and predict."""
    # 224 models are usually resized to 256, then center-cropped back to 224.
    resize_to = int(round(image_size / 0.875))
    return transforms.Compose(
        [
            transforms.Resize(resize_to),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def train_transform(image_size: int, augment: dict) -> transforms.Compose:
    """Random crops, flips, color changes, and RandAugment or TrivialAugment."""
    low = float(augment["crop_scale_min"])
    if not 0 < low <= 1:
        raise ValueError("augment.crop_scale_min must be greater than 0 and at most 1.")

    steps: list = [
        transforms.RandomResizedCrop(
            image_size,
            scale=(low, 1.0),
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
    ]
    # Vertical flips are off on purpose. Bottles, shoes, and mugs have an
    # up-and-down, and phone photos of them are rarely upside down.
    if augment.get("hflip", True):
        steps.append(transforms.RandomHorizontalFlip())

    jitter = float(augment.get("color_jitter", 0.0))
    if jitter > 0:
        # Hue stays off so a red bottle is not taught to look blue.
        steps.append(
            transforms.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter)
        )

    use_rand = bool(augment.get("randaugment", False))
    use_trivial = bool(augment.get("trivial_augment", False))
    if use_rand and use_trivial:
        print("Both RandAugment and TrivialAugment are on. Using RandAugment only.")
        use_trivial = False
    if use_rand:
        steps.append(transforms.RandAugment(num_ops=2, magnitude=7))
    elif use_trivial:
        steps.append(transforms.TrivialAugmentWide())

    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(steps)
