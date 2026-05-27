"""
Augmentation pipeline for the defect detection dataset.
Reads from data/images/train + data/labels/train and writes
augmented copies to data/augmented/images + data/augmented/labels.

Run after 01_reorganize_data.py.

Justification for chosen transforms:
  - HorizontalFlip: defects are orientation-invariant
  - VerticalFlip: uncommon in natural images → low probability
  - RandomBrightnessContrast: simulate varying lighting conditions (key challenge)
  - GaussNoise: simulate sensor noise from low-quality site photos
  - Rotate(limit=15): minor camera tilt during site inspection
  - RandomScale: simulate different distances from defect
  - Blur: simulate motion blur or low-resolution captures
  - CLAHE: simulate over/under-exposed images
  - RandomRain/Fog: simulate adverse weather conditions
"""

import argparse
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / 'data'
AUG_IMG_DIR  = DATA_DIR / 'augmented' / 'images'
AUG_LBL_DIR  = DATA_DIR / 'augmented' / 'labels'

IMAGE_EXTS = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']

AUGMENT_TRANSFORM = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.15),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.4),
        A.RandomScale(scale_limit=0.2, p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        A.CLAHE(clip_limit=4.0, p=0.2),
        A.RandomRain(blur_value=3, p=0.1),
        A.RandomFog(fog_coef_range=(0.1, 0.3), p=0.1),
    ],
    bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.3,
    ),
)


def _iter_images(directory: Path):
    seen = set()
    for ext in IMAGE_EXTS:
        for img in directory.glob(ext):
            key = img.name.lower()
            if key not in seen:
                seen.add(key)
                yield img


def read_yolo_labels(label_path: Path) -> tuple[list[int], list[list[float]]]:
    """Returns (class_ids, bboxes) where bbox = [cx, cy, w, h] normalised."""
    class_ids, bboxes = [], []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                cls  = int(parts[0])
                bbox = list(map(float, parts[1:]))
                # Clamp to [0, 1] to avoid albumentations validation errors
                bbox = [max(0.0, min(1.0, v)) for v in bbox]
                class_ids.append(cls)
                bboxes.append(bbox)
    return class_ids, bboxes


def write_yolo_labels(label_path: Path, class_ids: list[int], bboxes: list):
    lines = [f"{cls} {' '.join(f'{v:.6f}' for v in bbox)}\n"
             for cls, bbox in zip(class_ids, bboxes)]
    label_path.write_text(''.join(lines))


def augment_dataset(n_augments: int = 2, seed: int = 42):
    """
    Generates n_augments augmented versions of each training image.
    Skips images whose label files are empty or missing.
    """
    AUG_IMG_DIR.mkdir(parents=True, exist_ok=True)
    AUG_LBL_DIR.mkdir(parents=True, exist_ok=True)

    img_dir = DATA_DIR / 'images' / 'train'
    lbl_dir = DATA_DIR / 'labels' / 'train'

    images = sorted(_iter_images(img_dir))
    print(f"Augmenting {len(images)} training images × {n_augments} versions…")

    skipped, saved = 0, 0
    for img_path in images:
        class_ids, bboxes = read_yolo_labels(lbl_dir / (img_path.stem + '.txt'))
        if not bboxes:
            skipped += 1
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            skipped += 1
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for i in range(n_augments):
            result    = AUGMENT_TRANSFORM(image=image, bboxes=bboxes, class_labels=class_ids)
            aug_img   = result['image']
            aug_boxes = result['bboxes']
            aug_cls   = result['class_labels']

            if not aug_boxes:
                continue

            stem    = f"{img_path.stem}_aug{i}"
            out_img = AUG_IMG_DIR / (stem + '.jpg')
            out_lbl = AUG_LBL_DIR / (stem + '.txt')

            cv2.imwrite(str(out_img), cv2.cvtColor(np.array(aug_img), cv2.COLOR_RGB2BGR))
            write_yolo_labels(out_lbl, list(aug_cls), list(aug_boxes))
            saved += 1

    print(f"Done. Saved {saved} augmented images. Skipped {skipped} images (no labels).")
    print(f"Augmented images → {AUG_IMG_DIR}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Augment training images')
    parser.add_argument('--n', type=int, default=2,
                        help='Number of augmented copies per image (default: 2)')
    args = parser.parse_args()
    augment_dataset(n_augments=args.n)
