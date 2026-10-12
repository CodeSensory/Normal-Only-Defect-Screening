"""Dataset layout checks, split, and image loading."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

import configs


@dataclass(frozen=True)
class ImageRecord:
    relative_path: str
    split: str
    defect_type: str


def list_png_files(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.png"))


def relative_to_data_root(path: Path, data_root: Path) -> str:
    return path.relative_to(data_root).as_posix()


def count_dataset(data_root: Path) -> dict[str, int]:
    train_good = list_png_files(data_root / "train" / "good")
    test_good = list_png_files(data_root / "test" / "good")
    counts = {
        "train_all": len(train_good),
        "test_good": len(test_good),
        "test_defect_total": 0,
    }
    for defect in configs.DEFECT_TYPES:
        test_defect = list_png_files(data_root / "test" / defect)
        mask_defect = list_png_files(data_root / "ground_truth" / defect)
        counts[f"test_{defect}"] = len(test_defect)
        counts[f"mask_{defect}"] = len(mask_defect)
        counts["test_defect_total"] += len(test_defect)
    return counts


def verify_counts(data_root: Path, require_test: bool = True) -> dict[str, int]:
    counts = count_dataset(data_root)
    expected = {
        "train_all": configs.TRAIN_NORMAL_COUNT + configs.VAL_NORMAL_COUNT,
    }
    if require_test:
        expected["test_good"] = configs.TEST_GOOD_COUNT
        expected["test_defect_total"] = configs.TEST_DEFECT_COUNT
        for defect in configs.DEFECT_TYPES:
            expected[f"test_{defect}"] = configs.DEFECTS_PER_TYPE
            expected[f"mask_{defect}"] = configs.DEFECTS_PER_TYPE
    mismatches = {
        key: (counts[key], expected[key])
        for key in expected
        if counts[key] != expected[key]
    }
    if mismatches:
        lines = [f"  {key}: found {found}, expected {exp}" for key, (found, exp) in mismatches.items()]
        raise RuntimeError(
            "Dataset file counts do not match the plan. Update README with actual layout and fix paths.\n"
            + "\n".join(lines)
        )
    return counts


def build_split(data_root: Path) -> dict:
    train_files = list_png_files(data_root / "train" / "good")
    rel_train = [relative_to_data_root(path, data_root) for path in train_files]
    rng = np.random.default_rng(configs.RANDOM_SEED)
    order = rng.permutation(len(rel_train))
    shuffled = [rel_train[i] for i in order]
    split = {
        "seed": configs.RANDOM_SEED,
        "train": shuffled[: configs.TRAIN_NORMAL_COUNT],
        "val": shuffled[configs.TRAIN_NORMAL_COUNT :],
        "test_good": [
            relative_to_data_root(path, data_root)
            for path in list_png_files(data_root / "test" / "good")
        ],
        "test_defect": {
            defect: [
                relative_to_data_root(path, data_root)
                for path in list_png_files(data_root / "test" / defect)
            ]
            for defect in configs.DEFECT_TYPES
        },
    }
    return split


def save_split(split: dict, path: Path = configs.SPLIT_JSON) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2), encoding="utf-8")


def load_split(path: Path = configs.SPLIT_JSON) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_image_path(data_root: Path, relative_path: str) -> Path:
    return data_root / Path(relative_path)


def load_rgb_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_rgb(image: Image.Image, size: int = configs.INPUT_SIZE) -> Image.Image:
    return image.resize((size, size), Image.BILINEAR)


def image_to_tensor01(image: Image.Image) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32) / configs.PIXEL_SCALE
    return np.transpose(array, (2, 0, 1))


def apply_brightness(tensor01: np.ndarray, factor: float) -> np.ndarray:
    scaled = tensor01 * factor
    return np.clip(scaled, 0.0, 1.0)


def normalize_imagenet(tensor01: np.ndarray) -> np.ndarray:
    mean = np.array(configs.IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std = np.array(configs.IMAGENET_STD, dtype=np.float32)[:, None, None]
    return (tensor01 - mean) / std


def load_mask_nearest(path: Path, size: int = configs.INPUT_SIZE) -> np.ndarray:
    mask = Image.open(path).convert("L")
    mask = mask.resize((size, size), Image.NEAREST)
    return np.asarray(mask, dtype=np.uint8)


def mask_path_for_defect_image(data_root: Path, relative_test_path: str) -> Path:
    parts = Path(relative_test_path).parts
    defect_type = parts[1]
    filename = parts[-1]
    mask_name = filename.replace(".png", "_mask.png")
    return data_root / "ground_truth" / defect_type / mask_name


def records_from_split(split: dict, include_test: bool) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for rel in split["train"]:
        records.append(ImageRecord(rel, "train", "good"))
    for rel in split["val"]:
        records.append(ImageRecord(rel, "val", "good"))
    if not include_test:
        return records
    for rel in split["test_good"]:
        records.append(ImageRecord(rel, "test", "good"))
    for defect in configs.DEFECT_TYPES:
        for rel in split["test_defect"][defect]:
            records.append(ImageRecord(rel, "test", defect))
    return records


def print_counts(data_root: Path) -> None:
    counts = verify_counts(data_root)
    print("Dataset counts OK:")
    for key, value in sorted(counts.items()):
        print(f"  {key}: {value}")


def save_four_type_examples(data_root: Path, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(configs.DEFECT_TYPES), figsize=(12, 3))
    for axis, defect in zip(axes, configs.DEFECT_TYPES):
        first = list_png_files(data_root / "test" / defect)[0]
        axis.imshow(load_rgb_image(first))
        axis.set_title(configs.DEFECT_LABELS[defect])
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify MVTec transistor layout and build split.json")
    parser.add_argument("--data-root", type=Path, default=configs.DATA_ROOT)
    parser.add_argument("--write-split", action="store_true")
    parser.add_argument("--examples", action="store_true")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise SystemExit(f"Data root not found: {data_root}")
    print_counts(data_root)
    if args.write_split:
        split = build_split(data_root)
        save_split(split)
        print(f"Wrote {configs.SPLIT_JSON}")
    if args.examples:
        save_four_type_examples(data_root, configs.FIGURES_DIR / "examples_four_types.png")
        print(f"Wrote {configs.FIGURES_DIR / 'examples_four_types.png'}")


if __name__ == "__main__":
    main()
