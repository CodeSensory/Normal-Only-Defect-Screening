"""Brightness robustness scores (day 5); thresholds are not recomputed."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

import configs
import data
from score_embedding import load_resnet_backbone, score_image_patches, tensor01_to_imagenet_batch
from score_embedding import extract_layer3_features
from train_autoencoder import load_tensors, load_trained_model, score_autoencoder


def append_brightness_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "split", "defect_type", "method", "brightness", "score"],
        )
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def load_test_records(split: dict) -> list[tuple[str, str]]:
    records = [(rel, "good") for rel in split["test_good"]]
    for defect in configs.DEFECT_TYPES:
        for rel in split["test_defect"][defect]:
            records.append((rel, defect))
    return records


def tensor_with_brightness(data_root: Path, rel: str, factor: float) -> torch.Tensor:
    image = data.load_rgb_image(data.resolve_image_path(data_root, rel))
    image = data.resize_rgb(image)
    array = data.apply_brightness(data.image_to_tensor01(image), factor)
    return torch.from_numpy(array).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=configs.DATA_ROOT)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    split = data.load_split()
    ae_model = load_trained_model(device)
    bank_payload = torch.load(configs.PATCH_BANK_FILE, map_location="cpu", weights_only=False)
    bank = bank_payload["bank"]
    resnet = load_resnet_backbone(device)

    rows: list[dict] = []
    test_records = load_test_records(split)
    for factor in configs.BRIGHTNESS_FACTORS:
        for rel, defect_type in test_records:
            tensor = tensor_with_brightness(data_root, rel, factor)
            ae_score, _ = score_autoencoder(ae_model, tensor, device)
            batch = tensor01_to_imagenet_batch(tensor.to(device))
            layer3 = extract_layer3_features(resnet, batch)
            patches = layer3.squeeze(0).permute(1, 2, 0).reshape(-1, layer3.shape[1])
            emb_score, _ = score_image_patches(patches, bank.to(device))
            rows.append(
                {
                    "path": rel,
                    "split": "test",
                    "defect_type": defect_type,
                    "method": configs.METHOD_AUTOENCODER,
                    "brightness": factor,
                    "score": ae_score[0],
                }
            )
            rows.append(
                {
                    "path": rel,
                    "split": "test",
                    "defect_type": defect_type,
                    "method": configs.METHOD_EMBEDDING,
                    "brightness": factor,
                    "score": emb_score,
                }
            )

    if configs.SCORES_BRIGHTNESS_RAW_CSV.exists():
        configs.SCORES_BRIGHTNESS_RAW_CSV.unlink()
    append_brightness_rows(rows, configs.SCORES_BRIGHTNESS_RAW_CSV)


if __name__ == "__main__":
    main()
