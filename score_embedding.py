"""Method B: fixed ResNet-18 patch distances."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torchvision.models import ResNet18_Weights, resnet18

import configs
import data


def load_resnet_backbone(device: torch.device) -> torch.nn.Module:
    try:
        weights = ResNet18_Weights.IMAGENET1K_V1
    except AttributeError as error:
        raise RuntimeError("torchvision ResNet18_Weights.IMAGENET1K_V1 is required") from error
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model.to(device)


def extract_layer3_features(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    features: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        features["layer3"] = output

    handle = model.layer3.register_forward_hook(hook)
    with torch.no_grad():
        model(batch)
    handle.remove()
    return features["layer3"]


def tensor01_to_imagenet_batch(tensor01: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(configs.IMAGENET_MEAN, device=tensor01.device).view(1, 3, 1, 1)
    std = torch.tensor(configs.IMAGENET_STD, device=tensor01.device).view(1, 3, 1, 1)
    return (tensor01 - mean) / std


def build_patch_bank(train_tensors: torch.Tensor, device: torch.device) -> torch.Tensor:
    model = load_resnet_backbone(device)
    vectors = []
    for index in range(train_tensors.shape[0]):
        batch = tensor01_to_imagenet_batch(train_tensors[index : index + 1].to(device))
        layer3 = extract_layer3_features(model, batch)
        flat = layer3.squeeze(0).permute(1, 2, 0).reshape(-1, layer3.shape[1])
        vectors.append(flat.cpu())
    stacked = torch.cat(vectors, dim=0)
    rng = np.random.default_rng(configs.RANDOM_SEED)
    indices = rng.permutation(stacked.shape[0])[: configs.PATCH_BANK_SIZE]
    bank = stacked[indices]
    bank = bank / bank.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return bank


def score_image_patches(patches: torch.Tensor, bank: torch.Tensor) -> tuple[float, np.ndarray]:
    normalized = patches / patches.norm(dim=1, keepdim=True).clamp_min(1e-12)
    distances = torch.cdist(normalized.unsqueeze(0), bank.unsqueeze(0)).squeeze(0)
    min_distances = distances.min(dim=1).values
    patch_map = min_distances.reshape(16, 16).cpu().numpy()
    return float(min_distances.max().item()), patch_map


def append_scores(rows: list[dict], path: Path) -> None:
    from train_autoencoder import read_score_rows, write_score_rows

    kept = [row for row in read_score_rows(path) if row.get("method") != configs.METHOD_EMBEDDING]
    merged = kept + [{**row, "method": configs.METHOD_EMBEDDING} for row in rows]
    write_score_rows(path, merged)


def load_train_tensors(data_root: Path, split: dict) -> torch.Tensor:
    from train_autoencoder import load_tensors

    return load_tensors(data_root, split["train"])


def score_split(
    data_root: Path,
    split: dict,
    bank: torch.Tensor,
    device: torch.device,
    include_test: bool,
) -> list[dict]:
    model = load_resnet_backbone(device)
    rows: list[dict] = []

    def score_paths(relative_paths: list[str], split_name: str, defect_type: str) -> None:
        from train_autoencoder import load_tensors

        if not relative_paths:
            return
        tensor = load_tensors(data_root, relative_paths)
        for rel, index in zip(relative_paths, range(tensor.shape[0])):
            batch = tensor01_to_imagenet_batch(tensor[index : index + 1].to(device))
            layer3 = extract_layer3_features(model, batch)
            patches = layer3.squeeze(0).permute(1, 2, 0).reshape(-1, layer3.shape[1])
            score, _ = score_image_patches(patches, bank.to(device))
            rows.append(
                {"path": rel, "split": split_name, "defect_type": defect_type, "score": score}
            )

    score_paths(split["val"], "val", "good")
    if include_test:
        score_paths(split["test_good"], "test", "good")
        for defect in configs.DEFECT_TYPES:
            score_paths(split["test_defect"][defect], "test", defect)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=configs.DATA_ROOT)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    if not configs.SPLIT_JSON.exists():
        raise SystemExit("Run train_autoencoder.py or data.py --write-split first.")
    split = data.load_split()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_tensor = load_train_tensors(data_root, split)
    bank = build_patch_bank(train_tensor, device)
    configs.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"bank": bank, "size": configs.PATCH_BANK_SIZE}, configs.PATCH_BANK_FILE)

    rows = score_split(data_root, split, bank, device, include_test=True)
    append_scores(rows, configs.SCORES_RAW_CSV)


if __name__ == "__main__":
    main()
