"""Train method A and write autoencoder scores to scores_raw.csv."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import configs
import data
from autoencoder_model import ConvAutoencoder


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_tensors(data_root: Path, relative_paths: list[str]) -> torch.Tensor:
    arrays = []
    for rel in relative_paths:
        image = data.load_rgb_image(data.resolve_image_path(data_root, rel))
        image = data.resize_rgb(image)
        arrays.append(data.image_to_tensor01(image))
    return torch.from_numpy(np.stack(arrays))


def train_one_run(
    train_tensor: torch.Tensor,
    val_tensor: torch.Tensor,
    bottleneck_channels: int,
    device: torch.device,
) -> tuple[ConvAutoencoder, list[dict[str, float]]]:
    model = ConvAutoencoder(bottleneck_channels).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=configs.AE_LEARNING_RATE,
        weight_decay=configs.AE_WEIGHT_DECAY,
    )
    loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=configs.AE_BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(configs.RANDOM_SEED),
    )
    best_state = None
    best_val = float("inf")
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, configs.AE_MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = nn.functional.mse_loss(recon, batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_recon = model(val_tensor.to(device))
            val_loss = nn.functional.mse_loss(val_recon, val_tensor.to(device)).item()

        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= configs.AE_EARLY_STOP_PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, history


def mean_absolute_error(model: ConvAutoencoder, val_tensor: torch.Tensor, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        recon = model(val_tensor.to(device))
        return float(torch.mean(torch.abs(recon - val_tensor.to(device))).item())


def save_loss_csv(history: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)


def score_autoencoder(
    model: ConvAutoencoder,
    tensor: torch.Tensor,
    device: torch.device,
) -> tuple[list[float], list[np.ndarray]]:
    model.eval()
    image_scores: list[float] = []
    maps: list[np.ndarray] = []
    with torch.no_grad():
        for index in range(tensor.shape[0]):
            batch = tensor[index : index + 1].to(device)
            recon = model(batch)
            diff = (batch - recon) ** 2
            image_scores.append(float(diff.mean().item()))
            patch_map = diff.mean(dim=1).squeeze(0).cpu().numpy()
            maps.append(patch_map)
    return image_scores, maps


def read_score_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_score_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "split", "defect_type", "method", "score"],
        )
        writer.writeheader()
        writer.writerows(rows)


def append_scores(rows: list[dict], path: Path, method: str) -> None:
    kept = [row for row in read_score_rows(path) if row.get("method") != method]
    merged = kept + [{**row, "method": method} for row in rows]
    write_score_rows(path, merged)


def write_run_meta(meta: dict, path: Path = configs.RUN_META_JSON) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def train_pipeline(
    data_root: Path,
    device: torch.device,
    score_after_train: bool,
    require_test_layout: bool,
    split_path: Path | None,
) -> None:
    data.verify_counts(data_root, require_test=require_test_layout)
    if split_path and split_path.exists():
        split = data.load_split(split_path)
    else:
        split = data.build_split(data_root)
        data.save_split(split)

    train_tensor = load_tensors(data_root, split["train"])
    val_tensor = load_tensors(data_root, split["val"])

    bottleneck = configs.BOTTLENECK_CHANNELS
    narrowed = False
    model, history = train_one_run(train_tensor, val_tensor, bottleneck, device)
    mae = mean_absolute_error(model, val_tensor, device)
    if mae < configs.AE_COPY_MAE_THRESHOLD:
        narrowed = True
        bottleneck = configs.BOTTLENECK_CHANNELS_NARROW
        model, history = train_one_run(train_tensor, val_tensor, bottleneck, device)
        mae = mean_absolute_error(model, val_tensor, device)

    configs.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_loss_csv(history, configs.AE_LOSS_CSV)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "bottleneck_channels": bottleneck,
            "narrowed_bottleneck": narrowed,
            "val_mae": mae,
            "train_device": str(device),
        },
        configs.AUTOENCODER_WEIGHTS,
    )
    write_run_meta({"train_device": str(device), "narrowed_bottleneck": narrowed})

    if not score_after_train:
        return

    write_scores_for_split(model, split, data_root, device)


def write_scores_for_split(
    model: ConvAutoencoder,
    split: dict,
    data_root: Path,
    device: torch.device,
) -> None:
    rows: list[dict] = []
    for rel in split["val"]:
        tensor = load_tensors(data_root, [rel])
        score, _ = score_autoencoder(model, tensor, device)
        rows.append({"path": rel, "split": "val", "defect_type": "good", "score": score[0]})
    test_paths = list(split["test_good"])
    for defect in configs.DEFECT_TYPES:
        test_paths.extend(split["test_defect"][defect])
    if test_paths:
        tensor = load_tensors(data_root, test_paths)
        scores, _ = score_autoencoder(model, tensor, device)
        for rel, value in zip(test_paths, scores):
            defect_type = "good" if rel in split["test_good"] else Path(rel).parts[1]
            rows.append({"path": rel, "split": "test", "defect_type": defect_type, "score": value})
    append_scores(rows, configs.SCORES_RAW_CSV, configs.METHOD_AUTOENCODER)


def load_trained_model(device: torch.device) -> ConvAutoencoder:
    checkpoint = torch.load(configs.AUTOENCODER_WEIGHTS, map_location=device, weights_only=False)
    model = ConvAutoencoder(checkpoint["bottleneck_channels"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=configs.DATA_ROOT)
    parser.add_argument("--scores-only", action="store_true")
    parser.add_argument("--train-only", action="store_true", help="Kaggle: no test scoring")
    parser.add_argument("--split-file", type=Path, default=configs.SPLIT_JSON)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(configs.RANDOM_SEED)

    if args.scores_only:
        split = data.load_split()
        model = load_trained_model(device)
        write_scores_for_split(model, split, data_root, device)
        return

    train_pipeline(
        data_root,
        device,
        score_after_train=not args.train_only,
        require_test_layout=not args.train_only,
        split_path=args.split_file,
    )


if __name__ == "__main__":
    main()
