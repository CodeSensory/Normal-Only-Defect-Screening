"""Thresholds, metrics, figures, and summary.md from scores_raw.csv."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score

import configs
import data
from score_embedding import extract_layer3_features, load_resnet_backbone, score_image_patches, tensor01_to_imagenet_batch
from train_autoencoder import load_tensors, load_trained_model, score_autoencoder


def read_scores(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_scores_csv(rows: list[dict], thresholds: dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "split", "defect_type", "method", "score", "threshold", "prediction"],
        )
        writer.writeheader()
        for row in rows:
            threshold = thresholds[row["method"]]
            prediction = "defect" if float(row["score"]) >= threshold else "good"
            writer.writerow(
                {
                    **row,
                    "threshold": threshold,
                    "prediction": prediction,
                }
            )


def threshold_from_validation(rows: list[dict]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for method in (configs.METHOD_AUTOENCODER, configs.METHOD_EMBEDDING):
        val_scores = [
            float(row["score"])
            for row in rows
            if row["method"] == method and row["split"] == "val"
        ]
        thresholds[method] = float(np.quantile(val_scores, configs.THRESHOLD_QUANTILE, method="linear"))
    return thresholds


def test_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["split"] == "test"]


def metrics_for_method(rows: list[dict], method: str, threshold: float) -> dict:
    subset = [row for row in test_rows(rows) if row["method"] == method]
    y_true = [0 if row["defect_type"] == "good" else 1 for row in subset]
    y_pred = [1 if float(row["score"]) >= threshold else 0 for row in subset]
    y_score = [float(row["score"]) for row in subset]
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    recall_by_type = {}
    for defect in configs.DEFECT_TYPES:
        defect_rows = [row for row in subset if row["defect_type"] == defect]
        hits = sum(1 for row in defect_rows if float(row["score"]) >= threshold)
        recall_by_type[defect] = hits
    return {
        "confusion": matrix,
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "recall_by_type": recall_by_type,
    }


def threshold_curve(subset: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = np.array([float(row["score"]) for row in subset], dtype=np.float64)
    percentiles = np.linspace(0.0, 1.0, 101)
    thresholds = np.quantile(scores, percentiles, method="linear")
    false_positives = []
    false_negatives = []
    y_true = np.array([0 if row["defect_type"] == "good" else 1 for row in subset])
    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(int)
        false_positives.append(int(np.sum((y_true == 0) & (y_pred == 1))))
        false_negatives.append(int(np.sum((y_true == 1) & (y_pred == 0))))
    return thresholds, np.array(false_positives), np.array(false_negatives)


def plot_loss_curve() -> None:
    epochs = []
    train_loss = []
    val_loss = []
    with configs.AE_LOSS_CSV.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot(epochs, train_loss, label="train")
    axis.plot(epochs, val_loss, label="val")
    axis.set_xlabel("epoch")
    axis.set_ylabel("MSE")
    axis.legend()
    fig.tight_layout()
    fig.savefig(configs.FIGURES_DIR / "ae_loss.png", dpi=150)
    plt.close(fig)


def plot_reconstructions(data_root: Path, split: dict, device: torch.device) -> None:
    model = load_trained_model(device)
    val_paths = sorted(split["val"])[:4]
    tensor = load_tensors(data_root, val_paths)
    model.eval()
    with torch.no_grad():
        recon = model(tensor.to(device)).cpu()
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for index in range(4):
        input_img = np.transpose(tensor[index].numpy(), (1, 2, 0))
        output_img = np.transpose(recon[index].numpy(), (1, 2, 0))
        axes[0, index].imshow(np.clip(input_img, 0, 1))
        axes[0, index].axis("off")
        axes[1, index].imshow(np.clip(output_img, 0, 1))
        axes[1, index].axis("off")
    fig.tight_layout()
    fig.savefig(configs.FIGURES_DIR / "ae_recon.png", dpi=150)
    plt.close(fig)


def normalize_map(values: np.ndarray) -> np.ndarray:
    low = values.min()
    high = values.max()
    if high <= low:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def upsample_map(patch_map: np.ndarray, size: int = configs.INPUT_SIZE) -> np.ndarray:
    tensor = torch.from_numpy(patch_map).float().unsqueeze(0).unsqueeze(0)
    upsampled = torch.nn.functional.interpolate(tensor, size=(size, size), mode="nearest")
    return upsampled.squeeze().numpy()


def anomaly_maps_for_image(
    data_root: Path,
    rel: str,
    device: torch.device,
    ae_model,
    resnet,
    bank: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensor = load_tensors(data_root, [rel])
    _, ae_maps = score_autoencoder(ae_model, tensor, device)
    ae_map = normalize_map(ae_maps[0])
    batch = tensor01_to_imagenet_batch(tensor.to(device))
    layer3 = extract_layer3_features(resnet, batch)
    patches = layer3.squeeze(0).permute(1, 2, 0).reshape(-1, layer3.shape[1])
    _, emb_patch = score_image_patches(patches, bank.to(device))
    emb_map = normalize_map(upsample_map(emb_patch))
    rgb = np.transpose(tensor[0].numpy(), (1, 2, 0))
    return rgb, ae_map, emb_map


def plot_type_maps(data_root: Path, split: dict, device: torch.device) -> None:
    ae_model = load_trained_model(device)
    resnet = load_resnet_backbone(device)
    bank = torch.load(configs.PATCH_BANK_FILE, map_location=device, weights_only=False)["bank"]
    configs.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for defect in configs.DEFECT_TYPES:
        rel = sorted(split["test_defect"][defect])[0]
        rgb, ae_map, emb_map = anomaly_maps_for_image(data_root, rel, device, ae_model, resnet, bank)
        mask_path = data.mask_path_for_defect_image(data_root, rel)
        mask = data.load_mask_nearest(mask_path)
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        axes[0].imshow(rgb)
        axes[0].set_title("input")
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("mask")
        axes[2].imshow(rgb)
        axes[2].imshow(ae_map, cmap="magma", alpha=0.55)
        axes[2].set_title("AE")
        axes[3].imshow(rgb)
        axes[3].imshow(emb_map, cmap="magma", alpha=0.55)
        axes[3].set_title("ResNet-18")
        for axis in axes:
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(configs.FIGURES_DIR / f"maps_{defect}.png", dpi=150)
        plt.close(fig)


def pick_hit_miss(rows: list[dict], method: str, threshold: float) -> tuple[str | None, str | None]:
    defect_rows = [row for row in rows if row["method"] == method and row["defect_type"] != "good"]
    hits = [row for row in defect_rows if float(row["score"]) >= threshold]
    misses = [row for row in defect_rows if float(row["score"]) < threshold]
    hit = None
    miss = None
    if hits:
        hit = sorted(hits, key=lambda row: (-(float(row["score"]) - threshold), row["path"]))[0]["path"]
    if misses:
        miss = sorted(misses, key=lambda row: (-(threshold - float(row["score"])), row["path"]))[0]["path"]
    return hit, miss


def plot_panel(
    axis,
    data_root: Path,
    rel: str | None,
    method: str,
    device: torch.device,
    ae_model,
    resnet,
    bank: torch.Tensor,
    empty_text: str,
) -> str:
    if rel is None:
        axis.text(0.5, 0.5, empty_text, ha="center", va="center")
        axis.axis("off")
        return empty_text
    rgb, ae_map, emb_map = anomaly_maps_for_image(data_root, rel, device, ae_model, resnet, bank)
    heat = ae_map if method == configs.METHOD_AUTOENCODER else emb_map
    axis.imshow(rgb)
    axis.imshow(heat, cmap="magma", alpha=0.55)
    axis.axis("off")
    return Path(rel).name


def plot_hit_miss(data_root: Path, rows: list[dict], thresholds: dict[str, float], device: torch.device) -> None:
    ae_model = load_trained_model(device)
    resnet = load_resnet_backbone(device)
    bank = torch.load(configs.PATCH_BANK_FILE, map_location=device, weights_only=False)["bank"]
    for method, label in (
        (configs.METHOD_AUTOENCODER, "autoencoder"),
        (configs.METHOD_EMBEDDING, "embedding"),
    ):
        hit, miss = pick_hit_miss(rows, method, thresholds[method])
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        hit_title = plot_panel(
            axes[0], data_root, hit, method, device, ae_model, resnet, bank, "no hit"
        )
        miss_title = plot_panel(
            axes[1], data_root, miss, method, device, ae_model, resnet, bank, "miss 0"
        )
        axes[0].set_title(f"hit: {hit_title}")
        axes[1].set_title(f"miss: {miss_title}")
        fig.tight_layout()
        fig.savefig(configs.FIGURES_DIR / f"hit_miss_{label}.png", dpi=150)
        plt.close(fig)


def plot_confusion(metrics: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for axis, (method, label) in zip(
        axes,
        ((configs.METHOD_AUTOENCODER, "AE"), (configs.METHOD_EMBEDDING, "ResNet-18")),
    ):
        matrix = metrics[method]["confusion"]
        axis.imshow(matrix, cmap="Blues")
        axis.set_title(label)
        axis.set_xticks([0, 1], labels=["pred good", "pred defect"])
        axis.set_yticks([0, 1], labels=["true good", "true defect"])
        for (row_index, col_index), value in np.ndenumerate(matrix):
            axis.text(col_index, row_index, str(value), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(configs.FIGURES_DIR / "confusion.png", dpi=150)
    plt.close(fig)


def plot_recall(metrics: dict[str, dict]) -> None:
    labels = [configs.DEFECT_LABELS[key] for key in configs.DEFECT_TYPES]
    x = np.arange(len(labels))
    width = 0.35
    ae = [metrics[configs.METHOD_AUTOENCODER]["recall_by_type"][key] for key in configs.DEFECT_TYPES]
    emb = [metrics[configs.METHOD_EMBEDDING]["recall_by_type"][key] for key in configs.DEFECT_TYPES]
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.bar(x - width / 2, ae, width, label="AE")
    axis.bar(x + width / 2, emb, width, label="ResNet-18")
    axis.set_xticks(x, labels, rotation=15)
    axis.set_ylabel("hits / 10")
    axis.set_ylim(0, 10)
    axis.legend()
    fig.tight_layout()
    fig.savefig(configs.FIGURES_DIR / "recall_by_type.png", dpi=150)
    plt.close(fig)


def plot_threshold_curves(rows: list[dict], thresholds: dict[str, float]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, (method, label) in zip(
        axes,
        ((configs.METHOD_AUTOENCODER, "AE"), (configs.METHOD_EMBEDDING, "ResNet-18")),
    ):
        subset = [row for row in test_rows(rows) if row["method"] == method]
        thresh, fp, fn = threshold_curve(subset)
        axis.plot(thresh, fp, label="false positives")
        axis.plot(thresh, fn, label="false negatives")
        axis.axvline(thresholds[method], color="black", linestyle="--", label="val 95%")
        axis.set_title(label)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(configs.FIGURES_DIR / "threshold_curve.png", dpi=150)
    plt.close(fig)


def count_ae_parameters() -> int:
    checkpoint = torch.load(configs.AUTOENCODER_WEIGHTS, map_location="cpu", weights_only=False)
    model = load_trained_model(torch.device("cpu"))
    del checkpoint
    return sum(parameter.numel() for parameter in model.parameters())


def measure_timing(data_root: Path, split: dict, device: torch.device) -> list[dict]:
    test_paths = list(split["test_good"])
    for defect in configs.DEFECT_TYPES:
        test_paths.extend(split["test_defect"][defect])
    tensor = load_tensors(data_root, test_paths)
    ae_model = load_trained_model(device)
    resnet = load_resnet_backbone(device)
    bank = torch.load(configs.PATCH_BANK_FILE, map_location=device, weights_only=False)["bank"]

    def time_call(callable_fn, warmup: int = configs.TIMING_WARMUP_IMAGES) -> float:
        for index in range(warmup):
            callable_fn(index)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for index in range(tensor.shape[0]):
            callable_fn(index)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return elapsed_ms / tensor.shape[0]

    def ae_step(index: int) -> None:
        batch = tensor[index : index + 1].to(device)
        with torch.no_grad():
            _ = ae_model(batch)

    def emb_feature(index: int) -> torch.Tensor:
        batch = tensor01_to_imagenet_batch(tensor[index : index + 1].to(device))
        with torch.no_grad():
            return extract_layer3_features(resnet, batch)

    cached_patches: list[torch.Tensor] = []
    with torch.no_grad():
        for index in range(tensor.shape[0]):
            layer3 = emb_feature(index)
            cached_patches.append(layer3.squeeze(0).permute(1, 2, 0).reshape(-1, layer3.shape[1]))

    def emb_distance(index: int) -> None:
        score_image_patches(cached_patches[index], bank)

    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor() or "cpu"
    timing = [
        {
            "method": configs.METHOD_AUTOENCODER,
            "segment": "forward",
            "mean_ms": time_call(ae_step),
            "device": device_name,
        },
        {
            "method": configs.METHOD_EMBEDDING,
            "segment": "feature_extract",
            "mean_ms": time_call(emb_feature),
            "device": device_name,
        },
        {
            "method": configs.METHOD_EMBEDDING,
            "segment": "nearest_distance",
            "mean_ms": time_call(emb_distance),
            "device": device_name,
        },
    ]
    configs.TIMING_JSON.parent.mkdir(parents=True, exist_ok=True)
    configs.TIMING_JSON.write_text(json.dumps(timing, indent=2), encoding="utf-8")
    return timing


def brightness_section(
    base_rows: list[dict],
    thresholds: dict[str, float],
    brightness_rows: list[dict] | None,
) -> str:
    if not brightness_rows:
        return "밝기 변화 실험은 아직 실행하지 않았다. `python score_brightness.py` 후 `--finalize`로 갱신한다.\n"
    lines = []
    base_predictions = {
        (row["path"], row["method"]): float(row["score"]) >= thresholds[row["method"]]
        for row in base_rows
        if row["split"] == "test"
    }
    method_labels = {
        configs.METHOD_AUTOENCODER: "AE",
        configs.METHOD_EMBEDDING: "ResNet-18",
    }
    for factor in configs.BRIGHTNESS_FACTORS:
        for method, label in method_labels.items():
            changed_good = 0
            changed_defect = 0
            subset = [
                row
                for row in brightness_rows
                if float(row["brightness"]) == factor and row["method"] == method
            ]
            for row in subset:
                new_pred = float(row["score"]) >= thresholds[row["method"]]
                old_pred = base_predictions[(row["path"], row["method"])]
                if new_pred == old_pred:
                    continue
                if row["defect_type"] == "good":
                    changed_good += 1
                else:
                    changed_defect += 1
            lines.append(
                f"- 배율 {factor}, {label}: 정상 60장 중 판정 변경 {changed_good}장, "
                f"불량 40장 중 판정 변경 {changed_defect}장"
            )
    return "\n".join(lines) + "\n"


def write_summary(
    thresholds: dict[str, float],
    metrics: dict[str, dict],
    timing: list[dict] | None,
    include_brightness: bool,
    brightness_rows: list[dict] | None,
    base_rows: list[dict],
) -> None:
    meta = {}
    if configs.RUN_META_JSON.exists():
        meta = json.loads(configs.RUN_META_JSON.read_text(encoding="utf-8"))
    train_device = meta.get("train_device", "unknown")
    inference_device = timing[0]["device"] if timing else platform.processor() or "cpu"
    ae_params = count_ae_parameters()

    lines = [
        "# Summary",
        "",
        "## 기기와 버전",
        f"- Python: {sys.version.split()[0]}",
        f"- PyTorch: {torch.__version__}",
        f"- 학습 기기 (AE): {train_device}",
        f"- 추론 기기: {inference_device}",
        f"- AE 학습 파라미터 수: {ae_params}",
        "- ResNet-18 학습 파라미터 수: 0 (고정 특징)",
        "",
        "## 문턱값 (검증 95%)",
        f"- AE: {thresholds[configs.METHOD_AUTOENCODER]:.6f}",
        f"- ResNet-18: {thresholds[configs.METHOD_EMBEDDING]:.6f}",
        "",
        "## 혼동행렬 (양성=불량)",
    ]
    for method, label in ((configs.METHOD_AUTOENCODER, "AE"), (configs.METHOD_EMBEDDING, "ResNet-18")):
        matrix = metrics[method]["confusion"]
        lines.append(f"- {label}: TN={matrix[0,0]} FP={matrix[0,1]} FN={matrix[1,0]} TP={matrix[1,1]}")
    lines.extend(
        [
            "",
            "## 오탐과 미탐",
            f"- AE: 오탐 {metrics[configs.METHOD_AUTOENCODER]['false_positives']}장, 미탐 {metrics[configs.METHOD_AUTOENCODER]['false_negatives']}장",
            f"- ResNet-18: 오탐 {metrics[configs.METHOD_EMBEDDING]['false_positives']}장, 미탐 {metrics[configs.METHOD_EMBEDDING]['false_negatives']}장",
            "",
            "## 유형별 재현율 (맞힌 장수 / 10)",
        ]
    )
    for defect in configs.DEFECT_TYPES:
        label = configs.DEFECT_LABELS[defect]
        ae_hits = metrics[configs.METHOD_AUTOENCODER]["recall_by_type"][defect]
        emb_hits = metrics[configs.METHOD_EMBEDDING]["recall_by_type"][defect]
        lines.append(f"- {label}: AE {ae_hits}/10, ResNet-18 {emb_hits}/10")
    lines.extend(
        [
            "",
            "## AUROC (테스트 100장)",
            f"- AE: {metrics[configs.METHOD_AUTOENCODER]['auroc']:.4f}",
            f"- ResNet-18: {metrics[configs.METHOD_EMBEDDING]['auroc']:.4f}",
            "",
            "## 추론 시간 (배치 1, 이미지당 ms)",
        ]
    )
    if timing:
        for entry in timing:
            lines.append(f"- {entry['method']} / {entry['segment']}: {entry['mean_ms']:.3f} ms ({entry['device']})")
    else:
        lines.append("- 아직 측정하지 않았다. `python evaluate.py --timing` 실행.")
    lines.extend(["", "## 밝기"])
    if include_brightness and configs.SCORES_BRIGHTNESS_RAW_CSV.exists():
        brightness_rows = brightness_rows or read_scores(configs.SCORES_BRIGHTNESS_RAW_CSV)
        lines.append(brightness_section(base_rows, thresholds, brightness_rows).rstrip())
    else:
        lines.append("밝기 변화 실험은 아직 실행하지 않았다.")
    configs.SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluate(data_root: Path, device: torch.device, make_figures: bool, timing: bool, finalize: bool) -> None:
    rows = read_scores(configs.SCORES_RAW_CSV)
    thresholds = threshold_from_validation(rows)
    write_scores_csv(rows, thresholds, configs.SCORES_CSV)
    metrics = {
        configs.METHOD_AUTOENCODER: metrics_for_method(rows, configs.METHOD_AUTOENCODER, thresholds[configs.METHOD_AUTOENCODER]),
        configs.METHOD_EMBEDDING: metrics_for_method(rows, configs.METHOD_EMBEDDING, thresholds[configs.METHOD_EMBEDDING]),
    }
    timing_rows = None
    if timing:
        split = data.load_split()
        timing_rows = measure_timing(data_root, split, device)
    brightness_rows = None
    if finalize and configs.SCORES_BRIGHTNESS_RAW_CSV.exists():
        brightness_rows = read_scores(configs.SCORES_BRIGHTNESS_RAW_CSV)
    write_summary(
        thresholds,
        metrics,
        timing_rows,
        include_brightness=finalize,
        brightness_rows=brightness_rows,
        base_rows=rows,
    )
    if not make_figures:
        return
    configs.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    split = data.load_split()
    plot_loss_curve()
    plot_reconstructions(data_root, split, device)
    plot_confusion(metrics)
    plot_recall(metrics)
    plot_threshold_curves(rows, thresholds)
    plot_type_maps(data_root, split, device)
    plot_hit_miss(data_root, rows, thresholds, device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=configs.DATA_ROOT)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--finalize", action="store_true", help="Include brightness results in summary.md")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if not configs.SCORES_RAW_CSV.exists():
        raise SystemExit("scores_raw.csv not found. Run train_autoencoder.py and score_embedding.py first.")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_evaluate(
        args.data_root.resolve(),
        device,
        make_figures=not args.no_figures,
        timing=args.timing,
        finalize=args.finalize,
    )


if __name__ == "__main__":
    main()
