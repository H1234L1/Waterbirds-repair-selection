from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

from waterbirds_dataset import GROUP_NAMES, WaterbirdsDataset, dataset_statistics, load_metadata


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_transforms(config: dict):
    train_cfg, eval_cfg = config["preprocessing"]["train"], config["preprocessing"]["evaluation"]
    mean, std = train_cfg["normalize_mean"], train_cfg["normalize_std"]
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(train_cfg["random_resized_crop"], interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(train_cfg["random_horizontal_flip"]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(eval_cfg["resize"], interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(eval_cfg["center_crop"]),
        transforms.ToTensor(),
        transforms.Normalize(eval_cfg["normalize_mean"], eval_cfg["normalize_std"]),
    ])
    return train_transform, eval_transform


def make_model(config: dict) -> nn.Module:
    model_cfg = config["model"]
    if model_cfg["architecture"] != "resnet50":
        raise ValueError("Phase 0 supports exactly one architecture: resnet50")
    try:
        weights = models.ResNet50_Weights[model_cfg["pretrained_weights"]]
    except KeyError as exc:
        raise ValueError(f"unknown ResNet-50 weights: {model_cfg['pretrained_weights']}") from exc
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, model_cfg["num_classes"])
    return model


def configure_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("waterbirds_baseline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(output_dir / "train.log", encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total = correct = 0
    group_total = np.zeros(4, dtype=np.int64)
    group_correct = np.zeros(4, dtype=np.int64)
    with torch.inference_mode():
        for images, labels, _places, groups in loader:
            predictions = model(images.to(device, non_blocking=True)).argmax(dim=1).cpu()
            matches = predictions.eq(labels)
            total += labels.numel()
            correct += int(matches.sum())
            for group_id in range(4):
                mask = groups.eq(group_id)
                group_total[group_id] += int(mask.sum())
                group_correct[group_id] += int(matches[mask].sum())
    group_metrics = {}
    for group_id, name in GROUP_NAMES.items():
        count = int(group_total[group_id])
        group_metrics[name] = {
            "accuracy": float(group_correct[group_id] / count) if count else None,
            "correct": int(group_correct[group_id]),
            "count": count,
        }
    accuracies = [value["accuracy"] for value in group_metrics.values() if value["accuracy"] is not None]
    return {
        "overall_accuracy": float(correct / total),
        "correct": correct,
        "count": total,
        "group_metrics": group_metrics,
        "worst_group_accuracy": float(min(accuracies)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one plain ERM ResNet-50 baseline on Waterbirds.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    seed = int(config["seed"])
    set_seed(seed)
    torch_cache_dir = Path(config["torch_cache_dir"]).expanduser().resolve()
    torch_cache_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(torch_cache_dir))
    logger.info("Torch pretrained-weight cache: %s", torch_cache_dir)
    root, metadata = load_metadata(config["data_dir"])
    stats = dataset_statistics(metadata)
    logger.info("Dataset root: %s", root)
    logger.info("Dataset statistics:\n%s", json.dumps(stats, indent=2))
    with (output_dir / "dataset_statistics.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    train_transform, eval_transform = make_transforms(config)
    datasets = {
        "train": WaterbirdsDataset(root, "train", train_transform, metadata),
        "val": WaterbirdsDataset(root, "val", eval_transform, metadata),
        "test": WaterbirdsDataset(root, "test", eval_transform, metadata),
    }
    train_cfg = config["training"]
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=(split == "train"),
            num_workers=int(train_cfg["num_workers"]),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=int(train_cfg["num_workers"]) > 0,
            worker_init_fn=worker_init_fn,
            generator=generator if split == "train" else None,
        )
        for split, dataset in datasets.items()
    }
    requested_device = train_cfg.get("device", "auto")
    device = torch.device("cuda" if requested_device == "auto" and torch.cuda.is_available() else requested_device if requested_device != "auto" else "cpu")
    logger.info("Device: %s", device)
    model = make_model(config).to(device)
    if train_cfg["optimizer"] != "SGD":
        raise ValueError("Configured optimizer must be SGD for this baseline")
    optimizer = torch.optim.SGD(
        model.parameters(), lr=float(train_cfg["learning_rate"]),
        momentum=float(train_cfg["momentum"]), weight_decay=float(train_cfg["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()
    amp_enabled = bool(train_cfg["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history = []
    best_score = -1.0
    best_epoch = -1
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        running_loss = sample_count = 0
        for images, labels, _places, _groups in loaders["train"]:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach()) * labels.size(0)
            sample_count += labels.size(0)
        validation = evaluate(model, loaders["val"], device)
        record = {"epoch": epoch, "train_loss": running_loss / sample_count, "validation": validation}
        history.append(record)
        logger.info("Epoch %d | train_loss=%.6f | val_accuracy=%.4f | val_worst_group=%.4f", epoch, record["train_loss"], validation["overall_accuracy"], validation["worst_group_accuracy"])
        score = validation[train_cfg["selection_metric"]]
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "config": config, "validation": validation}, output_dir / "best_model.pt")

    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_validation = evaluate(model, loaders["val"], device)
    test = evaluate(model, loaders["test"], device)
    metrics = {"best_epoch": best_epoch, "selection_metric": train_cfg["selection_metric"], "validation": final_validation, "test": test, "training_history": history}
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    logger.info("Final metrics:\n%s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

