"""Two-stage fine-tuning, then a single look at the held-out test set.

Stage 1 trains only the new classification layer. The pretrained layers keep
the visual features they already learned. Stage 2 unfreezes everything and
continues with a smaller learning rate.

Validation accuracy decides which epoch is saved. The test set is not read
until that choice is finished. Reporting test accuracy earlier would let it
influence the checkpoint, and the number would no longer be honest.
"""

from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.checkpoint import save_checkpoint
from src.config import dump_config
from src.data import (
    build_manifest,
    datasets_from_manifest,
    read_manifest,
    samples_for_split,
    write_manifest,
    ImageListDataset,
)
from src.device import describe_device, get_device
from src.duplicates import find_cross_split_duplicates, report_duplicates
from src.inference import forward_logits
from src.model import (
    apply_train_mode,
    build_model,
    freeze_backbone,
    load_trained_model,
    parameter_counts,
    unfreeze_all,
)
from src.reporting import (
    NO_TEST_MESSAGE,
    clear_test_reports,
    print_test_report,
    write_report,
)
from src.transforms import eval_transform, train_transform
from src.utils import set_seed, warn_once, write_json


class EarlyStopper:
    """Stop a stage when validation accuracy stops improving.

    The comparison is strict: an equal score keeps the earlier checkpoint.
    ``patience`` of 0 turns early stopping off.
    """

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best = -1.0
        self.waits = 0

    def step(self, score: float) -> bool:
        if score > self.best:
            self.best = score
            self.waits = 0
            return True
        self.waits += 1
        return False

    def should_stop(self) -> bool:
        if self.patience <= 0:
            return False
        return self.waits >= self.patience


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> DataLoader:
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
    )


def _uses_batchnorm(model: nn.Module) -> bool:
    return any(
        isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm))
        for module in model.modules()
    )


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    from contextlib import nullcontext

    return nullcontext()


def _make_scaler(enabled: bool):
    if not enabled:
        return None
    return torch.amp.GradScaler("cuda")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler,
    use_amp: bool,
    skip_single_batch: bool,
    head_only: bool,
) -> tuple[float, float]:
    """One pass over the training photos. Weights are updated here."""
    apply_train_mode(model, head_only=head_only)
    total_loss = 0.0
    correct = 0
    seen = 0
    progress = tqdm(loader, desc="train", leave=False, disable=not sys.stderr.isatty())
    for images, labels in progress:
        if skip_single_batch and images.size(0) < 2:
            # BatchNorm computes a variance. A single image makes that zero
            # and can break the step, so this rare leftover batch is skipped.
            warn_once(
                "single-batch",
                "Skipping a 1-image batch because this model uses BatchNorm. "
                "Use a batch_size of at least 2.",
            )
            continue
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        batch = labels.size(0)
        total_loss += loss.item() * batch
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        seen += batch
    if seen == 0:
        raise RuntimeError(
            "No training batch was usable. Set train.batch_size to at least 2 "
            "and make sure the training folder has photos."
        )
    return total_loss / seen, correct / seen


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    tta: bool,
) -> dict:
    """Score a loader without updating weights.

    The loader must not be shuffled, so ``paths`` stay aligned with the
    dataset order. Eval mode keeps BatchNorm from learning the validation
    or test photos.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    seen = 0
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    progress = tqdm(loader, desc="eval", leave=False, disable=not sys.stderr.isatty())
    for images, batch_labels in progress:
        images = images.to(device, non_blocking=device.type == "cuda")
        batch_labels = batch_labels.to(device, non_blocking=device.type == "cuda")
        logits = forward_logits(model, images, tta=tta)
        loss = criterion(logits, batch_labels)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)
        batch = batch_labels.size(0)
        total_loss += loss.item() * batch
        correct += int((pred == batch_labels).sum().item())
        seen += batch
        labels.extend(int(value) for value in batch_labels.detach().cpu().tolist())
        predictions.extend(int(value) for value in pred.detach().cpu().tolist())
        probabilities.extend(probs.detach().cpu().tolist())
    if seen == 0:
        raise RuntimeError("There are no photos to score.")
    paths = [sample[0] for sample in loader.dataset.samples]
    if len(paths) != len(labels):
        raise RuntimeError("Internal error: predictions and file paths got out of order.")
    return {
        "loss": total_loss / seen,
        "accuracy": correct / seen,
        "labels": labels,
        "predictions": predictions,
        "probabilities": probabilities,
        "paths": paths,
    }


def _payload(
    model: nn.Module,
    cfg: dict,
    class_names: list[str],
    stage: str,
    epoch: int,
    val_accuracy: float,
) -> dict:
    # CPU tensors so the same file opens on a laptop that has no GPU.
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    return {
        "model_state": state,
        "model_name": cfg["model"]["name"],
        "dropout": float(cfg["model"]["dropout"]),
        "image_size": int(cfg["image_size"]),
        "max_side": int(cfg["data"]["max_side"]),
        "class_names": list(class_names),
        "output_dir": str(cfg["output_dir"]),
        "stage": stage,
        "epoch": epoch,
        "val_accuracy": val_accuracy,
        "config": cfg,
        "torch_version": torch.__version__,
    }


def _run_stage(
    stage: str,
    epochs: int,
    learning_rate: float,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_criterion: nn.Module,
    eval_criterion: nn.Module,
    device: torch.device,
    cfg: dict,
    class_names: list[str],
    output_dir: Path,
    history: list[dict],
    use_amp: bool,
    scaler,
) -> None:
    if epochs <= 0:
        print(f"[{stage}] skipped (epochs is 0).", flush=True)
        return
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("There are no trainable parameters.")
    trainable_count, total_count = parameter_counts(model)
    print(
        f"[{stage}] trainable parameters: {trainable_count:,} / {total_count:,}",
        flush=True,
    )
    # A new optimizer is required when the stage changes. The stage-1
    # optimizer only knows about the last layer. After unfreezing, the
    # backbone parameters have to be added, which means a new optimizer.
    weight_decay = float(cfg["train"]["weight_decay"])
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    stopper = EarlyStopper(int(cfg["train"]["patience"]))
    skip_single = _uses_batchnorm(model)

    for epoch in range(1, epochs + 1):
        # Validation only inside this loop. The test set is not touched here.
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            train_criterion,
            optimizer,
            device,
            scaler,
            use_amp,
            skip_single_batch=skip_single,
            head_only=stage == "head",
        )
        validation = collect_predictions(
            model, val_loader, eval_criterion, device, tta=False
        )
        improved = stopper.step(validation["accuracy"])
        print(
            f"[{stage}] epoch {epoch}/{epochs}  "
            f"train_loss={train_loss:.3f} train_acc={train_acc:.1%}  "
            f"val_loss={validation['loss']:.3f} val_acc={validation['accuracy']:.1%}  "
            f"lr={current_lr:.6f}",
            flush=True,
        )
        history.append(
            {
                "stage": stage,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": validation["loss"],
                "val_acc": validation["accuracy"],
                "lr": current_lr,
            }
        )
        write_json(output_dir / "history.json", history)
        if improved:
            save_checkpoint(
                output_dir / "best.pt",
                _payload(
                    model,
                    cfg,
                    class_names,
                    stage,
                    epoch,
                    validation["accuracy"],
                ),
            )
            print(
                f"  saved best.pt because validation accuracy improved to "
                f"{validation['accuracy']:.1%}",
                flush=True,
            )
        if stopper.should_stop():
            print(
                "Early stopping: validation accuracy did not improve for "
                f"{stopper.patience} epochs.",
                flush=True,
            )
            break
        scheduler.step()


def _score_test(
    model: nn.Module,
    test_dataset: ImageListDataset,
    class_names: list[str],
    device: torch.device,
    cfg: dict,
    tta: bool,
    report_dir: Path,
) -> dict | None:
    if len(test_dataset) == 0:
        print(NO_TEST_MESSAGE, flush=True)
        clear_test_reports(report_dir)
        return None
    loader = make_loader(
        test_dataset,
        batch_size=int(cfg["eval"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        device=device,
        seed=int(cfg["seed"]),
    )
    # Plain cross-entropy on the test set. Label smoothing is a training
    # trick, and using it here would make the loss harder to read.
    criterion = nn.CrossEntropyLoss()
    collected = collect_predictions(model, loader, criterion, device, tta=tta)
    metrics = write_report(
        report_dir,
        class_names,
        collected["labels"],
        collected["predictions"],
        collected["probabilities"],
        collected["paths"],
        tta=tta,
    )
    print_test_report(metrics, report_dir)
    return metrics


def fit(cfg: dict, data_root: Path, csv_path: str | None) -> Path:
    """Train, save ``best.pt``, and score the test set once."""
    cfg = copy.deepcopy(cfg)
    cfg["data_root"] = str(data_root.resolve())
    if csv_path:
        cfg["data"]["csv"] = csv_path
    seed = int(cfg["seed"])
    set_seed(seed)
    device = get_device(str(cfg["device"]))
    print(f"Device: {describe_device(device)}", flush=True)

    manifest = build_manifest(data_root, cfg, csv_path)
    print(f"Layout: {manifest.layout}", flush=True)
    for message in manifest.warnings:
        print(f"Warning: {message}", flush=True)
    class_names = manifest.class_names
    print(f"Classes ({len(class_names)}): {', '.join(class_names)}", flush=True)
    counts = {"train": 0, "val": 0, "test": 0}
    for row in manifest.rows:
        counts[row["split"]] += 1
    print(
        f"Images: train={counts['train']} val={counts['val']} test={counts['test']}",
        flush=True,
    )

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(output_dir / "split_manifest.csv", manifest.rows)
    write_json(output_dir / "class_names.json", class_names)
    dump_config(output_dir / "config.used.yaml", cfg)

    if cfg["duplicates"]["enabled"]:
        print(
            f"Checking {len(manifest.rows)} photos for near-duplicates across splits...",
            flush=True,
        )
        pairs = find_cross_split_duplicates(
            manifest.rows,
            hash_size=int(cfg["duplicates"]["hash_size"]),
            max_distance=int(cfg["duplicates"]["max_distance"]),
            max_side=int(cfg["data"]["max_side"]),
        )
        report_duplicates(pairs, output_dir / "duplicates.csv")
    else:
        print("Near-duplicate check: skipped (duplicates.enabled is false).", flush=True)

    max_side = int(cfg["data"]["max_side"])
    image_size = int(cfg["image_size"])
    if cfg["model"]["name"] == "vit_b_16" and image_size != 224:
        print(
            "Warning: vit_b_16 was designed for image_size 224. "
            f"This run uses {image_size}.",
            flush=True,
        )
    train_ds, val_ds, test_ds = datasets_from_manifest(
        manifest.rows,
        class_names,
        train_transform(image_size, cfg["augment"]),
        eval_transform(image_size),
        max_side=max_side,
    )
    train_loader = make_loader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        device=device,
        seed=seed,
    )
    val_loader = make_loader(
        val_ds,
        batch_size=int(cfg["eval"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        device=device,
        seed=seed,
    )

    model = build_model(
        name=cfg["model"]["name"],
        num_classes=len(class_names),
        pretrained=bool(cfg["model"]["pretrained"]),
        dropout=float(cfg["model"]["dropout"]),
    )
    model.to(device)
    print(
        f"Model: {cfg['model']['name']} "
        f"(ImageNet weights: {'yes' if cfg['model']['pretrained'] and cfg['model']['name'] != 'tiny' else 'no'})",
        flush=True,
    )

    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    if cfg["train"]["amp"] and device.type != "cuda":
        print(
            "Mixed precision stays off: it is only used on NVIDIA GPUs. Training in full precision.",
            flush=True,
        )
    scaler = _make_scaler(use_amp)
    smoothing = float(cfg["train"]["label_smoothing"])
    train_criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
    eval_criterion = nn.CrossEntropyLoss()
    history: list[dict] = []

    print(
        f"Stage 1: train the new head for up to {int(cfg['train']['head_epochs'])} epochs.",
        flush=True,
    )
    freeze_backbone(model)
    _run_stage(
        stage="head",
        epochs=int(cfg["train"]["head_epochs"]),
        learning_rate=float(cfg["train"]["head_lr"]),
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_criterion=train_criterion,
        eval_criterion=eval_criterion,
        device=device,
        cfg=cfg,
        class_names=class_names,
        output_dir=output_dir,
        history=history,
        use_amp=use_amp,
        scaler=scaler,
    )

    print(
        f"Stage 2: train the whole model for up to {int(cfg['train']['full_epochs'])} epochs.",
        flush=True,
    )
    unfreeze_all(model)
    _run_stage(
        stage="full",
        epochs=int(cfg["train"]["full_epochs"]),
        learning_rate=float(cfg["train"]["full_lr"]),
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_criterion=train_criterion,
        eval_criterion=eval_criterion,
        device=device,
        cfg=cfg,
        class_names=class_names,
        output_dir=output_dir,
        history=history,
        use_amp=use_amp,
        scaler=scaler,
    )

    best_path = output_dir / "best.pt"
    if not best_path.is_file():
        raise RuntimeError("Training finished without saving a checkpoint.")
    # last.pt is the final epoch, which can be worse than the best one.
    last_record = history[-1] if history else {"stage": "none", "epoch": 0, "val_acc": 0.0}
    save_checkpoint(
        output_dir / "last.pt",
        _payload(
            model,
            cfg,
            class_names,
            str(last_record["stage"]),
            int(last_record["epoch"]),
            float(last_record["val_acc"]),
        ),
    )

    # Reload the best weights before touching the test set. The in-memory
    # model is the last epoch, and that epoch was not necessarily the best.
    best_model, _checkpoint = load_trained_model(best_path, device)
    _score_test(
        best_model,
        test_ds,
        class_names,
        device,
        cfg,
        tta=bool(cfg["eval"]["tta"]),
        report_dir=output_dir,
    )
    print(f"Best checkpoint: {best_path}", flush=True)
    return best_path


def evaluate_checkpoint(
    checkpoint_path: Path,
    cfg: dict,
    tta: bool,
    batch_size: int,
    num_workers: int,
) -> dict | None:
    """Score the saved test split with the best checkpoint.

    The file list is the manifest written during training, so evaluate.py
    sees the same held-out photos train.py scored.
    """
    device = get_device(str(cfg.get("device", "auto")))
    model, checkpoint = load_trained_model(checkpoint_path, device)
    trained_name = checkpoint["model_name"]
    if cfg["model"]["name"] != trained_name:
        print(
            f"config model is {cfg['model']['name']}, but this checkpoint is {trained_name}. "
            f"Using {trained_name}.",
            flush=True,
        )
    image_size = int(checkpoint["image_size"])
    if int(cfg["image_size"]) != image_size:
        print(
            f"config image_size is {cfg['image_size']}, but this checkpoint was trained at "
            f"{image_size}. Using {image_size}.",
            flush=True,
        )
    print(
        f"Device: {describe_device(device)}",
        flush=True,
    )
    print(
        f"Model: {trained_name}  image_size={image_size}  "
        f"classes={len(checkpoint['class_names'])}",
        flush=True,
    )
    manifest_path = checkpoint_path.resolve().parent / "split_manifest.csv"
    rows = read_manifest(manifest_path)
    class_names = list(checkpoint["class_names"])
    samples = samples_for_split(rows, "test", class_names)
    dataset = ImageListDataset(
        samples,
        transform=eval_transform(image_size),
        max_side=int(checkpoint.get("max_side", cfg["data"]["max_side"])),
    )
    saved_cfg = checkpoint.get("config", cfg)
    score_cfg = copy.deepcopy(saved_cfg)
    score_cfg["eval"]["batch_size"] = batch_size
    score_cfg["train"]["num_workers"] = num_workers
    score_cfg["device"] = cfg.get("device", "auto")
    report_dir = checkpoint_path.resolve().parent
    return _score_test(
        model,
        dataset,
        class_names,
        device,
        score_cfg,
        tta=tta,
        report_dir=report_dir,
    )
