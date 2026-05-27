from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

try:
    from clearml import Task, Logger as ClearMLLogger
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False

from models.lighting.model   import LightingClassifier, export_onnx, LABEL_NAMES
from models.lighting.dataset import LightingDataset


PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 10
LR_PHASE1     = 1e-2
LR_PHASE2     = 5e-4
MOMENTUM      = 0.9
WEIGHT_DECAY  = 1e-4
BATCH_SIZE    = 32
VAL_SPLIT     = 0.15
PATIENCE      = 5


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def per_class_accuracy(all_preds: list, all_labels: list, num_classes: int = 3) -> dict:
    preds  = torch.tensor(all_preds)
    labels = torch.tensor(all_labels)
    result = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            result[LABEL_NAMES[c]] = 0.0
        else:
            result[LABEL_NAMES[c]] = (preds[mask] == labels[mask]).float().mean().item()
    return result


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device) -> dict:
    model.train(True)
    total_loss, total_acc, n = 0.0, 0.0, 0

    for frames, labels in loader:
        frames, labels = frames.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(frames)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs          = frames.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy(logits, labels) * bs
        n          += bs

    return {"loss": total_loss / n, "acc": total_acc / n}


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    model.train(False)
    total_loss, n = 0.0, 0
    all_preds, all_labels = [], []

    for frames, labels in loader:
        frames, labels = frames.to(device), labels.to(device)
        logits  = model(frames)
        loss    = criterion(logits, labels)
        preds   = logits.argmax(dim=1)

        total_loss  += loss.item() * frames.size(0)
        n           += frames.size(0)
        all_preds   .extend(preds.cpu().tolist())
        all_labels  .extend(labels.cpu().tolist())

    acc      = sum(p == l for p, l in zip(all_preds, all_labels)) / n
    per_cls  = per_class_accuracy(all_preds, all_labels)

    return {
        "loss":      total_loss / n,
        "acc":       acc,
        "per_class": per_cls,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv",   required=True)
    parser.add_argument("--output-dir", default="models/weights")
    parser.add_argument("--epochs",     type=int, default=PHASE1_EPOCHS + PHASE2_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--no-clearml", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    use_clearml = CLEARML_AVAILABLE and not args.no_clearml
    clearml_task = None
    if use_clearml:
        clearml_task = Task.init(
            project_name="VitalLens",
            task_name="lighting-mobilenetv3-finetune",
            task_type=Task.TaskTypes.training,
            tags=["lighting-classifier", "mobilenetv3", "ubfc-synthetic"],
        )
        clearml_task.connect({
            "model": "MobileNetV3-Small",
            "phase1_epochs": PHASE1_EPOCHS, "phase2_epochs": PHASE2_EPOCHS,
            "lr_phase1": LR_PHASE1, "lr_phase2": LR_PHASE2,
            "batch_size": args.batch_size, "dataset": args.data_csv,
        })
    full_dataset = LightingDataset(args.data_csv, augment=False)
    indices      = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=VAL_SPLIT, random_state=42,
        stratify=[full_dataset.samples[i]["label"] for i in indices],
    )

    train_dataset = LightingDataset(args.data_csv, augment=True)
    val_dataset   = LightingDataset(args.data_csv, augment=False)

    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_idx),
        batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")

    model     = LightingClassifier(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()

    best_val_acc    = 0.0
    best_ckpt_path  = output_dir / "lighting_best.pt"
    epochs_no_improve = 0
    total_epochs    = args.epochs

    for epoch in range(1, total_epochs + 1):

        # --- Phase transition ---
        if epoch == 1:
            # Phase 1: frozen backbone
            model.freeze_backbone()
            optimizer = torch.optim.SGD(
                model.trainable_params(),
                lr=LR_PHASE1, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=PHASE1_EPOCHS
            )

        elif epoch == PHASE1_EPOCHS + 1:
            # Phase 2: unfreeze top 3 layers
            model.unfreeze_top_layers(3)
            optimizer = torch.optim.SGD(
                model.trainable_params(),
                lr=LR_PHASE2, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=PHASE2_EPOCHS
            )

        # --- Train + evaluate ---
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        phase = 1 if epoch <= PHASE1_EPOCHS else 2

        print(
            f"Epoch {epoch:3d}/{total_epochs} | Phase {phase} | "
            f"Train loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f} | "
            f"Val loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f} | "
            f"lr={lr:.6f}"
        )
        print(f"  Per-class val acc: {val_metrics['per_class']}")

        if use_clearml:
            logger = ClearMLLogger.current_logger()
            logger.report_scalar("loss", "train", train_metrics["loss"], epoch)
            logger.report_scalar("loss", "val",   val_metrics["loss"],   epoch)
            logger.report_scalar("accuracy", "train", train_metrics["acc"], epoch)
            logger.report_scalar("accuracy", "val",   val_metrics["acc"],   epoch)
            logger.report_scalar("lr", "lr", lr, epoch)
            for cls, acc in val_metrics["per_class"].items():
                logger.report_scalar("per_class_acc", cls, acc, epoch)

        # --- Checkpoint best model ---
        if val_metrics["acc"] > best_val_acc:
            best_val_acc      = val_metrics["acc"]
            epochs_no_improve = 0
            model.save(best_ckpt_path)
            print(f"  Best val acc: {best_val_acc:.4f} — saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE and epoch > PHASE1_EPOCHS:
                print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    print(f"\nBest val accuracy: {best_val_acc:.4f}")

    results = {
        "best_val_acc": best_val_acc,
        "total_epochs": epoch,
        "device":       str(device),
    }
    with open(output_dir / "lighting_results.json", "w") as f:
        json.dump(results, f, indent=2)

    best_model = LightingClassifier.load(best_ckpt_path, device=str(device))
    export_onnx(best_model, output_path=output_dir / "vitallens_lighting.onnx")

    if use_clearml:
        clearml_task.upload_artifact("lighting_onnx", output_dir / "vitallens_lighting.onnx")
        clearml_task.close()

    print("Training complete.")


if __name__ == "__main__":
    main()
