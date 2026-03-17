from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import segmentation_models_pytorch as smp
except ImportError as exc:  # pragma: no cover - runtime guidance
    raise ImportError(
        "segmentation-models-pytorch が見つかりません。"
        "`uv add segmentation-models-pytorch torch torchvision pillow numpy tqdm` を実行してください。"
    ) from exc


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Batch:
    image: torch.Tensor
    mask: torch.Tensor


class SegmentationDataset(Dataset):
    """images/ と masks/ の同名ファイルペアを読み込む簡易データセット。"""

    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        image_size: int,
        num_classes: int,
    ) -> None:
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.num_classes = num_classes

        image_paths = sorted([p for p in images_dir.iterdir() if p.is_file()])
        self.pairs: List[Tuple[Path, Path]] = []
        for img_path in image_paths:
            mask_path = masks_dir / img_path.name
            if mask_path.exists():
                self.pairs.append((img_path, mask_path))

        if not self.pairs:
            raise ValueError(
                f"画像/マスクのペアが見つかりません: images={images_dir}, masks={masks_dir}"
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Batch:
        image_path, mask_path = self.pairs[idx]

        image = Image.open(image_path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        mask = Image.open(mask_path).convert("L").resize(
            (self.image_size, self.image_size), Image.NEAREST
        )

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_np = (image_np - IMAGENET_MEAN) / IMAGENET_STD
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()

        mask_np = np.asarray(mask, dtype=np.int64)
        if self.num_classes == 1:
            mask_np = (mask_np > 127).astype(np.float32)
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
        else:
            mask_np = np.clip(mask_np, 0, self.num_classes - 1)
            mask_tensor = torch.from_numpy(mask_np)

        return Batch(image=image_tensor, mask=mask_tensor)


def collate_fn(batch: List[Batch]) -> Batch:
    images = torch.stack([b.image for b in batch], dim=0)
    masks = torch.stack([b.mask for b in batch], dim=0)
    return Batch(image=images, mask=masks)


def build_model(encoder_name: str, num_classes: int) -> nn.Module:
    return smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1 if num_classes == 1 else num_classes,
        activation=None,
    )


def freeze_batch_norm_layers(model: nn.Module) -> None:
    """小バッチ学習でBatchNormが不安定になるのを防ぐ。"""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False


def compute_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    if num_classes == 1:
        return nn.BCEWithLogitsLoss()(logits, target)
    return nn.CrossEntropyLoss()(logits, target.long())


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    freeze_batch_norm: bool,
) -> float:
    model.train()
    if freeze_batch_norm:
        freeze_batch_norm_layers(model)

    total_loss = 0.0

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch.image.to(device)
        masks = batch.mask.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = compute_loss(logits, masks, num_classes)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    stats: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    for batch in tqdm(loader, desc="valid", leave=False):
        images = batch.image.to(device)
        masks = batch.mask.to(device)

        logits = model(images)
        loss = compute_loss(logits, masks, num_classes)
        total_loss += loss.item()

        if num_classes == 1:
            preds = (torch.sigmoid(logits) > 0.5).long()
            target = masks.long()
            tp, fp, fn, tn = smp.metrics.get_stats(
                preds,
                target,
                mode="binary",
            )
        else:
            preds = torch.argmax(logits, dim=1)
            target = masks.long()
            tp, fp, fn, tn = smp.metrics.get_stats(
                preds,
                target,
                mode="multiclass",
                num_classes=num_classes,
            )

        stats.append((tp, fp, fn, tn))

    tp = torch.cat([s[0] for s in stats], dim=0)
    fp = torch.cat([s[1] for s in stats], dim=0)
    fn = torch.cat([s[2] for s in stats], dim=0)
    tn = torch.cat([s[3] for s in stats], dim=0)
    iou = float(smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item())
    return {"loss": total_loss / max(len(loader), 1), "iou": iou}


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: Dict[str, str | int | float],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
        },
        checkpoint_path,
    )


def run_train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_ds = SegmentationDataset(
        images_dir=Path(args.train_images),
        masks_dir=Path(args.train_masks),
        image_size=args.image_size,
        num_classes=args.num_classes,
    )
    valid_ds = SegmentationDataset(
        images_dir=Path(args.valid_images),
        masks_dir=Path(args.valid_masks),
        image_size=args.image_size,
        num_classes=args.num_classes,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    model = build_model(args.encoder, args.num_classes).to(device)
    freeze_batch_norm = args.batch_size == 1
    if freeze_batch_norm:
        print("batch_size=1 のため BatchNorm を凍結して学習します")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: List[Dict[str, float]] = []
    best_iou = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.num_classes,
            freeze_batch_norm=freeze_batch_norm,
        )
        valid_metrics = validate_one_epoch(model, valid_loader, device, args.num_classes)

        log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_metrics["loss"],
            "valid_iou": valid_metrics["iou"],
        }
        history.append(log)
        print(json.dumps(log, ensure_ascii=False))

        if valid_metrics["iou"] > best_iou:
            best_iou = valid_metrics["iou"]
            save_checkpoint(
                checkpoint_path=Path(args.output_dir) / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=vars(args),
            )

    metrics_path = Path(args.output_dir) / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"metrics saved: {metrics_path}")


@torch.no_grad()
def run_predict(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    num_classes = int(checkpoint["config"]["num_classes"])
    encoder = str(checkpoint["config"]["encoder"])
    image_size = int(checkpoint["config"]["image_size"])

    model = build_model(encoder, num_classes)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    image = Image.open(args.input).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    normalized = (image_np - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    logits = model(tensor)

    if num_classes == 1:
        mask = (torch.sigmoid(logits)[0, 0] > 0.5).cpu().numpy().astype(np.uint8) * 255
    else:
        class_map = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
        mask = (class_map * (255 // max(num_classes - 1, 1))).astype(np.uint8)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(output_path)
    print(f"mask saved: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepLabv3+ 実験スクリプト")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="DeepLabv3+ の学習")
    train.add_argument("--train-images", required=True)
    train.add_argument("--train-masks", required=True)
    train.add_argument("--valid-images", required=True)
    train.add_argument("--valid-masks", required=True)
    train.add_argument("--output-dir", default="experiment/runs/deeplabv3plus")
    train.add_argument("--encoder", default="resnet50")
    train.add_argument("--num-classes", type=int, default=1)
    train.add_argument("--image-size", type=int, default=512)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--num-workers", type=int, default=2)
    train.add_argument("--cpu", action="store_true")

    predict = subparsers.add_parser("predict", help="学習済み重みで推論")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--input", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--cpu", action="store_true")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "predict":
        run_predict(args)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()