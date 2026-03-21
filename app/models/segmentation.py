from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.config import (
    SEGMENTATION_CHECKPOINT_PATH,
    SEGMENTATION_CONFIDENCE_THRESHOLD,
    SEGMENTATION_DEVICE,
    SEGMENTATION_ENABLED,
    SEGMENTATION_IMAGE_SIZE,
)


@dataclass
class SegmentationResult:
    foreground_map: np.ndarray
    avg_foreground: float


class SegmentationService:
    """Lazy segmentation predictor using a DeepLabV3+ checkpoint from experiment script."""

    def __init__(self) -> None:
        self._initialized = False
        self._available = False
        self._device = None
        self._torch = None
        self._model = None
        self._mean = (0.485, 0.456, 0.406)
        self._std = (0.229, 0.224, 0.225)
        self._num_classes = 1
        self._image_size = SEGMENTATION_IMAGE_SIZE

    def _lazy_init(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        if not SEGMENTATION_ENABLED:
            return

        ckpt = Path(SEGMENTATION_CHECKPOINT_PATH) if SEGMENTATION_CHECKPOINT_PATH else None
        if ckpt is None or not ckpt.exists():
            return

        try:
            import torch  # type: ignore
            import segmentation_models_pytorch as smp  # type: ignore
        except Exception:
            return

        device_name = SEGMENTATION_DEVICE.strip()
        if not device_name:
            device_name = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            checkpoint = torch.load(str(ckpt), map_location=device_name)
            cfg = checkpoint.get("config", {})
            self._num_classes = int(cfg.get("num_classes", 1))
            self._image_size = int(cfg.get("image_size", self._image_size))
            encoder = str(cfg.get("encoder", "resnet50"))

            model = smp.DeepLabV3Plus(
                encoder_name=encoder,
                encoder_weights=None,
                in_channels=3,
                classes=1 if self._num_classes == 1 else self._num_classes,
                activation=None,
            )
            model.load_state_dict(checkpoint["model"])
            model.to(device_name)
            model.eval()

            self._torch = torch
            self._device = device_name
            self._model = model
            self._available = True
        except Exception:
            self._available = False

    @property
    def available(self) -> bool:
        self._lazy_init()
        return self._available

    def predict_foreground_map(self, image: Image.Image) -> SegmentationResult | None:
        self._lazy_init()
        if not self._available or self._model is None or self._torch is None:
            return None

        torch = self._torch
        src = image.convert("RGB")
        resized = src.resize((self._image_size, self._image_size), Image.BILINEAR)

        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr = (arr - self._mean) / self._std
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)

        if self._num_classes == 1:
            probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)
        else:
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.float32)
            # Treat non-background classes as foreground.
            probs = 1.0 - probs[0]

        binary = (probs >= float(SEGMENTATION_CONFIDENCE_THRESHOLD)).astype(np.float32)
        mask = Image.fromarray((binary * 255).astype(np.uint8), mode="L").resize(src.size, Image.BILINEAR)
        foreground_map = np.asarray(mask, dtype=np.float32) / 255.0

        return SegmentationResult(
            foreground_map=foreground_map,
            avg_foreground=float(foreground_map.mean()),
        )


_SEGMENTATION_SERVICE = SegmentationService()


def get_segmentation_service() -> SegmentationService:
    return _SEGMENTATION_SERVICE
