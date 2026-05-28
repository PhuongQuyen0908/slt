"""
models/encoders.py

DINOv2-based encoders cho pipeline/precompute.py.

FaceEncoder:  DINOv2-Small fine-tuned trên face crops → (T, 384)
HandEncoder:  DINOv2-Small fine-tuned trên hand crops → (T, 384)
encode_batch: helper function để batch encode crops từ .npz files

Paper Section 3.2: DINOv2-Small (ViT-S/14 + 4 registers)
  - CLS token output: 384-dim
  - Input: 224×224 RGB crops
  - Checkpoints: dinov2_vits14_reg, fine-tuned trên sign language crops
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
from pathlib import Path
from typing import Optional

# DINOv2 normalization (ImageNet stats)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DINOV2_MODEL = "dinov2_vits14_reg"  # ViT-Small + 4 registers (theo paper)
FEATURE_DIM = 384  # DINOv2-Small CLS token dimension


# ---------------------------------------------------------------------------
# Preprocessing transform
# ---------------------------------------------------------------------------


def get_transform() -> T.Compose:
    """
    Standard ImageNet normalization cho DINOv2.
    Input: uint8 numpy (H, W, C) hoặc float tensor (C, H, W) trong [0,1]
    """
    return T.Compose(
        [
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# ---------------------------------------------------------------------------
# Base DINOv2 Encoder
# ---------------------------------------------------------------------------


class DINOv2Encoder(nn.Module):
    """
    Wrapper cho DINOv2-Small.

    Load từ:
      1. Fine-tuned checkpoint (face hoặc hand specific)
      2. Fallback: original dinov2_vits14_reg từ torch.hub
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        super().__init__()

        # Load DINOv2-Small từ torch.hub
        # img_size=224 để khớp với checkpoint (224/14=16 patches → pos_embed [1,257,384])
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            DINOV2_MODEL,
            pretrained=(checkpoint_path is None),
            img_size=224,
            verbose=False,
        )

        # Load fine-tuned weights nếu có checkpoint
        if checkpoint_path is not None:
            ckpt_path = Path(checkpoint_path)
            if ckpt_path.exists():
                state = torch.load(str(ckpt_path), map_location="cpu")
                # Handle các format checkpoint khác nhau
                if "teacher" in state:
                    # DINOv2 training checkpoint: lấy teacher backbone
                    state = state["teacher"]
                    state = {
                        k.replace("backbone.", ""): v
                        for k, v in state.items()
                        if k.startswith("backbone.")
                    }
                elif "model" in state:
                    state = state["model"]
                self.backbone.load_state_dict(state, strict=False)
                print(f"[Encoder] Loaded checkpoint: {ckpt_path}")
            else:
                print(
                    f"[Encoder] WARNING: checkpoint not found at {ckpt_path}. "
                    f"Using pretrained dinov2_vits14_reg."
                )

        self.transform = get_transform()
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, H, W) float32 tensor in [0, 1]
        Returns:
            features: (N, 384) CLS token
        """
        # Normalize
        x = self.transform(x)

        # Forward → CLS token
        features = self.backbone(x)  # (N, 384)
        return features


# ---------------------------------------------------------------------------
# Face and Hand Encoders
# ---------------------------------------------------------------------------


class FaceEncoder(DINOv2Encoder):
    """
    DINOv2-F: fine-tuned trên face crops của sign language videos.
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        super().__init__(checkpoint_path=checkpoint_path)
        print(
            f"[FaceEncoder] Ready. checkpoint={'custom' if checkpoint_path else 'pretrained'}"
        )


class HandEncoder(DINOv2Encoder):
    """
    DINOv2-H: fine-tuned trên hand crops (left + right pooled) của sign language videos.
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        super().__init__(checkpoint_path=checkpoint_path)
        print(
            f"[HandEncoder] Ready. checkpoint={'custom' if checkpoint_path else 'pretrained'}"
        )


# ---------------------------------------------------------------------------
# Batch encode helper
# ---------------------------------------------------------------------------


def encode_batch(
    encoder: DINOv2Encoder,
    crops: torch.Tensor,  # (T, C, H, W) float32 in [0, 1]
    batch_size: int = 128,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Encode một chuỗi crops theo batch để tránh OOM.

    Args:
        encoder:    FaceEncoder hoặc HandEncoder (frozen, eval mode)
        crops:      (T, C, H, W) float32 tensor in [0, 1]
        batch_size: số crops mỗi lần forward
        device:     'cuda' hoặc 'cpu'

    Returns:
        features: (T, 384) float32 trên CPU
    """
    encoder = encoder.to(device)
    encoder.eval()

    T_total = crops.shape[0]
    features = []

    with torch.no_grad():
        for start in range(0, T_total, batch_size):
            end = min(start + batch_size, T_total)
            batch = crops[start:end].to(device)  # (B, C, H, W)
            feat = encoder(batch)  # (B, 384)
            features.append(feat.cpu())

    return torch.cat(features, dim=0)  # (T, 384)
