"""
models/visual.py

Phase 1: DINOv2 self-supervised fine-tuning trên sign language crops.

Theo SignMusketeers paper Section 3.2:
  - Backbone: DINOv2-Small (ViT-S/14 + 4 registers)
  - Separate fine-tuning cho FaceEncoder và HandEncoder
  - Training: iBOT (image-level DINO + patch-level BEiT) objective
  - Student-Teacher với EMA teacher update (momentum=0.9995)
  - 1M crops mỗi loại, batch 256, lr 3e-4

Key classes:
  DINOTrainer        - Training loop wrapper
  MultiCropTransform - Multi-scale crop augmentation cho DINO
  iBOTLoss           - DINO (global) + iBOT (local patch) combined loss

Usage:
  trainer = DINOTrainer(encoder_type="face", data_dir="data/crops/face")
  trainer.train(steps=10000, checkpoint_dir="checkpoints/")
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from pathlib import Path
from typing import Optional, List, Tuple
from torch.utils.data import Dataset, DataLoader

DINOV2_MODEL = "dinov2_vits14_reg"
FEATURE_DIM = 384  # DINOv2-Small output dim
PATCH_SIZE = 14  # ViT-S/14 patch size
CROP_SIZE = 224  # input size


# ---------------------------------------------------------------------------
# Multi-crop augmentation (DINO protocol)
# ---------------------------------------------------------------------------


class MultiCropTransform:
    """
    Tạo 2 global views (224×224) + N local views (96×96) từ mỗi ảnh.
    Theo DINO paper: global crops 0.5–1.0 scale, local crops 0.05–0.5 scale.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        num_local_crops: int = 8,
        global_crop_scale: Tuple[float, float] = (0.5, 1.0),
        local_crop_scale: Tuple[float, float] = (0.05, 0.5),
    ):
        self.num_local_crops = num_local_crops

        flip_and_color = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
            ]
        )

        normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
            ]
        )

        # Global view 1 (with solarize)
        self.global1 = T.Compose(
            [
                T.RandomResizedCrop(CROP_SIZE, scale=global_crop_scale),
                flip_and_color,
                T.RandomSolarize(threshold=128, p=0.2),
                normalize,
            ]
        )

        # Global view 2 (no solarize)
        self.global2 = T.Compose(
            [
                T.RandomResizedCrop(CROP_SIZE, scale=global_crop_scale),
                flip_and_color,
                normalize,
            ]
        )

        # Local views
        self.local = T.Compose(
            [
                T.RandomResizedCrop(96, scale=local_crop_scale),
                flip_and_color,
                normalize,
            ]
        )

    def __call__(self, img) -> List[torch.Tensor]:
        crops = [self.global1(img), self.global2(img)]
        crops += [self.local(img) for _ in range(self.num_local_crops)]
        return crops


# ---------------------------------------------------------------------------
# Crop dataset (loads .png/.jpg crops saved by pipeline/extract.py)
# ---------------------------------------------------------------------------


class CropDataset(Dataset):
    """
    Đọc các frame crop images từ thư mục.
    Mỗi sample là 1 frame crop (face hoặc hand).
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, data_dir: str, transform=None):
        self.paths = [
            p for p in Path(data_dir).rglob("*") if p.suffix.lower() in self.EXTENSIONS
        ]
        if len(self.paths) == 0:
            raise ValueError(f"No images found in {data_dir}")
        self.transform = transform
        print(f"[CropDataset] {len(self.paths)} crops in {data_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        from PIL import Image

        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            return self.transform(img)
        return img


# ---------------------------------------------------------------------------
# EMA teacher update
# ---------------------------------------------------------------------------


def update_teacher_ema(
    student: nn.Module,
    teacher: nn.Module,
    momentum: float,
) -> None:
    """Exponential moving average: theta_t = m * theta_t + (1-m) * theta_s"""
    with torch.no_grad():
        for param_s, param_t in zip(student.parameters(), teacher.parameters()):
            param_t.data.mul_(momentum).add_((1.0 - momentum) * param_s.data)


# ---------------------------------------------------------------------------
# Projection head
# ---------------------------------------------------------------------------


class DINOProjectionHead(nn.Module):
    """
    3-layer MLP projection head (DINO paper Section 4).
    Student: with BN, Teacher: without (output normalized).
    """

    def __init__(
        self,
        in_dim: int = FEATURE_DIM,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        out_dim: int = 65536,
        use_bn: bool = False,
        norm_last: bool = True,
    ):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        layers += [nn.Linear(hidden_dim, hidden_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))

        self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last:
            self.last_layer.weight_g.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return self.last_layer(x)


# ---------------------------------------------------------------------------
# DINO / iBOT Loss (class token + patch token)
# ---------------------------------------------------------------------------


class iBOTLoss(nn.Module):
    """
    Combined DINO (class token) + iBOT (patch tokens) loss.

    DINO loss: cross-entropy between student and teacher softmax distributions.
    iBOT loss: same on masked patch tokens.

    Paper: iBOT: Image BERT Pre-Training with Online Tokenizer (arXiv 2111.07832)
    """

    def __init__(
        self,
        out_dim: int = 65536,
        patch_out_dim: int = 8192,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        teacher_temp_warmup_steps: int = 2000,
        center_momentum: float = 0.9,
        ibot_weight: float = 1.0,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.teacher_temp_warmup_steps = teacher_temp_warmup_steps
        self.center_momentum = center_momentum
        self.ibot_weight = ibot_weight
        self.out_dim = out_dim
        self.patch_out_dim = patch_out_dim

        # Centering vectors (EMA)
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("patch_center", torch.zeros(1, patch_out_dim))

    def _teacher_temp(self, step: int) -> float:
        """Linear warmup từ 0.04→0.04 (stable sau warmup)."""
        if step < self.teacher_temp_warmup_steps:
            return (
                0.02
                + (self.teacher_temp - 0.02) * step / self.teacher_temp_warmup_steps
            )
        return self.teacher_temp

    @torch.no_grad()
    def _update_center(self, teacher_cls: torch.Tensor) -> None:
        """EMA update của centering vector."""
        batch_center = teacher_cls.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (
            1.0 - self.center_momentum
        )

    def forward(
        self,
        student_cls: torch.Tensor,  # (B * n_crops, out_dim)
        teacher_cls: torch.Tensor,  # (B * 2, out_dim)   — only global views
        student_patch: Optional[torch.Tensor] = None,  # (B, N_patches, patch_out_dim)
        teacher_patch: Optional[torch.Tensor] = None,  # (B, N_patches, patch_out_dim)
        step: int = 0,
    ) -> torch.Tensor:
        """
        Trả về scalar loss = DINO_cls_loss + ibot_weight * iBOT_patch_loss.
        """
        t_temp = self._teacher_temp(step)

        # --- DINO class-token loss ---
        teacher_softmax = F.softmax(
            (teacher_cls - self.center) / t_temp, dim=-1
        ).detach()
        n_crops = student_cls.shape[0] // teacher_cls.shape[0]
        loss_cls = torch.tensor(0.0, device=student_cls.device)
        count_cls = 0

        # Cross all crops (student) × global views (teacher)
        batch_size = teacher_cls.shape[0]
        for t_idx in range(2):  # 2 global teacher views
            t_soft = teacher_softmax[
                t_idx * batch_size : (t_idx + 1) * batch_size
            ]  # (B, D)
            for s_idx in range(n_crops):
                if s_idx == t_idx:
                    continue  # không predict chính mình
                s_logit = student_cls[
                    s_idx * batch_size : (s_idx + 1) * batch_size
                ]  # (B, D)
                s_log = F.log_softmax(s_logit / self.student_temp, dim=-1)
                loss_cls += -(t_soft * s_log).sum(dim=-1).mean()
                count_cls += 1

        if count_cls > 0:
            loss_cls /= count_cls

        # EMA update center
        self._update_center(teacher_cls)

        # --- iBOT patch loss ---
        loss_patch = torch.tensor(0.0, device=student_cls.device)
        if student_patch is not None and teacher_patch is not None:
            t_patch_soft = F.softmax(
                (teacher_patch - self.patch_center) / t_temp, dim=-1
            ).detach()
            s_patch_log = F.log_softmax(student_patch / self.student_temp, dim=-1)
            loss_patch = -(t_patch_soft * s_patch_log).sum(dim=-1).mean()

            with torch.no_grad():
                batch_center_patch = teacher_patch.mean(dim=(0, 1), keepdim=True)
                self.patch_center = (
                    self.patch_center * self.center_momentum
                    + batch_center_patch.squeeze() * (1.0 - self.center_momentum)
                )

        return loss_cls + self.ibot_weight * loss_patch


# ---------------------------------------------------------------------------
# DINOTrainer
# ---------------------------------------------------------------------------


class DINOTrainer:
    """
    Phase 1 trainer: fine-tune DINOv2-Small trên sign language crops.

    Theo paper Section 3.2:
    - encoder_type: 'face' hoặc 'hand'
    - 1M crops per encoder type
    - batch_size=256, lr=3e-4, cosine schedule
    - EMA momentum: 0.9995

    Example:
        trainer = DINOTrainer(encoder_type='face', data_dir='data/crops/face')
        trainer.train(steps=10000, out_dir='checkpoints/face')
    """

    def __init__(
        self,
        encoder_type: str = "face",  # 'face' | 'hand'
        data_dir: Optional[str] = None,
        batch_size: int = 256,
        lr: float = 3e-4,
        ema_momentum: float = 0.9995,
        teacher_temp: float = 0.04,
        out_dim: int = 65536,
        device: str = "cuda",
        num_workers: int = 4,
    ):
        self.encoder_type = encoder_type
        self.batch_size = batch_size
        self.lr = lr
        self.ema_momentum = ema_momentum
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # --- Student and Teacher backbones ---
        self.student = torch.hub.load(
            "facebookresearch/dinov2",
            DINOV2_MODEL,
            pretrained=True,
            verbose=False,
        ).to(self.device)

        self.teacher = torch.hub.load(
            "facebookresearch/dinov2",
            DINOV2_MODEL,
            pretrained=True,
            verbose=False,
        ).to(self.device)

        # Teacher: no gradients, initialized from student
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.load_state_dict(self.student.state_dict())

        # --- Projection heads ---
        self.student_head = DINOProjectionHead(
            in_dim=FEATURE_DIM, out_dim=out_dim, use_bn=True
        ).to(self.device)

        self.teacher_head = DINOProjectionHead(
            in_dim=FEATURE_DIM, out_dim=out_dim, use_bn=False
        ).to(self.device)
        for p in self.teacher_head.parameters():
            p.requires_grad_(False)

        # --- Loss ---
        self.criterion = iBOTLoss(out_dim=out_dim, teacher_temp=teacher_temp).to(
            self.device
        )

        # --- Data ---
        transform = MultiCropTransform()
        if data_dir is not None:
            dataset = CropDataset(data_dir, transform=transform)
            self.loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=True,
                pin_memory=True,
            )
        else:
            self.loader = None

        # --- Optimizer ---
        params = list(self.student.parameters()) + list(self.student_head.parameters())
        self.optimizer = torch.optim.AdamW(
            params, lr=lr, weight_decay=0.04, betas=(0.9, 0.95)
        )

    def _cosine_schedule(self, step: int, total_steps: int, base_lr: float) -> float:
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * step / total_steps))

    def train(
        self,
        steps: int = 10_000,
        out_dir: str = "checkpoints",
        save_every: int = 1000,
        log_every: int = 100,
    ) -> None:
        """
        Training loop.

        Args:
            steps:      total training steps
            out_dir:    directory to save checkpoints
            save_every: save checkpoint every N steps
            log_every:  print loss every N steps
        """
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if self.loader is None:
            raise ValueError(
                "data_dir not provided. Pass data_dir when creating DINOTrainer."
            )

        self.student.train()
        self.student_head.train()

        loader_iter = iter(self.loader)
        global_step = 0

        print(
            f"[DINOTrainer] Starting Phase 1 — encoder_type={self.encoder_type}, device={self.device}"
        )

        while global_step < steps:
            try:
                crops_list = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.loader)
                crops_list = next(loader_iter)

            # crops_list: list of (B, C, H, W) tensors — [global1, global2, local...]
            crops = [c.to(self.device) for c in crops_list]
            n_crops = len(crops)
            n_global = 2

            # --- Forward all crops through student backbone ---
            all_cls = []
            for crop in crops:
                cls_token = self.student(crop)  # (B, 384)
                proj = self.student_head(cls_token)  # (B, out_dim)
                all_cls.append(proj)
            student_cls = torch.cat(all_cls, dim=0)  # (B * n_crops, out_dim)

            # --- Forward global crops through teacher backbone ---
            with torch.no_grad():
                teacher_cls_list = []
                for crop in crops[:n_global]:
                    t_cls = self.teacher(crop)
                    t_proj = self.teacher_head(t_cls)
                    teacher_cls_list.append(t_proj)
                teacher_cls = torch.cat(teacher_cls_list, dim=0)  # (B * 2, out_dim)

            # --- Loss ---
            loss = self.criterion(
                student_cls=student_cls,
                teacher_cls=teacher_cls,
                step=global_step,
            )

            # --- Backward ---
            # Cosine LR decay
            new_lr = self._cosine_schedule(global_step, steps, self.lr)
            for pg in self.optimizer.param_groups:
                pg["lr"] = new_lr

            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 3.0)
            self.optimizer.step()

            # --- EMA teacher update ---
            update_teacher_ema(self.student, self.teacher, self.ema_momentum)
            update_teacher_ema(self.student_head, self.teacher_head, self.ema_momentum)

            global_step += 1

            if global_step % log_every == 0:
                print(
                    f"[Step {global_step}/{steps}]  loss={loss.item():.4f}  lr={new_lr:.2e}"
                )

            if global_step % save_every == 0:
                ckpt_path = out_path / f"{self.encoder_type}_step{global_step}.pt"
                torch.save(
                    {
                        "model": self.student.state_dict(),
                        "head": self.student_head.state_dict(),
                        "step": global_step,
                    },
                    str(ckpt_path),
                )
                print(f"[DINOTrainer] Saved checkpoint → {ckpt_path}")

        # Final checkpoint
        final_path = out_path / f"{self.encoder_type}_final.pt"
        torch.save({"model": self.student.state_dict(), "step": steps}, str(final_path))
        print(f"[DINOTrainer] Training complete → {final_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 1: DINOv2 fine-tuning on sign language crops"
    )
    parser.add_argument("--encoder_type", choices=["face", "hand"], required=True)
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory of crop images"
    )
    parser.add_argument("--out_dir", type=str, default="checkpoints/")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=100)
    args = parser.parse_args()

    trainer = DINOTrainer(
        encoder_type=args.encoder_type,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        num_workers=args.workers,
    )
    trainer.train(
        steps=args.steps,
        out_dir=args.out_dir,
        save_every=args.save_every,
        log_every=args.log_every,
    )
