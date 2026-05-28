"""
pipeline/precompute.py

Bước 3 của pipeline: chạy DINOv2-F và DINOv2-H trên tất cả crops,
lưu features ra disk. QUAN TRỌNG với 1-2 GPU RTX 3090.

Tại sao cần precompute:
  - DINOv2 encoders bị FROZEN trong Phase 2
  - Chạy encoder mỗi epoch = lãng phí ~70% compute
  - Precompute 1 lần → training loop chỉ load features từ disk
  - Tiết kiệm ~3x training time

Input:  thư mục .npz từ extract.py (chứa raw crops)
Output: thư mục .npz chứa features

Structure của output .npz:
  face_feats:  (T, 384) float32
  lhand_feats: (T, 384) float32
  rhand_feats: (T, 384) float32
  pose_vecs:   (T, 14)  float32    (copy từ extracted)
  conf_face:   (T,)     float32    (copy từ extracted)
  conf_lhand:  (T,)     float32
  conf_rhand:  (T,)     float32
  conf_pose:   (T,)     float32

Usage:
  python -m pipeline.precompute \
      --extracted_dir data/extracted/how2sign/train \
      --out_dir       data/features/how2sign/train \
      --face_ckpt     checkpoints/dinov2_face.pt \
      --hand_ckpt     checkpoints/dinov2_hand.pt \
      --batch_size    128 \
      --device        cuda
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.encoders import FaceEncoder, HandEncoder, encode_batch


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"precompute_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset để load crops từ .npz
# ---------------------------------------------------------------------------


class CropDataset(Dataset):
    """
    Load face/lhand/rhand crops từ .npz files.
    Dùng trong DataLoader để batch encode hiệu quả.
    """

    def __init__(self, npz_paths: list[Path]):
        self.paths = npz_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        data = np.load(str(path))
        return {
            "path": str(path),
            "face_crops": data["face_crops"],  # (T, 224, 224, 3) uint8
            "lhand_crops": data["lhand_crops"],
            "rhand_crops": data["rhand_crops"],
            "pose_vecs": data["pose_vecs"],  # (T, 14) float32
            "conf_face": data["conf_face"],  # (T,)
            "conf_lhand": data["conf_lhand"],
            "conf_rhand": data["conf_rhand"],
            "conf_pose": data["conf_pose"],
        }


def crops_to_tensor(crops: np.ndarray) -> torch.Tensor:
    """
    Convert (T, H, W, C) uint8 numpy → (T, C, H, W) float32 tensor [0,1].
    """
    t = torch.from_numpy(crops).float() / 255.0  # (T, H, W, C)
    t = t.permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)
    return t


# ---------------------------------------------------------------------------
# Per-clip precompute
# ---------------------------------------------------------------------------


def precompute_clip(
    npz_path: Path,
    out_path: Path,
    face_encoder: FaceEncoder,
    hand_encoder: HandEncoder,
    encode_batch_size: int,
    device: str,
) -> tuple[bool, str | None]:
    """
    Precompute features cho 1 clip.
    Encode face, lhand, rhand crops qua DINOv2 → lưu .npz.
    """
    try:
        data = np.load(str(npz_path))

        face_crops = crops_to_tensor(data["face_crops"])  # (T, 3, 224, 224)
        lhand_crops = crops_to_tensor(data["lhand_crops"])
        rhand_crops = crops_to_tensor(data["rhand_crops"])

        # Encode từng stream
        with torch.no_grad():
            face_feats = encode_batch(
                face_encoder, face_crops, batch_size=encode_batch_size, device=device
            )
            lhand_feats = encode_batch(
                hand_encoder, lhand_crops, batch_size=encode_batch_size, device=device
            )
            rhand_feats = encode_batch(
                hand_encoder, rhand_crops, batch_size=encode_batch_size, device=device
            )

        # Lưu features + copy pose và confidence
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(out_path),
            face_feats=face_feats.numpy().astype(np.float32),  # (T, 384)
            lhand_feats=lhand_feats.numpy().astype(np.float32),
            rhand_feats=rhand_feats.numpy().astype(np.float32),
            pose_vecs=data["pose_vecs"],  # (T, 14)
            conf_face=data["conf_face"],  # (T,)
            conf_lhand=data["conf_lhand"],
            conf_rhand=data["conf_rhand"],
            conf_pose=data["conf_pose"],
        )

        return True, None

    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Main precompute loop
# ---------------------------------------------------------------------------


def precompute_all(
    extracted_dir: str,
    out_dir: str,
    face_ckpt: str | None = None,
    hand_ckpt: str | None = None,
    encode_batch_size: int = 128,
    device: str = "cuda",
    overwrite: bool = False,
):
    """
    Precompute DINOv2 features cho tất cả clips.

    encode_batch_size=128: fit trong 24GB VRAM với DINOv2-Small.
    Giảm xuống 64 nếu OOM.
    """
    extracted_dir = Path(extracted_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_dir)

    # Load encoders
    logger.info(f"Loading encoders | device={device}")
    logger.info(f"  face_ckpt: {face_ckpt or 'pretrained DINOv2 (no fine-tuning)'}")
    logger.info(f"  hand_ckpt: {hand_ckpt or 'pretrained DINOv2 (no fine-tuning)'}")

    face_encoder = FaceEncoder(checkpoint_path=face_ckpt).to(device)
    hand_encoder = HandEncoder(checkpoint_path=hand_ckpt).to(device)
    face_encoder.eval()
    hand_encoder.eval()

    # Tìm tất cả .npz files
    npz_paths = sorted(list(extracted_dir.glob("*.npz")))
    if not npz_paths:
        logger.error(f"No .npz files in {extracted_dir}")
        return

    logger.info(f"Found {len(npz_paths)} clips to precompute")

    # Build task list
    tasks = []
    skipped = 0
    for npz_path in npz_paths:
        out_path = out_dir / npz_path.name  # Giữ nguyên filename
        if out_path.exists() and not overwrite:
            skipped += 1
            continue
        tasks.append((npz_path, out_path))

    logger.info(f"To precompute: {len(tasks)} | Already done: {skipped}")

    if not tasks:
        logger.info("Nothing to do.")
        return

    # Precompute
    success = failed = 0
    failed_clips = []

    for npz_path, out_path in tqdm(tasks, desc="Precomputing"):
        ok, err = precompute_clip(
            npz_path=npz_path,
            out_path=out_path,
            face_encoder=face_encoder,
            hand_encoder=hand_encoder,
            encode_batch_size=encode_batch_size,
            device=device,
        )

        if ok:
            success += 1
            if success % 500 == 0:
                logger.info(f"Progress: {success}/{len(tasks)} clips done")
        else:
            failed += 1
            failed_clips.append((npz_path.stem, err))
            logger.warning(f"FAIL | {npz_path.stem} | {err}")

    # Log failures
    if failed_clips:
        fail_log = out_dir / "failed_precompute.txt"
        with open(fail_log, "w") as f:
            for stem, err in failed_clips:
                f.write(f"{stem}\t{err}\n")

    # Estimate disk usage
    total_gb = sum(p.stat().st_size for p in out_dir.glob("*.npz")) / 1e9

    logger.info("=" * 60)
    logger.info(f"DONE | success={success} | failed={failed} | skipped={skipped}")
    logger.info(f"Total disk usage: {total_gb:.1f} GB")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--face_ckpt", default=None)
    parser.add_argument("--hand_ckpt", default=None)
    parser.add_argument("--encode_batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    precompute_all(
        extracted_dir=args.extracted_dir,
        out_dir=args.out_dir,
        face_ckpt=args.face_ckpt,
        hand_ckpt=args.hand_ckpt,
        encode_batch_size=args.encode_batch_size,
        device=args.device,
        overwrite=args.overwrite,
    )
