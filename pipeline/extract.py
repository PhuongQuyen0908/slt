"""
pipeline/extract.py

Bước 2 của pipeline: đọc videos, chạy FrameParser, lưu raw crops + pose + confidence.

Input:  thư mục chứa .mp4 clips
Output: thư mục chứa .npz files, mỗi file tương ứng 1 clip

Structure của mỗi .npz:
  face_crops:  (T, 224, 224, 3) uint8
  lhand_crops: (T, 224, 224, 3) uint8
  rhand_crops: (T, 224, 224, 3) uint8
  pose_vecs:   (T, 14) float32
  conf_face:   (T,) float32
  conf_lhand:  (T,) float32
  conf_rhand:  (T,) float32
  conf_pose:   (T,) float32

Usage:
  python -m pipeline.extract \
      --video_dir data/raw/how2sign/train \
      --out_dir   data/extracted/how2sign/train \
      --stride    2 \
      --workers   4
"""

import argparse
import logging
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from data.frame_parser import FrameParser, FrameResult


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"extract_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-clip extraction
# ---------------------------------------------------------------------------


def extract_clip(
    video_path: Path,
    out_path: Path,
    stride: int = 2,
    min_frames: int = 4,
) -> tuple[bool, str]:
    """
    Extract một video clip → lưu .npz.

    Returns (success, error_message)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, "cannot_open"

    parser = FrameParser()

    face_crops = []
    lhand_crops = []
    rhand_crops = []
    pose_vecs = []
    conf_face = []
    conf_lhand = []
    conf_rhand = []
    conf_pose = []

    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % stride == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result: FrameResult = parser.parse(frame_rgb)

                face_crops.append(result.face_crop)
                lhand_crops.append(result.lhand_crop)
                rhand_crops.append(result.rhand_crop)
                pose_vecs.append(result.pose_vec)
                conf_face.append(result.confidences["face"])
                conf_lhand.append(result.confidences["lhand"])
                conf_rhand.append(result.confidences["rhand"])
                conf_pose.append(result.confidences["pose"])

            frame_idx += 1

    finally:
        cap.release()
        parser.close()

    T = len(face_crops)
    if T < min_frames:
        return False, f"too_short ({T} frames)"

    # Lưu .npz
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        face_crops=np.stack(face_crops, axis=0).astype(np.uint8),  # (T,224,224,3)
        lhand_crops=np.stack(lhand_crops, axis=0).astype(np.uint8),
        rhand_crops=np.stack(rhand_crops, axis=0).astype(np.uint8),
        pose_vecs=np.stack(pose_vecs, axis=0).astype(np.float32),  # (T,14)
        conf_face=np.array(conf_face, dtype=np.float32),  # (T,)
        conf_lhand=np.array(conf_lhand, dtype=np.float32),
        conf_rhand=np.array(conf_rhand, dtype=np.float32),
        conf_pose=np.array(conf_pose, dtype=np.float32),
    )

    return True, None


# ---------------------------------------------------------------------------
# Worker function (dùng với multiprocessing)
# ---------------------------------------------------------------------------


def _worker(args):
    """Worker đơn giản — không có thread nội bộ, không có timeout.
    Timeout được xử lý bởng main process qua apply_async().get(timeout=).
    """
    video_path, out_path, stride = args
    try:
        ok, err = extract_clip(video_path, out_path, stride=stride)
        return str(video_path.stem), ok, err
    except Exception as e:
        return str(video_path.stem), False, str(e)


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------


def extract_all(
    video_dir: str,
    out_dir: str,
    stride: int = 2,
    workers: int = 4,
    overwrite: bool = False,
    task_timeout: int = 300,
):
    """
    Extract tất cả clips trong video_dir.

    workers=4: mỗi worker xử lý 1 clip độc lập.
    task_timeout: số giây tối đa cho một clip (default 300s = 5 phút).
                  Clip nào quá timeout sẽ bị bỏ qua và ghi vào failed_clips.txt.
    MediaPipe chạy trên CPU nên parallel workers giúp ích nhiều.
    """
    video_dir = Path(video_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_dir)

    # Tìm tất cả video files
    video_paths = sorted(list(video_dir.glob("*.mp4")))
    if not video_paths:
        logger.error(f"No .mp4 files found in {video_dir}")
        return

    logger.info(
        f"Found {len(video_paths)} videos | stride={stride} | workers={workers}"
    )

    # Build task list (skip đã extract nếu không overwrite)
    tasks = []
    skipped = 0
    for vp in video_paths:
        out_path = out_dir / f"{vp.stem}.npz"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue
        tasks.append((vp, out_path, stride))

    logger.info(f"To extract: {len(tasks)} | Already done: {skipped}")

    if not tasks:
        logger.info("Nothing to do.")
        return

    # Run với mp.Pool + per-task timeout từ main process (apply_async + get(timeout=)).
    # Khi timeout: terminate pool → tạo lại pool → tiếp tục. Tránh double-MediaPipe
    # vì mỗi lần terminate toàn bộ worker bị kill, không có instance cũ còn sót.
    success = failed = 0
    failed_clips = []

    if workers > 1:
        remaining = list(tasks)
        with tqdm(total=len(tasks), desc="Extracting") as pbar:
            while remaining:
                pool = mp.Pool(processes=workers)
                # Mỗi batch: gửi trước workers*2 tasks để pipeline được fill
                batch_size = min(workers * 2, len(remaining))
                batch = remaining[:batch_size]
                remaining = remaining[batch_size:]

                async_results = [
                    (str(t[0].stem), t[1], pool.apply_async(_worker, (t,)))
                    for t in batch
                ]

                pool_ok = True
                for idx, (stem, out_path_i, ar) in enumerate(async_results):
                    try:
                        stem_r, ok, err = ar.get(timeout=task_timeout)
                        if ok:
                            success += 1
                        else:
                            failed += 1
                            failed_clips.append((stem_r, err))
                        pbar.update(1)
                    except mp.TimeoutError:
                        logger.warning(f"TIMEOUT (skipped): {stem}")
                        pool.terminate()
                        pool.join()
                        failed += 1
                        failed_clips.append((stem, f"timeout>{task_timeout}s"))
                        pbar.update(1)
                        pool_ok = False
                        # Xóa .npz có thể bị viết dở của các task chưa collect xong
                        for j in range(idx, len(batch)):
                            p_out = batch[j][1]
                            if p_out.exists():
                                p_out.unlink(missing_ok=True)
                        # Đưa lại task chưa xử lý vào hàng đợi (sẽ retry lần sau)
                        remaining = batch[idx + 1 :] + remaining
                        break
                    except Exception as e:
                        failed += 1
                        failed_clips.append((stem, str(e)))
                        pbar.update(1)

                if pool_ok:
                    pool.close()
                    pool.join()
    else:
        # Single process (dễ debug)
        for task in tqdm(tasks, desc="Extracting"):
            stem, ok, err = _worker(task)
            if ok:
                success += 1
            else:
                failed += 1
                failed_clips.append((stem, err))

    # Log failures
    if failed_clips:
        fail_log = out_dir / "failed_clips.txt"
        with open(fail_log, "w") as f:
            for stem, err in failed_clips:
                f.write(f"{stem}\t{err}\n")
        logger.warning(f"Failed clips logged to {fail_log}")

    logger.info("=" * 60)
    logger.info(f"DONE | success={success} | failed={failed} | skipped={skipped}")
    coverage = (success + skipped) / len(video_paths) * 100
    logger.info(f"Coverage: {coverage:.1f}%")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--task_timeout",
        type=int,
        default=300,
        help="Timeout (giây) cho mỗi video clip (default: 300)",
    )
    args = parser.parse_args()

    extract_all(
        video_dir=args.video_dir,
        out_dir=args.out_dir,
        stride=args.stride,
        workers=args.workers,
        overwrite=args.overwrite,
        task_timeout=args.task_timeout,
    )
