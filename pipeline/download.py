"""
pipeline/download.py

Download video clips từ CSV metadata.

How2Sign CSV format (tab-separated):
  SENTENCE_NAME  VIDEO_ID  START  END  SENTENCE

YouTube-ASL CSV format:
  video_id  start  end  text

Usage:
  python -m pipeline.download --dataset how2sign \
      --csv data/how2sign_train.csv \
      --out data/raw/how2sign/train

  python -m pipeline.download --dataset youtube_asl \
      --csv data/youtube_asl.csv \
      --out data/raw/youtube_asl
"""

import argparse
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core download function
# ---------------------------------------------------------------------------


def download_clip(
    video_id: str,
    start: float,
    end: float,
    out_path: Path,
    buffer: float = 1.0,
    timeout: int = 120,
    retries: int = 2,
) -> tuple[bool, str | None]:
    """
    Download một clip từ YouTube dùng yt-dlp.

    buffer: thêm 1s vào đầu và cuối để đảm bảo không bị thiếu frame.
    Dùng max 480p để tiết kiệm disk (đủ cho 224×224 crops).
    """
    start_sec = max(0.0, float(start) - buffer)
    duration = (float(end) - float(start)) + buffer * 2.0

    url = f"https://www.youtube.com/watch?v={video_id}"

    cmd = [
        "yt-dlp",
        "-f",
        "bestvideo[height<=480][ext=mp4]+bestaudio/best[ext=m4a]/best",
        "--download-sections",
        f"*{start_sec:.2f}-{start_sec + duration:.2f}",
        "--merge-output-format",
        "mp4",
        "-o",
        str(out_path),
        "--quiet",
        "--no-playlist",
        "--no-warnings",
        "--retries",
        "3",
    ]

    for attempt in range(retries + 1):
        try:
            result = subprocess.run(
                cmd + [url],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                # Verify file thực sự tồn tại và có size > 0
                if out_path.exists() and out_path.stat().st_size > 1024:
                    return True, None
                else:
                    return False, "empty_file"

            error = (
                result.stderr.strip().splitlines()[-1] if result.stderr else "unknown"
            )

            # Một số lỗi không đáng retry
            if any(
                x in error for x in ["Private video", "Video unavailable", "removed"]
            ):
                return False, f"unavailable: {error}"

            if attempt < retries:
                time.sleep(2**attempt)  # Exponential backoff

        except subprocess.TimeoutExpired:
            if attempt < retries:
                time.sleep(5)
                continue
            return False, "timeout"
        except Exception as e:
            return False, str(e)

    return False, f"failed after {retries + 1} attempts: {error}"


# ---------------------------------------------------------------------------
# How2Sign downloader
# ---------------------------------------------------------------------------


def download_how2sign(csv_path: str, output_dir: str, split: str = "train"):
    """
    Download How2Sign clips.

    CSV columns: SENTENCE_NAME, VIDEO_ID, START (or START_REALIGNED), END (or END_REALIGNED), SENTENCE
    Output: output_dir/{SENTENCE_NAME}.mp4
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)

    df = pd.read_csv(csv_path, sep="\t")
    # Normalize column names (some CSVs có whitespace)
    df.columns = df.columns.str.strip()

    # Handle both original (START/END) and realigned (START_REALIGNED/END_REALIGNED) columns
    if "START_REALIGNED" in df.columns and "START" not in df.columns:
        df = df.rename(columns={"START_REALIGNED": "START", "END_REALIGNED": "END"})

    required = {"SENTENCE_NAME", "VIDEO_ID", "START", "END"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Got: {list(df.columns)}")

    total = len(df)
    success = failed = skipped = 0

    logger.info(f"How2Sign {split} | {total} clips | output: {output_dir}")

    for idx, row in df.iterrows():
        name = row["SENTENCE_NAME"]
        out_path = output_dir / f"{name}.mp4"

        if out_path.exists() and out_path.stat().st_size > 1024:
            skipped += 1
            if skipped % 100 == 0:
                logger.info(
                    f"[{idx + 1}/{total}] SKIP (already downloaded {skipped} clips)"
                )
            continue

        ok, error = download_clip(
            video_id=str(row["VIDEO_ID"]),
            start=row["START"],
            end=row["END"],
            out_path=out_path,
        )

        if ok:
            success += 1
            size_mb = out_path.stat().st_size / 1e6
            logger.info(f"[{idx + 1}/{total}] OK   | {name} | {size_mb:.1f} MB")
        else:
            failed += 1
            if out_path.exists():
                out_path.unlink()
            logger.warning(f"[{idx + 1}/{total}] FAIL | {name} | {error}")

    _log_summary(logger, total, success, failed, skipped)


# ---------------------------------------------------------------------------
# YouTube-ASL downloader
# ---------------------------------------------------------------------------


def download_youtube_asl(csv_path: str, output_dir: str):
    """
    Download YouTube-ASL clips.

    CSV columns: video_id, start, end, text
    Output: output_dir/{video_id}_{start:.1f}_{end:.1f}.mp4
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    total = len(df)
    success = failed = skipped = 0

    logger.info(f"YouTube-ASL | {total} clips | output: {output_dir}")

    for idx, row in df.iterrows():
        video_id = str(row["video_id"])
        start = float(row["start"])
        end = float(row["end"])

        # Clip name: video_id + timestamps để unique
        clip_name = f"{video_id}_{start:.1f}_{end:.1f}"
        out_path = output_dir / f"{clip_name}.mp4"

        if out_path.exists() and out_path.stat().st_size > 1024:
            skipped += 1
            continue

        ok, error = download_clip(
            video_id=video_id,
            start=start,
            end=end,
            out_path=out_path,
        )

        if ok:
            success += 1
            if success % 500 == 0:
                size_mb = out_path.stat().st_size / 1e6
                logger.info(f"[{idx + 1}/{total}] OK | {clip_name} | {size_mb:.1f} MB")
        else:
            failed += 1
            if out_path.exists():
                out_path.unlink()
            if failed % 100 == 0:
                logger.warning(f"[{idx + 1}/{total}] FAIL | {clip_name} | {error}")

    _log_summary(logger, total, success, failed, skipped)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _log_summary(logger, total, success, failed, skipped):
    logger.info("=" * 60)
    logger.info(
        f"DONE | total={total} | success={success} | "
        f"failed={failed} | skipped={skipped}"
    )
    coverage = (success + skipped) / total * 100 if total > 0 else 0
    logger.info(f"Coverage: {coverage:.1f}%")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["how2sign", "youtube_asl"])
    parser.add_argument("--csv", required=True, help="Path to CSV metadata file")
    parser.add_argument("--out", required=True, help="Output directory for videos")
    parser.add_argument("--split", default="train", help="Split name (for logging)")
    args = parser.parse_args()

    if args.dataset == "how2sign":
        download_how2sign(args.csv, args.out, args.split)
    else:
        download_youtube_asl(args.csv, args.out)
