"""
scripts/create_mini_dataset.py

Tạo ~10% subset của How2Sign dataset:
- Lọc những clip có file feature tương ứng
- Sample 10% ngẫu nhiên (seed=42)
- Copy .npz files vào data/new/how2sign/{train,val}/
- Ghi 2 file CSV (same format, tab-separated) ra project root:
    how2sign_train.csv  (mini train)
    how2sign_val.csv    (mini val)

Usage:
    python scripts/create_mini_dataset.py
    python scripts/create_mini_dataset.py --ratio 0.1 --seed 42
"""

import argparse
import csv
import os
import random
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ratio", type=float, default=0.1, help="Fraction to keep (default 0.10)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_csv(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            if row.get("SENTENCE_NAME") and row.get("SENTENCE"):
                rows.append(row)
    return rows


def write_csv(rows: list[dict], out_path: Path, fieldnames: list[str]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def create_split_subset(
    csv_path: Path,
    features_dir: Path,
    out_features_dir: Path,
    ratio: float,
    seed: int,
    out_csv_path: Path,
):
    print(f"\n--- Processing {csv_path.name} ---")

    rows = load_csv(csv_path)
    print(f"  Total CSV rows: {len(rows)}")

    # Filter to rows that have existing .npz feature files
    valid = []
    for row in rows:
        clip_id = row["SENTENCE_NAME"]
        if (features_dir / f"{clip_id}.npz").exists():
            valid.append(row)

    print(f"  Rows with matching .npz files: {len(valid)}")

    # Sample ratio% 
    rng = random.Random(seed)
    n_sample = max(1, int(len(valid) * ratio))
    selected = rng.sample(valid, n_sample)
    print(f"  Selected ({ratio*100:.0f}%): {n_sample}")

    # Copy .npz files
    out_features_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for row in selected:
        clip_id = row["SENTENCE_NAME"]
        src = features_dir / f"{clip_id}.npz"
        dst = out_features_dir / f"{clip_id}.npz"
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1

    print(f"  Copied {copied} new .npz files → {out_features_dir}")

    # Write mini CSV (same tab-separated format)
    fieldnames = list(rows[0].keys())
    write_csv(selected, out_csv_path, fieldnames)
    print(f"  Written CSV → {out_csv_path}")

    return len(selected)


def main():
    args = parse_args()

    root = Path(__file__).parent.parent  # project root

    orig_features_base = root / "data" / "features" / "how2sign"
    orig_data_dir = root / "data"

    new_features_base = root / "data" / "new" / "how2sign"

    # Train split
    n_train = create_split_subset(
        csv_path=orig_data_dir / "how2sign_train.csv",
        features_dir=orig_features_base / "train",
        out_features_dir=new_features_base / "train",
        ratio=args.ratio,
        seed=args.seed,
        out_csv_path=root / "how2sign_train.csv",
    )

    # Val split
    n_val = create_split_subset(
        csv_path=orig_data_dir / "how2sign_val.csv",
        features_dir=orig_features_base / "val",
        out_features_dir=new_features_base / "val",
        ratio=args.ratio,
        seed=args.seed,
        out_csv_path=root / "how2sign_val.csv",
    )

    print(f"\nDone! Mini dataset: {n_train} train clips, {n_val} val clips.")
    print(f"  Features: data/new/how2sign/{{train,val}}/")
    print(f"  CSVs:     how2sign_train.csv, how2sign_val.csv (project root)")


if __name__ == "__main__":
    main()
