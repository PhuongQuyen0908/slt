"""
evaluate.py

Evaluation utilities cho SignMusketeers Phase 2.

Functions:
  evaluate_bleu()             — Dùng trong train.py (trả BLEU-4 float)
  evaluate_from_checkpoint()  — Full eval + in bảng so sánh giống paper

BLEU được tính bằng sacrebleu 1.4.14.
"""

import math
import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
import sacrebleu
import yaml
from tqdm import tqdm

from data.dataset import How2SignDataset
from data.collate_fn import collate_fn


# ---------------------------------------------------------------------------
# Paper baselines — Supervised Training Schedule: H2S
# Nguồn: ACL 2025 Findings, Table so sánh (từ paper đính kèm)
# ---------------------------------------------------------------------------

PAPER_BASELINES = [
    {
        "method": "Lin et al. (2023)",
        "bleu1": 14.9,
        "bleu2": 7.3,
        "bleu3": 3.9,
        "bleu4": 2.2,
    },
    {
        "method": "Tarrés et al. (2023)",
        "bleu1": 34.0,
        "bleu2": 19.3,
        "bleu3": 12.2,
        "bleu4": 8.0,
    },
    {
        "method": "Uthus et al. (2023)",
        "bleu1": 15.0,
        "bleu2": 5.1,
        "bleu3": 2.3,
        "bleu4": 1.2,
    },
    {
        "method": "SSVP-SLT (Rust et al., 2024)",
        "bleu1": 38.1,
        "bleu2": 23.7,
        "bleu3": 16.3,
        "bleu4": 11.7,
    },
    {
        "method": "SignMusketeers (paper)",
        "bleu1": 18.8,
        "bleu2": 8.1,
        "bleu3": 4.2,
        "bleu4": 2.4,
    },
]


# ---------------------------------------------------------------------------
# BLEU computation
# ---------------------------------------------------------------------------


def _bleu_n(bp: float, precisions: list, n: int) -> float:
    """
    Tính BLEU-N từ bp và precisions (0–100 scale) của sacrebleu.

    Công thức: BP × exp(1/N × Σ log(pᵢ/100)) × 100
    """
    ps = [p / 100.0 for p in precisions[:n]]
    if any(p <= 0.0 for p in ps):
        return 0.0
    log_avg = sum(math.log(p) for p in ps) / n
    return round(bp * math.exp(log_avg) * 100.0, 1)


def compute_bleu_scores(hypotheses: list, references: list) -> dict:
    """
    Tính BLEU-1, BLEU-2, BLEU-3, BLEU-4 bằng sacrebleu.

    Args:
        hypotheses: list of generated sentences (str)
        references: list of reference sentences (str), cùng thứ tự

    Returns:
        dict với keys 'bleu1', 'bleu2', 'bleu3', 'bleu4'  (float, 0–100 scale)
    """
    result = sacrebleu.corpus_bleu(hypotheses, [references])
    bp = result.bp
    precisions = result.precisions  # [p1, p2, p3, p4] trong 0–100 range

    return {
        "bleu1": _bleu_n(bp, precisions, 1),
        "bleu2": _bleu_n(bp, precisions, 2),
        "bleu3": _bleu_n(bp, precisions, 3),
        "bleu4": round(result.score, 1),  # sacrebleu tự tính BLEU-4
    }


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_translations(
    model: torch.nn.Module,
    dataloader: DataLoader,
    tokenizer: T5Tokenizer,
    device: torch.device,
    max_length: int = 128,
    num_beams: int = 5,
) -> tuple:
    """
    Chạy beam search trên toàn DataLoader.

    Returns:
        (hypotheses, references) — hai list of strings, cùng độ dài
    """
    model.eval()
    hypotheses = []
    references = []

    for batch in tqdm(dataloader, desc="Generating", leave=False):
        # Chuyển tensors lên device, giữ nguyên strings
        batch_gpu = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        labels = batch_gpu.pop("labels")  # (B, L)
        batch_gpu.pop("clip_ids", None)  # strings, không cần

        # Beam search
        generated_ids = model.generate(
            **batch_gpu,
            max_length=max_length,
            num_beams=num_beams,
        )

        # Decode predictions
        hyps = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Decode references (-100 là padding → thay bằng pad_token_id)
        ref_ids = labels.clone()
        ref_ids[ref_ids == -100] = tokenizer.pad_token_id
        refs = tokenizer.batch_decode(ref_ids, skip_special_tokens=True)

        hypotheses.extend(hyps)
        references.extend(refs)

    return hypotheses, references


# ---------------------------------------------------------------------------
# evaluate_bleu — dùng bởi train.py
# ---------------------------------------------------------------------------


def evaluate_bleu(
    model: torch.nn.Module,
    dataloader: DataLoader,
    tokenizer: T5Tokenizer,
    device: torch.device,
    max_length: int = 128,
    num_beams: int = 5,
) -> float:
    """
    Evaluate model trên DataLoader, trả về BLEU-4.
    Được gọi bởi train.py sau mỗi test_every steps.
    """
    hypotheses, references = generate_translations(
        model, dataloader, tokenizer, device, max_length, num_beams
    )
    scores = compute_bleu_scores(hypotheses, references)
    return scores["bleu4"]


# ---------------------------------------------------------------------------
# print_comparison_table
# ---------------------------------------------------------------------------


def print_comparison_table(
    scores: dict,
    split: str,
    model_label: str = "This Run",
    step: int = None,
):
    """
    In bảng so sánh theo format của paper.

    Args:
        scores:      dict với 'bleu1','bleu2','bleu3','bleu4'
        split:       'val' hoặc 'test'
        model_label: tên hiển thị cho dòng kết quả hiện tại
        step:        checkpoint step (optional)
    """
    col_w = 8
    method_w = 32

    header_label = f"Supervised Training Schedule: H2S  |  Split: {split.upper()}" + (
        f"  |  Step: {step}" if step is not None else ""
    )
    sep = "─" * (method_w + col_w * 4 + 3)

    print(f"\n{sep}")
    print(header_label)
    print(sep)
    print(
        f"{'METHOD':<{method_w}}"
        f"{'BLEU-1':>{col_w}}"
        f"{'BLEU-2':>{col_w}}"
        f"{'BLEU-3':>{col_w}}"
        f"{'BLEU':>{col_w}}"
    )
    print(sep)

    for b in PAPER_BASELINES:
        marker = " ◀" if b["method"] == "SignMusketeers (paper)" else ""
        print(
            f"{b['method']:<{method_w}}"
            f"{b['bleu1']:>{col_w}.1f}"
            f"{b['bleu2']:>{col_w}.1f}"
            f"{b['bleu3']:>{col_w}.1f}"
            f"{b['bleu4']:>{col_w}.1f}"
            f"{marker}"
        )

    print(sep)

    label = f"{model_label} (ours)"
    delta4 = scores["bleu4"] - 2.4  # diff với SignMusketeers paper
    delta_str = f"  ({delta4:+.1f} vs paper)"
    print(
        f"{label:<{method_w}}"
        f"{scores['bleu1']:>{col_w}.1f}"
        f"{scores['bleu2']:>{col_w}.1f}"
        f"{scores['bleu3']:>{col_w}.1f}"
        f"{scores['bleu4']:>{col_w}.1f}"
        f"{delta_str}"
    )
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# evaluate_from_checkpoint — dùng bởi main.py
# ---------------------------------------------------------------------------


def evaluate_from_checkpoint(
    ckpt_path: str,
    model_type: str,
    config: dict,
    data_dir: str,
    features_dir: str,
    split: str = "val",
    device: str = "cuda",
) -> float:
    """
    Load checkpoint, chạy evaluation, in bảng so sánh.

    Args:
        ckpt_path:    đường dẫn đến file .pt
        model_type:   'baseline' hoặc 'ca_csa'
        config:       dict cfg['model'] từ YAML
        data_dir:     thư mục chứa how2sign_{split}.csv
        features_dir: thư mục gốc chứa train/val/test sub-folders
        split:        'val' hoặc 'test'
        device:       'cuda' hoặc 'cpu'

    Returns:
        BLEU-4 score (float)
    """
    from models.baseline import SignMusketeersBaseline
    from models.ca_csa import SignMusketeersCACSA

    _device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {_device} | Checkpoint: {ckpt_path}")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    if model_type == "baseline":
        model = SignMusketeersBaseline(
            t5_model_name=config["t5_model_name"],
            fusion_dropout=config.get("dropout", 0.1),
            label_smoothing=config.get("label_smoothing", 0.2),
        )
    elif model_type == "ca_csa":
        model = SignMusketeersCACSA(
            t5_model_name=config["t5_model_name"],
            stream_dim=config.get("stream_dim", 256),
            num_heads=config.get("num_heads", 4),
            num_layers=config.get("num_layers", 2),
            ffn_ratio=config.get("ffn_dim_ratio", 4),
            alpha=config.get("alpha", 0.5),
            dropout=config.get("dropout", 0.1),
            label_smoothing=config.get("label_smoothing", 0.2),
        )
    else:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. Use 'baseline' or 'ca_csa'."
        )

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model = model.to(_device)
    model.eval()

    ckpt_step = ckpt.get("step")
    ckpt_bleu = ckpt.get("best_bleu", 0.0)
    print(f"[Eval] Step: {ckpt_step} | Recorded best BLEU: {ckpt_bleu:.2f}")

    # ------------------------------------------------------------------
    # Tokenizer + Dataset
    # ------------------------------------------------------------------
    t5_name = config["t5_model_name"]
    tokenizer = T5Tokenizer.from_pretrained(t5_name)

    features_split_dir = os.path.join(features_dir, split)
    use_conf = model_type == "ca_csa"

    dataset = How2SignDataset(
        data_dir=data_dir,
        features_dir=features_split_dir,
        split=split,
        tokenizer=tokenizer,
        max_frames=512,
        use_confidence=use_conf,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=(_device.type == "cuda"),
    )

    print(f"[Eval] Evaluating {len(dataset)} clips on split='{split}'...")

    # ------------------------------------------------------------------
    # Generate + Score
    # ------------------------------------------------------------------
    hypotheses, references = generate_translations(
        model,
        dataloader,
        tokenizer,
        _device,
        max_length=128,
        num_beams=5,
    )
    scores = compute_bleu_scores(hypotheses, references)

    model_label = "CA-CSA" if model_type == "ca_csa" else "Baseline"
    print_comparison_table(scores, split=split, model_label=model_label, step=ckpt_step)

    return scores["bleu4"]
