"""
train.py

Training loop for SignMusketeers Phase 2 (supervised SLT).
Supports baseline and CA-CSA models.

Usage:
    python train.py --config configs/baseline.yaml
    python train.py --config configs/ca_csa.yaml
    python train.py --config configs/baseline.yaml --resume logs/baseline/ckpt_step_1000.pt
"""

import os
import time
import random
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from transformers import T5Tokenizer, get_cosine_schedule_with_warmup
import yaml

from data.dataset import How2SignDataset
from data.collate_fn import collate_fn
from models.baseline import SignMusketeersBaseline
from models.ca_csa import SignMusketeersCACSA
from evaluate import evaluate_bleu


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
    )
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        fh = logging.FileHandler(output_dir / "train.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict) -> torch.nn.Module:
    name = cfg["model"]["name"]
    if name == "baseline":
        return SignMusketeersBaseline(
            t5_model_name=cfg["model"]["t5_model_name"],
            fusion_dropout=cfg["model"]["dropout"],
            label_smoothing=cfg["model"].get("label_smoothing", 0.2),
        )
    elif name == "ca_csa":
        return SignMusketeersCACSA(
            t5_model_name=cfg["model"]["t5_model_name"],
            stream_dim=cfg["model"].get("stream_dim", 256),
            num_heads=cfg["model"].get("num_heads", 4),
            num_layers=cfg["model"].get("num_layers", 2),
            ffn_ratio=cfg["model"].get("ffn_dim_ratio", 4),
            alpha=cfg["model"].get("alpha", 0.5),
            dropout=cfg["model"]["dropout"],
            label_smoothing=cfg["model"].get("label_smoothing", 0.2),
        )
    else:
        raise ValueError(
            f"Unknown model name: {name!r}. Choose 'baseline' or 'ca_csa'."
        )


def compute_loss(
    logits: torch.Tensor,  # (B, L, vocab_size)
    labels: torch.Tensor,  # (B, L)  — -100 at padding positions
    label_smoothing: float,
) -> torch.Tensor:
    """Cross-entropy with label smoothing, ignoring -100 positions."""
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        label_smoothing=label_smoothing,
    )


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    step: int,
    best_bleu: float,
    output_dir: Path,
    name: str = None,
) -> Path:
    ckpt_path = output_dir / (name or f"ckpt_step_{step}.pt")
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_bleu": best_bleu,
        },
        ckpt_path,
    )
    return ckpt_path


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train(config_path: str, resume: str = None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    tr = cfg["training"]
    log_cfg = cfg.get("logging", {})
    output_dir = Path(log_cfg.get("output_dir", "logs/run"))
    logger = setup_logging(output_dir)

    set_seed(tr.get("seed", 42))
    logger.info(f"Config:     {config_path}")
    logger.info(f"Output dir: {output_dir}")

    # ------------------------------------------------------------------
    # Device + gradient accumulation
    # Paper: batch_size_per_gpu=16, num_gpus=8 → effective_batch=128
    # Single-GPU: simulate via gradient accumulation
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    batch_per_gpu = tr["batch_size_per_gpu"]
    num_gpus = tr.get("num_gpus", 1)
    effective_batch = batch_per_gpu * num_gpus
    # Cap actual batch to what fits in memory
    actual_gpu_batch = batch_per_gpu
    grad_accum = max(1, effective_batch // actual_gpu_batch)
    logger.info(
        f"Effective batch: {effective_batch} | "
        f"per-GPU: {actual_gpu_batch} | "
        f"grad_accum: {grad_accum}"
    )

    # ------------------------------------------------------------------
    # Tokenizer + Datasets
    # ------------------------------------------------------------------
    t5_model_name = cfg["model"]["t5_model_name"]
    logger.info(f"Loading tokenizer: {t5_model_name}")
    tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

    data_cfg = cfg["data"]
    features_base = data_cfg["features_dir"]
    features_train = os.path.join(features_base, "train")
    features_val = os.path.join(features_base, "val")
    use_conf = data_cfg.get("use_confidence", False)
    max_frames = data_cfg.get("max_frames", 512)
    data_dir = data_cfg.get("data_dir", ".")
    num_workers = tr.get("num_workers", 4)

    logger.info("Building datasets...")
    train_dataset = How2SignDataset(
        data_dir=data_dir,
        features_dir=features_train,
        split="train",
        tokenizer=tokenizer,
        max_frames=max_frames,
        use_confidence=use_conf,
    )
    val_dataset = How2SignDataset(
        data_dir=data_dir,
        features_dir=features_val,
        split="val",
        tokenizer=tokenizer,
        max_frames=max_frames,
        use_confidence=use_conf,
    )
    logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=actual_gpu_batch,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=actual_gpu_batch,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    logger.info(f"Building model: {cfg['model']['name']}")
    model = build_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = (
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    )
    logger.info(
        f"Params: {total_params:.1f}M total | {trainable_params:.1f}M trainable"
    )

    # ------------------------------------------------------------------
    # Optimizer + LR schedule
    # ------------------------------------------------------------------
    train_steps = tr["train_steps"]
    lr = tr["learning_rate"]
    weight_decay = tr.get("weight_decay", 0.1)
    warmup_steps = max(1, int(0.05 * train_steps))  # 5% warmup

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),
        eps=1e-6,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=train_steps,
    )

    # ------------------------------------------------------------------
    # Mixed precision
    # ------------------------------------------------------------------
    fp16 = tr.get("fp16", False) and device.type == "cuda"
    scaler = GradScaler("cuda") if fp16 else None
    logger.info(f"Mixed precision (fp16): {fp16}")

    # Training hyperparams
    label_smoothing = cfg["model"].get("label_smoothing", 0.2)
    grad_clip = tr.get("grad_clipping", 1.0)
    log_every = log_cfg.get("log_every", 100)
    save_every = log_cfg.get("save_every", 500)
    test_every = log_cfg.get("test_every", 500)
    early_stop_patience = log_cfg.get("early_stop_patience", 0)  # 0 = disabled
    no_improve_count = 0
    gen_cfg = cfg.get("generation", {})
    max_gen_length = gen_cfg.get("max_length", 128)
    num_beams = gen_cfg.get("num_beams", 5)

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    global_step = 0
    best_bleu = 0.0
    if resume:
        logger.info(f"Resuming from: {resume}")
        ckpt = torch.load(resume, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        global_step = ckpt["step"]
        best_bleu = ckpt.get("best_bleu", 0.0)
        logger.info(f"Resumed at step {global_step} | best BLEU={best_bleu:.2f}")

    # ------------------------------------------------------------------
    # Training loop (step-based with cycling DataLoader)
    # ------------------------------------------------------------------
    model.train()
    optimizer.zero_grad()
    data_iter = iter(train_loader)
    running_loss = 0.0
    t0 = time.time()

    logger.info(
        f"Training: {train_steps} steps | "
        f"warmup: {warmup_steps} | "
        f"label_smoothing: {label_smoothing}"
    )

    while global_step < train_steps:
        # ---- Gradient accumulation loop ----
        accum_loss = 0.0
        skip_step = False
        for _ in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            # Move tensors to device; leave strings (clip_ids) as-is
            batch_gpu = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            labels = batch_gpu.pop("labels")
            batch_gpu.pop("clip_ids", None)

            if fp16:
                with autocast("cuda"):
                    outputs = model(**batch_gpu, labels=labels)
                    step_loss = (
                        compute_loss(outputs["logits"], labels, label_smoothing)
                        / grad_accum
                    )
                # Check NaN BEFORE backward to keep scaler state clean
                if torch.isnan(step_loss):
                    skip_step = True
                    optimizer.zero_grad()
                    break
                scaler.scale(step_loss).backward()
            else:
                outputs = model(**batch_gpu, labels=labels)
                step_loss = (
                    compute_loss(outputs["logits"], labels, label_smoothing)
                    / grad_accum
                )
                if torch.isnan(step_loss):
                    skip_step = True
                    optimizer.zero_grad()
                    break
                step_loss.backward()

            accum_loss += step_loss.item()

        if skip_step:
            continue

        # ---- Optimizer step ----
        if fp16:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()
        optimizer.zero_grad()
        global_step += 1
        running_loss += accum_loss

        # ---- Logging ----
        if global_step % log_every == 0:
            elapsed = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]
            avg_loss = running_loss / log_every
            logger.info(
                f"step={global_step}/{train_steps} | "
                f"loss={avg_loss:.4f} | "
                f"lr={lr_now:.2e} | "
                f"{elapsed:.0f}s/{log_every} steps"
            )
            running_loss = 0.0
            t0 = time.time()

        # ---- Periodic checkpoint ----
        if global_step % save_every == 0:
            ckpt_path = save_checkpoint(
                model, optimizer, scheduler, global_step, best_bleu, output_dir
            )
            logger.info(f"Checkpoint: {ckpt_path}")

        # ---- Validation ----
        if global_step % test_every == 0 or global_step == train_steps:
            model.eval()
            logger.info(f"--- Validation @ step {global_step} ---")
            bleu = evaluate_bleu(
                model=model,
                dataloader=val_loader,
                tokenizer=tokenizer,
                device=device,
                max_length=max_gen_length,
                num_beams=num_beams,
            )
            logger.info(f"BLEU={bleu:.2f} | best={best_bleu:.2f}")

            if bleu > best_bleu:
                best_bleu = bleu
                no_improve_count = 0
                ckpt_path = save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    global_step,
                    best_bleu,
                    output_dir,
                    name="best_model.pt",
                )
                logger.info(f"New best BLEU={best_bleu:.2f} → {ckpt_path}")
            else:
                no_improve_count += 1
                if early_stop_patience > 0 and no_improve_count >= early_stop_patience:
                    logger.info(
                        f"Early stopping: no improvement for {no_improve_count} validations "
                        f"(patience={early_stop_patience}). Best BLEU={best_bleu:.2f}"
                    )
                    model.train()
                    break
            model.train()

    # ---- Final checkpoint ----
    final_ckpt = save_checkpoint(
        model,
        optimizer,
        scheduler,
        global_step,
        best_bleu,
        output_dir,
        name="final_model.pt",
    )
    logger.info(f"Done. Best BLEU={best_bleu:.2f} | Saved: {final_ckpt}")
    return best_bleu


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SignMusketeers Phase 2")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--resume", default=None, help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()
    train(args.config, resume=args.resume)
