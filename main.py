"""
main.py

Unified entry point for SignMusketeers pipeline.

Commands:
  train       — Run Phase 2 supervised training
  eval        — Evaluate a saved checkpoint on val/test

Examples:
    python main.py train --config configs/baseline.yaml
    python main.py train --config configs/ca_csa.yaml --resume logs/baseline/ckpt_step_1000.pt
    python main.py eval  --config configs/baseline.yaml --ckpt logs/baseline/best_model.pt
    python main.py eval  --config configs/ca_csa.yaml   --ckpt logs/ca_csa/best_model.pt --split test
"""

import argparse
import sys


def cmd_train(args):
    from train import train

    train(args.config, resume=args.resume)


def cmd_eval(args):
    import yaml
    from evaluate import evaluate_from_checkpoint

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    result = evaluate_from_checkpoint(
        ckpt_path=args.ckpt,
        model_type=model_cfg["name"],
        config=model_cfg,
        data_dir=cfg["data"].get("data_dir", "."),
        features_dir=cfg["data"]["features_dir"],
        split=args.split,
        device=args.device,
    )
    print(f"BLEU ({args.split}): {result:.2f}")


def main():
    parser = argparse.ArgumentParser(prog="main", description="SignMusketeers pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # -- train --
    p_train = sub.add_parser("train", help="Run Phase 2 training")
    p_train.add_argument("--config", required=True, help="Path to YAML config")
    p_train.add_argument(
        "--resume", default=None, help="Path to checkpoint to resume from"
    )

    # -- eval --
    p_eval = sub.add_parser("eval", help="Evaluate a checkpoint")
    p_eval.add_argument("--config", required=True, help="Path to YAML config")
    p_eval.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    p_eval.add_argument(
        "--split",
        default="test",
        choices=["val", "test"],
        help="Dataset split to evaluate on",
    )
    p_eval.add_argument("--device", default="cuda", help="Device: cuda or cpu")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)


if __name__ == "__main__":
    main()
