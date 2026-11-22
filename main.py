import argparse
import random
import numpy as np
import torch

from src.config import BATCH_SIZE, IMG_SIZE, SEED
from src.evaluate import evaluate_model
from src.train import train_model


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Brain Tumor Classification (CPU)")
    parser.add_argument(
        "--model",
        type=str,
        default="simple_cnn",
        choices=["simple_cnn", "efficientnet_b0", "both"],
        help="Model yang dijalankan: simple_cnn, efficientnet_b0, atau both",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train_eval",
        choices=["train", "eval", "train_eval"],
        help="Mode eksekusi: train, eval, atau train_eval",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Jumlah epoch training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--img-size", type=int, default=IMG_SIZE, help="Ukuran sisi gambar")
    parser.add_argument(
        "--unfreeze",
        action="store_true",
        help="Aktifkan fine-tuning backbone EfficientNet-B0 (default: freeze)",
    )

    args = parser.parse_args()
    set_seed(SEED)

    models_to_run = ["simple_cnn", "efficientnet_b0"] if args.model == "both" else [args.model]
    freeze_backbone = not args.unfreeze

    for model_name in models_to_run:
        if args.mode in {"train", "train_eval"}:
            train_model(
                model_name=model_name,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                img_size=args.img_size,
                freeze_backbone=freeze_backbone,
            )
        if args.mode in {"eval", "train_eval"}:
            evaluate_model(
                model_name=model_name,
                batch_size=args.batch_size,
                img_size=args.img_size,
            )


if __name__ == "__main__":
    main()