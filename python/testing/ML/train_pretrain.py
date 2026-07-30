"""
Stage 1: pretrain the backbone on AkashPrasadMishra/multilingual-keywords-900,
using MFCCs that match the C++ MFCCProcessorCore pipeline.

Usage:
    python train_pretrain.py --min-count 100 --max-words 500 \
        --sample-rate 16000 --num-coeffs 13 --epochs 30 --out pretrained.pt

--sample-rate / --frame-length / --hop-length / --num-mel-bands / --num-coeffs
MUST match whatever you configure MFCCProcessorCore with in C++, or the
features the deployed model sees won't match what it was trained on.
"""

import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from python.Testing.ML.cached_mfcc import CachedMFCCDataset
from python.Testing.ML.hf_dataset import (
    load_keywords_dataset,
    select_words,
    HFKeywordsDataset,
    pad_collate,
)
from python.Testing.ML.mfcc import make_extractor
from python.Testing.ML.model import Net

from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()


def run_epoch(model, loader, criterion, optimizer, device):
    train = optimizer is not None
    model.train(train)
    total_loss, total_correct, total_n = 0.0, 0, 0

    for specs, labels, _lengths in loader:
        specs, labels = specs.to(device), labels.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(specs)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_n += labels.size(0)

    return total_loss / total_n, total_correct / total_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="AkashPrasadMishra/multilingual-keywords-900")
    ap.add_argument("--min-count", type=int, default=100)
    ap.add_argument("--max-words", type=int, default=500)
    ap.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="optional subset of language codes, e.g. en ja tr",
    )
    ap.add_argument("--val-fraction", type=float, default=0.1)
    # MFCC config -- must match your C++ MFCCProcessorCore setup
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--frame-length", type=int, default=512)
    ap.add_argument("--hop-length", type=int, default=256)
    ap.add_argument("--num-mel-bands", type=int, default=40)
    ap.add_argument("--num-coeffs", type=int, default=13)
    # training
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--resume",
        default=None,
        help="path to checkpoint to warm-start from (weights only)",
    )
    ap.add_argument(
        "--restart-lr",
        type=float,
        default=7e-4,
        help="LR to use for the new cosine cycle when resuming",
    )
    ap.add_argument(
        "--patience",
        type=int,
        default=8,
        help="early-stop if val_acc doesn't improve for this many epochs",
    )
    ap.add_argument("--out", default="pretrained.pt")
    ap.add_argument(
        "--no-cpp-backend",
        action="store_true",
        help="force the pure-numpy MFCC extractor even if the "
        "keyword_detection C++ extension is built and importable",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds, audio_col, word_col, lang_col = load_keywords_dataset(
        args.repo_id, sample_rate=args.sample_rate
    )

    words = select_words(
        ds,
        word_col,
        min_count=args.min_count,
        max_words=args.max_words,
        languages=args.languages,
        lang_col=lang_col,
    )
    print(
        f"Pretraining on {len(words)} words: {words[:10]}{'...' if len(words) > 10 else ''}"
    )

    words_set = set(words)
    ds = ds.filter(lambda w: w in words_set, input_columns=[word_col])
    split = ds.train_test_split(test_size=args.val_fraction, seed=0)
    train_hf, val_hf = split["train"], split["test"]

    extractor = make_extractor(
        prefer_cpp=not args.no_cpp_backend,
        sample_rate=args.sample_rate,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
        num_mel_bands=args.num_mel_bands,
        num_coeffs=args.num_coeffs,
    )
    print(f"Using MFCC backend: {type(extractor).__name__}")

    train_ds = CachedMFCCDataset(
        train_hf,
        words,
        audio_col,
        word_col,
        cache_dir="./train-cache",
        sample_rate=args.sample_rate,
        train=True,
        extractor=extractor,
    )
    val_ds = CachedMFCCDataset(
        val_hf,
        words,
        audio_col,
        word_col,
        cache_dir="./val-cache",
        sample_rate=args.sample_rate,
        train=False,
        extractor=extractor,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=pad_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=pad_collate,
    )

    model = Net(n_classes=len(words), n_mels=args.num_coeffs).to(device)

    init_lr = args.restart_lr if args.resume else args.lr

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=init_lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_val_acc = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        # sanity check: class list must match exactly, or the head is misaligned
        assert ckpt["classes"] == words, (
            "class list mismatch between checkpoint and current run — "
            "make sure --min-count/--max-words/--languages match the original run"
        )
        model.load_state_dict(ckpt["model_state"])
        best_val_acc = ckpt.get("val_acc", 0.0)
        print(
            f"resumed weights from {args.resume} (val_acc={best_val_acc:.3f}); "
            f"starting a fresh {args.epochs}-epoch cosine cycle at lr={args.restart_lr}"
        )

    no_improve = 0
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        print(
            f"epoch {epoch+1:03d}/{args.epochs} "
            f"lr {current_lr:.2e} "
            f"train_loss {train_loss:.3f} train_acc {train_acc:.3f} "
            f"val_loss {val_loss:.3f} val_acc {val_acc:.3f} "
            f"gap {train_acc-val_acc:+.3f} "
            f"({time.time()-t0:.1f}s)"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": words,
                    "mfcc_config": dict(
                        sample_rate=args.sample_rate,
                        frame_length=args.frame_length,
                        hop_length=args.hop_length,
                        num_mel_bands=args.num_mel_bands,
                        num_coeffs=args.num_coeffs,
                    ),
                    "val_acc": val_acc,
                },
                args.out,
            )
            print(f"  -> saved new best checkpoint ({val_acc:.3f}) to {args.out}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  -> no improvement for {args.patience} epochs, stopping early")
                break

    print(f"done. best val_acc={best_val_acc:.3f}, checkpoint at {args.out}")


if __name__ == "__main__":
    main()
