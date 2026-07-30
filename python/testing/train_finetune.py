# train_finetune_cv.py

import argparse
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ML.hf_dataset import pad_collate
from ML.mfcc import make_extractor
from ML.model import Net

from splits.common_voice import (
    load_split,
)


class ClipDataset(Dataset):
    """
    Dataset backed by a fixed Common Voice enrollment split.

    This intentionally uses only the enrollment clips.
    The exact same clips are used as DTW templates.
    """

    def __init__(self, clips, classes, sample_rate, extractor):
        self.clips = clips
        self.classes = classes
        self.class_to_idx = {word: i for i, word in enumerate(classes)}
        self.sample_rate = sample_rate
        self.extractor = extractor

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        clip = self.clips[index]
        import librosa

        audio, _ = librosa.load(
            clip.path,
            sr=self.sample_rate,
            mono=True,
        )
        audio = audio.astype("float32")
        features = self.extractor(audio)
        # pad_collate expects:
        #   tensor [time, coeffs]
        #   label
        #   length
        features = torch.from_numpy(features).float()
        label = self.class_to_idx[clip.word]
        return features, label, features.shape[0]


def build_datasets(split, classes, extractor, sample_rate):
    enrollment_clips = []
    for word in classes:
        enrollment, _test = split[word]
        enrollment_clips.extend(enrollment)
    dataset = ClipDataset(
        enrollment_clips,
        classes,
        sample_rate,
        extractor,
    )
    return dataset


def run_epoch(model, loader, criterion, optimizer, device):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for specs, labels, _lengths in loader:
        specs = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(specs)
            loss = criterion(
                logits,
                labels,
            )
            if train:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_n += labels.size(0)
    return (
        total_loss / total_n,
        total_correct / total_n,
    )


# --------------------------------------------------------------------------
# Reusable pieces (pulled out of main() so other scripts, e.g. a DTW-vs-CNN
# comparison harness, can drive fine-tuning in-process without shelling out
# to this file per run).
# --------------------------------------------------------------------------
def build_model_from_pretrained(pretrained_state, n_classes, n_mels, device):
    """
    Build a fresh Net(n_classes, n_mels) and load everything except the
    final `fc` head from `pretrained_state` (a model_state dict, e.g.
    ckpt["model_state"]).

    Returns (model, missing_keys) where missing_keys is the list of
    parameter names that were NOT found in the pretrained state (expected
    to just be the new fc.* head).
    """
    model = Net(n_classes=n_classes, n_mels=n_mels).to(device)
    own_state = model.state_dict()
    backbone_state = {
        k: v
        for k, v in pretrained_state.items()
        if k in own_state and not k.startswith("fc.")
    }
    missing = model.load_state_dict(backbone_state, strict=False)
    return model, missing.missing_keys


def set_frozen(module, frozen):
    for p in module.parameters():
        p.requires_grad = not frozen


def freeze_backbone(model):
    """
    Freeze stem + blocks[0] + blocks[1], leave blocks[2] and fc trainable.
    Also pins frozen BatchNorm layers in eval() mode so their running
    stats don't drift during fine-tuning on a tiny enrollment set.
    """
    set_frozen(model.stem, True)
    set_frozen(model.blocks[0], True)
    set_frozen(model.blocks[1], True)

    set_frozen(model.blocks[2], False)
    set_frozen(model.fc, False)

    for module in [model.stem, model.blocks[0], model.blocks[1]]:
        for m in module.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def finetune_loop(
    model,
    train_loader,
    device,
    epochs,
    lr_head,
    lr_full,
    weight_decay,
    patience,
    unfreeze_at=None,
    verbose=True,
):
    """
    Runs the same train-acc-based early-stopping loop as the original
    main(), but returns the best state_dict in memory instead of writing
    a checkpoint to disk. Caller decides what to do with the result
    (save it, evaluate it, discard it, ...).

    Returns:
        best_state (dict): state_dict of the best (highest train-acc) model
        best_train_acc (float)
        epochs_run (int): how many epochs actually ran before stopping
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr_head, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_train_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    epochs_since_improve = 0
    epochs_run = 0

    for epoch in range(epochs):
        if unfreeze_at is not None and epoch == unfreeze_at:
            if verbose:
                print(f"Epoch {epoch}: unfreezing full backbone")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr_full, weight_decay=weight_decay
            )

        t0 = time.time()
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device
        )
        epochs_run = epoch + 1

        if verbose:
            print(
                f"epoch {epoch + 1:03d}/{epochs} "
                f"loss={train_loss:.4f} "
                f"acc={train_acc:.4f} "
                f"({time.time() - t0:.1f}s)"
            )

        if train_acc > best_train_acc:
            best_train_acc = train_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improve = 0
            if verbose:
                print("  -> new best")
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                if verbose:
                    print("Early stopping.")
                break

    return best_state, best_train_acc, epochs_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", required=True)
    ap.add_argument(
        "--split", required=True, help="JSON split generated by the shared split script"
    )
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-full", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--unfreeze-at", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="finetuned_model.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    # --------------------------------------------------
    # Load pretrained checkpoint
    # --------------------------------------------------
    ckpt = torch.load(
        args.pretrained,
        map_location=device,
    )
    mfcc_cfg = ckpt.get("mfcc_config")
    if mfcc_cfg is None:
        raise RuntimeError("Checkpoint does not contain mfcc_config.")
    print(
        "Using MFCC configuration:",
        mfcc_cfg,
    )
    extractor = make_extractor(**mfcc_cfg)
    print(
        "MFCC backend:",
        type(extractor).__name__,
    )
    # --------------------------------------------------
    # Load exact enrollment/test split
    # --------------------------------------------------
    split = load_split(Path(args.split))
    classes = sorted(split.keys())
    print(f"Fine-tuning on {len(classes)} words")
    total_enrollment = sum(len(enrollment) for enrollment, _ in split.values())
    total_test = sum(len(test) for _, test in split.values())
    print(f"Enrollment samples: {total_enrollment}")
    print(f"Test samples: {total_test}")

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------
    train_ds = build_datasets(
        split,
        classes,
        extractor,
        mfcc_cfg["sample_rate"],
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=pad_collate,
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model, missing_keys = build_model_from_pretrained(
        ckpt["model_state"],
        n_classes=len(classes),
        n_mels=mfcc_cfg["num_coeffs"],
        device=device,
    )
    print("Loaded pretrained backbone.")
    print("Missing keys:", missing_keys)

    # --------------------------------------------------
    # Freeze plan
    # --------------------------------------------------
    freeze_backbone(model)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: " f"{sum(p.numel() for p in trainable):,}")
    print(f"Total parameters: " f"{sum(p.numel() for p in model.parameters()):,}")

    # --------------------------------------------------
    # Training
    # --------------------------------------------------
    best_state, best_train_acc, epochs_run = finetune_loop(
        model,
        train_loader,
        device,
        epochs=args.epochs,
        lr_head=args.lr_head,
        lr_full=args.lr_full,
        weight_decay=args.weight_decay,
        patience=args.patience,
        unfreeze_at=args.unfreeze_at,
        verbose=True,
    )
    model.load_state_dict(best_state)

    torch.save(
        {
            "model_state": best_state,
            "classes": classes,
            "mfcc_config": mfcc_cfg,
            "val_acc": None,
            "train_acc": best_train_acc,
            "split": str(Path(args.split).resolve()),
        },
        args.out,
    )
    print(f"Done. Best enrollment training accuracy: {best_train_acc:.4f}")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
