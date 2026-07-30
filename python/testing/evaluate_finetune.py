# evaluate_finetune.py

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch

from ML.mfcc import make_extractor
from ML.model import Net

from splits.common_voice import load_split


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_features(path, sample_rate, extractor, num_coeffs):
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    audio = audio.astype(np.float32)
    features = np.asarray(extractor(audio))
    if features.ndim != 2:
        raise ValueError(
            f"Expected 2D features from extractor.compute(), got shape {features.shape} for {path}"
        )
    # Normalize to [T, num_coeffs]. Different extractor backends /
    # calling conventions (extractor.compute() vs extractor()) have been
    # observed to return either [T, C] or [C, T] -- use the known
    # num_coeffs (from the checkpoint's mfcc_config) as ground truth
    # rather than assuming an orientation.
    if features.shape[1] != num_coeffs and features.shape[0] == num_coeffs:
        features = features.T
    elif features.shape[1] != num_coeffs:
        raise ValueError(
            f"Feature shape {features.shape} doesn't match expected num_coeffs="
            f"{num_coeffs} on either axis for {path}"
        )
    return torch.from_numpy(features).float()


def pad_features(features):
    """
    Pad a list of [T, C] tensors into
    [B, 1, C, T].
    """
    max_t = max(x.shape[0] for x in features)
    coeffs = features[0].shape[1]
    batch = torch.zeros(len(features), 1, coeffs, max_t, dtype=torch.float32)

    for i, x in enumerate(features):
        batch[i, 0, :, : x.shape[0]] = x.T
    return batch


@torch.no_grad()
def evaluate(
    model, extractor, split, classes, sample_rate, device, batch_size, num_coeffs
):
    class_to_idx = {word: i for i, word in enumerate(classes)}
    idx_to_class = {i: word for word, i in class_to_idx.items()}
    test_samples = []

    for word in classes:
        _enrollment, test = split[word]
        for clip in test:
            test_samples.append(clip)
    detailed_rows = []
    model.eval()

    for start in range(0, len(test_samples), batch_size):
        batch_clips = test_samples[start : start + batch_size]
        features = [
            load_features(clip.path, sample_rate, extractor, num_coeffs)
            for clip in batch_clips
        ]

        batch = pad_features(features).to(device)
        logits = model(batch)
        probabilities = torch.softmax(logits, dim=1)

        confidences, predictions = probabilities.max(dim=1)

        for clip, prediction, confidence in zip(
            batch_clips, predictions.cpu().numpy(), confidences.cpu().numpy()
        ):
            predicted_word = idx_to_class[int(prediction)]
            correct = predicted_word == clip.word
            detailed_rows.append(
                {
                    "true_word": clip.word,
                    "predicted_word": predicted_word,
                    "correct": correct,
                    "confidence": float(confidence),
                    "clip_path": clip.path,
                    "age": clip.age,
                    "gender": clip.gender,
                    "accent": clip.accent,
                }
            )
    return detailed_rows


def aggregate(
    detailed_rows,
    group_keys,
):
    buckets = defaultdict(lambda: {"correct": 0, "total": 0})

    for row in detailed_rows:
        key = tuple(row[k] for k in group_keys)
        bucket = buckets[key]
        bucket["total"] += 1
        bucket["correct"] += int(row["correct"])

    rows = []
    for key, bucket in buckets.items():
        total = bucket["total"]
        correct = bucket["correct"]
        accuracy = correct / total if total else 0.0
        row = dict(zip(group_keys, key))
        row["num_test_samples"] = total
        row["correct"] = correct
        row["accuracy"] = round(accuracy, 4)
        row["word_error_rate"] = round(1.0 - accuracy, 4)
        rows.append(row)

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--output-dir", default="./results_finetune")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.model, map_location=device)
    classes = ckpt["classes"]
    mfcc_cfg = ckpt["mfcc_config"]
    extractor = make_extractor(**mfcc_cfg)
    split = load_split(Path(args.split))
    model = Net(n_classes=len(classes), n_mels=mfcc_cfg["num_coeffs"]).to(device)
    model.load_state_dict(ckpt["model_state"])

    print(f"Evaluating {len(classes)} classes")
    detailed_rows = evaluate(
        model=model,
        extractor=extractor,
        split=split,
        classes=classes,
        sample_rate=mfcc_cfg["sample_rate"],
        device=device,
        batch_size=args.batch_size,
        num_coeffs=mfcc_cfg["num_coeffs"],
    )

    total = len(detailed_rows)
    correct = sum(int(row["correct"]) for row in detailed_rows)
    accuracy = correct / total if total else 0.0
    print(f"Accuracy: " f"{accuracy:.4f} " f"({correct}/{total})")
    print(f"WER: " f"{1.0 - accuracy:.4f}")
    out_dir = Path(args.output_dir)
    detailed_fields = [
        "true_word",
        "predicted_word",
        "correct",
        "confidence",
        "clip_path",
        "age",
        "gender",
        "accent",
    ]
    write_csv(
        out_dir / "results_detailed.csv",
        detailed_rows,
        detailed_fields,
    )
    summary_rows = aggregate(detailed_rows, [])

    # Since group_keys=[] gives one
    # global bucket, write manually.
    summary_rows = [
        {
            "num_test_samples": total,
            "correct": correct,
            "accuracy": round(
                accuracy,
                4,
            ),
            "word_error_rate": round(
                1.0 - accuracy,
                4,
            ),
        }
    ]

    write_csv(
        out_dir / "results_summary.csv",
        summary_rows,
        [
            "num_test_samples",
            "correct",
            "accuracy",
            "word_error_rate",
        ],
    )

    gender_rows = aggregate(
        [row for row in detailed_rows if row["gender"] not in ("unknown", "", None)],
        ["gender"],
    )

    write_csv(
        out_dir / "results_by_gender.csv",
        gender_rows,
        [
            "gender",
            "num_test_samples",
            "correct",
            "accuracy",
            "word_error_rate",
        ],
    )
    print(f"Wrote results to " f"{out_dir.resolve()}")


if __name__ == "__main__":
    main()
