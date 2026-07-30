"""
compare_dtw_cnn.py

DTW vs. few-shot-CNN comparison harness.

For a grid of (n_classes x samples_per_class) it repeatedly:
  1. Draws a random enrollment/test split (same helper `create_split` used
     by make_splits.py), for several trials per grid point.
  2. Evaluates DTW (via the `keyword_detection` C++ library, same calling
     convention as evaluate_keyword_detection.py) on that split, enrolling
     one DTW template set per word.
  3. Fine-tunes a fresh copy of the pretrained CNN backbone (same freeze
     plan / training loop as train_finetune.py) on the *same* split's
     enrollment clips, then evaluates it on the same test clips
     (evaluate_finetune.py's evaluate()).

Both methods see the exact same clips for a given (locale, n_classes,
samples_per_class, trial) -- that's what makes the comparison fair.

--------------------------------------------------------------------------
IMPORTANT: adjust the import block below to match your project layout.
This script assumes it lives next to (or importable alongside):
    - train_finetune.py   (build_datasets, build_model_from_pretrained,
                               freeze_backbone, finetune_loop)
    - evaluate_finetune.py   (evaluate, aggregate)
    - evaluate_keyword_detection.py (build_feature_extractor, build_matcher,
                               parse_classify_result)
    - common_voice.py / splits/common_voice.py (ClipInfo, discover_locales,
                               load_locale_clips, create_split)
    - ML/mfcc.py, ML/hf_dataset.py  (make_extractor, pad_collate)
    - the `keyword_detection` (kd) compiled extension on sys.path
--------------------------------------------------------------------------

Outputs (written to --output-dir):
  - results_trials.csv       one row per (locale, method, n_classes,
                              samples_per_class, trial): accuracy/WER/timing.
                              This is the file you want for plotting.
  - results_utterances.csv   one row per test utterance per trial per
                              method (skip with --no-detailed; this file
                              can get very large for a big grid).
  - results_summary.csv      accuracy mean/std aggregated across trials,
                              per (locale, method, n_classes,
                              samples_per_class). Regenerated at the end of
                              a run, or on demand with --summarize-only.

Safe to interrupt and resume: pass --resume and already-completed
(locale, method, n_classes, samples_per_class, trial) rows found in an
existing results_trials.csv are skipped.
"""

import argparse
import csv
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import librosa
import torch
from torch.utils.data import DataLoader

# ---- project imports: adjust if your layout differs -----------------
try:
    from splits.common_voice import (
        discover_locales,
        load_locale_clips,
        create_split,
    )
except ImportError:
    from common_voice import (
        discover_locales,
        load_locale_clips,
        create_split,
    )

from ML.mfcc import make_extractor
from ML.hf_dataset import pad_collate

from python.testing.train_finetune import (
    build_datasets,
    build_model_from_pretrained,
    freeze_backbone,
    finetune_loop,
)
from evaluate_finetune import evaluate as evaluate_cnn

from evaluate_keyword_detection import (
    build_feature_extractor,
    build_matcher,
    parse_classify_result,
)

# -----------------------------------------------------------------------


TRIALS_FIELDS = [
    "locale",
    "method",
    "n_classes_requested",
    "n_classes_actual",
    "samples_per_class",
    "trial",
    "seed",
    "num_enrollment_total",
    "num_test_samples",
    "correct",
    "accuracy",
    "word_error_rate",
    "eval_time_sec",
    "train_time_sec",
    "epochs_run",
    "best_train_acc",
    "dtw_feature",
    "dtw_distance",
    "dtw_classify",
    "dtw_sample_rate",
    "dtw_num_coeffs",
]

UTTER_FIELDS = [
    "locale",
    "method",
    "n_classes_requested",
    "samples_per_class",
    "trial",
    "true_word",
    "predicted_word",
    "correct",
    "confidence",
    "best_distance",
    "clip_path",
    "age",
    "gender",
    "accent",
]


def seed_for(base_seed, n_classes, samples_per_class, trial):
    """Deterministic, distinct seed per grid point x trial."""
    return base_seed + trial * 100_003 + samples_per_class * 1_009 + n_classes * 97


# --------------------------------------------------------------------------
# DTW side
# --------------------------------------------------------------------------
class DTWAudioCache:
    """Caches decoded PCM + computed DTW features per clip path within one
    locale's run, since the same clips recur across many grid points/trials."""

    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self._pcm = {}
        self._feat = {}

    def clear(self):
        self._pcm.clear()
        self._feat.clear()

    def get_pcm(self, path):
        if path not in self._pcm:
            audio, _ = librosa.load(path, sr=self.sample_rate, mono=True)
            self._pcm[path] = audio.astype(np.float32)
        return self._pcm[path]

    def get_features(self, path, feature_extractor):
        if path not in self._feat:
            self._feat[path] = feature_extractor.compute(self.get_pcm(path))
        return self._feat[path]


def dtw_evaluate_split(
    split,
    classes,
    feature_name,
    distance_name,
    classify_name,
    sample_rate,
    num_coeffs,
    band_width,
    cache,
):
    feature_extractor = build_feature_extractor(feature_name, sample_rate, num_coeffs)
    matcher = build_matcher(distance_name, classify_name, band_width)

    for word in classes:
        enrollment, _test = split[word]
        for clip in enrollment:
            feats = cache.get_features(clip.path, feature_extractor)
            matcher.add_template(word, feats)

    rows = []
    correct = 0
    total = 0
    for word in classes:
        _enrollment, test = split[word]
        for clip in test:
            feats = cache.get_features(clip.path, feature_extractor)
            try:
                result = matcher.classify_with_best_score(feats)
                predicted, distance = parse_classify_result(result)
            except Exception as e:  # noqa: BLE001
                print(f"    [warn] DTW classify failed on {clip.path}: {e}")
                predicted, distance = None, None

            is_correct = predicted == word
            correct += int(is_correct)
            total += 1
            rows.append(
                {
                    "true_word": word,
                    "predicted_word": predicted,
                    "correct": is_correct,
                    "confidence": "",
                    "best_distance": distance,
                    "clip_path": clip.path,
                    "age": clip.age,
                    "gender": clip.gender,
                    "accent": clip.accent,
                }
            )

    return correct, total, rows


# --------------------------------------------------------------------------
# CNN side
# --------------------------------------------------------------------------
def safe_loader_batch_size(n, batch_size):
    """
    Pick an effective batch size + drop_last so we never hand a batch of
    size 1 to the model in training mode -- blocks[2]'s BatchNorm2d layers
    stay trainable under the freeze plan, and BatchNorm2d raises on a
    batch of 1 during training. This bites at small n_classes x
    samples_per_class totals where batch_size doesn't evenly divide the
    enrollment set size.
    """
    bs = min(batch_size, n)
    drop_last = bs > 1 and n % bs == 1
    return bs, drop_last


def cnn_evaluate_split(
    split,
    classes,
    pretrained_state,
    mfcc_cfg,
    extractor,
    device,
    batch_size,
    epochs,
    lr_head,
    lr_full,
    weight_decay,
    patience,
    unfreeze_at,
    workers,
):
    train_ds = build_datasets(split, classes, extractor, mfcc_cfg["sample_rate"])
    effective_bs, drop_last = safe_loader_batch_size(len(train_ds), batch_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=effective_bs,
        shuffle=True,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        drop_last=drop_last,
        collate_fn=pad_collate,
    )

    model, _missing = build_model_from_pretrained(
        pretrained_state,
        n_classes=len(classes),
        n_mels=mfcc_cfg["num_coeffs"],
        device=device,
    )
    freeze_backbone(model)

    t0 = time.time()
    best_state, best_train_acc, epochs_run = finetune_loop(
        model,
        train_loader,
        device,
        epochs=epochs,
        lr_head=lr_head,
        lr_full=lr_full,
        weight_decay=weight_decay,
        patience=patience,
        unfreeze_at=unfreeze_at,
        verbose=False,
    )
    train_time = time.time() - t0
    model.load_state_dict(best_state)

    detailed_rows = evaluate_cnn(
        model=model,
        extractor=extractor,
        split=split,
        classes=classes,
        sample_rate=mfcc_cfg["sample_rate"],
        device=device,
        batch_size=batch_size,
        num_coeffs=mfcc_cfg["num_coeffs"],
    )
    correct = sum(int(r["correct"]) for r in detailed_rows)
    total = len(detailed_rows)

    rows = [
        {
            "true_word": r["true_word"],
            "predicted_word": r["predicted_word"],
            "correct": r["correct"],
            "confidence": r["confidence"],
            "best_distance": "",
            "clip_path": r["clip_path"],
            "age": r["age"],
            "gender": r["gender"],
            "accent": r["accent"],
        }
        for r in detailed_rows
    ]

    return correct, total, rows, best_train_acc, epochs_run, train_time


# --------------------------------------------------------------------------
# Summary aggregation (mean/std accuracy across trials)
# --------------------------------------------------------------------------
def write_summary(trials_path: Path, summary_path: Path):
    buckets = defaultdict(list)
    with open(trials_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                row["locale"],
                row["method"],
                row["n_classes_requested"],
                row["samples_per_class"],
            )
            try:
                buckets[key].append(float(row["accuracy"]))
            except (ValueError, KeyError):
                continue

    rows = []
    for (locale, method, n_classes, spc), accs in buckets.items():
        rows.append(
            {
                "locale": locale,
                "method": method,
                "n_classes_requested": n_classes,
                "samples_per_class": spc,
                "num_trials": len(accs),
                "accuracy_mean": round(statistics.mean(accs), 4),
                "accuracy_std": (
                    round(statistics.pstdev(accs), 4) if len(accs) > 1 else 0.0
                ),
                "accuracy_min": round(min(accs), 4),
                "accuracy_max": round(max(accs), 4),
            }
        )

    rows.sort(
        key=lambda r: (
            r["locale"],
            r["n_classes_requested"],
            r["samples_per_class"],
            r["method"],
        )
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "locale",
                "method",
                "n_classes_requested",
                "samples_per_class",
                "num_trials",
                "accuracy_mean",
                "accuracy_std",
                "accuracy_min",
                "accuracy_max",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {summary_path.resolve()} ({len(rows)} rows)")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument(
        "--locales",
        nargs="*",
        default=None,
        help="default: all locales found under --corpus-dir",
    )
    ap.add_argument("--tsv-names", nargs="*", default=["validated.tsv"])
    ap.add_argument("--min-up-votes", type=int, default=2)
    ap.add_argument("--max-down-votes", type=int, default=0)

    ap.add_argument(
        "--pretrained", required=True, help="checkpoint from train_pretrain.py"
    )

    ap.add_argument("--n-classes", type=int, nargs="+", required=True)
    ap.add_argument(
        "--samples-per-class",
        type=int,
        nargs="+",
        required=True,
        help="enrollment clips per word (this is what few-shot CNN trains on AND what DTW enrolls as templates)",
    )
    ap.add_argument("--max-test-per-word", type=int, default=20)
    ap.add_argument(
        "--trials", type=int, default=5, help="random resample repeats per grid point"
    )
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--output-dir", default="./results_dtw_vs_cnn")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--no-detailed",
        action="store_true",
        help="skip per-utterance CSV (results_trials.csv is still written)",
    )
    ap.add_argument("--dtw-only", action="store_true")
    ap.add_argument("--cnn-only", action="store_true")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="just report grid size + per-locale word availability, then exit",
    )
    ap.add_argument(
        "--summarize-only",
        action="store_true",
        help="skip running trials; just (re)build results_summary.csv from an existing results_trials.csv",
    )

    # DTW config -- fixed single config, defaults mirror evaluate_keyword_detection.py
    ap.add_argument("--dtw-feature", default="mfcc", choices=["mfcc", "pncc"])
    ap.add_argument(
        "--dtw-distance", default="euclidean", choices=["euclidean", "cosine"]
    )
    ap.add_argument(
        "--dtw-classify", default="best_match", choices=["best_match", "average"]
    )
    ap.add_argument("--dtw-band-width", type=float, default=None)
    ap.add_argument(
        "--dtw-sample-rate",
        type=int,
        default=None,
        help="default: same sample_rate as the pretrained CNN checkpoint, for a fair comparison",
    )
    ap.add_argument(
        "--dtw-num-coeffs",
        type=int,
        default=None,
        help="default: same num_coeffs as the pretrained CNN checkpoint",
    )

    # CNN config -- defaults mirror train_finetune.py
    ap.add_argument("--cnn-batch-size", type=int, default=16)
    ap.add_argument("--cnn-epochs", type=int, default=60)
    ap.add_argument("--cnn-lr-head", type=float, default=1e-3)
    ap.add_argument("--cnn-lr-full", type=float, default=1e-4)
    ap.add_argument("--cnn-weight-decay", type=float, default=1e-4)
    ap.add_argument("--cnn-patience", type=int, default=12)
    ap.add_argument("--cnn-unfreeze-at", type=int, default=None)
    ap.add_argument(
        "--cnn-workers",
        type=int,
        default=0,
        help="DataLoader worker processes. Default 0: with hundreds/thousands of "
        "tiny trials, per-trial worker spawn overhead (esp. on Windows) usually "
        "costs more than it saves.",
    )

    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    trials_path = out_dir / "results_trials.csv"
    utter_path = out_dir / "results_utterances.csv"
    summary_path = out_dir / "results_summary.csv"

    if args.summarize_only:
        if not trials_path.exists():
            print(f"No {trials_path} found -- nothing to summarize.")
            sys.exit(1)
        write_summary(trials_path, summary_path)
        return

    corpus_dir = Path(args.corpus_dir)
    if not corpus_dir.exists():
        print(f"Corpus dir not found: {corpus_dir}")
        sys.exit(1)

    locales = args.locales or discover_locales(corpus_dir)
    if not locales:
        print("No locales found (expected subfolders containing a 'clips' folder).")
        sys.exit(1)

    n_grid_points = len(args.n_classes) * len(args.samples_per_class)
    n_runs = (
        n_grid_points
        * args.trials
        * len(locales)
        * (1 if (args.dtw_only or args.cnn_only) else 2)
    )
    print(f"Locales: {locales}")
    print(
        f"Grid: n_classes={args.n_classes} x samples_per_class={args.samples_per_class} "
        f"= {n_grid_points} points, x {args.trials} trials x {len(locales)} locales "
        f"~= {n_runs} total (method) runs"
    )

    if args.dry_run:
        max_spc = max(args.samples_per_class)
        min_spc = min(args.samples_per_class)
        for locale in locales:
            by_word = load_locale_clips(
                corpus_dir,
                locale,
                args.tsv_names,
                args.min_up_votes,
                args.max_down_votes,
            )
            eligible_min = sum(1 for c in by_word.values() if len(c) >= min_spc + 1)
            eligible_max = sum(1 for c in by_word.values() if len(c) >= max_spc + 1)
            print(
                f"[{locale}] {len(by_word)} distinct words | "
                f"eligible at samples_per_class={min_spc}: {eligible_min} | "
                f"eligible at samples_per_class={max_spc}: {eligible_max} "
                f"(need >= max(n_classes)={max(args.n_classes)} to cover your biggest grid point)"
            )
        return

    # -------------------- load pretrained CNN backbone once --------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.pretrained, map_location=device)
    mfcc_cfg = ckpt.get("mfcc_config")
    if mfcc_cfg is None:
        raise RuntimeError("Pretrained checkpoint does not contain mfcc_config.")
    pretrained_state = ckpt["model_state"]
    cnn_extractor = make_extractor(**mfcc_cfg)
    print(
        f"CNN pretrained on {len(ckpt.get('classes', []))} words, MFCC config: {mfcc_cfg}"
    )

    dtw_sample_rate = args.dtw_sample_rate or mfcc_cfg["sample_rate"]
    dtw_num_coeffs = args.dtw_num_coeffs or mfcc_cfg["num_coeffs"]
    print(
        f"DTW feature config: feature={args.dtw_feature} distance={args.dtw_distance} "
        f"classify={args.dtw_classify} sample_rate={dtw_sample_rate} num_coeffs={dtw_num_coeffs} "
        f"(matched to the CNN checkpoint's MFCC config unless overridden)"
    )

    # -------------------- resume bookkeeping --------------------
    completed = set()
    if args.resume and trials_path.exists():
        with open(trials_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                completed.add(
                    (
                        row["locale"],
                        row["method"],
                        int(row["n_classes_requested"]),
                        int(row["samples_per_class"]),
                        int(row["trial"]),
                    )
                )
        print(
            f"Resuming: {len(completed)} (locale, method, n_classes, spc, trial) rows already done."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    trials_is_new = not (args.resume and trials_path.exists())
    trials_f = open(
        trials_path, "w" if trials_is_new else "a", newline="", encoding="utf-8"
    )
    trials_writer = csv.DictWriter(trials_f, fieldnames=TRIALS_FIELDS)
    if trials_is_new:
        trials_writer.writeheader()

    utter_f = None
    utter_writer = None
    if not args.no_detailed:
        utter_is_new = not (args.resume and utter_path.exists())
        utter_f = open(
            utter_path, "w" if utter_is_new else "a", newline="", encoding="utf-8"
        )
        utter_writer = csv.DictWriter(utter_f, fieldnames=UTTER_FIELDS)
        if utter_is_new:
            utter_writer.writeheader()

    def write_utter_rows(rows, locale, method, n_classes, spc, trial):
        if utter_writer is None:
            return
        for r in rows:
            r = dict(r)
            r.update(
                {
                    "locale": locale,
                    "method": method,
                    "n_classes_requested": n_classes,
                    "samples_per_class": spc,
                    "trial": trial,
                }
            )
            utter_writer.writerow(r)
        utter_f.flush()

    run_dtw = not args.cnn_only
    run_cnn = not args.dtw_only

    try:
        for locale in locales:
            print(f"\n=== Locale: {locale} ===")
            by_word = load_locale_clips(
                corpus_dir,
                locale,
                args.tsv_names,
                args.min_up_votes,
                args.max_down_votes,
            )
            dtw_cache = DTWAudioCache(dtw_sample_rate)

            for n_classes in args.n_classes:
                for spc in args.samples_per_class:
                    for trial in range(args.trials):
                        seed = seed_for(args.seed, n_classes, spc, trial)
                        key_dtw = (locale, "dtw", n_classes, spc, trial)
                        key_cnn = (locale, "cnn", n_classes, spc, trial)

                        need_dtw = run_dtw and key_dtw not in completed
                        need_cnn = run_cnn and key_cnn not in completed
                        if not need_dtw and not need_cnn:
                            continue

                        try:
                            split = create_split(
                                by_word,
                                num_enrollment=spc,
                                max_test_per_word=args.max_test_per_word,
                                n_classes=n_classes,
                                seed=seed,
                            )
                        except EOFError as e:
                            print(
                                f"  [skip] n_classes={n_classes} spc={spc} trial={trial}: {e}"
                            )
                            continue

                        classes = sorted(split.keys())
                        n_classes_actual = len(classes)
                        if n_classes_actual < 2:
                            print(
                                f"  [skip] n_classes={n_classes} spc={spc} trial={trial}: "
                                f"only {n_classes_actual} usable word(s) after filtering"
                            )
                            continue

                        num_enrollment_total = sum(len(split[w][0]) for w in classes)
                        num_test_total = sum(len(split[w][1]) for w in classes)

                        base_row = {
                            "locale": locale,
                            "n_classes_requested": n_classes,
                            "n_classes_actual": n_classes_actual,
                            "samples_per_class": spc,
                            "trial": trial,
                            "seed": seed,
                            "num_enrollment_total": num_enrollment_total,
                        }

                        if need_dtw:
                            t0 = time.time()
                            correct, total, rows = dtw_evaluate_split(
                                split,
                                classes,
                                args.dtw_feature,
                                args.dtw_distance,
                                args.dtw_classify,
                                dtw_sample_rate,
                                dtw_num_coeffs,
                                args.dtw_band_width,
                                dtw_cache,
                            )
                            elapsed = time.time() - t0
                            acc = correct / total if total else 0.0
                            print(
                                f"  n_classes={n_classes} spc={spc} trial={trial} "
                                f"DTW acc={acc:.3f} ({correct}/{total}) [{elapsed:.1f}s]"
                            )
                            trials_writer.writerow(
                                {
                                    **base_row,
                                    "method": "dtw",
                                    "num_test_samples": total,
                                    "correct": correct,
                                    "accuracy": round(acc, 4),
                                    "word_error_rate": round(1 - acc, 4),
                                    "eval_time_sec": round(elapsed, 2),
                                    "train_time_sec": "",
                                    "epochs_run": "",
                                    "best_train_acc": "",
                                    "dtw_feature": args.dtw_feature,
                                    "dtw_distance": args.dtw_distance,
                                    "dtw_classify": args.dtw_classify,
                                    "dtw_sample_rate": dtw_sample_rate,
                                    "dtw_num_coeffs": dtw_num_coeffs,
                                }
                            )
                            trials_f.flush()
                            write_utter_rows(rows, locale, "dtw", n_classes, spc, trial)

                        if need_cnn:
                            (
                                correct,
                                total,
                                rows,
                                best_train_acc,
                                epochs_run,
                                train_time,
                            ) = cnn_evaluate_split(
                                split,
                                classes,
                                pretrained_state,
                                mfcc_cfg,
                                cnn_extractor,
                                device,
                                batch_size=args.cnn_batch_size,
                                epochs=args.cnn_epochs,
                                lr_head=args.cnn_lr_head,
                                lr_full=args.cnn_lr_full,
                                weight_decay=args.cnn_weight_decay,
                                patience=args.cnn_patience,
                                unfreeze_at=args.cnn_unfreeze_at,
                                workers=args.cnn_workers,
                            )
                            acc = correct / total if total else 0.0
                            print(
                                f"  n_classes={n_classes} spc={spc} trial={trial} "
                                f"CNN acc={acc:.3f} ({correct}/{total}) "
                                f"train_acc={best_train_acc:.3f} epochs={epochs_run} [{train_time:.1f}s]"
                            )
                            trials_writer.writerow(
                                {
                                    **base_row,
                                    "method": "cnn",
                                    "num_test_samples": total,
                                    "correct": correct,
                                    "accuracy": round(acc, 4),
                                    "word_error_rate": round(1 - acc, 4),
                                    "eval_time_sec": "",
                                    "train_time_sec": round(train_time, 2),
                                    "epochs_run": epochs_run,
                                    "best_train_acc": round(best_train_acc, 4),
                                    "dtw_feature": "",
                                    "dtw_distance": "",
                                    "dtw_classify": "",
                                    "dtw_sample_rate": "",
                                    "dtw_num_coeffs": "",
                                }
                            )
                            trials_f.flush()
                            write_utter_rows(rows, locale, "cnn", n_classes, spc, trial)

            dtw_cache.clear()
    finally:
        trials_f.close()
        if utter_f is not None:
            utter_f.close()

    print(f"\nWrote {trials_path.resolve()}")
    if utter_writer is not None:
        print(f"Wrote {utter_path.resolve()}")

    write_summary(trials_path, summary_path)


if __name__ == "__main__":
    main()
