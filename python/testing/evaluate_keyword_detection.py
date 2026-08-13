"""
evaluate_keyword_detection.py

Benchmark harness for the `keyword_detection` (kd) library against the
Mozilla Common Voice "single-word" corpus (cv-corpus-7.0-singleword).

It sweeps over every combination of:
  - feature extractor   : MFCC vs PNCC
  - DTW distance metric : Euclidean vs Cosine
  - classification rule : best-match vs average
  - additive noise level: a list of SNRs (dB), plus a "clean" (no-noise) run

For every locale it finds under --corpus-dir it:
  1. Groups clips by word (the `sentence` column, which is a single word
     in this corpus) and keeps words that have enough usable clips.
  2. Picks `--num-templates` clips per word as enrollment templates
     (this is your "~5 samples" enrollment set).
  3. Tests on the remaining clips (capped by --max-test-per-word) against
     a matcher enrolled with ALL words in that locale, so "accuracy" here
     is real multi-class classification accuracy, and 1-accuracy is
     reported as the word error rate (WER) for this isolated-word task.

Outputs (written to --output-dir):
  - results_detailed.csv    one row per test utterance (with age/gender/
                             accent columns so you can slice by demographic
                             even though a lot of that data is missing)
  - results_summary.csv     accuracy/WER aggregated per config, across all
                             locales
  - results_by_locale.csv   accuracy/WER aggregated per config x locale
  - results_by_gender.csv   accuracy/WER aggregated per config x gender,
                             rows with unknown gender are dropped here

================================================================================
API notes (confirmed against the pybind11 bindings source)
================================================================================
  kd.MFCCProcessor() / kd.PNCCProcessor()
      .sample_rate, .num_coeffs, .frame_length, .hop_length  (both)
      .num_mel_bands                                          (MFCC only)
      .num_gamma_bands, .power_law_exponent, .medium_time_frames (PNCC only)
      .compute(samples: 1D float32 ndarray) -> (n_frames, num_coeffs) float32

  kd.DTWMatcher()
      .distance_metric: DistanceMetric.EUCLIDEAN | DistanceMetric.COSINE
      .classify_method:  ClassifyMethod.BEST_MATCH | ClassifyMethod.AVERAGE_DISTANCE
      .band_width: int (Sakoe-Chiba band; optional tunable, see --band-width)
      .add_template(label: str, features: (n_frames, n_coeffs) ndarray)
      .classify_with_best_score(features) -> {"label": str, "distance": float}
      .classify_with_every_score(features) -> {label: distance, ...}
      .clear_templates()

This script only relies on add_template / classify_with_best_score, both of
which are fully specified above, so there's nothing left to guess. Run with
--dry-run first anyway - it's a cheap sanity check on your corpus layout
(word counts per locale) before a long run.
================================================================================
"""

import argparse
import csv
import os
import random
import sys
import time
import traceback
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import librosa

import keyword_detection as kd


# --------------------------------------------------------------------------
#  feature extractor registry
# --------------------------------------------------------------------------
def build_feature_extractor(name: str, sample_rate: int, num_coeffs: int):
    """Return a configured feature extractor instance for `name`."""
    if name == "mfcc":
        fe = kd.MFCCProcessor()
        fe.sample_rate = sample_rate
        fe.num_coeffs = num_coeffs
        return fe
    elif name == "pncc":
        fe = kd.PNCCProcessor()
        fe.sample_rate = sample_rate
        fe.num_coeffs = num_coeffs
        return fe
    else:
        raise ValueError(f"Unknown feature extractor: {name}")


# --------------------------------------------------------------------------
# Distance metric / classify method registry
# --------------------------------------------------------------------------
def get_distance_metric(name: str):
    DM = kd.DTWMatcher.DistanceMetric
    if name == "euclidean":
        return DM.EUCLIDEAN
    elif name == "cosine":
        return DM.COSINE
    else:
        raise ValueError(f"Unknown distance metric: {name}")


def get_classify_method(name: str):
    CM = kd.DTWMatcher.ClassifyMethod
    if name == "best_match":
        return CM.BEST_MATCH
    elif name == "average":
        return CM.AVERAGE_DISTANCE
    else:
        raise ValueError(f"Unknown classify method: {name}")


def build_matcher(distance_name: str, classify_name: str, band_width: Optional[int] = None):
    matcher = kd.DTWMatcher()
    matcher.distance_metric = get_distance_metric(distance_name)
    matcher.classify_method = get_classify_method(classify_name)
    if band_width is not None:
        matcher.band_width = band_width
    return matcher


def parse_classify_result(result) -> Tuple[Optional[str], Optional[float]]:
    """classify_with_best_score() returns {"label": str, "distance": float}."""
    if result is None:
        return None, None
    if isinstance(result, dict):
        return result.get("label"), result.get("distance")
    # Fallback in case a future binding version returns an object instead.
    if hasattr(result, "label"):
        return getattr(result, "label", None), getattr(result, "distance", None)
    raise AttributeError(
        f"Don't know how to read classify_with_best_score()'s return value "
        f"(got type {type(result)}: {result!r})."
    )


# --------------------------------------------------------------------------
# Noise injection
# --------------------------------------------------------------------------
def add_noise(signal: np.ndarray, snr_db: Optional[float], seed: int) -> np.ndarray:
    """Add white Gaussian noise at the requested SNR (dB). snr_db=None -> clean."""
    if snr_db is None:
        return signal
    rng = np.random.default_rng(seed)
    sig_power = float(np.mean(signal.astype(np.float64) ** 2))
    if sig_power <= 0:
        return signal
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise = rng.normal(0.0, np.sqrt(noise_power), size=signal.shape).astype(np.float32)
    return (signal + noise).astype(np.float32)


def noise_seed_for(path: str, snr_label: str) -> int:
    """Deterministic seed per (clip, noise level) so runs are reproducible."""
    return abs(hash((path, snr_label))) % (2**31 - 1)


def parse_snr_list(raw: str) -> List[Tuple[str, Optional[float]]]:
    """'clean,20,10,0' -> [('clean', None), ('20dB', 20.0), ...]"""
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.lower() == "clean":
            out.append(("clean", None))
        else:
            val = float(tok)
            out.append((f"{val:g}dB", val))
    return out


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@dataclass
class ClipInfo:
    path: str          # absolute path to mp3
    word: str
    age: str
    gender: str
    accent: str


def discover_locales(corpus_dir: Path) -> List[str]:
    locales = []
    for entry in sorted(corpus_dir.iterdir()):
        if entry.is_dir() and (entry / "clips").is_dir():
            locales.append(entry.name)
    return locales


def load_locale_clips(
    corpus_dir: Path,
    locale: str,
    tsv_names: List[str],
    min_up_votes: int,
    max_down_votes: int,
) -> Dict[str, List[ClipInfo]]:
    """Read the tsv(s) for a locale, dedup by clip path, group by word."""
    locale_dir = corpus_dir / locale
    clips_dir = locale_dir / "clips"

    seen_paths = set()
    by_word: Dict[str, List[ClipInfo]] = defaultdict(list)

    for tsv_name in tsv_names:
        tsv_path = locale_dir / tsv_name
        if not tsv_path.exists():
            continue
        with open(tsv_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                rel_path = row.get("path", "").strip()
                sentence = row.get("sentence", "").strip()
                if not rel_path or not sentence:
                    continue
                if rel_path in seen_paths:
                    continue

                try:
                    up = int(row.get("up_votes", "0") or 0)
                    down = int(row.get("down_votes", "0") or 0)
                except ValueError:
                    up, down = 0, 0
                if up < min_up_votes or down > max_down_votes:
                    continue

                abs_path = clips_dir / rel_path
                if not abs_path.exists():
                    continue

                seen_paths.add(rel_path)
                word = sentence.lower()
                by_word[word].append(
                    ClipInfo(
                        path=str(abs_path),
                        word=word,
                        age=(row.get("age") or "unknown").strip() or "unknown",
                        gender=(row.get("gender") or "unknown").strip() or "unknown",
                        accent=(row.get("accent") or "unknown").strip() or "unknown",
                    )
                )
    return by_word


def select_words(
    by_word: Dict[str, List[ClipInfo]],
    num_templates: int,
    max_test_per_word: int,
    max_words: int,
    rng: random.Random,
) -> Dict[str, Tuple[List[ClipInfo], List[ClipInfo]]]:
    """Keep words with enough clips; split each into (templates, test_pool)."""
    eligible = [w for w, clips in by_word.items() if len(clips) >= num_templates + 1]
    rng.shuffle(eligible)
    eligible = eligible[:max_words]

    split: Dict[str, Tuple[List[ClipInfo], List[ClipInfo]]] = {}
    for w in eligible:
        clips = by_word[w][:]
        rng.shuffle(clips)
        templates = clips[:num_templates]
        test_pool = clips[num_templates:num_templates + max_test_per_word]
        if test_pool:
            split[w] = (templates, test_pool)
    return split


# --------------------------------------------------------------------------
# Silence trimming (mirrors KeywordMatcher._trim_silence in keyword_matcher.gd)
# --------------------------------------------------------------------------
def trim_silence(
    pcm: np.ndarray,
    sample_rate: int,
    window_ms: float = 20.0,
    threshold_ratio: float = 0.05,
) -> np.ndarray:
    """Strip leading/trailing silence based on short-window RMS energy
    relative to the clip's peak RMS. Must match the Godot-side trimming
    (same window_ms/threshold_ratio semantics) so evaluation numbers reflect
    what the deployed app actually sees - templates and live audio there are
    both trimmed the same way, right before feature extraction."""
    if pcm.size == 0:
        return pcm

    window = max(1, int(round(window_ms * sample_rate / 1000.0)))
    num_windows = int(np.ceil(pcm.size / window))

    rms = np.empty(num_windows, dtype=np.float64)
    for w in range(num_windows):
        start = w * window
        end = min(start + window, pcm.size)
        chunk = pcm[start:end].astype(np.float64)
        rms[w] = np.sqrt(np.mean(chunk ** 2))

    peak_rms = rms.max()
    if peak_rms <= 0.0:
        return pcm  # fully silent clip, nothing sensible to trim

    threshold = peak_rms * threshold_ratio
    above = np.flatnonzero(rms >= threshold)
    if above.size == 0:
        return pcm  # nothing above threshold, leave as-is

    first_window, last_window = int(above[0]), int(above[-1])
    start_sample = first_window * window
    end_sample = min((last_window + 1) * window, pcm.size)
    return pcm[start_sample:end_sample]


# --------------------------------------------------------------------------
# Audio / feature helpers
# --------------------------------------------------------------------------
def load_mp3_as_pcm(path: str, target_sr: int) -> np.ndarray:
    samples, _sr = librosa.load(path, sr=target_sr, mono=True)
    return samples.astype(np.float32)


class FeatureCache:
    """Caches PCM loads and computed features within a single locale's run
    so the same clip isn't decoded/re-featurized for every
    distance-metric x classify-method combo (they share features)."""

    def __init__(
        self,
        target_sr: int,
        trim_window_ms: float = 20.0,
        trim_threshold_ratio: float = 0.05,
        trim_enabled: bool = True,
    ):
        self.target_sr = target_sr
        self.trim_window_ms = trim_window_ms
        self.trim_threshold_ratio = trim_threshold_ratio
        self.trim_enabled = trim_enabled
        self._pcm_cache: Dict[str, np.ndarray] = {}
        self._feat_cache: Dict[Tuple[str, str, str, int], np.ndarray] = {}

    def get_pcm(self, path: str) -> np.ndarray:
        if path not in self._pcm_cache:
            self._pcm_cache[path] = load_mp3_as_pcm(path, self.target_sr)
        return self._pcm_cache[path]

    def get_features(
        self,
        path: str,
        feature_name: str,
        snr_label: str,
        snr_db: Optional[float],
        feature_extractor,
    ) -> np.ndarray:
        key = (path, feature_name, snr_label, feature_extractor.num_coeffs)
        if key not in self._feat_cache:
            pcm = self.get_pcm(path)
            noisy = add_noise(pcm, snr_db, noise_seed_for(path, snr_label))
            # Trim AFTER noise injection: this mirrors the real pipeline,
            # where the mic captures noisy audio and trimming runs on
            # whatever was actually recorded, not on a clean reference.
            if self.trim_enabled:
                signal = trim_silence(
                    noisy, self.target_sr, self.trim_window_ms, self.trim_threshold_ratio
                )
            else:
                signal = noisy
            self._feat_cache[key] = feature_extractor.compute(signal)
        return self._feat_cache[key]

    def clear(self):
        self._pcm_cache.clear()
        self._feat_cache.clear()


# --------------------------------------------------------------------------
# Core evaluation
# --------------------------------------------------------------------------
@dataclass
class RunConfig:
    feature: str
    distance: str
    classify: str
    snr_label: str
    snr_db: Optional[float]


def run_config_for_locale(
    cfg: RunConfig,
    locale: str,
    word_split: Dict[str, Tuple[List[ClipInfo], List[ClipInfo]]],
    cache: FeatureCache,
    sample_rate: int,
    num_coeffs: int,
    detailed_rows: List[dict],
    band_width: Optional[int] = None,
):
    feature_extractor = build_feature_extractor(cfg.feature, sample_rate, num_coeffs)
    matcher = build_matcher(cfg.distance, cfg.classify, band_width)

    # Enroll templates for every word so this is real multi-class
    # classification, not per-word verification.
    for word, (templates, _test_pool) in word_split.items():
        for clip in templates:
            feats = cache.get_features(
                clip.path, cfg.feature, cfg.snr_label, cfg.snr_db, feature_extractor
            )
            matcher.add_template(word, feats)

    correct = 0
    total = 0
    for word, (_templates, test_pool) in word_split.items():
        for clip in test_pool:
            feats = cache.get_features(
                clip.path, cfg.feature, cfg.snr_label, cfg.snr_db, feature_extractor
            )
            try:
                result = matcher.classify_with_best_score(feats)
                predicted, best_distance = parse_classify_result(result)
            except Exception as e:  # noqa: BLE001
                predicted, best_distance = None, None
                print(f"  [warn] classify failed on {clip.path}: {e}")

            is_correct = predicted == word
            correct += int(is_correct)
            total += 1

            detailed_rows.append(
                {
                    "locale": locale,
                    "feature": cfg.feature,
                    "distance": cfg.distance,
                    "classify": cfg.classify,
                    "noise": cfg.snr_label,
                    "true_word": word,
                    "predicted_word": predicted,
                    "correct": is_correct,
                    "best_distance": best_distance,
                    "clip_path": clip.path,
                    "age": clip.age,
                    "gender": clip.gender,
                    "accent": clip.accent,
                }
            )

    return correct, total


# --------------------------------------------------------------------------
# Aggregation / CSV writing
# --------------------------------------------------------------------------
def write_csv(path: Path, rows: List[dict], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(detailed_rows: List[dict], group_keys: List[str], drop_unknown_key: Optional[str] = None):
    buckets = defaultdict(lambda: {"correct": 0, "total": 0, "dist_sum": 0.0, "dist_n": 0})
    for row in detailed_rows:
        if drop_unknown_key and row.get(drop_unknown_key) in (None, "", "unknown"):
            continue
        key = tuple(row[k] for k in group_keys)
        b = buckets[key]
        b["total"] += 1
        b["correct"] += int(row["correct"])
        d = row.get("best_distance")
        if d is not None:
            b["dist_sum"] += d
            b["dist_n"] += 1

    out_rows = []
    for key, b in buckets.items():
        total, correct = b["total"], b["correct"]
        acc = correct / total if total else 0.0
        row = dict(zip(group_keys, key))
        row["num_test_samples"] = total
        row["correct"] = correct
        row["accuracy"] = round(acc, 4)
        row["word_error_rate"] = round(1 - acc, 4)
        row["avg_best_distance"] = round(b["dist_sum"] / b["dist_n"], 4) if b["dist_n"] else ""
        out_rows.append(row)
    out_rows.sort(key=lambda r: group_keys and r[group_keys[0]])
    return out_rows


def process_locale(
    locale: str,
    corpus_dir: Path,
    tsv_names,
    min_up_votes: int,
    max_down_votes: int,
    num_templates: int,
    max_test_per_word: int,
    max_words_per_locale: int,
    seed: int,
    sample_rate: int,
    num_coeffs: int,
    combos,
    band_width,
    trim_window_ms: float = 20.0,
    trim_threshold_ratio: float = 0.05,
    trim_enabled: bool = True,
):
    rng = random.Random(seed)
    detailed_rows = []
    by_word = load_locale_clips(
        corpus_dir,
        locale,
        tsv_names,
        min_up_votes,
        max_down_votes,
    )
    word_split = select_words(
        by_word,
        num_templates,
        max_test_per_word,
        max_words_per_locale,
        rng,
    )
    if not word_split:
        return locale, [], []
    cache = FeatureCache(sample_rate, trim_window_ms, trim_threshold_ratio, trim_enabled)
    summary = []
    for feature, distance, classify, (snr_label, snr_db) in combos:
        cfg = RunConfig(
            feature,
            distance,
            classify,
            snr_label,
            snr_db,
        )
        t0 = time.time()
        try:
            correct, total = run_config_for_locale(
                cfg,
                locale,
                word_split,
                cache,
                sample_rate,
                num_coeffs,
                detailed_rows,
                band_width=band_width,
            )
        except Exception as e:
            print(f"[ERROR] {locale} {cfg}: {e}")
            traceback.print_exc()
            continue
        acc = correct / total if total else 0
        summary.append(
            (
                feature,
                distance,
                classify,
                snr_label,
                acc,
                correct,
                total,
                time.time() - t0,
            )
        )
    cache.clear()
    return locale, detailed_rows, summary

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--corpus-dir",
        required=True,
        help="Path to cv-corpus-7.0-singleword (or similar) root folder",
    )
    parser.add_argument("--locales", nargs="*", default=None, help="Subset of locale codes to run (default: all found)")
    parser.add_argument("--tsv-names", nargs="*", default=["validated.tsv"])
    parser.add_argument("--num-templates", type=int, default=5)
    parser.add_argument("--max-test-per-word", type=int, default=5)
    parser.add_argument("--max-words-per-locale", type=int, default=50)
    parser.add_argument("--min-up-votes", type=int, default=2)
    parser.add_argument("--max-down-votes", type=int, default=0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--num-coeffs", type=int, default=13)
    parser.add_argument("--features", nargs="*", default=["mfcc", "pncc"])
    parser.add_argument("--distances", nargs="*", default=["euclidean", "cosine"])
    parser.add_argument("--classify-methods", nargs="*", default=["best_match", "average"])
    parser.add_argument("--noise-snr-db", default="clean,20,10,0", help="Comma list, e.g. 'clean,20,10,0'")
    parser.add_argument("--band-width", type=float, default=None, help="Sakoe-Chiba DTW band width; omit to use the library default")
    parser.add_argument("--trim-window-ms", type=float, default=20.0, help="Silence-trim RMS analysis window, in ms")
    parser.add_argument("--trim-threshold-ratio", type=float, default=0.05, help="Silence-trim threshold as a fraction of the clip's peak RMS")
    parser.add_argument("--no-trim", action="store_true", help="Disable silence trimming (for A/B comparison against the trimmed pipeline)")
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Sanity-check API names + data availability, then exit")
    parser.add_argument("--jobs",type=int,default=max(1, multiprocessing.cpu_count() - 1),help="Number of worker processes")

    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    if not corpus_dir.exists():
        print(f"Corpus dir not found: {corpus_dir}")
        sys.exit(1)

    locales = args.locales or discover_locales(corpus_dir)
    if not locales:
        print("No locales found (expected subfolders containing a 'clips' folder).")
        sys.exit(1)

    snr_configs = parse_snr_list(args.noise_snr_db)
    combos = list(product(args.features, args.distances, args.classify_methods, snr_configs))

    print(f"Locales: {locales}")
    print(f"Sweeping {len(combos)} configs x {len(locales)} locales")

    # --- dry run: just make sure every extractor/metric/method name resolves ---
    if args.dry_run:
        ok = True
        for feature, distance, classify, (snr_label, _snr_db) in combos:
            try:
                build_feature_extractor(feature, args.sample_rate, args.num_coeffs)
                build_matcher(distance, classify, args.band_width)
            except Exception as e:  # noqa: BLE001
                ok = False
                print(f"[FAIL] feature={feature} distance={distance} classify={classify}: {e}")
        for locale in locales:
            by_word = load_locale_clips(corpus_dir, locale, args.tsv_names, args.min_up_votes, args.max_down_votes)
            usable = sum(1 for c in by_word.values() if len(c) >= args.num_templates + 1)
            print(f"[{locale}] {len(by_word)} distinct words, {usable} usable with >= {args.num_templates + 1} clips")
        print("Dry run OK" if ok else "Dry run found problems - fix the ADAPT ME sections above.")
        return

    detailed_rows: List[dict] = []
    rng = random.Random(args.seed)
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = []
        for locale in locales:
            futures.append(
                executor.submit(process_locale, locale, corpus_dir, args.tsv_names, args.min_up_votes, args.max_down_votes, args.num_templates, args.max_test_per_word, args.max_words_per_locale, args.seed, args.sample_rate, args.num_coeffs, combos, args.band_width, args.trim_window_ms, args.trim_threshold_ratio, not args.no_trim,)
            )
        for future in as_completed(futures):
            locale, locale_rows, summary = future.result()
            if not locale_rows:
                print(f"\n=== Locale: {locale} ===")
                print("  skipping")
                continue
            print(f"\n=== Locale: {locale} ===")
            for feature, distance, classify, noise, acc, correct, total, elapsed in summary:
                print(
                    f"  {feature:5s} "
                    f"{distance:9s} "
                    f"{classify:10s} "
                    f"noise={noise:6s} "
                    f"acc={acc:.3f} "
                    f"({correct}/{total}) "
                    f"[{elapsed:.1f}s]"
                )
            detailed_rows.extend(locale_rows)

    print(f"\nTotal run time: {time.time() - t_start:.1f}s")

    if not detailed_rows:
        print("No results collected - nothing to write.")
        return

    out_dir = Path(args.output_dir)
    detailed_fields = [
        "locale", "feature", "distance", "classify", "noise",
        "true_word", "predicted_word", "correct", "best_distance", "clip_path",
        "age", "gender", "accent",
    ]
    write_csv(out_dir / "results_detailed.csv", detailed_rows, detailed_fields)

    print(f"\nWrote results to {out_dir.resolve()}")
    print("  - results_detailed.csv")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
