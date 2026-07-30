# common_voice.py

import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class ClipInfo:
    path: str
    word: str
    age: str = "unknown"
    gender: str = "unknown"
    accent: str = "unknown"


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
    """
    Load Common Voice clips for one locale and group them by word.

    Returns:
        {
            "word1": [ClipInfo(...), ...],
            "word2": [...],
        }
    """

    locale_dir = corpus_dir / locale
    clips_dir = locale_dir / "clips"

    seen_paths = set()
    by_word = defaultdict(list)

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
                    up_votes = int(row.get("up_votes", "0") or 0)
                    down_votes = int(row.get("down_votes", "0") or 0)
                except ValueError:
                    up_votes = 0
                    down_votes = 0

                if up_votes < min_up_votes:
                    continue

                if down_votes > max_down_votes:
                    continue

                abs_path = clips_dir / rel_path

                if not abs_path.exists():
                    continue

                seen_paths.add(rel_path)

                word = sentence.lower()

                by_word[word].append(
                    ClipInfo(
                        path=str(abs_path.resolve()),
                        word=word,
                        age=(row.get("age") or "unknown").strip() or "unknown",
                        gender=(row.get("gender") or "unknown").strip() or "unknown",
                        accent=(row.get("accent") or "unknown").strip() or "unknown",
                    )
                )

    return dict(by_word)


def create_split(
    by_word: Dict[str, List[ClipInfo]],
    num_enrollment: int,
    max_test_per_word: int,
    n_classes: int,
    seed: int,
) -> Dict[str, Tuple[List[ClipInfo], List[ClipInfo]]]:
    """
    Create a deterministic enrollment/test split.

    Every selected word gets:
        num_enrollment clips for training/enrollment
        up to max_test_per_word clips for testing

    The same split can be used by both DTW and neural fine-tuning.
    """

    rng = random.Random(seed)

    eligible = [
        word for word, clips in by_word.items() if len(clips) >= num_enrollment + 1
    ]

    eligible.sort()
    rng.shuffle(eligible)
    if len(eligible) < n_classes:
        raise EOFError(f"Not enough words to have {n_classes} classes")
    eligible = eligible[:n_classes]

    split = {}

    for word in sorted(eligible):
        clips = by_word[word].copy()
        rng.shuffle(clips)

        enrollment = clips[:num_enrollment]

        test = clips[num_enrollment : num_enrollment + max_test_per_word]

        if test:
            split[word] = (enrollment, test)

    return split


def save_split(
    split: Dict[str, Tuple[List[ClipInfo], List[ClipInfo]]],
    path: Path,
):
    data = {}

    for word, (enrollment, test) in split.items():
        data[word] = {
            "enrollment": [asdict(clip) for clip in enrollment],
            "test": [asdict(clip) for clip in test],
        }

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_split(
    path: Path,
) -> Dict[str, Tuple[List[ClipInfo], List[ClipInfo]]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    split = {}

    for word, values in data.items():
        enrollment = [ClipInfo(**clip) for clip in values["enrollment"]]

        test = [ClipInfo(**clip) for clip in values["test"]]

        split[word] = (enrollment, test)

    return split
