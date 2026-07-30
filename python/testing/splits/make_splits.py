# make_split.py

import argparse
from pathlib import Path

from common_voice import (
    load_locale_clips,
    create_split,
    save_split,
    discover_locales,
)


def create_a_split(args):

    corpus_dir = Path(args.corpus_dir)
    locales = args.locales
    if locales == None:
        locales = discover_locales(corpus_dir)
    for locale in locales:
        by_word = load_locale_clips(
            corpus_dir,
            locale,
            args.tsv_names,
            args.min_up_votes,
            args.max_down_votes,
        )
        try:
            split = create_split(
                by_word,
                num_enrollment=args.num_enrollment,
                max_test_per_word=args.max_test_per_word,
                n_classes=args.n_classes,
                seed=args.seed,
            )
        except:
            continue

        save_split(
            split,
            Path(
                args.out_dir,
                f"{args.num_enrollment}_samples_per_word",
                f"{args.n_classes}_classes",
                f"{locale}_split.json",
            ),
        )

        print(f"Created split with " f"{len(split)} words")
        print(f"Enrollment samples: " f"{sum(len(x[0]) for x in split.values())}")
        print(f"Test samples: " f"{sum(len(x[1]) for x in split.values())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--locales", type=list[str], default=None)  # None is all locales
    ap.add_argument("--tsv-names", nargs="+", default=["validated.tsv"])
    ap.add_argument("--num-enrollment", type=int, default=5)
    ap.add_argument("--max-test-per-word", type=int, default=5)
    ap.add_argument("--n_classes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()
    create_a_split(args)


if __name__ == "__main__":
    main()
