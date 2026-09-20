#!/usr/bin/env python3
"""MimicVerse harvester v1.5 — panel + score + snapshot in one run."""

from __future__ import annotations

import argparse
import os
import sys

print("MimicVerse harvester v1.5")

from mimicverse.harvest import harvest
from mimicverse.snapshot import persist, build_snapshot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--posts", type=int, default=20)
    parser.add_argument("--transformer", action="store_true",
                        help="Batch-score with GoEmotions if the model can load")
    parser.add_argument("--no-sleep", action="store_true")
    args = parser.parse_args()

    use_transformer = args.transformer or os.getenv("MIMICVERSE_TRANSFORMER") == "1"

    rows = harvest(posts_per_sub=args.posts, sleep=not args.no_sleep)
    if not rows:
        print("no posts harvested")
        return 1

    snap, scored = build_snapshot(rows, use_transformer=use_transformer)
    path = persist(snap, scored)
    print(f"snapshot -> {path}")
    print(snap["street_line"])
    print(f"n={snap['n_posts']} source={snap['score_source']} dominant={snap['overall']['dominant']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
