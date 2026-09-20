#!/usr/bin/env python3
"""Idempotent HarvestScroll updater. Safe if v1.5 persist() already wrote the row."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SCROLL_PATH = DATA_DIR / "HarvestScroll.csv"


def main() -> None:
    files = sorted(p.name for p in DATA_DIR.glob("reddit_*.csv"))
    if not files:
        print("No reddit_*.csv files.")
        return

    if SCROLL_PATH.exists():
        scroll = pd.read_csv(SCROLL_PATH)
    else:
        SCROLL_PATH.parent.mkdir(parents=True, exist_ok=True)
        scroll = pd.DataFrame(
            columns=["seq_id", "file_name", "timestamp_utc", "posts", "subreddits", "harvester_type", "size_bytes"]
        )

    logged = set(scroll["file_name"].astype(str)) if "file_name" in scroll.columns else set()
    new_rows = []
    last_seq = int(scroll["seq_id"].max()) if "seq_id" in scroll.columns and len(scroll) else 0

    for fname in files:
        if fname in logged:
            continue
        path = DATA_DIR / fname
        posts = max(0, sum(1 for _ in path.open(encoding="utf-8", errors="ignore")) - 1)
        subreddits = ""
        try:
            sr = pd.read_csv(path, usecols=["subreddit"], dtype=str)
            subreddits = int(sr["subreddit"].nunique())
        except Exception:
            pass
        last_seq += 1
        new_rows.append(
            {
                "seq_id": last_seq,
                "file_name": fname,
                "timestamp_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "posts": posts,
                "subreddits": subreddits,
                "harvester_type": "v1.5-panel",
                "size_bytes": path.stat().st_size,
            }
        )

    if not new_rows:
        print("HarvestScroll already up to date.")
        return

    updated = pd.concat([scroll, pd.DataFrame(new_rows)], ignore_index=True)
    updated.to_csv(SCROLL_PATH, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Logged {len(new_rows)} new harvest(s).")


if __name__ == "__main__":
    main()
