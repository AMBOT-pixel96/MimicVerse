#!/usr/bin/env python3
"""
scripts/harvest_logger.py

Idempotent logger that ensures data/HarvestScroll.csv lists every reddit_*.csv
recorded in data/. Appends only new entries. Safe for manual or CI runs.
"""

import os
import sys
import csv
import pandas as pd
from datetime import datetime
from pathlib import Path
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SCROLL_PATH = DATA_DIR / "HarvestScroll.csv"

def git(cmd):
    return subprocess.run(["git"] + cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)

def list_reddit_files():
    files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith("reddit_") and f.endswith(".csv")])
    return files

def file_stats(fname):
    p = DATA_DIR / fname
    # number of posts = number of data rows (minus possible header)
    try:
        # read with pandas (fast enough) but only count lines to avoid memory explosion
        # pandas.read_csv with usecols will still read; we can do a streaming count
        with open(p, "r", encoding="utf-8", errors="ignore") as fh:
            # subtract header if present
            total_lines = sum(1 for _ in fh)
            posts = max(0, total_lines - 1)
    except Exception:
        posts = None

    # get unique subreddits (try streaming first)
    subreddits = None
    try:
        # read only subreddit column if present - faster
        for sep in [",", ";", "\t"]:
            try:
                # attempt reading first line to detect header
                df_sample = pd.read_csv(p, nrows=3, sep=sep)
                if "subreddit" in df_sample.columns:
                    # read subreddit column only (lower memory)
                    sr = pd.read_csv(p, usecols=["subreddit"], squeeze=True, dtype=str, sep=sep)
                    subreddits = int(sr.nunique(dropna=True))
                    break
            except Exception:
                continue
    except Exception:
        subreddits = None

    size_bytes = p.stat().st_size
    return posts if posts is not None else "", subreddits if subreddits is not None else "", size_bytes

def load_scroll():
    if not SCROLL_PATH.exists():
        # create with header
        SCROLL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SCROLL_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["seq_id","file_name","timestamp_utc","posts","subreddits","harvester_type","size_bytes"])
        return pd.read_csv(SCROLL_PATH)
    else:
        return pd.read_csv(SCROLL_PATH)

def main():
    os.chdir(REPO_ROOT)
    files = list_reddit_files()
    if not files:
        print("No reddit_*.csv files found in data/ — nothing to do.")
        return

    scroll = load_scroll()

    # Build dictionary of already-logged file names for quick lookup
    logged = set(scroll['file_name'].astype(str).tolist()) if 'file_name' in scroll.columns else set()

    new_rows = []
    for fname in files:
        if fname in logged:
            continue
        # compute stats
        posts, subreddits, size_bytes = file_stats(fname)
        # determine harvester_type by simple heuristic: if size huge -> auto, else maybe manual
        harvester_type = "auto"
        # set timestamp_utc to file modified time (UTC)
        ts = datetime.utcfromtimestamp((DATA_DIR / fname).stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        new_rows.append({
            "seq_id": None,
            "file_name": fname,
            "timestamp_utc": ts,
            "posts": posts,
            "subreddits": subreddits,
            "harvester_type": harvester_type,
            "size_bytes": int(size_bytes)
        })

    if not new_rows:
        print("No new files to log. HarvestScroll already up-to-date.")
        return

    # append rows with proper seq_id
    last_seq = int(scroll['seq_id'].max()) if 'seq_id' in scroll.columns and not scroll.empty else 0
    for i, r in enumerate(new_rows, start=1):
        r['seq_id'] = last_seq + i

    # append to CSV
    updated = pd.concat([scroll, pd.DataFrame(new_rows)], ignore_index=True, sort=False)
    updated.to_csv(SCROLL_PATH, index=False)

    # git add/commit/push only if changes exist
    git("add data/HarvestScroll.csv".split())
    status = git(["status","--porcelain"]).stdout.strip()
    if status == "":
        print("No git changes after update (unexpected).")
        return

    # commit as bot
    git(["config","user.name","MimicVerse Bot"])
    git(["config","user.email","bot@users.noreply.github.com"])
    commit_msg = f"chore: update HarvestScroll — {datetime.utcnow().isoformat()} (auto)"
    git(["commit","-m", commit_msg])
    # push (GITHUB_TOKEN is available in workflow environment)
    push = git(["push"])
    if push.returncode == 0:
        print("HarvestScroll updated and pushed successfully.")
    else:
        print("Failed to push HarvestScroll. stdout/stderr:")
        print(push.stdout)
        print(push.stderr)

if __name__ == "__main__":
    main()