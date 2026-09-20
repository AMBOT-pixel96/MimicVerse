"""Write small mood snapshots + timeseries. Git-friendly."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import __version__
from .score import PRIMARY, aggregate, prophet_line, score_records, top_themes

DATA = Path("data")
SNAP_DIR = DATA / "snapshots"
TIMESERIES = DATA / "mood_timeseries.csv"
LATEST = DATA / "latest_snapshot.json"
SCROLL = DATA / "HarvestScroll.csv"


def build_snapshot(records: list[dict], use_transformer: bool = False) -> dict:
    scored = score_records(records, use_transformer=use_transformer)
    overall = aggregate(scored)["all"]
    by_region = aggregate(scored, key="region")
    by_sub = aggregate(scored, key="subreddit")
    titles = [r.get("title") or "" for r in scored]
    themes = top_themes(titles)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    file_stem = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")

    snap = {
        "version": __version__,
        "timestamp_utc": ts,
        "file_stem": file_stem,
        "n_posts": len(scored),
        "n_subreddits": len({r.get("subreddit") for r in scored}),
        "n_regions": len(by_region),
        "score_source": scored[0]["score_source"] if scored else "none",
        "methodology": {
            "claim": "Reddit-stratified world panel, not planetary ground truth.",
            "panel": "curated regional + topic subs, equal seats, not r/popular",
            "unit": "hot titles + short selftext; no comment trees",
            "emotions": list(PRIMARY),
            "limitations": [
                "Reddit is English-heavy and urban-platform-skewed",
                "hot ranking is engagement, not population",
                "lexicon scoring misses sarcasm and code-switching",
            ],
        },
        "overall": overall,
        "by_region": by_region,
        "by_subreddit_top": dict(
            sorted(by_sub.items(), key=lambda kv: kv[1]["n"], reverse=True)[:25]
        ),
        "themes": [{"term": t, "n": n} for t, n in themes],
        "street_line": prophet_line(overall, themes, by_region),
    }
    return snap, scored


def persist(snap: dict, scored: list[dict], keep_raw: int = 8) -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    SNAP_DIR.mkdir(parents=True, exist_ok=True)

    path = SNAP_DIR / f"snapshot_{snap['file_stem']}.json"
    path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    LATEST.write_text(json.dumps(snap, indent=2), encoding="utf-8")

    # slim raw for the dashboard drill-down — not 7MB comment dumps
    slim = pd.DataFrame(scored)
    keep_cols = [
        c
        for c in [
            "harvested_at",
            "region",
            "subreddit",
            "title",
            "score",
            "num_comments",
            "created_utc",
            *PRIMARY,
            "polarity",
            "score_source",
        ]
        if c in slim.columns
    ]
    raw_path = DATA / f"reddit_{snap['file_stem']}.csv"
    slim[keep_cols].to_csv(raw_path, index=False, encoding="utf-8")

    row = {
        "timestamp_utc": snap["timestamp_utc"],
        "file_stem": snap["file_stem"],
        "n_posts": snap["n_posts"],
        "n_subreddits": snap["n_subreddits"],
        "score_source": snap["score_source"],
        "dominant": snap["overall"]["dominant"],
        "polarity": snap["overall"]["polarity"],
        **{e: snap["overall"]["mood"][e] for e in PRIMARY},
        "street_line": snap["street_line"],
    }
    if TIMESERIES.exists():
        ts = pd.read_csv(TIMESERIES)
        ts = pd.concat([ts, pd.DataFrame([row])], ignore_index=True)
    else:
        ts = pd.DataFrame([row])
    ts.to_csv(TIMESERIES, index=False, encoding="utf-8")

    _update_scroll(snap, raw_path)
    _prune_raw(keep_raw)
    return path


def _update_scroll(snap: dict, raw_path: Path) -> None:
    row = {
        "seq_id": None,
        "file_name": raw_path.name,
        "timestamp_utc": snap["timestamp_utc"].replace("T", " ").replace("Z", ""),
        "posts": snap["n_posts"],
        "subreddits": snap["n_subreddits"],
        "harvester_type": "v1.5-panel",
        "size_bytes": raw_path.stat().st_size if raw_path.exists() else 0,
    }
    if SCROLL.exists():
        scroll = pd.read_csv(SCROLL)
        if raw_path.name in set(scroll.get("file_name", pd.Series(dtype=str)).astype(str)):
            return
        row["seq_id"] = int(scroll["seq_id"].max()) + 1 if "seq_id" in scroll.columns and len(scroll) else 1
        scroll = pd.concat([scroll, pd.DataFrame([row])], ignore_index=True)
    else:
        row["seq_id"] = 1
        scroll = pd.DataFrame([row])
    scroll.to_csv(SCROLL, index=False, encoding="utf-8")


def _prune_raw(keep: int) -> None:
    files = sorted(DATA.glob("reddit_*.csv"))
    extra = files[:-keep] if len(files) > keep else []
    for f in extra:
        f.unlink(missing_ok=True)
