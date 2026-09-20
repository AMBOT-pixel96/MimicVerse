"""PRAW harvest over the stratified world panel. Titles + short selftext only."""

from __future__ import annotations

import os
import random
import time
from datetime import datetime, timezone

from .panel import POSTS_PER_SUB, MAX_SELFTEXT_CHARS, MAX_TITLE_CHARS, flatten_panel


def load_reddit_creds() -> tuple[str, str, str]:
    client_id = client_secret = user_agent = None
    try:
        import streamlit as st

        if hasattr(st, "secrets") and "reddit" in st.secrets:
            client_id = st.secrets["reddit"]["client_id"]
            client_secret = st.secrets["reddit"]["client_secret"]
            user_agent = st.secrets["reddit"]["user_agent"]
    except Exception:
        pass

    client_id = client_id or os.getenv("REDDIT_CLIENT_ID")
    client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = (
        os.getenv("REDDIT_USER_AGENT")
        or user_agent
        or "MimicVerse/1.5 by u/ripped_geek"
    )

    if not all([client_id, client_secret, user_agent]):
        raise SystemExit("Missing Reddit credentials (env or Streamlit secrets).")
    return client_id, client_secret, user_agent


def harvest(posts_per_sub: int = POSTS_PER_SUB, sleep: bool = True) -> list[dict]:
    import praw

    client_id, client_secret, user_agent = load_reddit_creds()
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

    rows: list[dict] = []
    harvested_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for region, sub_name in flatten_panel():
        try:
            subreddit = reddit.subreddit(sub_name)
            print(f"harvest {region}/{sub_name}")
            for submission in subreddit.hot(limit=posts_per_sub):
                if getattr(submission, "stickied", False):
                    continue
                title = (submission.title or "")[:MAX_TITLE_CHARS]
                if len(title.split()) < 3:
                    continue
                rows.append(
                    {
                        "harvested_at": harvested_at,
                        "region": region,
                        "subreddit": sub_name,
                        "title": title,
                        "selftext": (submission.selftext or "")[:MAX_SELFTEXT_CHARS],
                        "score": int(getattr(submission, "score", 0) or 0),
                        "num_comments": int(getattr(submission, "num_comments", 0) or 0),
                        "created_utc": float(getattr(submission, "created_utc", 0) or 0),
                        "permalink": str(getattr(submission, "permalink", "") or ""),
                    }
                )
                if sleep:
                    time.sleep(random.uniform(0.15, 0.4))
            if sleep:
                time.sleep(random.uniform(0.4, 0.9))
        except Exception as exc:
            print(f"skip {sub_name}: {exc}")
            continue

    print(f"harvest complete — {len(rows)} posts")
    return rows
