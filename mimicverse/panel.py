"""Stratified Reddit panel.

Popular() is the US/meme front page. A world-mood claim needs
equal-ish regional seats, not whatever Reddit is boosting today.
"""

from __future__ import annotations

# region -> subreddits. Keep lists short so one harvest finishes
# inside a GitHub Actions job without comment-tree explosion.
WORLD_PANEL: dict[str, list[str]] = {
    "global_news": [
        "worldnews",
        "news",
        "geopolitics",
        "internationalpolitics",
        "economics",
        "climate",
        "environment",
    ],
    "south_asia": [
        "india",
        "pakistan",
        "bangladesh",
        "nepal",
        "srilanka",
        "indiaSpeaks",
    ],
    "east_asia": [
        "japan",
        "korea",
        "china",
        "hongkong",
        "taiwan",
        "singapore",
    ],
    "se_asia_oceania": [
        "philippines",
        "indonesia",
        "malaysia",
        "australia",
        "newzealand",
        "pacific",
    ],
    "middle_east_africa": [
        "MiddleEast",
        "iran",
        "israel",
        "lebanon",
        "nigeria",
        "southafrica",
        "Africa",
        "egypt",
    ],
    "europe": [
        "europe",
        "unitedkingdom",
        "france",
        "germany",
        "italy",
        "spain",
        "ukraine",
        "russia",
        "poland",
        "nordiccountries",
    ],
    "americas": [
        "canada",
        "mexico",
        "brazil",
        "argentina",
        "chile",
        "colombia",
        "usa",
        "nyc",
    ],
    "street_topics": [
        "technology",
        "science",
        "soccer",
        "sports",
        "movies",
        "music",
        "AskReddit",
    ],
}

# Caps that keep Actions under ~15–25 min with title-only harvest.
POSTS_PER_SUB = 20
MAX_TITLE_CHARS = 300
MAX_SELFTEXT_CHARS = 400


def flatten_panel() -> list[tuple[str, str]]:
    """Return (region, subreddit) pairs, de-duplicated."""
    seen: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for region, subs in WORLD_PANEL.items():
        for sub in subs:
            key = sub.lower()
            if key in seen:
                continue
            seen.add(key)
            pairs.append((region, sub))
    return pairs


def region_for(subreddit: str) -> str:
    target = subreddit.lower()
    for region, subs in WORLD_PANEL.items():
        if any(s.lower() == target for s in subs):
            return region
    return "unmapped"
