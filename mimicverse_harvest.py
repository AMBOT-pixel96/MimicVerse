# ============================================================
# 🌌 MimicVerse Data Harvester v0.5 — Failsafe Hybrid
# Handles both Streamlit Cloud & GitHub Actions seamlessly
# ============================================================

import os
import json
import datetime
import pandas as pd
import praw

# Optional Streamlit import (not mandatory)
try:
    import streamlit as st
except ImportError:
    st = None

print("🚀 Starting MimicVerse Harvester v0.5 (Failsafe Mode)")

# ------------------------------------------------------------
# 🔐 Load Reddit credentials (with verification)
CLIENT_ID = CLIENT_SECRET = USER_AGENT = None

if st and hasattr(st, "secrets"):
    try:
        if "reddit" in st.secrets:
            CLIENT_ID = st.secrets["reddit"]["client_id"]
            CLIENT_SECRET = st.secrets["reddit"]["client_secret"]
            USER_AGENT = st.secrets["reddit"]["user_agent"]
            print("🔑 Using Streamlit Secrets for credentials")
    except Exception as e:
        print(f"⚠️ Streamlit secrets unavailable: {e}")

# Fall back to environment vars if Streamlit secrets missing
if not all([CLIENT_ID, CLIENT_SECRET, USER_AGENT]):
    print("🔄 Falling back to GitHub Action environment secrets")
    CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
    CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
    USER_AGENT = os.getenv("REDDIT_USER_AGENT")

# Final sanity check
if not all([CLIENT_ID, CLIENT_SECRET, USER_AGENT]):
    print("❌ Missing Reddit credentials. Exiting.")
    exit(1)

# ------------------------------------------------------------
# 🧠 Initialize Reddit API
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)
print("✅ Reddit authentication successful.")

# ------------------------------------------------------------
# 🗓 Setup paths & filenames
TODAY = datetime.date.today().strftime("%Y-%m-%d")
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
csv_name = f"{DATA_DIR}/reddit_{TODAY}.csv"
meta_name = f"{DATA_DIR}/metadata.json"

# ------------------------------------------------------------
# 🌎 Discover top 10 subreddits (debug scale for Actions)
top_subs = []
try:
    print("🌍 Discovering subreddits...")
    for sub in reddit.subreddits.popular(limit=10):
        top_subs.append(sub.display_name)
    print(f"✅ Found {len(top_subs)} subreddits: {top_subs[:5]}...")
except Exception as e:
    print(f"⚠️ Could not fetch subreddit list: {e}")
    top_subs = ["AskReddit", "worldnews", "technology"]

# ------------------------------------------------------------
# 🔍 Fetch threads + comments
data_rows = []
for sub_name in top_subs:
    try:
        subreddit = reddit.subreddit(sub_name)
        print(f"🔍 Harvesting: {sub_name}")
        for submission in subreddit.hot(limit=10):  # smaller limit for debug
            if submission.stickied:
                continue
            submission.comments.replace_more(limit=0)
            comments = [c.body for c in submission.comments[:5] if hasattr(c, "body")]
            data_rows.append({
                "subreddit": sub_name,
                "title": submission.title,
                "selftext": submission.selftext or "",
                "comments": " || ".join(comments),
                "score": submission.score,
                "created_utc": submission.created_utc
            })
    except Exception as e:
        print(f"⚠️ Error in {sub_name}: {e}")
        continue

# ------------------------------------------------------------
# 💾 Save harvested data
if not data_rows:
    print("⚠️ No posts found. Reddit may have rate-limited or credentials invalid.")
else:
    df = pd.DataFrame(data_rows)
    df.to_csv(csv_name, index=False, encoding="utf-8")
    print(f"✅ Harvest complete — {len(df)} posts saved to {csv_name}")

# ------------------------------------------------------------
# 🧭 Update metadata
metadata = {"date": TODAY, "subreddits": top_subs}
with open(meta_name, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print("🧭 Metadata updated.")

# ------------------------------------------------------------
# 🧹 Optional: prune old CSVs
files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
if len(files) > 30:
    os.remove(os.path.join(DATA_DIR, files[0]))
    print(f"🧹 Purged oldest dataset: {files[0]}")

print("🌙 Harvester complete.")