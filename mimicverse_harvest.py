# ============================================================
# 🌌 MimicVerse Data Harvester v0.7 — Controlled Power-Up Edition
# ------------------------------------------------------------
# Collects emotional Reddit data across 100 subreddits, with
# safety throttling and timestamped filenames for multiple runs.
# ============================================================

import os
import json
import time
import random
import datetime
import pandas as pd
import praw

# Optional Streamlit import
try:
    import streamlit as st
except ImportError:
    st = None

print("🚀 Starting MimicVerse Harvester v0.7 (Controlled Power-Up Edition)")

# ------------------------------------------------------------
# 🔐 Load Reddit credentials (from Streamlit or environment)
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

if not all([CLIENT_ID, CLIENT_SECRET, USER_AGENT]):
    print("🔄 Falling back to GitHub Action environment secrets")
    CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
    CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
    USER_AGENT = os.getenv("REDDIT_USER_AGENT")

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
now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H%M")
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
csv_name = f"{DATA_DIR}/reddit_{timestamp}.csv"
meta_name = f"{DATA_DIR}/metadata.json"
history_name = f"{DATA_DIR}/harvest_history.json"

# ------------------------------------------------------------
# 🌍 Discover popular subreddits
top_subs = []
try:
    print("🌍 Discovering subreddits...")
    for sub in reddit.subreddits.popular(limit=100):  # Increased limit
        top_subs.append(sub.display_name)
    print(f"✅ Found {len(top_subs)} subreddits (sample): {top_subs[:10]}")
except Exception as e:
    print(f"⚠️ Could not fetch subreddit list: {e}")
    top_subs = ["AskReddit", "worldnews", "technology"]

# ------------------------------------------------------------
# 🔍 Fetch threads + comments with throttling
data_rows = []
for sub_name in top_subs:
    try:
        subreddit = reddit.subreddit(sub_name)
        print(f"🔍 Harvesting: {sub_name}")
        for submission in subreddit.hot(limit=25):  # Increased threads per subreddit
            if submission.stickied:
                continue
            submission.comments.replace_more(limit=0)
            comments = [c.body for c in submission.comments[:20] if hasattr(c, "body")]
            data_rows.append({
                "subreddit": sub_name,
                "title": submission.title,
                "selftext": submission.selftext or "",
                "comments": " || ".join(comments),
                "score": submission.score,
                "created_utc": submission.created_utc
            })
            # 🧠 Random sleep to dodge API rate limits
            time.sleep(random.uniform(0.5, 1.5))
        # brief cooldown between subreddits
        time.sleep(random.uniform(2, 4))
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
metadata = {
    "timestamp": timestamp,
    "subreddits": top_subs,
    "post_count": len(data_rows)
}
with open(meta_name, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print("🧭 Metadata updated.")

# ------------------------------------------------------------
# 📜 Update harvest history
history_entry = {
    "run_timestamp": timestamp,
    "subreddits": len(top_subs),
    "posts": len(data_rows)
}

try:
    if os.path.exists(history_name):
        with open(history_name, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = []
    history.append(history_entry)
    with open(history_name, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print("🗓 Harvest history updated.")
except Exception as e:
    print(f"⚠️ Could not update harvest history: {e}")

# ------------------------------------------------------------
# 🧹 Optional: prune old CSVs (keep last 30)
files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
if len(files) > 30:
    os.remove(os.path.join(DATA_DIR, files[0]))
    print(f"🧹 Purged oldest dataset: {files[0]}")

print("🌙 Harvester complete. Doug has fed well tonight.")