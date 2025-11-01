# ============================================================
# 🌌 MimicVerse Data Harvester v0.6 — Infinite-Day Mode (Stable)
# Handles multiple harvests/day + auto Streamlit redeploy triggers
# ============================================================

import os
import json
import datetime
import pandas as pd
import praw

print("🚀 Starting MimicVerse Harvester v0.6 (Infinite-Day Mode)")

# ------------------------------------------------------------
# 🔐 Load Reddit credentials (dual-source: Streamlit / GH Actions)
try:
    import streamlit as st
except ImportError:
    st = None

CLIENT_ID = CLIENT_SECRET = USER_AGENT = None

if st and hasattr(st, "secrets"):
    try:
        if "reddit" in st.secrets:
            CLIENT_ID = st.secrets["reddit"]["client_id"]
            CLIENT_SECRET = st.secrets["reddit"]["client_secret"]
            USER_AGENT = st.secrets["reddit"]["user_agent"]
            print("🔑 Using Streamlit secrets for credentials")
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
TODAY = datetime.date.today().strftime("%Y-%m-%d")
TIME_NOW = datetime.datetime.now().strftime("%H%M")  # adds unique timestamp
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

csv_name = f"{DATA_DIR}/reddit_{TODAY}_{TIME_NOW}.csv"
meta_name = f"{DATA_DIR}/metadata.json"
history_log = f"{DATA_DIR}/harvest_history.json"

# ------------------------------------------------------------
# 🌎 Discover top subreddits
top_subs = []
try:
    print("🌍 Discovering top subreddits...")
    for sub in reddit.subreddits.popular(limit=10):
        top_subs.append(sub.display_name)
    print(f"✅ Found {len(top_subs)} subreddits: {', '.join(top_subs[:5])} ...")
except Exception as e:
    print(f"⚠️ Could not fetch subreddit list: {e}")
    top_subs = ["AskReddit", "worldnews", "technology"]

# ------------------------------------------------------------
# 🔍 Harvest posts
data_rows = []
for sub_name in top_subs:
    try:
        subreddit = reddit.subreddit(sub_name)
        print(f"🔍 Harvesting: {sub_name}")
        for submission in subreddit.hot(limit=10):  # adjustable batch size
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
    print("⚠️ No posts found. Reddit may have rate-limited the bot or credentials are invalid.")
else:
    df = pd.DataFrame(data_rows)
    df.to_csv(csv_name, index=False, encoding="utf-8")
    print(f"✅ Harvest complete — {len(df)} posts saved to {csv_name}")

# ------------------------------------------------------------
# 🧭 Update metadata
metadata = {
    "date": TODAY,
    "time": TIME_NOW,
    "subreddits": top_subs,
    "records": len(data_rows)
}
with open(meta_name, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print("🧭 Metadata updated successfully.")

# ------------------------------------------------------------
# 🧾 Append to harvest history log
history_entry = {
    "timestamp": f"{TODAY} {TIME_NOW}",
    "file": os.path.basename(csv_name),
    "posts": len(data_rows)
}
history = []
if os.path.exists(history_log):
    try:
        with open(history_log, "r", encoding="utf-8") as f:
            history = json.load(f)
    except json.JSONDecodeError:
        history = []
history.append(history_entry)
with open(history_log, "w", encoding="utf-8") as f:
    json.dump(history[-50:], f, indent=2)  # keep last 50 runs
print("📊 Harvest history updated.")

# ------------------------------------------------------------
# 🧹 Optional cleanup (keep 30 newest CSVs)
files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith("reddit_") and f.endswith(".csv")])
if len(files) > 30:
    os.remove(os.path.join(DATA_DIR, files[0]))
    print(f"🧹 Purged oldest dataset: {files[0]}")

print("🌙 Harvester complete — data safely stored and ready for MimicVerse.")