# ============================================================
# 🧠 MimicVerse Data Harvester v0.3
# Pulls top posts + comments from trending subreddits
# ============================================================

import praw
import streamlit as st
import pandas as pd
import json
import datetime
import os

# ------------------------------------------------------------
# 🔐 Load Reddit credentials from Streamlit Secrets
reddit = praw.Reddit(
    client_id=st.secrets["reddit"]["client_id"],
    client_secret=st.secrets["reddit"]["client_secret"],
    user_agent=st.secrets["reddit"]["user_agent"]
)

# ------------------------------------------------------------
# 🗓 Setup paths & filenames
TODAY = datetime.date.today().strftime("%Y-%m-%d")
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

csv_name = f"{DATA_DIR}/reddit_{TODAY}.csv"
meta_name = f"{DATA_DIR}/metadata.json"

# ------------------------------------------------------------
# 🌎 Discover top 500 subreddits dynamically
top_subs = []
for sub in reddit.subreddits.popular(limit=500):
    top_subs.append(sub.display_name)

# ------------------------------------------------------------
# 🔍 For each subreddit → get 50 threads + 30 comments
data_rows = []
for sub_name in top_subs:
    subreddit = reddit.subreddit(sub_name)
    for submission in subreddit.hot(limit=50):
        title = submission.title
        selftext = submission.selftext or ""
        comments = []
        submission.comments.replace_more(limit=0)
        for c in submission.comments[:30]:
            if hasattr(c, "body"):
                comments.append(c.body)
        data_rows.append({
            "subreddit": sub_name,
            "title": title,
            "selftext": selftext,
            "comments": " || ".join(comments),
            "score": submission.score,
            "created_utc": submission.created_utc
        })

# ------------------------------------------------------------
# 💾 Save harvested data
df = pd.DataFrame(data_rows)
df.to_csv(csv_name, index=False)

# ------------------------------------------------------------
# 🧭 Update metadata
metadata = {
    "date": TODAY,
    "subreddits": top_subs
}
with open(meta_name, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Harvest complete — {len(df)} posts saved to {csv_name}")

# ------------------------------------------------------------
# 🧹 Optional: prune if >30 CSVs (90-day rotation)
files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
if len(files) > 30:
    os.remove(os.path.join(DATA_DIR, files[0]))
    print(f"🧹 Purged oldest dataset: {files[0]}")
