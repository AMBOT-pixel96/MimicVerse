# ============================================================
# 🌌 MimicVerse Dashboard v0.1
# Streamlit UI for global Reddit sentiment + trend analysis
# ============================================================

import streamlit as st
import pandas as pd
import json
import os
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime

# ------------------------------------------------------------
# 🎨 Streamlit Page Setup
st.set_page_config(
    page_title="MimicVerse Dashboard",
    page_icon="🌙",
    layout="wide"
)

st.title("🌌 MimicVerse Dashboard")
st.caption("AI that interprets the global subconscious — one subreddit at a time.")

# ------------------------------------------------------------
# 🧭 Load Latest Dataset
DATA_DIR = "data"
csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])

if not csv_files:
    st.warning("No harvested data found yet. Wait for the next nightly run 🌙")
    st.stop()

latest_csv = os.path.join(DATA_DIR, csv_files[-1])
st.success(f"Loaded dataset: `{latest_csv}`")

df = pd.read_csv(latest_csv)

# ------------------------------------------------------------
# 💭 Sentiment Analysis
st.header("🧠 Mood Mix of the World 🌍")

def get_sentiment(text):
    if not isinstance(text, str) or text.strip() == "":
        return None
    blob = TextBlob(text)
    return blob.sentiment.polarity

df["sentiment"] = df["title"].apply(get_sentiment)
sentiment_score = df["sentiment"].dropna()

if len(sentiment_score) > 0:
    mood_mix = {
        "Positive": (sentiment_score > 0.1).sum(),
        "Neutral": ((sentiment_score >= -0.1) & (sentiment_score <= 0.1)).sum(),
        "Negative": (sentiment_score < -0.1).sum()
    }

    fig, ax = plt.subplots()
    ax.pie(
        mood_mix.values(),
        labels=mood_mix.keys(),
        autopct='%1.1f%%',
        startangle=90,
        shadow=True
    )
    st.pyplot(fig)
else:
    st.info("No sentiment data available in the current dataset.")

# ------------------------------------------------------------
# 🔤 Trending Words (WordCloud)
st.header("🔮 Trending Words")
all_text = " ".join(df["title"].dropna().tolist())

if len(all_text) > 100:
    wordcloud = WordCloud(width=800, height=400, background_color="black", colormap="viridis").generate(all_text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
else:
    st.info("Not enough text data for word cloud generation.")

# ------------------------------------------------------------
# 🕓 Activity Timeline
st.header("📅 Evolution Timeline")
df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
df["date"] = df["created_utc"].dt.date
timeline = df.groupby("date").size().reset_index(name="posts")

if not timeline.empty:
    st.line_chart(timeline, x="date", y="posts", use_container_width=True)
else:
    st.info("Timeline not available yet.")

# ------------------------------------------------------------
# 🌐 Source Subreddits
st.header("🌐 Source Tracker")
meta_file = os.path.join(DATA_DIR, "metadata.json")

if os.path.exists(meta_file):
    with open(meta_file, "r") as f:
        meta = json.load(f)
    st.write(f"**Top subreddits as of {meta['date']}**")
    st.dataframe(pd.DataFrame(meta["subreddits"], columns=["Subreddit"]))
else:
    st.info("Metadata not found.")

# ------------------------------------------------------------
# 🪩 Footer
st.markdown("---")
st.caption("🪄 Built by **ripped_geek** | Powered by the collective consciousness of Reddit")