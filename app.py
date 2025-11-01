# ============================================================
# 🌀 MimicVerse v0.3 — World Pulse Edition 🌎
# The Reddit-Trained Chaos Oracle
# ============================================================

import streamlit as st
import pandas as pd
import markovify
import json
import glob
import random
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 🧠 LOAD COMBINED MEMORY
# ------------------------------------------------------------
st.set_page_config(page_title="MimicVerse 🌎", layout="wide")

data_files = sorted(glob.glob("data/reddit_*.csv"))
if data_files:
    df = pd.concat([pd.read_csv(f) for f in data_files], ignore_index=True)
    st.sidebar.success(f"🧠 Loaded {len(df):,} Reddit posts/comments from {len(data_files)} data files")
else:
    st.sidebar.warning("No data files found. Upload or generate Reddit CSVs first!")
    df = pd.DataFrame(columns=["title", "selftext"])

# ------------------------------------------------------------
# 🎨 HEADER
# ------------------------------------------------------------
st.title("🌀 MimicVerse — The Reddit-Trained Chaos Oracle")
st.caption("A self-learning mirror of humanity’s collective mood — powered by Reddit 🧠")

# ------------------------------------------------------------
# 💬 MIMIC ENGINE
# ------------------------------------------------------------
def train_markov_model(texts):
    text_blob = " ".join([str(t) for t in texts if isinstance(t, str)])
    return markovify.Text(text_blob)

if not df.empty:
    model = train_markov_model(df["title"].dropna().tolist() + df["selftext"].dropna().tolist())
else:
    model = None

st.subheader("💭 Ask the Internet Brain:")
prompt = st.text_input("What do you want to ask the collective consciousness?")
if prompt and model:
    with st.spinner("The hive mind is thinking..."):
        reply = model.make_sentence(tries=100)
        if not reply:
            reply = "Bro, Reddit’s silent on that one 💀"
    st.success(f"🤖 **MimicVerse:** {reply}")

# ------------------------------------------------------------
# 🌍 MOOD MIX OF THE WORLD
# ------------------------------------------------------------
st.markdown("---")
st.subheader("🌍 Mood Mix of the World")

if not df.empty:
    # quick sentiment proxy by keyword frequency
    emotions = {
        "Joy 😁": df["selftext"].str.count("love|happy|joy|fun").sum(),
        "Anger 😡": df["selftext"].str.count("hate|angry|rage|annoy").sum(),
        "Sadness 😢": df["selftext"].str.count("sad|cry|hurt|lonely").sum(),
        "Sarcasm 😏": df["selftext"].str.count("lol|lmao|smh|idk").sum(),
        "Lust 😳": df["selftext"].str.count("hot|sexy|crush|date").sum(),
        "Neutral 😐": len(df)
    }

    total = sum(emotions.values())
    if total == 0: total = 1
    sizes = [round((v/total)*100, 1) for v in emotions.values()]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=emotions.keys(), autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)
    st.caption("Sentiment distribution based on keyword analysis — updated automatically 🔄")
else:
    st.info("No mood data available yet. Fetch Reddit data first!")

# ------------------------------------------------------------
# 🧭 TREND TRACKER (TOP SUBREDDITS)
# ------------------------------------------------------------
st.markdown("---")
st.subheader("🧭 Trend Tracker — Top 500 Subreddits")

try:
    with open("data/metadata.json") as f:
        meta = json.load(f)
        date = meta.get("date", "unknown date")
        subs = meta.get("subreddits", [])
        st.write(f"**Top 500 Subreddits as on {date}:**")
        st.text_area("Source", "\n".join(subs), height=200)
except Exception:
    st.warning("⚠️ No metadata.json found yet. Run your harvester first!")

# ------------------------------------------------------------
# 🕰️ FOOTER / CREDITS
# ------------------------------------------------------------
st.markdown("---")
st.caption("✨ Built with ❤️ and chaos by Amlan Mishra | MimicVerse v0.3")
