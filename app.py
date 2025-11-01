# ============================================================
# 🌌 MimicVerse v1.0 — Global Reddit Mood Dashboard (Patched)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import json, os, random
import altair as alt
from datetime import datetime
from textblob import TextBlob
import text2emotion as te
from wordcloud import WordCloud
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF
import markovify
from collections import Counter

# ============================================================
# 🧩 NLTK Punkt Safety Patch (for Streamlit Cloud)
import nltk

NLTK_DIR = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(NLTK_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DIR)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=NLTK_DIR)

# ============================================================
# 🧭 Page Config
st.set_page_config(
    page_title="🌌 MimicVerse",
    page_icon="🧠",
    layout="wide"
)
st.title("🌌 **MimicVerse – The Global Reddit Mood Dashboard**")
st.caption("AI that learns from the world’s collective chatter – one thread at a time 🌀")

# ============================================================
# 🧩 Load Latest Dataset
DATA_DIR = "data"
files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")], reverse=True)

if not files:
    st.warning("⚠️ No dataset found yet. Wait for the nightly harvester to run!")
    st.stop()

latest_csv = os.path.join(DATA_DIR, files[0])
meta_file = os.path.join(DATA_DIR, "metadata.json")

df = pd.read_csv(latest_csv)
meta = json.load(open(meta_file)) if os.path.exists(meta_file) else {}

st.sidebar.header("🗓️ Data Overview")
st.sidebar.write(f"**Dataset:** {os.path.basename(latest_csv)}")
st.sidebar.write(f"**Posts:** {len(df):,}")
st.sidebar.write(f"**Subreddits:** {len(df['subreddit'].unique())}")
st.sidebar.write(f"**Harvested:** {meta.get('date', datetime.now().strftime('%Y-%m-%d'))}")

# ============================================================
# 🧠 Sentiment + Emotion Analysis
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

st.markdown("### 🧭 Mood Mix of the World 🌍")
df["text"] = df["title"].fillna('') + " " + df["selftext"].fillna('')
df["sentiment"] = df["text"].apply(get_sentiment)

emotions = {"Happy": 0, "Angry": 0, "Surprise": 0, "Sad": 0, "Fear": 0}
for t in df["text"].sample(min(100, len(df))):  # sample for speed
    try:
        emo = te.get_emotion(t)
        for k in emo:
            emotions[k] += emo[k]
    except Exception:
        continue

# Pie chart
emo_data = pd.DataFrame({"Emotion": emotions.keys(), "Value": emotions.values()})
chart = alt.Chart(emo_data).mark_arc(innerRadius=50).encode(
    theta="Value",
    color="Emotion",
    tooltip=["Emotion", "Value"]
)
st.altair_chart(chart, use_container_width=True)

# ============================================================
# 📈 Trend Pulse (Top Emerging Keywords)
st.markdown("### 📈 Trend Pulse")
kw_model = KeyBERT(model='all-MiniLM-L6-v2')
docs = df["text"].dropna().tolist()
keywords = []
for text in random.sample(docs, min(50, len(docs))):
    try:
        kws = kw_model.extract_keywords(text, top_n=3)
        keywords.extend([k[0] for k in kws])
    except Exception:
        pass
freq = Counter(keywords)
top_kw = pd.DataFrame(freq.most_common(10), columns=["Keyword", "Frequency"])
st.bar_chart(top_kw.set_index("Keyword"))

# ============================================================
# 💬 Word on the Street (Markov chain summary)
st.markdown("### 💬 Word on the Street")
joined = ". ".join(df["title"].dropna().tolist()[:500])
try:
    text_model = markovify.Text(joined)
    st.info(f"🗣️ *“{text_model.make_sentence() or 'Feels like everyone’s tired but still pretending to hustle.'}”*")
except Exception:
    st.info("🗣️ *No quote generated this time (insufficient data)*")

# ============================================================
# 🔥 Emotional Index by Subreddit
st.markdown("### 🔥 Emotional Index (Sentiment by Subreddit)")
sent_df = df.groupby("subreddit")["sentiment"].mean().sort_values(ascending=False).head(10)
st.bar_chart(sent_df)

# ============================================================
# 🧩 Meme Cluster (Language Tone Groups - simplified demo)
st.markdown("### 🧩 Meme Cluster")
vectorizer = CountVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(df["text"].fillna(''))
nmf = NMF(n_components=3, random_state=42).fit(X)
top_words = np.argsort(nmf.components_, axis=1)[:, -10:]
terms = np.array(vectorizer.get_feature_names_out())
clusters = {f"Cluster {i+1}": ", ".join(terms[tw]) for i, tw in enumerate(top_words)}
st.json(clusters)

# ============================================================
# ☁️ Word Cloud
st.markdown("### ☁️ Global Word Cloud")
all_text = " ".join(df["text"].astype(str).tolist())
wordcloud = WordCloud(width=1200, height=400, background_color="black", colormap="plasma").generate(all_text)
st.image(wordcloud.to_array(), use_container_width=True)

# ============================================================
# 📦 Footer
st.markdown("---")
st.caption("© 2025 MimicVerse | Built by Amlan Mishra 🧠 | Reddit data harvested via PRAW | Visualization powered by Streamlit + Altair + NLP Models")