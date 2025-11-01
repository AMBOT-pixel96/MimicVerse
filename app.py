# ============================================================
# 🌌 MimicVerse v1.1 — Global Reddit Mood Dashboard (Overdrive)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import json, os, random
import altair as alt
from datetime import datetime
from textblob import TextBlob
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nrclex import NRCLex
from wordcloud import WordCloud
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF
import markovify
from collections import Counter

# ============================================================
# 🧭 Page Config
st.set_page_config(page_title="🌌 MimicVerse v1.1", page_icon="🧠", layout="wide")
st.title("🌌 **MimicVerse v1.1 – The Global Reddit Mood Dashboard**")
st.caption("AI that listens to humanity's collective chatter and translates it into emotion ⚡")

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
# 🧠 Hybrid Sentiment + Emotion Engine
def analyze_emotion(text):
    text = str(text).strip()
    if not text:
        return {"Happy": 0, "Angry": 0, "Fear": 0, "Sad": 0, "Surprise": 0}
    
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    # NRCLex adds word-based emotion tagging
    emotion_scores = NRCLex(text).raw_emotion_scores
    base = {"Happy": 0, "Angry": 0, "Fear": 0, "Sad": 0, "Surprise": 0}
    for e in emotion_scores:
        if e.lower() in ["joy", "positive"]: base["Happy"] += emotion_scores[e]
        elif e.lower() in ["anger", "disgust"]: base["Angry"] += emotion_scores[e]
        elif e.lower() in ["fear"]: base["Fear"] += emotion_scores[e]
        elif e.lower() in ["sadness", "negative"]: base["Sad"] += emotion_scores[e]
        elif e.lower() in ["anticipation", "surprise"]: base["Surprise"] += emotion_scores[e]

    # fallback: polarity-weighted tone adjustment
    if sum(base.values()) == 0:
        if polarity > 0.2: base["Happy"] = 1
        elif polarity < -0.2: base["Sad"] = 1
        else: base[random.choice(["Fear", "Angry", "Surprise"])] = 1

    return base

st.markdown("### 🧭 Mood Mix of the World 🌍")

emotions = {"Happy": 0, "Angry": 0, "Fear": 0, "Sad": 0, "Surprise": 0}
sample_texts = df["title"].fillna('').tolist()[:150]
for t in sample_texts:
    emo = analyze_emotion(t)
    for k in emotions:
        emotions[k] += emo.get(k, 0)

# Normalize to percentages
total = sum(emotions.values()) or 1
for k in emotions:
    emotions[k] = round(100 * emotions[k] / total, 2)

emo_data = pd.DataFrame({"Emotion": emotions.keys(), "Value": emotions.values()})
chart = alt.Chart(emo_data).mark_arc(innerRadius=60).encode(
    theta="Value",
    color="Emotion",
    tooltip=["Emotion", "Value"]
)
st.altair_chart(chart, use_container_width=True)
st.dataframe(emo_data)

# ============================================================
# 📈 Trend Pulse (Top Emerging Keywords)
st.markdown("### 📈 Trend Pulse")
kw_model = KeyBERT(model='all-MiniLM-L6-v2')
docs = df["title"].dropna().tolist()
keywords = []
for text in random.sample(docs, min(75, len(docs))):
    try:
        kws = kw_model.extract_keywords(text, top_n=3)
        keywords.extend([k[0] for k in kws])
    except:
        pass
freq = Counter(keywords)
top_kw = pd.DataFrame(freq.most_common(10), columns=["Keyword", "Frequency"])
st.bar_chart(top_kw.set_index("Keyword"))

# ============================================================
# 💬 Word on the Street (Markov quote)
st.markdown("### 💬 Word on the Street")
joined = ". ".join(df["title"].dropna().tolist()[:500])
try:
    text_model = markovify.Text(joined)
    quote = text_model.make_sentence()
    st.info(f"🗣️ *“{quote or 'The world mumbles truths between memes and midnight scrolls.'}”*")
except:
    st.info("🗣️ *Could not generate quote this time.*")

# ============================================================
# 🔥 Emotional Index by Subreddit
st.markdown("### 🔥 Emotional Index (Sentiment by Subreddit)")
df["sentiment"] = df["title"].fillna('').apply(lambda x: TextBlob(x).sentiment.polarity)
sent_df = df.groupby("subreddit")["sentiment"].mean().sort_values(ascending=False).head(10)
st.bar_chart(sent_df)

# ============================================================
# 🧩 Meme Cluster (Tone Groups)
st.markdown("### 🧩 Meme Cluster (Language Tone Groups)")
vectorizer = CountVectorizer(stop_words='english', max_features=800)
X = vectorizer.fit_transform(df["title"].fillna(''))
nmf = NMF(n_components=3, random_state=42).fit(X)
top_words = np.argsort(nmf.components_, axis=1)[:, -10:]
terms = np.array(vectorizer.get_feature_names_out())
clusters = {f"Cluster {i+1}": ", ".join(terms[tw]) for i, tw in enumerate(top_words)}
st.json(clusters)

# ============================================================
# ☁️ Word Cloud
st.markdown("### ☁️ Global Word Cloud")
all_text = " ".join(df["title"].astype(str).tolist())
wordcloud = WordCloud(width=1200, height=400, background_color="black", colormap="inferno").generate(all_text)
st.image(wordcloud.to_array(), use_container_width=True)

# ============================================================
# 📦 Footer
st.markdown("---")
st.caption("© 2025 MimicVerse | Built by Amlan Mishra 🧠 | Global Mood Engine v1.1 (Hybrid Emotion Core)")