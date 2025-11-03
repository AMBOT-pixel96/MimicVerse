# ============================================================
# 🌌 MimicVerse v1.4.0 — The Global Reddit Mood Dashboard (Oracle Engine Awakens)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import json, os, random, zipfile, requests, re
import altair as alt
from datetime import datetime
from pathlib import Path
from collections import Counter

# ============================================================
# ☁️ Model Setup
# ============================================================

MODEL_ZIP_URL = "https://github.com/AMBOT-pixel96/MimicVerse/releases/download/v1.2-model/goemotions_model.zip"
MODEL_DIR = Path("models/goemotions_model")
MODEL_ZIP_PATH = Path("models/goemotions_model.zip")
MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)

@st.cache_resource(show_spinner="📦 Preparing GoEmotions model...")
def prepare_goemotions_model():
    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        return str(MODEL_DIR)
    with requests.get(MODEL_ZIP_URL, stream=True) as r:
        r.raise_for_status()
        with open(MODEL_ZIP_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=32768):
                f.write(chunk)
    with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(MODEL_DIR)
    os.remove(MODEL_ZIP_PATH)
    return str(MODEL_DIR)

prepare_goemotions_model()

# ============================================================
# 🧠 NLTK + TextBlob
# ============================================================

import nltk
from textblob import download_corpora
NLTK_DIR = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(NLTK_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DIR)
for pkg in ["punkt", "wordnet", "omw-1.4"]:
    try: nltk.data.find(f"tokenizers/{pkg}")
    except LookupError: nltk.download(pkg, download_dir=NLTK_DIR)
try: download_corpora.download_all()
except: pass

# ============================================================
# ⚙️ Imports
# ============================================================

from textblob import TextBlob
from nrclex import NRCLex
from wordcloud import WordCloud
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF
import markovify
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import torch.nn.functional as F

# ============================================================
# 🧠 Load Model
# ============================================================

@st.cache_resource
def load_goemotions():
    base_dir = Path("models/goemotions_model")
    for folder in [base_dir] + [f for f in base_dir.iterdir() if f.is_dir()]:
        if (folder / "config.json").exists():
            tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(str(folder), local_files_only=True)
            return tokenizer, model
    raise FileNotFoundError("GoEmotions model missing")

tokenizer, model = load_goemotions()

# ============================================================
# 🔮 Oracle Engine — DistilGPT-2 Integration (Makarov Reborn)
# ============================================================

@st.cache_resource(show_spinner="🧠 Summoning Oracle Engine...")
def summon_oracle():
    try:
        return pipeline("text-generation", model="distilgpt2")
    except Exception as e:
        st.warning(f"⚠️ Could not load DistilGPT-2 ({e}), fallback to Makarov.")
        return None

oracle_engine = summon_oracle()

def makarov_oracle_quote(context="human emotion"):
    """Generates a Reddit-style witty one-liner.
       Uses DistilGPT-2 if available, else falls back to Markovify chaos."""
    try:
        if oracle_engine:
            prompt = f"Write a short witty Reddit-style quote about {context}:"
            result = oracle_engine(prompt, max_length=40, do_sample=True,
                                   top_p=0.9, temperature=0.8)
            return result[0]["generated_text"]
        else:
            joined = ". ".join(df["title"].dropna().tolist()[:400])
            text_model = markovify.Text(joined)
            quote = text_model.make_sentence()
            return quote or "The world mumbles truths between memes and midnight scrolls."
    except Exception:
        return "Even silence has a punchline somewhere in the algorithm."

# ============================================================
# 🎛️ Page Config
# ============================================================

st.set_page_config(page_title="🌌 MimicVerse", page_icon="🧠", layout="wide")
st.title("🌌 **MimicVerse – The Global Reddit Mood Dashboard**")
st.caption("AI that listens to humanity's collective chatter and translates it into emotion ⚡")

# ============================================================
# 🧾 Load Harvest Scroll + Detect Latest Files
# ============================================================

DATA_DIR = "data"
scroll_path = os.path.join(DATA_DIR, "HarvestScroll.csv")

if not os.path.exists(scroll_path):
    st.error("⚠️ No Harvest Scroll found. Run the harvester first.")
    st.stop()

scroll = pd.read_csv(scroll_path)
scroll = scroll.sort_values(by="timestamp_utc", ascending=True).reset_index(drop=True)

def extract_timestamp(filename):
    match = re.search(r"reddit_(\d{4}-\d{2}-\d{2})_(\d{4})\.csv", str(filename))
    if match:
        date_str, time_str = match.groups()
        return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H%M")
    return datetime.min

csv_files = [f for f in os.listdir(DATA_DIR) if f.startswith("reddit_") and f.endswith(".csv")]
csv_files = sorted(csv_files, key=extract_timestamp, reverse=True)

if not csv_files:
    st.error("⚠️ No reddit CSVs found in data/")
    st.stop()

latest_file = csv_files[0]
previous_file = csv_files[1] if len(csv_files) > 1 else None
latest_csv = os.path.join(DATA_DIR, latest_file)
prev_csv = os.path.join(DATA_DIR, previous_file) if previous_file else None

df = pd.read_csv(latest_csv)

st.sidebar.header("📜 Harvest Scroll")
st.sidebar.markdown(f"**Latest Harvest:** `{latest_file}`")
st.sidebar.write(f"**Total Harvests Logged:** {len(csv_files)}")
st.sidebar.write(f"**Posts:** {len(df):,}")
st.sidebar.write(f"**Subreddits:** {len(df['subreddit'].unique())}")

if previous_file:
    st.sidebar.info(f"📊 Comparing with previous harvest: `{previous_file}`")
else:
    st.sidebar.warning("⚙️ Waiting for at least two harvests to compute delta map.")

# ============================================================
# 🧠 Emotion Analyzer
# ============================================================

go_labels = [
    'admiration','amusement','anger','annoyance','approval','caring','confusion',
    'curiosity','desire','disappointment','disapproval','disgust','embarrassment',
    'excitement','fear','gratitude','grief','joy','love','nervousness','optimism',
    'pride','realization','relief','remorse','sadness','surprise','neutral'
]

def analyze_emotion(text):
    text = str(text).strip()
    if not text:
        return {k: 0 for k in ['joy','anger','fear','sadness','surprise']}
    nrc = NRCLex(text)
    lex_map = {'joy':'joy','positive':'joy','anger':'anger','disgust':'anger','fear':'fear',
               'sadness':'sadness','negative':'sadness','surprise':'surprise'}
    base = {v:0 for v in ['joy','anger','fear','sadness','surprise']}
    for e, val in nrc.raw_emotion_scores.items():
        if e in lex_map: base[lex_map[e]] += val

    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    ge_dict = {go_labels[i]: float(probs[i]) for i in range(len(go_labels))}
    collapse_map = {
        'joy': ['joy','amusement','excitement','optimism','love','relief','gratitude','pride'],
        'anger': ['anger','annoyance','disapproval','disgust'],
        'fear': ['fear','nervousness'],
        'sadness': ['sadness','grief','remorse','disappointment'],
        'surprise': ['surprise','realization','curiosity']
    }
    ge_reduced = {k: sum(ge_dict.get(e,0) for e in v) for k,v in collapse_map.items()}
    return {k: 0.3 * base[k] + 0.7 * ge_reduced.get(k,0) for k in base}

# ============================================================
# 🌍 Mood Mix
# ============================================================

st.markdown("### 🧭 Mood Mix of the World 🌍")
emotions = {k: 0 for k in ['joy','anger','fear','sadness','surprise']}
texts = df["title"].fillna('').tolist()[:250]
progress = st.progress(0)
for i, t in enumerate(texts):
    emo = analyze_emotion(t)
    for k in emotions: emotions[k] += emo[k]
    progress.progress((i + 1) / len(texts))
progress.empty()

total = sum(emotions.values()) or 1
emo_data = pd.DataFrame({"Emotion": emotions.keys(), "Value": [round(100*v/total,2) for v in emotions.values()]})
chart = alt.Chart(emo_data).mark_arc(innerRadius=60).encode(theta="Value", color="Emotion", tooltip=["Emotion","Value"])
st.altair_chart(chart, use_container_width=True)
st.dataframe(emo_data)

# ============================================================
# 🌈 Mood Delta Map
# ============================================================

if previous_file and os.path.exists(prev_csv):
    df_prev = pd.read_csv(prev_csv)

    def mood_snapshot(df):
        emos = {k:0 for k in ['joy','anger','fear','sadness','surprise']}
        for t in df["title"].fillna('').tolist()[:250]:
            emo = analyze_emotion(t)
            for k in emos: emos[k] += emo[k]
        total = sum(emos.values()) or 1
        return {k: emos[k]/total for k in emos}

    m_latest = mood_snapshot(df)
    m_prev = mood_snapshot(df_prev)
    delta = {k: round(100*(m_latest[k]-m_prev[k]),2) for k in m_latest}

    st.markdown("### 🌈 Mood Delta Map (Latest vs Previous Harvest)")
    delta_df = pd.DataFrame({"Emotion": delta.keys(), "Change (%)": delta.values()})
    chart = alt.Chart(delta_df).mark_bar().encode(
        x="Emotion",
        y="Change (%)",
        color=alt.condition(alt.datum["Change (%)"] > 0, alt.value("green"), alt.value("red")),
        tooltip=["Emotion", "Change (%)"]
    )
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(delta_df)
else:
    st.info("Waiting for a previous harvest to compute the delta map.")

# ============================================================
# 💬 Word on the Street — Oracle Mode
# ============================================================

st.markdown("### 💬 Word on the Street")
dominant_emotion = max(emotions, key=emotions.get)
quote = makarov_oracle_quote(dominant_emotion)
st.info(f"🗣️ *“{quote.strip()}”*")

# ============================================================
# 📈 Trend Pulse / Word Cloud / Index
# ============================================================

st.markdown("### 📈 Trend Pulse")
kw_model = KeyBERT(model='all-MiniLM-L6-v2')
docs = df["title"].dropna().tolist()
keywords = []
for text in random.sample(docs, min(75, len(docs))):
    try: keywords.extend([k[0] for k in kw_model.extract_keywords(text, top_n=3)])
    except: pass
freq = Counter(keywords)
st.bar_chart(pd.DataFrame(freq.most_common(10), columns=["Keyword","Frequency"]).set_index("Keyword"))

st.markdown("### 🔥 Emotional Index by Subreddit")
df["sentiment"] = df["title"].fillna('').apply(lambda x: TextBlob(x).sentiment.polarity)
st.bar_chart(df.groupby("subreddit")["sentiment"].mean().sort_values(ascending=False).head(10))

st.markdown("### ☁️ Global Word Cloud")
wc = WordCloud(width=1200, height=400, background_color="black", colormap="inferno").generate(" ".join(df["title"].astype(str)))
st.image(wc.to_array(), use_container_width=True)

# ============================================================
# 📦 Footer
# ============================================================

st.markdown("---")
st.caption("© 2025 MimicVerse | Built by [Amlan Mishra 🧠](https://www.reddit.com/u/ripped_geek/s/DCuDNlO8Lk) | Global Mood Engine v1.4.0 (Oracle Engine Awakens)")