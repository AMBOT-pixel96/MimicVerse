# ============================================================
# 🌌 MimicVerse v1.3 — The Global Reddit Mood Dashboard
# (Harvest Scroll–Aware + Emotion Delta Map)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import json, os, random, zipfile, requests
import altair as alt
from datetime import datetime
from pathlib import Path
from collections import Counter

# ============================================================
# ☁️ GoEmotions Model Loader (Persistent Cached Download)
# ============================================================

MODEL_ZIP_URL = "https://github.com/AMBOT-pixel96/MimicVerse/releases/download/v1.2-model/goemotions_model.zip"
MODEL_DIR = Path("models/goemotions_model")
MODEL_ZIP_PATH = Path("models/goemotions_model.zip")
MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)

@st.cache_resource(show_spinner="📦 Downloading & caching GoEmotions model... ⏳")
def prepare_goemotions_model():
    """Download once, cache forever."""
    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        return str(MODEL_DIR)
    try:
        with requests.get(MODEL_ZIP_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_ZIP_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=32768):
                    f.write(chunk)
        with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(MODEL_DIR)
        os.remove(MODEL_ZIP_PATH)
        return str(MODEL_DIR)
    except Exception as e:
        st.error(f"❌ Failed to prepare GoEmotions model: {e}")
        st.stop()

prepare_goemotions_model()

# ============================================================
# 🧠 NLTK + TextBlob Global Patch
# ============================================================

import nltk
from textblob import download_corpora
from nltk.data import path as nltk_data_path

NLTK_DIR = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(NLTK_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DIR)
nltk_data_path.append(NLTK_DIR)
os.environ["NLTK_DATA"] = NLTK_DIR

for pkg in ["punkt", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, download_dir=NLTK_DIR)

os.environ["TEXTBLOB_DATA_DIR"] = NLTK_DIR
try:
    download_corpora.download_all()
except TypeError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# ============================================================
# 🧩 Main Imports
# ============================================================

from textblob import TextBlob
from nrclex import NRCLex
from wordcloud import WordCloud
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF
import markovify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# ============================================================
# 🧠 Load GoEmotions Model (Flat or Nested Safe)
# ============================================================

@st.cache_resource
def load_goemotions():
    base_dir = Path("models/goemotions_model")
    if (base_dir / "config.json").exists():
        model_path = base_dir
    else:
        candidates = list(base_dir.glob("**/config.json"))
        model_path = candidates[0].parent if candidates else None
    if not model_path:
        raise FileNotFoundError("⚠️ No valid GoEmotions model found.")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True)
    return tokenizer, model

tokenizer, model = load_goemotions()

# ============================================================
# 🧭 Page Config
# ============================================================

st.set_page_config(page_title="🌌 MimicVerse v1.3", page_icon="🧠", layout="wide")
st.title("🌌 **MimicVerse v1.3 – The Global Reddit Mood Dashboard**")
st.caption("AI that listens to humanity’s chatter and tracks emotional evolution ⚡")

# ============================================================
# 🧾 Harvest Scroll Awareness
# ============================================================

DATA_DIR = "data"
scroll_path = os.path.join(DATA_DIR, "HarvestScroll.csv")
latest_csv = prev_csv = None

if os.path.exists(scroll_path):
    scroll_df = pd.read_csv(scroll_path)
    if len(scroll_df) >= 1:
        latest_csv = os.path.join(DATA_DIR, scroll_df.iloc[-1]["file_name"])
    if len(scroll_df) >= 2:
        prev_csv = os.path.join(DATA_DIR, scroll_df.iloc[-2]["file_name"])

st.sidebar.header("🗓️ Harvest Scroll")
if latest_csv:
    st.sidebar.write(f"**Latest Harvest:** `{os.path.basename(latest_csv)}`")
else:
    st.sidebar.warning("⚠️ No harvest data found.")

if prev_csv:
    st.sidebar.write(f"**Previous Harvest:** `{os.path.basename(prev_csv)}`")
else:
    st.sidebar.info("Waiting for second harvest to compute delta map.")

# ============================================================
# 🧩 Load Latest Dataset
# ============================================================

if not latest_csv or not os.path.exists(latest_csv):
    st.warning("⚠️ Latest dataset not found. Run a harvest first.")
    st.stop()

df = pd.read_csv(latest_csv)
st.sidebar.write(f"**Posts:** {len(df):,}")
if "subreddit" in df.columns:
    st.sidebar.write(f"**Subreddits:** {len(df['subreddit'].unique())}")

# ============================================================
# 🧠 Hybrid Emotion Engine (NRCLex + GoEmotions)
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
    lex_map = {'joy':'joy','positive':'joy','anger':'anger','disgust':'anger',
               'fear':'fear','sadness':'sadness','negative':'sadness','surprise':'surprise'}
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
# 🧭 Mood Mix of the World 🌍
# ============================================================

st.markdown("### 🧭 Mood Mix of the World 🌍")
emotions = {k: 0 for k in ['joy','anger','fear','sadness','surprise']}
sample_texts = df["title"].fillna('').tolist()[:150]
progress = st.progress(0)
for i, t in enumerate(sample_texts):
    emo = analyze_emotion(t)
    for k in emotions:
        emotions[k] += emo.get(k, 0)
    progress.progress((i + 1) / len(sample_texts))
progress.empty()

total = sum(emotions.values()) or 1
for k in emotions:
    emotions[k] = round(100 * emotions[k] / total, 2)

emo_data = pd.DataFrame({"Emotion": emotions.keys(), "Value": emotions.values()})
chart = alt.Chart(emo_data).mark_arc(innerRadius=60).encode(
    theta="Value", color="Emotion", tooltip=["Emotion", "Value"]
)
st.altair_chart(chart, use_container_width=True)
st.dataframe(emo_data)

# ============================================================
# 🌈 Mood Delta Map — One File Ago Comparison
# ============================================================

st.markdown("### 🌈 Mood Delta Map – One File Ago Comparison")

if prev_csv and os.path.exists(prev_csv):
    try:
        df_prev = pd.read_csv(prev_csv)

        current_counts = df["subreddit"].value_counts(normalize=True) * 100
        prev_counts = df_prev["subreddit"].value_counts(normalize=True) * 100

        delta_df = pd.DataFrame({
            "subreddit": sorted(set(current_counts.index) | set(prev_counts.index))
        })
        delta_df["latest"] = delta_df["subreddit"].map(current_counts).fillna(0)
        delta_df["previous"] = delta_df["subreddit"].map(prev_counts).fillna(0)
        delta_df["delta"] = delta_df["latest"] - delta_df["previous"]

        chart = (
            alt.Chart(delta_df)
            .transform_fold(["previous", "latest"], as_=["Harvest", "Value"])
            .mark_area(opacity=0.8)
            .encode(
                x=alt.X("Harvest:N", title="Harvest Comparison"),
                y=alt.Y("Value:Q", title="Subreddit Representation (%)"),
                color=alt.Color("subreddit:N", legend=alt.Legend(title="Subreddit")),
                tooltip=["subreddit", "latest", "previous", "delta"]
            )
            .properties(title="🌈 Emotional Representation Shift", width=800, height=400)
        )
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(delta_df.sort_values("delta", ascending=False))
    except Exception as e:
        st.error(f"💥 Delta computation failed: {e}")
else:
    st.info("⏳ Waiting for multiple harvests to compute deltas.")

# ============================================================
# 📈 Trend Pulse
# ============================================================

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

# 💬 Word on the Street
st.markdown("### 💬 Word on the Street")
joined = ". ".join(df["title"].dropna().tolist()[:500])
try:
    text_model = markovify.Text(joined)
    quote = text_model.make_sentence()
    st.info(f"🗣️ *“{quote or 'The world mumbles truths between memes and midnight scrolls.'}”*")
except:
    st.info("🗣️ *Could not generate quote this time.*")

# 🔥 Emotional Index by Subreddit
st.markdown("### 🔥 Emotional Index (Sentiment by Subreddit)")
df["sentiment"] = df["title"].fillna('').apply(lambda x: TextBlob(x).sentiment.polarity)
sent_df = df.groupby("subreddit")["sentiment"].mean().sort_values(ascending=False).head(10)
st.bar_chart(sent_df)

# 🧩 Meme Cluster
st.markdown("### 🧩 Meme Cluster (Language Tone Groups)")
vectorizer = CountVectorizer(stop_words='english', max_features=800)
X = vectorizer.fit_transform(df["title"].fillna(''))
nmf = NMF(n_components=3, random_state=42).fit(X)
terms = np.array(vectorizer.get_feature_names_out())
clusters = {f"Cluster {i+1}": ", ".join(terms[np.argsort(nmf.components_[i])[-10:]]) for i in range(3)}
st.json(clusters)

# ☁️ Word Cloud
st.markdown("### ☁️ Global Word Cloud")
wordcloud = WordCloud(width=1200, height=400, background_color="black", colormap="inferno").generate(" ".join(df["title"].astype(str)))
st.image(wordcloud.to_array(), use_container_width=True)

# 📦 Footer
st.markdown("---")
st.caption("© 2025 MimicVerse | Built by Amlan Mishra 🧠 | Global Mood Engine v1.3 (Harvest Scroll Aware)")