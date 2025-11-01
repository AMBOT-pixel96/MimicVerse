# ============================================================
# 🌌 MimicVerse v1.1.3 — Global Reddit Mood Dashboard (Corpus Calming Update)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import json, os, random
import altair as alt
from datetime import datetime

# ============================================================
# 🧠 NLTK + TextBlob Global Patch (prevents MissingCorpusError)
import nltk
from textblob import download_corpora
from nltk.data import path as nltk_data_path

NLTK_DIR = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(NLTK_DIR, exist_ok=True)

# Make sure everyone (nltk + textblob + nrclex) uses the same directory
nltk.data.path.append(NLTK_DIR)
nltk_data_path.append(NLTK_DIR)
os.environ["NLTK_DATA"] = NLTK_DIR

# Download missing corpora safely
for pkg in ["punkt", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, download_dir=NLTK_DIR)

# Ensure TextBlob has all its corpora in the same place
# download_corpora.download_all() no longer takes arguments
os.environ["TEXTBLOB_DATA_DIR"] = NLTK_DIR
try:
    download_corpora.download_all()
except TypeError:
    # fallback if version mismatched
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# ============================================================
# 🧩 Main Imports (after patch)
from textblob import TextBlob
from nrclex import NRCLex
from wordcloud import WordCloud
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF
import markovify
from collections import Counter

# ============================================================
# 🧭 Page Config
st.set_page_config(page_title="🌌 MimicVerse v1.1.3", page_icon="🧠", layout="wide")
st.title("🌌 **MimicVerse v1.1.3 – The Global Reddit Mood Dashboard**")
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
# 🧠 Hybrid Emotion Engine (NRCLex + GoEmotions)
# ============================================================

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

@st.cache_resource
def load_goemotions():
    model_name = "bhadresh-savani/distilbert-base-uncased-go-emotions"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_goemotions()
go_labels = [
    'admiration','amusement','anger','annoyance','approval','caring',
    'confusion','curiosity','desire','disappointment','disapproval',
    'disgust','embarrassment','excitement','fear','gratitude','grief',
    'joy','love','nervousness','optimism','pride','realization',
    'relief','remorse','sadness','surprise','neutral'
]

def analyze_emotion(text):
    text = str(text).strip()
    if not text:
        return {k: 0 for k in ['joy','anger','fear','sadness','surprise']}

    # ---------- NRCLex (lexical) ----------
    nrc = NRCLex(text)
    nrc_scores = nrc.raw_emotion_scores

    lex_map = {'joy':'joy','positive':'joy','anger':'anger','disgust':'anger',
               'fear':'fear','sadness':'sadness','negative':'sadness','surprise':'surprise'}
    base = {v:0 for v in ['joy','anger','fear','sadness','surprise']}
    for e, val in nrc_scores.items():
        if e in lex_map:
            base[lex_map[e]] += val

    # ---------- GoEmotions (contextual) ----------
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    ge_dict = {go_labels[i]: float(probs[i]) for i in range(len(go_labels))}

    # Collapse GoEmotions 27 → 5 major categories
    collapse_map = {
        'joy': ['joy','amusement','excitement','optimism','love','relief','gratitude','pride'],
        'anger': ['anger','annoyance','disapproval','disgust'],
        'fear': ['fear','nervousness'],
        'sadness': ['sadness','grief','remorse','disappointment'],
        'surprise': ['surprise','realization','curiosity']
    }
    ge_reduced = {k: sum(ge_dict.get(e,0) for e in v) for k,v in collapse_map.items()}

    # ---------- Merge ----------
    final = {}
    for k in base:
        final[k] = 0.3 * base[k] + 0.7 * ge_reduced.get(k,0)

    return final

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
    except Exception:
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
except Exception:
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
st.caption("© 2025 MimicVerse | Built by Amlan Mishra 🧠 | Global Mood Engine v1.1.3 (Corpus Calming Update)")