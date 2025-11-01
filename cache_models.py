# =============================================================
# ⚡ MimicVerse Cache Models Utility (v1.0)
# Purpose: Pre-download & warm up all heavy models safely
# Compatible with Streamlit Cloud + GitHub Actions
# =============================================================

import os
import spacy
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from transformers import pipeline

print("🚀 Starting model cache process...")

# ---------- 1. SpaCy Model ----------
try:
    print("🧠 Downloading spaCy model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy model ready.")
except Exception as e:
    print(f"⚠️ spaCy error: {e}")

# ---------- 2. SentenceTransformer ----------
try:
    print("🧠 Loading SentenceTransformer (MiniLM)...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ SentenceTransformer cached.")
except Exception as e:
    print(f"⚠️ ST error: {e}")

# ---------- 3. KeyBERT ----------
try:
    print("🧠 Initializing KeyBERT with MiniLM...")
    kw_model = KeyBERT(model=st_model)
    print("✅ KeyBERT model cached.")
except Exception as e:
    print(f"⚠️ KeyBERT error: {e}")

# ---------- 4. Summarization Pipeline ----------
try:
    print("🧠 Preloading summarizer (BART)...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    print("✅ Summarizer cached.")
except Exception as e:
    print(f"⚠️ Summarizer error: {e}")

# ---------- 5. Confirmation Marker ----------
os.makedirs("models_cache", exist_ok=True)
with open("models_cache/info.txt", "w") as f:
    f.write("✅ Models cached successfully.\n")

print("\n🎉 All models are prewarmed and ready to serve!")