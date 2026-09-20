MimicVerse v1.5
Reddit-stratified world-panel mood, scored at harvest time.
This is not the mood of the planet. It is the mood of a fixed seat map of regional and topic subreddits, using hot titles. Quote it that way.
What changed from v1.4.1
Before
After
subreddits.popular(100)
Curated regional panel (South Asia, East Asia, Europe, Americas, MENA/Africa, topics)
7MB comment dumps in git
Slim title CSV + JSON snapshot (~KB)
Streamlit runs GoEmotions 250–500 times
Harvest writes data/latest_snapshot.json; app only reads
Softmax on a multi-label net
Optional GoEmotions uses sigmoid
DistilGPT-2 “prophet”
One line built from actual dominant emotion + themes + loudest region
App dies without HarvestScroll
Snapshot first, CSV fallback, scroll last
Logger path wrong in harvest workflow
python scripts/harvest_logger.py
Run
pip install -r requirements.txt
export REDDIT_CLIENT_ID=...
export REDDIT_CLIENT_SECRET=...
export REDDIT_USER_AGENT="MimicVerse/1.5 by u/ripped_geek"
python mimicverse_harvest.py --posts 20
streamlit run app.py
Transformer path (heavier, optional):
pip install transformers torch
python mimicverse_harvest.py --transformer
Files that matter
mimicverse/panel.py — the world seat map
mimicverse/harvest.py — PRAW, titles only
mimicverse/score.py — NRCLex default, optional batched GoEmotions
mimicverse/snapshot.py — JSON + timeseries + prune
data/latest_snapshot.json — what the dashboard reads
data/mood_timeseries.csv — harvest-to-harvest deltas
data/snapshots/ — history
Honesty rules
English-heavy. Urban platform skew. Engagement ranking ≠ population.
Lexicons miss sarcasm.
Do not title a screenshot “mood of Earth.” Title it “Reddit world panel, this window.”