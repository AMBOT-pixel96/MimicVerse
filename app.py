# MimicVerse v1.5 — dashboard over precomputed snapshots.
# No live 250-pass transformer loop. If there is no snapshot yet,
# fall back to the latest slim reddit_*.csv and score lexicon-only.

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from mimicverse.score import PRIMARY, prophet_line, score_records, top_themes, aggregate

DATA = Path("data")
LATEST = DATA / "latest_snapshot.json"
SNAP_DIR = DATA / "snapshots"
TIMESERIES = DATA / "mood_timeseries.csv"

st.set_page_config(page_title="MimicVerse", page_icon="🌍", layout="wide")
st.title("🌍 MimicVerse — world-panel mood")
st.caption(
    "Stratified Reddit panel scored at harvest time. "
    "This is **not** planetary ground truth — it is the street as Reddit ranks it."
)


def load_snapshot() -> dict | None:
    if LATEST.exists():
        return json.loads(LATEST.read_text(encoding="utf-8"))
    files = sorted(SNAP_DIR.glob("snapshot_*.json")) if SNAP_DIR.exists() else []
    if files:
        return json.loads(files[-1].read_text(encoding="utf-8"))
    return None


def fallback_from_csv() -> dict | None:
    csvs = sorted(DATA.glob("reddit_*.csv"))
    if not csvs:
        return None
    df = pd.read_csv(csvs[-1])
    if "title" not in df.columns:
        return None
    records = df.to_dict(orient="records")
    # lexicon only — dashboard must stay cheap
    scored = score_records(records[:400], use_transformer=False)
    overall = aggregate(scored)["all"]
    by_region = aggregate(scored, key="region") if "region" in df.columns else {}
    themes = top_themes([r.get("title") or "" for r in scored])
    return {
        "timestamp_utc": csvs[-1].stem,
        "n_posts": len(scored),
        "n_subreddits": df["subreddit"].nunique() if "subreddit" in df.columns else None,
        "n_regions": len(by_region),
        "score_source": "nrclex-fallback",
        "methodology": {"claim": "Fallback score over latest CSV. Run v1.5 harvester for real snapshots."},
        "overall": overall,
        "by_region": by_region,
        "by_subreddit_top": {},
        "themes": [{"term": t, "n": n} for t, n in themes],
        "street_line": prophet_line(overall, themes, by_region),
        "file_stem": csvs[-1].stem,
    }


snap = load_snapshot() or fallback_from_csv()
if not snap:
    st.error("No snapshot and no reddit_*.csv in data/. Run `python mimicverse_harvest.py`.")
    st.stop()

mood = snap.get("overall", {}).get("mood") or {k: 0 for k in PRIMARY}

left, mid, right = st.columns(3)
left.metric("Posts scored", snap.get("n_posts", "—"))
mid.metric("Subs / regions", f"{snap.get('n_subreddits', '—')} / {snap.get('n_regions', '—')}")
right.metric("Dominant", snap.get("overall", {}).get("dominant", "—"))

st.info(f"🗣️ {snap.get('street_line', '')}")
st.caption(
    f"Harvest `{snap.get('timestamp_utc', '')}` · source `{snap.get('score_source', '')}` · v{snap.get('version', '?')}"
)

st.markdown("### Mood mix")
emo_df = pd.DataFrame(
    {"Emotion": list(mood.keys()), "Share": [round(100 * float(v), 2) for v in mood.values()]}
)
chart = (
    alt.Chart(emo_df)
    .mark_arc(innerRadius=70)
    .encode(theta="Share", color="Emotion", tooltip=["Emotion", "Share"])
)
st.altair_chart(chart, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.markdown("### By region")
    regional = snap.get("by_region") or {}
    if regional:
        rows = []
        for region, payload in regional.items():
            rec = {"region": region, "n": payload.get("n", 0), "polarity": payload.get("polarity", 0)}
            rec.update({k: payload.get("mood", {}).get(k, 0) for k in PRIMARY})
            rows.append(rec)
        rdf = pd.DataFrame(rows).sort_values("n", ascending=False)
        st.dataframe(rdf, use_container_width=True, hide_index=True)
        long = rdf.melt(id_vars=["region"], value_vars=list(PRIMARY), var_name="emotion", value_name="share")
        st.altair_chart(
            alt.Chart(long)
            .mark_bar()
            .encode(x="region:N", y="share:Q", color="emotion:N", tooltip=["region", "emotion", "share"])
            .properties(height=280),
            use_container_width=True,
        )
    else:
        st.write("No regional split in this snapshot (old harvest format).")

with c2:
    st.markdown("### Themes")
    themes = snap.get("themes") or []
    if themes:
        tdf = pd.DataFrame(themes).rename(columns={"term": "theme", "n": "count"})
        st.bar_chart(tdf.set_index("theme"))
    else:
        st.write("No themes extracted.")

st.markdown("### Timeseries")
if TIMESERIES.exists():
    ts = pd.read_csv(TIMESERIES)
    if "timestamp_utc" in ts.columns:
        keep = [c for c in ["timestamp_utc", *PRIMARY, "polarity"] if c in ts.columns]
        plot = ts[keep].copy()
        plot["timestamp_utc"] = pd.to_datetime(plot["timestamp_utc"], errors="coerce")
        long = plot.melt(id_vars=["timestamp_utc"], var_name="series", value_name="value")
        st.altair_chart(
            alt.Chart(long)
            .mark_line(point=True)
            .encode(x="timestamp_utc:T", y="value:Q", color="series:N", tooltip=["timestamp_utc", "series", "value"])
            .properties(height=280),
            use_container_width=True,
        )
        if len(ts) >= 2 and all(e in ts.columns for e in PRIMARY):
            delta = {e: round(100 * (float(ts.iloc[-1][e]) - float(ts.iloc[-2][e])), 2) for e in PRIMARY}
            st.markdown("#### Delta vs previous harvest")
            ddf = pd.DataFrame({"Emotion": list(delta), "Change (pp)": list(delta.values())})
            st.altair_chart(
                alt.Chart(ddf)
                .mark_bar()
                .encode(
                    x="Emotion",
                    y="Change (pp)",
                    color=alt.condition(alt.datum["Change (pp)"] > 0, alt.value("#2ca02c"), alt.value("#d62728")),
                    tooltip=["Emotion", "Change (pp)"],
                ),
                use_container_width=True,
            )
else:
    st.write("Timeseries appears after the first v1.5 harvest.")

with st.expander("Methodology — read this before quoting the dashboard"):
    meth = snap.get("methodology") or {}
    st.write(meth.get("claim", ""))
    for key in ("panel", "unit", "emotions"):
        if key in meth:
            st.write(f"**{key}:** {meth[key]}")
    for line in meth.get("limitations", []):
        st.write(f"- {line}")

st.caption("MimicVerse v1.5 · built for Amlan · harvest scores the world, the app only reads.")
