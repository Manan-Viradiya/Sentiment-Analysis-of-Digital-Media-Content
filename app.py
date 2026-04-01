"""
app.py
------
Streamlit entry-point for "Sentiment Analysis of Digital Media Content".

Run with:
    streamlit run app.py
"""

import io
import json
import re
import string
from collections import Counter
from typing import List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sentiment_engine import predict_sentiment_with_scores
from youtube_handler import fetch_youtube_comments

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sentiment Analysis · Digital Media",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ---- fonts & base ---- */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=Space+Grotesk:wght@700&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    /* ---- header ---- */
    .app-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem 1.6rem;
        border-radius: 16px;
        margin-bottom: 1.8rem;
        color: #fff;
    }
    .app-header h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        margin: 0 0 .35rem;
        letter-spacing: -0.5px;
    }
    .app-header p { margin: 0; opacity: .75; font-size: .95rem; }

    /* ---- metric cards ---- */
    .metric-row { display: flex; gap: 1rem; margin-bottom: 1.6rem; }
    .metric-card {
        flex: 1;
        background: #1e1e2e;
        border: 1px solid #2e2e45;
        border-radius: 14px;
        padding: 1.1rem 1.4rem;
        color: #fff;
    }
    .metric-card .label { font-size: .78rem; opacity: .6; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card .value { font-size: 2rem; font-weight: 700; line-height: 1.2; margin-top: .25rem; }
    .metric-card .sub   { font-size: .8rem; opacity: .55; margin-top: .15rem; }

    /* ── sentiment badge colours ── */
    .badge-Positive { color: #4ade80; }
    .badge-Negative { color: #f87171; }
    .badge-Neutral  { color: #94a3b8; }

    /* ---- section titles ---- */
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 1.6rem 0 .8rem;
        letter-spacing: -.3px;
    }

    /* ---- sidebar ---- */
    section[data-testid="stSidebar"] { background: #13131f; }
    section[data-testid="stSidebar"] * { color: #cbd5e1; }

    /* ---- misc ---- */
    div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ──────────────────────────────────────────────────────────────────

SENTIMENT_COLORS = {
    "Positive": "#4ade80",
    "Negative": "#f87171",
    "Neutral":  "#94a3b8",
}

STOPWORDS = set(
    "i me my myself we our ours ourselves you your yours yourself yourselves "
    "he him his himself she her hers herself it its itself they them their "
    "theirs themselves what which who whom this that these those am is are "
    "was were be been being have has had having do does did doing a an the "
    "and but if or because as until while of at by for with about against "
    "between into through during before after above below to from up down "
    "in out on off over under again further then once here there when where "
    "why how all both each few more most other some such no nor not only own "
    "same so than too very s t can will just don should now d ll m o re ve "
    "y ain aren couldn didn doesn hadn hasn haven isn mightn mustn needn "
    "shan shouldn wasn weren won wouldn video youtube like comment just get "
    "one would could also even still really think know good great".split()
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def sanitise_text(text: str) -> str:
    """Lower-case, strip punctuation."""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def top_keywords(texts: List[str], n: int = 20) -> pd.DataFrame:
    """Return a DataFrame of the top-n keywords across all texts."""
    words = []
    for t in texts:
        words.extend(
            w for w in sanitise_text(t).split()
            if w not in STOPWORDS and len(w) > 2
        )
    counts = Counter(words).most_common(n)
    return pd.DataFrame(counts, columns=["Keyword", "Count"])


def load_file(uploaded_file) -> Optional[pd.DataFrame]:
    """Parse uploaded CSV or JSON into a DataFrame."""
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif name.endswith(".json"):
            raw = json.load(uploaded_file)
            df = pd.DataFrame(raw) if isinstance(raw, list) else pd.json_normalize(raw)
        else:
            st.error("Unsupported file type. Upload a .csv or .json file.")
            return None
    except Exception as exc:
        st.error(f"Could not parse file: {exc}")
        return None
    return df


def pick_text_column(df: pd.DataFrame) -> Optional[str]:
    """
    Heuristically find the column that contains comment/text data.
    Falls back to letting the user choose.
    """
    preferred = ["comment", "text", "content", "body", "message", "review", "tweet"]
    for col in df.columns:
        if col.lower() in preferred:
            return col
    # Let user decide
    return None


# ── Visualisation helpers ──────────────────────────────────────────────────────

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#cbd5e1", family="DM Sans"),
    margin=dict(t=40, b=20, l=20, r=20),
)


def pie_chart(df: pd.DataFrame) -> go.Figure:
    counts = df["Sentiment"].value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]
    fig = px.pie(
        counts,
        names="Sentiment",
        values="Count",
        color="Sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        hole=0.42,
        title="Sentiment Distribution",
    )
    fig.update_traces(textinfo="percent+label", pull=[0.04] * len(counts))
    fig.update_layout(**CHART_LAYOUT)
    return fig


def bar_chart(df: pd.DataFrame) -> go.Figure:
    counts = df["Sentiment"].value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]
    fig = px.bar(
        counts,
        x="Sentiment",
        y="Count",
        color="Sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        text="Count",
        title="Comment Count per Sentiment",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, **CHART_LAYOUT)
    return fig


def keyword_bar(kw_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        kw_df.sort_values("Count"),
        x="Count",
        y="Keyword",
        orientation="h",
        color="Count",
        color_continuous_scale=["#302b63", "#7c3aed", "#a78bfa"],
        title="Top Keywords",
    )
    fig.update_layout(coloraxis_showscale=False, **CHART_LAYOUT)
    return fig


def confidence_histogram(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x="Confidence",
        color="Sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        nbins=30,
        barmode="overlay",
        opacity=0.7,
        title="Confidence Score Distribution",
    )
    fig.update_layout(**CHART_LAYOUT)
    return fig


# ── Run pipeline ───────────────────────────────────────────────────────────────

def run_analysis(texts: List[str]) -> pd.DataFrame:
    """Call the sentiment engine and package results as a DataFrame."""
    with st.spinner("🤖  Running BERT inference …"):
        results = predict_sentiment_with_scores(texts)

    df = pd.DataFrame({
        "Comment":    texts,
        "Sentiment":  [r["label"] for r in results],
        "Confidence": [r["score"] for r in results],
    })
    return df


# ── Metric cards ───────────────────────────────────────────────────────────────

def render_metrics(df: pd.DataFrame, source: str):
    total     = len(df)
    majority  = df["Sentiment"].mode()[0]
    pos_pct   = round((df["Sentiment"] == "Positive").mean() * 100, 1)
    neg_pct   = round((df["Sentiment"] == "Negative").mean() * 100, 1)
    avg_conf  = round(df["Confidence"].mean() * 100, 1)

    badge_cls = f"badge-{majority}"
    st.markdown(
        f"""
        <div class="metric-row">
          <div class="metric-card">
            <div class="label">Total Comments</div>
            <div class="value">{total:,}</div>
            <div class="sub">Source: {source}</div>
          </div>
          <div class="metric-card">
            <div class="label">Majority Sentiment</div>
            <div class="value {badge_cls}">{majority}</div>
            <div class="sub">Most frequent label</div>
          </div>
          <div class="metric-card">
            <div class="label">Positive Rate</div>
            <div class="value badge-Positive">{pos_pct}%</div>
            <div class="sub">of all comments</div>
          </div>
          <div class="metric-card">
            <div class="label">Negative Rate</div>
            <div class="value badge-Negative">{neg_pct}%</div>
            <div class="sub">of all comments</div>
          </div>
          <div class="metric-card">
            <div class="label">Avg Confidence</div>
            <div class="value">{avg_conf}%</div>
            <div class="sub">model certainty</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────

def build_sidebar():
    st.sidebar.markdown("## ⚙️ Configuration")

    api_key = st.sidebar.text_input(
        "YouTube Data API Key",
        type="password",
        placeholder="AIza…",
        help="Required only for Live YouTube Analysis.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📥 Input Mode")

    mode = st.sidebar.radio(
        "Select input source",
        ["🎬 Live YouTube Analysis", "📂 File Upload"],
        label_visibility="collapsed",
    )

    max_comments = st.sidebar.slider(
        "Max comments to fetch (YouTube)",
        min_value=50,
        max_value=1000,
        value=200,
        step=50,
        help="Higher values consume more API quota.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<small style='opacity:.5'>Powered by BERT · Streamlit · Plotly</small>",
        unsafe_allow_html=True,
    )

    return api_key, mode, max_comments


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown(
        """
        <div class="app-header">
          <h1>Sentiment Analysis of Digital Media Content</h1>
          <p>Fine-tuned BERT model · YouTube API v3 · Real-time dashboard</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    api_key, mode, max_comments = build_sidebar()

    result_df: Optional[pd.DataFrame] = None
    source_label = "—"

    # ── Input: YouTube ─────────────────────────────────────────────────────────
    if mode == "🎬 Live YouTube Analysis":
        st.markdown('<div class="section-title">🎬 YouTube Video Analysis</div>', unsafe_allow_html=True)
        yt_url = st.text_input(
            "YouTube Video URL",
            placeholder="https://www.youtube.com/watch?v=…",
        )

        if st.button("Fetch & Analyse", type="primary") and yt_url:
            if not api_key:
                st.error("⚠️  Please enter your YouTube Data API Key in the sidebar.")
                st.stop()

            try:
                with st.spinner("📡  Fetching comments from YouTube …"):
                    comments, metadata = fetch_youtube_comments(
                        api_key=api_key,
                        video_url=yt_url,
                        max_comments=max_comments,
                    )

                if metadata:
                    st.info(
                        f"**{metadata.get('title', 'N/A')}** · "
                        f"{metadata.get('channel', 'N/A')} · "
                        f"{metadata.get('view_count', 0):,} views · "
                        f"{metadata.get('comment_count', 0):,} total comments"
                    )

                if not comments:
                    st.warning("No comments found for this video.")
                    st.stop()

                source_label = "YouTube"
                result_df = run_analysis(comments)

            except (ValueError, PermissionError) as exc:
                st.error(f"❌  {exc}")
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")

    # ── Input: File Upload ─────────────────────────────────────────────────────
    else:
        st.markdown('<div class="section-title">📂 Upload Comment File</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drop a .csv or .json file here",
            type=["csv", "json"],
        )

        if uploaded is not None:
            df_raw = load_file(uploaded)
            if df_raw is not None:
                st.markdown(f"**Preview** — {len(df_raw):,} rows × {len(df_raw.columns)} columns")
                st.dataframe(df_raw.head(5), use_container_width=True)

                # Column selection
                auto_col = pick_text_column(df_raw)
                text_col = st.selectbox(
                    "Select the column containing comment/text data:",
                    options=df_raw.columns.tolist(),
                    index=df_raw.columns.tolist().index(auto_col) if auto_col else 0,
                )

                if st.button("Analyse Sentiments", type="primary"):
                    texts = df_raw[text_col].dropna().astype(str).tolist()
                    if not texts:
                        st.warning("Selected column appears to be empty.")
                    else:
                        source_label = uploaded.name
                        result_df = run_analysis(texts)

    # ── Dashboard ──────────────────────────────────────────────────────────────
    if result_df is not None and not result_df.empty:

        # Metrics
        st.markdown("---")
        render_metrics(result_df, source_label)

        # Charts row 1
        col_pie, col_bar = st.columns(2)
        with col_pie:
            st.plotly_chart(pie_chart(result_df), use_container_width=True)
        with col_bar:
            st.plotly_chart(bar_chart(result_df), use_container_width=True)

        # Charts row 2
        col_kw, col_conf = st.columns(2)
        with col_kw:
            kw_df = top_keywords(result_df["Comment"].tolist(), n=15)
            if not kw_df.empty:
                st.plotly_chart(keyword_bar(kw_df), use_container_width=True)
            else:
                st.info("Not enough text for keyword analysis.")
        with col_conf:
            st.plotly_chart(confidence_histogram(result_df), use_container_width=True)

        # Per-sentiment keyword breakdown
        st.markdown('<div class="section-title">🔍 Keyword Breakdown by Sentiment</div>', unsafe_allow_html=True)
        tabs = st.tabs(["Positive 😊", "Negative 😠", "Neutral 😐"])
        for tab, label in zip(tabs, ["Positive", "Negative", "Neutral"]):
            with tab:
                subset = result_df[result_df["Sentiment"] == label]["Comment"].tolist()
                if subset:
                    kw = top_keywords(subset, n=12)
                    st.plotly_chart(keyword_bar(kw), use_container_width=True)
                else:
                    st.write("No comments in this category.")

        # Searchable comment table
        st.markdown('<div class="section-title">📋 Comment Explorer</div>', unsafe_allow_html=True)

        search = st.text_input("🔎 Search comments …", placeholder="Type a keyword …")
        filter_sentiment = st.multiselect(
            "Filter by sentiment",
            options=["Positive", "Negative", "Neutral"],
            default=["Positive", "Negative", "Neutral"],
        )

        display_df = result_df[result_df["Sentiment"].isin(filter_sentiment)].copy()
        if search:
            display_df = display_df[
                display_df["Comment"].str.contains(search, case=False, na=False)
            ]

        display_df["Confidence"] = (display_df["Confidence"] * 100).round(1).astype(str) + "%"
        st.dataframe(
            display_df.reset_index(drop=True),
            use_container_width=True,
            height=420,
        )
        st.caption(f"Showing {len(display_df):,} of {len(result_df):,} comments.")

        # Download
        st.markdown("---")
        csv_bytes = result_df.to_csv(index=False).encode()
        st.download_button(
            label="⬇️  Download Results as CSV",
            data=csv_bytes,
            file_name="sentiment_results.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
