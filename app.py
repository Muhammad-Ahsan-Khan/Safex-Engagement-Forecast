"""
app.py

Interactive Streamlit dashboard for the SafeX engagement forecasting model.
Lets a user pick post attributes and get a live predicted engagement rate,
run predictions on a batch of posts, compare two scenarios side by side,
and browse the charts and patterns behind the model.

Run with:  streamlit run app.py
"""

import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SafeX Engagement Forecast", page_icon="📊", layout="wide")

CUSTOM_CSS = """
<style>
:root {
    --safex-navy: #16324F;
    --safex-blue: #1B6FB0;
    --safex-blue-light: #2E86C1;
    --safex-tint: #EAF3FB;
    --safex-tint-2: #F5F9FD;
}

.block-container { padding-top: 1.5rem; max-width: 1200px; }

/* ---- Header banner ---- */
.safex-header {
    background: linear-gradient(120deg, var(--safex-navy) 0%, var(--safex-blue) 100%);
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 24px;
    color: #ffffff;
}
.safex-header h1 {
    color: #ffffff;
    margin: 0 0 6px 0;
    font-size: 1.9rem;
    font-weight: 700;
}
.safex-header p {
    color: #DCEBF7;
    margin: 0;
    font-size: 0.98rem;
}
.safex-badge {
    display: inline-block;
    background: rgba(255,255,255,0.16);
    color: #ffffff;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-bottom: 10px;
}

/* ---- Headings ---- */
h2, h3 { color: var(--safex-navy); }

/* ---- Metric cards ---- */
[data-testid="stMetric"] {
    background: var(--safex-tint-2);
    border: 1px solid #D6E7F5;
    border-left: 4px solid var(--safex-blue);
    border-radius: 10px;
    padding: 14px 16px 10px 16px;
}
[data-testid="stMetricLabel"] { font-weight: 600; color: var(--safex-blue); }
[data-testid="stMetricValue"] { color: var(--safex-navy); }

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 2px solid var(--safex-tint);
}
.stTabs [data-baseweb="tab"] {
    background-color: var(--safex-tint-2);
    border-radius: 8px 8px 0 0;
    color: var(--safex-navy);
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background-color: var(--safex-blue) !important;
    color: #ffffff !important;
    font-weight: 600;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: var(--safex-tint);
    border-right: 1px solid #D6E7F5;
}
[data-testid="stSidebar"] h3 { color: var(--safex-navy); }

/* ---- Buttons ---- */
.stButton button, .stDownloadButton button {
    background-color: var(--safex-blue);
    color: #ffffff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}
.stButton button:hover, .stDownloadButton button:hover {
    background-color: var(--safex-navy);
    color: #ffffff;
}

/* ---- Progress bar ---- */
[data-testid="stProgress"] > div > div { background-color: var(--safex-blue); }

/* ---- Dataframe header row ---- */
[data-testid="stDataFrame"] thead tr th {
    background-color: var(--safex-tint) !important;
    color: var(--safex-navy) !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="safex-header">
        <div class="safex-badge">SAFEX SOLUTIONS &nbsp;·&nbsp; AI / ML PROTOTYPE</div>
        <h1>📊 Engagement Forecast</h1>
        <p>Predicts a post's expected engagement rate — (likes + comments + shares) as a
        percentage of followers — from attributes known before it goes live.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

POST_TYPES = ["video", "carousel", "image", "text"]
PLATFORMS = ["instagram", "facebook", "linkedin", "twitter"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@st.cache_resource
def load_model():
    model = joblib.load("models/model.pkl")
    with open("models/feature_columns.json") as f:
        feature_cols = json.load(f)
    return model, feature_cols


@st.cache_data
def load_results():
    with open("results.json") as f:
        return json.load(f)


@st.cache_data
def load_training_data():
    return pd.read_csv("data/social_media_posts.csv")


def hour_bucket(h):
    if 0 <= h <= 6:
        return "early_morning"
    if 7 <= h <= 11:
        return "morning"
    if 12 <= h <= 17:
        return "afternoon"
    if 18 <= h <= 22:
        return "evening"
    return "late_night"


def build_feature_row(feature_cols, **attrs):
    row = {
        "hour_of_day": attrs["hour_of_day"],
        "is_weekend": 1 if attrs["day_of_week"] in ["Sat", "Sun"] else 0,
        "hashtag_count": attrs["hashtag_count"],
        "caption_length": attrs["caption_length"],
        "follower_count": attrs["follower_count"],
    }
    bucket = hour_bucket(attrs["hour_of_day"])
    for col in feature_cols:
        if col.startswith("platform_"):
            row[col] = int(col == f"platform_{attrs['platform']}")
        elif col.startswith("post_type_"):
            row[col] = int(col == f"post_type_{attrs['post_type']}")
        elif col.startswith("day_of_week_"):
            row[col] = int(col == f"day_of_week_{attrs['day_of_week']}")
        elif col.startswith("hour_bucket_"):
            row[col] = int(col == f"hour_bucket_{bucket}")
    return row


def predict_one(model, feature_cols, **attrs):
    row = build_feature_row(feature_cols, **attrs)
    X_input = pd.DataFrame([row]).reindex(columns=feature_cols, fill_value=0)
    return float(model.predict(X_input)[0]), row


def attribute_inputs(key_prefix):
    """Renders the post-attribute widgets and returns the chosen values."""
    post_type = st.selectbox("Post type", POST_TYPES, key=f"{key_prefix}_type")
    platform = st.selectbox("Platform", PLATFORMS, key=f"{key_prefix}_platform")
    day_of_week = st.selectbox("Day of week", DAYS, key=f"{key_prefix}_day")
    hour_of_day = st.slider("Hour posted (24h)", 0, 23, 19, key=f"{key_prefix}_hour")
    hashtag_count = st.slider("Number of hashtags", 0, 19, 7, key=f"{key_prefix}_hashtags")
    caption_length = st.slider("Caption length (characters)", 5, 399, 110, key=f"{key_prefix}_caption")
    follower_count = st.number_input(
        "Follower count", min_value=500, max_value=200_000, value=15_000, step=500, key=f"{key_prefix}_followers"
    )
    return dict(
        post_type=post_type, platform=platform, day_of_week=day_of_week,
        hour_of_day=hour_of_day, hashtag_count=hashtag_count,
        caption_length=caption_length, follower_count=follower_count,
    )


model, feature_cols = load_model()
results = load_results()
history_df = load_training_data()

with st.sidebar:
    st.markdown("### About")
    st.write(
        "AI/ML prototype built for SafeX Solutions. Trained on a synthetic "
        "dataset that mirrors real-world social posting patterns — see "
        "`case_study.md` for full methodology."
    )
    st.markdown("### Final model")
    st.write(f"**{results['winner'].replace('_', ' ').title()}**")
    st.write(f"R² = {results[results['winner']]['r2']:.3f}")
    st.write(f"Trained on {results['n_rows']:,} posts")

tab_predict, tab_compare, tab_batch, tab_model, tab_data = st.tabs(
    ["🔮 Predict", "⚖️ Compare scenarios", "📋 Batch predict", "📈 Model performance", "🗂️ Explore the data"]
)

with tab_predict:
    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.subheader("Post attributes")
        attrs = attribute_inputs("single")
        prediction, row = predict_one(model, feature_cols, **attrs)

    with col_right:
        st.subheader("Forecast")
        avg_rate = history_df["engagement_rate"].mean()
        st.metric(
            "Predicted engagement rate",
            f"{prediction:.2f}%",
            delta=f"{prediction - avg_rate:+.2f} pts vs. dataset average",
        )

        follower_count = attrs["follower_count"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Est. likes", f"{prediction / 100 * follower_count * 0.78:,.0f}")
        c2.metric("Est. comments", f"{prediction / 100 * follower_count * 0.10:,.0f}")
        c3.metric("Est. shares", f"{prediction / 100 * follower_count * 0.10:,.0f}")

        percentile = (history_df["engagement_rate"] < prediction).mean() * 100
        st.progress(min(max(percentile / 100, 0.0), 1.0))
        st.caption(f"This would outperform roughly {percentile:.0f}% of posts in the training set.")

        st.divider()
        st.markdown("**Quick read on this combination**")
        tips = []
        if attrs["hashtag_count"] < 4 or attrs["hashtag_count"] > 12:
            tips.append("Hashtag count is outside the 5–10 sweet spot seen in the data.")
        if not (80 <= attrs["caption_length"] <= 150):
            tips.append("Caption length is outside the ~80–150 character sweet spot.")
        if not (18 <= attrs["hour_of_day"] <= 22):
            tips.append("Posting outside the 6–10pm window tends to underperform.")
        if attrs["post_type"] == "text":
            tips.append("Text-only posts consistently score lowest — consider an image or video.")
        if not tips:
            tips.append("This combination lines up well with the strongest patterns in the data.")
        for t in tips:
            st.write(f"- {t}")

        export_row = {**row, "predicted_engagement_rate": round(prediction, 3)}
        st.download_button(
            "Download this prediction as CSV",
            data=pd.DataFrame([export_row]).to_csv(index=False),
            file_name="engagement_prediction.csv",
            mime="text/csv",
        )

with tab_compare:
    st.subheader("Compare two scenarios side by side")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Scenario A**")
        attrs_a = attribute_inputs("a")
        pred_a, _ = predict_one(model, feature_cols, **attrs_a)

    with col_b:
        st.markdown("**Scenario B**")
        attrs_b = attribute_inputs("b")
        pred_b, _ = predict_one(model, feature_cols, **attrs_b)

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Scenario A", f"{pred_a:.2f}%")
    m2.metric("Scenario B", f"{pred_b:.2f}%")
    winner = "A" if pred_a > pred_b else "B" if pred_b > pred_a else "Tie"
    m3.metric("Higher forecast", winner, delta=f"{abs(pred_a - pred_b):.2f} pts apart")

with tab_batch:
    st.subheader("Predict engagement for many posts at once")
    st.write(
        "Upload a CSV with columns `post_type, platform, day_of_week, hour_of_day, "
        "hashtag_count, caption_length, follower_count` to score a batch of drafts."
    )
    upload = st.file_uploader("CSV file", type="csv")
    if upload is not None:
        batch_df = pd.read_csv(upload)
        required = {"post_type", "platform", "day_of_week", "hour_of_day",
                    "hashtag_count", "caption_length", "follower_count"}
        missing = required - set(batch_df.columns)
        if missing:
            st.error(f"Missing required column(s): {', '.join(sorted(missing))}")
        else:
            preds = [
                predict_one(model, feature_cols, **r.to_dict())[0]
                for _, r in batch_df[list(required)].iterrows()
            ]
            batch_df["predicted_engagement_rate"] = [round(p, 3) for p in preds]
            st.dataframe(batch_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download predictions as CSV",
                data=batch_df.to_csv(index=False),
                file_name="batch_predictions.csv",
                mime="text/csv",
            )

with tab_model:
    st.subheader("Model comparison")
    rows = []
    for name, label in [
        ("linear_regression", "Linear Regression"),
        ("random_forest", "Random Forest"),
        ("gradient_boosting", "Gradient Boosting"),
    ]:
        if name in results:
            m = results[name]
            rows.append({"Model": label, "RMSE": m["rmse"], "MAE": m["mae"], "R²": m["r2"]})
    comparison_df = pd.DataFrame(rows).sort_values("RMSE")
    styled = (
        comparison_df.style
        .format({"RMSE": "{:.3f}", "MAE": "{:.3f}", "R²": "{:.3f}"})
        .background_gradient(cmap="Blues_r", subset=["RMSE", "MAE"])
        .background_gradient(cmap="Blues", subset=["R²"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption(f"Winning model: **{results['winner'].replace('_', ' ').title()}**, "
               f"trained on {results['n_rows']:,} posts with {results['n_features']} engineered features.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.image("images/predicted_vs_actual.png", use_container_width=True)
    with col_b:
        st.image("images/feature_importance.png", use_container_width=True)

with tab_data:
    st.subheader("Patterns in the training data")
    st.image("images/eda_overview.png", use_container_width=True)

    st.subheader("Best time to post, by platform")
    pivot = history_df.pivot_table(
        index="platform", columns="hour_of_day", values="engagement_rate", aggfunc="mean"
    ).round(2)
    st.dataframe(
        pivot.style.background_gradient(cmap="Blues", axis=None).format("{:.2f}"),
        use_container_width=True,
    )
    st.caption("Average engagement rate (%) by platform and hour of day. Darker = stronger.")

    st.subheader("Filter and browse posts")
    f1, f2 = st.columns(2)
    platform_filter = f1.multiselect("Platform", sorted(history_df["platform"].unique()))
    type_filter = f2.multiselect("Post type", sorted(history_df["post_type"].unique()))

    filtered = history_df.copy()
    if platform_filter:
        filtered = filtered[filtered["platform"].isin(platform_filter)]
    if type_filter:
        filtered = filtered[filtered["post_type"].isin(type_filter)]

    st.dataframe(filtered.head(200), use_container_width=True, hide_index=True)
    st.caption(f"Showing {min(len(filtered), 200)} of {len(filtered)} matching posts.")

st.divider()
with st.expander("About this project"):
    st.write(
        "Trained on a synthetic dataset built to mirror well-documented, real-world social "
        "media engagement patterns — not real SafeX client data, which wasn't available for "
        "this prototype. See `case_study.md` in the repository for full methodology, results, "
        "and limitations."
    )
