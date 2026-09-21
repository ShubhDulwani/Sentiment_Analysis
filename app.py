"""
Streamlit app: U.S. Election Tweet Sentiment Analysis
Run locally:  streamlit run app.py
"""

import json
import joblib
import pandas as pd
import streamlit as st

from preprocessing import preprocess

st.set_page_config(
    page_title="Election Tweet Sentiment Analysis",
    page_icon="🗳️",
    layout="wide",
)

SENTIMENT_COLOR = {"Positive": "#3ED598", "Neutral": "#8A93A6", "Negative": "#FF5470"}


@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
    models = {
        "Logistic Regression": joblib.load("models/logistic_regression.joblib"),
        "Naive Bayes": joblib.load("models/naive_bayes.joblib"),
        "SVM (Linear)": joblib.load("models/svm_linear.joblib"),
    }
    best_model_name = joblib.load("models/best_model_name.joblib")
    with open("outputs/summary.json") as f:
        summary = json.load(f)
    return vectorizer, models, best_model_name, summary


vectorizer, models, best_model_name, summary = load_artifacts()

st.title("🗳️ U.S. Election Tweet Sentiment Analysis")
st.caption(
    "NLP pipeline: NLTK preprocessing + VADER auto-labeling → TF-IDF → "
    "Logistic Regression / Naive Bayes / SVM"
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔮 Try it live", "📊 Dataset insights", "🧪 Model comparison", "📁 Analyze your own dataset"]
)

# ------------------------------------------------------------------
# Tab 1: Live prediction
# ------------------------------------------------------------------
with tab1:
    st.subheader("Classify your own text")
    col_input, col_settings = st.columns([3, 1])
    with col_settings:
        model_choice = st.selectbox("Model", list(models.keys()), index=list(models.keys()).index(best_model_name))
        st.caption(f"Best performing on test set: **{best_model_name}**")

    with col_input:
        user_text = st.text_area(
            "Enter a tweet or any short text about a political candidate",
            placeholder="e.g. This administration has done a fantastic job handling the economy!",
            height=120,
        )
        predict_clicked = st.button("Analyze sentiment", type="primary")

    if predict_clicked and user_text.strip():
        cleaned = preprocess(user_text)
        if not cleaned:
            st.warning("Text became empty after cleaning (e.g. it was only links/mentions). Try different text.")
        else:
            X = vectorizer.transform([cleaned])
            model = models[model_choice]
            pred = model.predict(X)[0]
            color = SENTIMENT_COLOR.get(pred, "#8A93A6")

            st.markdown(
                f"""
                <div style="padding:18px 22px;border-radius:8px;background:{color}22;
                border-left:4px solid {color};margin-top:10px;">
                    <span style="font-size:14px;color:#666;">Predicted sentiment</span><br/>
                    <span style="font-size:28px;font-weight:700;color:{color};">{pred}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                proba_df = pd.DataFrame({"Sentiment": model.classes_, "Confidence": proba}).sort_values(
                    "Confidence", ascending=False
                )
                st.bar_chart(proba_df.set_index("Sentiment"))

            with st.expander("See preprocessed text fed to the model"):
                st.code(cleaned or "(empty)")
    elif predict_clicked:
        st.warning("Please enter some text first.")

# ------------------------------------------------------------------
# Tab 2: Dataset insights
# ------------------------------------------------------------------
with tab2:
    st.subheader("Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Tweets analyzed", f"{summary['total_tweets_used']:,}")
    c2.metric("Candidates covered", len(summary["candidates"]))
    c3.metric("Best model", best_model_name)

    col_a, col_b = st.columns(2)
    with col_a:
        st.image("outputs/sentiment_distribution.png", use_container_width=True)
    with col_b:
        st.image("outputs/sentiment_by_candidate.png", use_container_width=True)

    st.subheader("Top TF-IDF terms by sentiment")
    col_p, col_neu, col_n = st.columns(3)
    with col_p:
        st.image("outputs/top_words_positive.png", use_container_width=True)
    with col_neu:
        st.image("outputs/top_words_neutral.png", use_container_width=True)
    with col_n:
        st.image("outputs/top_words_negative.png", use_container_width=True)

# ------------------------------------------------------------------
# Tab 3: Model comparison
# ------------------------------------------------------------------
with tab3:
    st.subheader("Model performance")
    results_df = pd.DataFrame(summary["model_results"]).T
    results_df.columns = ["Accuracy", "Macro F1"]
    st.dataframe((results_df * 100).round(1).astype(str) + "%", use_container_width=True)

    st.image("outputs/model_comparison.png", use_container_width=True)

    st.subheader("Confusion matrices")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("outputs/logistic_regression_confusion.png", use_container_width=True)
    with col2:
        st.image("outputs/naive_bayes_confusion.png", use_container_width=True)
    with col3:
        st.image("outputs/svm_linear_confusion.png", use_container_width=True)

st.divider()
st.caption("Built with Python, NLTK, scikit-learn & Streamlit.")

# ------------------------------------------------------------------
# Tab 4: Upload your own dataset and get sentiment for every row
# ------------------------------------------------------------------
with tab4:
    st.subheader("Upload a tweet dataset and get sentiment for every row")
    st.caption(
        "Upload a CSV or Excel file with a column of tweet/text content. "
        "Every row gets a predicted sentiment using the trained model."
    )

    bulk_model_choice = st.selectbox(
        "Model to use", list(models.keys()), index=list(models.keys()).index(best_model_name), key="bulk_model"
    )

    uploaded_file = st.file_uploader("Upload dataset", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Couldn't read the file: {e}")
            raw_df = None

        if raw_df is not None and len(raw_df) > 0:
            st.write(f"**{len(raw_df):,} rows** loaded. Columns: {', '.join(raw_df.columns.astype(str))}")

            # Guess the text column: prefer common names, else let the user pick
            likely_names = ["text", "tweet", "tweets", "content", "message", "body"]
            cols_lower = {c.lower(): c for c in raw_df.columns}
            default_col = next((cols_lower[n] for n in likely_names if n in cols_lower), raw_df.columns[0])

            text_col = st.selectbox(
                "Which column contains the tweet/text?",
                options=list(raw_df.columns),
                index=list(raw_df.columns).index(default_col),
            )

            run_bulk = st.button("Analyze full dataset", type="primary")

            if run_bulk:
                with st.spinner(f"Cleaning text and predicting sentiment for {len(raw_df):,} rows..."):
                    work_df = raw_df.copy()
                    work_df[text_col] = work_df[text_col].astype(str)
                    work_df["clean_text"] = work_df[text_col].apply(preprocess)

                    non_empty_mask = work_df["clean_text"].str.len() > 0
                    model = models[bulk_model_choice]

                    predictions = pd.Series(index=work_df.index, dtype=object)
                    if non_empty_mask.any():
                        X_bulk = vectorizer.transform(work_df.loc[non_empty_mask, "clean_text"])
                        predictions.loc[non_empty_mask] = model.predict(X_bulk)
                    predictions.loc[~non_empty_mask] = "Neutral"  # empty-after-cleaning fallback

                    work_df["predicted_sentiment"] = predictions

                st.success(f"Done! Predicted sentiment for {len(work_df):,} rows using {bulk_model_choice}.")

                # Summary chart
                dist = work_df["predicted_sentiment"].value_counts().reindex(
                    ["Positive", "Neutral", "Negative"]
                ).fillna(0)
                col_chart, col_metrics = st.columns([2, 1])
                with col_chart:
                    st.bar_chart(dist, color="#F5A623")
                with col_metrics:
                    for label in ["Positive", "Neutral", "Negative"]:
                        pct = (dist[label] / len(work_df) * 100) if len(work_df) else 0
                        st.metric(label, f"{int(dist[label]):,}", f"{pct:.1f}%")

                st.subheader("Results")
                display_cols = [c for c in raw_df.columns] + ["predicted_sentiment"]
                st.dataframe(work_df[display_cols], use_container_width=True, height=350)

                csv_bytes = work_df[display_cols].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download results as CSV",
                    data=csv_bytes,
                    file_name="sentiment_predictions.csv",
                    mime="text/csv",
                )
