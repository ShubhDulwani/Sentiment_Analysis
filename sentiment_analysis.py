"""
Sentiment Analysis of U.S. Election Tweets
===========================================
Pipeline:
  1. Load tweet data (user, Subject/candidate, text)
  2. Clean + preprocess text with NLTK (stopword removal, lemmatization)
  3. Auto-label sentiment (Positive/Negative/Neutral) using VADER,
     since the raw dataset has no ground-truth sentiment labels
  4. Vectorize text with TF-IDF
  5. Train & compare three classifiers: Logistic Regression, Naive Bayes, SVM
  6. Analyze sentiment trends per candidate (voter behavior patterns)
  7. Save charts + a text report to outputs/

Run:  python sentiment_analysis.py
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from preprocessing import preprocess

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DATA_PATH = "data/us_election_2024.xlsx"
OUTPUT_DIR = "outputs"
MODELS_DIR = "models"
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_TFIDF_FEATURES = 5000

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 130
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    df["Subject"] = df["Subject"].str.strip()
    return df


def label_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Use VADER (lexicon-based) to assign a sentiment label per tweet.

    This substitutes for missing ground-truth labels — a standard
    approach when working with unlabeled social-media text. VADER is
    tuned for short, informal social text (emojis, slang, punctuation
    emphasis), which fits tweets well.
    """
    sia = SentimentIntensityAnalyzer()

    def score_to_label(text: str) -> str:
        compound = sia.polarity_scores(text)["compound"]
        if compound >= 0.05:
            return "Positive"
        elif compound <= -0.05:
            return "Negative"
        return "Neutral"

    df["sentiment"] = df["text"].apply(score_to_label)
    return df


def preprocess_corpus(df: pd.DataFrame) -> pd.DataFrame:
    df["clean_text"] = df["text"].apply(preprocess)
    # Drop rows that became empty after cleaning (e.g., link-only tweets)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    return df


def plot_sentiment_distribution(df: pd.DataFrame):
    plt.figure(figsize=(6, 4.5))
    order = ["Positive", "Neutral", "Negative"]
    colors = {"Positive": "#3ED598", "Neutral": "#8A93A6", "Negative": "#FF5470"}
    counts = df["sentiment"].value_counts().reindex(order)
    bars = plt.bar(order, counts.values, color=[colors[o] for o in order])
    plt.title("Overall Sentiment Distribution (All Tweets)")
    plt.ylabel("Number of Tweets")
    for b in bars:
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 30,
                  str(int(b.get_height())), ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/sentiment_distribution.png")
    plt.close()


def plot_sentiment_by_candidate(df: pd.DataFrame):
    plt.figure(figsize=(7, 5))
    order = ["Positive", "Neutral", "Negative"]
    cross = pd.crosstab(df["Subject"], df["sentiment"])[order]
    cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100
    colors = ["#3ED598", "#8A93A6", "#FF5470"]
    cross_pct.plot(kind="bar", stacked=True, color=colors, figsize=(7, 5))
    plt.title("Sentiment Breakdown by Candidate (%)")
    plt.ylabel("Percentage of Tweets")
    plt.xlabel("")
    plt.xticks(rotation=0)
    plt.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/sentiment_by_candidate.png")
    plt.close()
    return cross, cross_pct


def plot_model_comparison(results: dict):
    plt.figure(figsize=(6.5, 4.5))
    names = list(results.keys())
    accs = [results[n]["accuracy"] * 100 for n in names]
    f1s = [results[n]["f1_macro"] * 100 for n in names]

    x = np.arange(len(names))
    width = 0.35
    plt.bar(x - width / 2, accs, width, label="Accuracy", color="#F5A623")
    plt.bar(x + width / 2, f1s, width, label="Macro F1", color="#3ED598")
    plt.xticks(x, names)
    plt.ylabel("Score (%)")
    plt.title("Model Comparison: TF-IDF Features")
    plt.ylim(0, 100)
    for i, v in enumerate(accs):
        plt.text(i - width / 2, v + 1, f"{v:.1f}", ha="center", fontsize=9)
    for i, v in enumerate(f1s):
        plt.text(i + width / 2, v + 1, f"{v:.1f}", ha="center", fontsize=9)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_comparison.png")
    plt.close()


def plot_confusion_matrix(cm, labels, model_name, filename):
    plt.figure(figsize=(5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}")
    plt.close()


def plot_top_words(df: pd.DataFrame, vectorizer: TfidfVectorizer, tfidf_matrix, sentiment: str, n=15):
    mask = (df["sentiment"] == sentiment).values
    if mask.sum() == 0:
        return
    sub_matrix = tfidf_matrix[mask]
    mean_scores = np.asarray(sub_matrix.mean(axis=0)).flatten()
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_idx = mean_scores.argsort()[-n:][::-1]
    top_words = feature_names[top_idx]
    top_vals = mean_scores[top_idx]

    color = {"Positive": "#3ED598", "Negative": "#FF5470", "Neutral": "#8A93A6"}[sentiment]
    plt.figure(figsize=(7, 5))
    plt.barh(top_words[::-1], top_vals[::-1], color=color)
    plt.title(f"Top {n} TF-IDF Terms — {sentiment} Tweets")
    plt.xlabel("Mean TF-IDF Score")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/top_words_{sentiment.lower()}.png")
    plt.close()


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)
    print(f"  {len(df)} tweets loaded. Candidates: {df['Subject'].value_counts().to_dict()}")

    print("Labeling sentiment with VADER...")
    df = label_sentiment(df)
    print(f"  Sentiment distribution: {df['sentiment'].value_counts().to_dict()}")

    print("Preprocessing text (clean, tokenize, lemmatize)...")
    df = preprocess_corpus(df)
    print(f"  {len(df)} tweets remain after cleaning.")

    print("Generating descriptive charts...")
    plot_sentiment_distribution(df)
    cross, cross_pct = plot_sentiment_by_candidate(df)

    # ---------------- TF-IDF ----------------
    print("Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=MAX_TFIDF_FEATURES, ngram_range=(1, 2), min_df=3)
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["sentiment"]

    for s in ["Positive", "Negative", "Neutral"]:
        plot_top_words(df, vectorizer, X, s)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # ---------------- Models ----------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Naive Bayes": MultinomialNB(),
        "SVM (Linear)": LinearSVC(random_state=RANDOM_STATE, max_iter=5000),
    }

    results = {}
    labels_order = ["Positive", "Neutral", "Negative"]

    print("\nTraining & evaluating models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1_macro = f1_score(y_test, preds, average="macro")
        report = classification_report(y_test, preds, output_dict=True)
        cm = confusion_matrix(y_test, preds, labels=labels_order)

        results[name] = {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "report": report,
        }

        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + "_confusion.png"
        plot_confusion_matrix(cm, labels_order, name, fname)

        print(f"  {name}: accuracy={acc:.4f}, macro-F1={f1_macro:.4f}")

    plot_model_comparison(results)

    # Best model
    best_model_name = max(results, key=lambda k: results[k]["accuracy"])
    print(f"\nBest model: {best_model_name} (accuracy={results[best_model_name]['accuracy']:.4f})")

    # ---------------- Persist vectorizer + models for the app ----------------
    print("Saving vectorizer and trained models to models/...")
    joblib.dump(vectorizer, f"{MODELS_DIR}/tfidf_vectorizer.joblib")
    for name, model in models.items():
        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(model, f"{MODELS_DIR}/{fname}.joblib")
    joblib.dump(best_model_name, f"{MODELS_DIR}/best_model_name.joblib")

    # ---------------- Save summary report ----------------
    summary = {
        "total_tweets_used": len(df),
        "candidates": df["Subject"].value_counts().to_dict(),
        "sentiment_distribution_overall": df["sentiment"].value_counts().to_dict(),
        "sentiment_by_candidate_pct": cross_pct.round(2).to_dict(),
        "model_results": {
            name: {"accuracy": round(r["accuracy"], 4), "f1_macro": round(r["f1_macro"], 4)}
            for name, r in results.items()
        },
        "best_model": best_model_name,
    }
    with open(f"{OUTPUT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll charts + summary.json saved to '{OUTPUT_DIR}/'")
    return df, results, summary


if __name__ == "__main__":
    main()
