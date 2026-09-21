# Sentiment Analysis of U.S. Election Tweets

Analyzes public sentiment toward Donald Trump and Joe Biden using tweet
data, comparing three classical ML models on TF-IDF features.

**Tech stack:** Python, NLTK, TF-IDF, Logistic Regression, Naive Bayes, SVM

## Dataset

`data/us_election_2024.xlsx` — 10,881 tweets with columns:
- `user` — tweet author handle
- `Subject` — candidate the tweet is about (Donald Trump / Joe Biden)
- `text` — raw tweet text

> Note: the sheet inside the file is named `us_election_2020` and the
> tweet content (COVID-19 references, 2020 campaign events) indicates
> this is 2020 U.S. election data, not 2024. The pipeline works
> identically regardless of year — update the file/labels if you have
> genuine 2024 data.

The raw data has **no sentiment labels**, so this project generates
them using a lexicon-based method (VADER) before training the
supervised classifiers — a standard approach for unlabeled social
text.

## Pipeline (`sentiment_analysis.py`)

1. **Load** — read the Excel file, drop empty rows
2. **Auto-label sentiment** — NLTK's VADER `SentimentIntensityAnalyzer`
   scores each tweet; compound score thresholds (±0.05) assign
   Positive / Neutral / Negative
3. **Preprocess** (`preprocessing.py`) — lowercase, strip URLs/@mentions/
   `#` symbols/punctuation/numbers, tokenize, remove stopwords,
   lemmatize (NLTK)
4. **Vectorize** — TF-IDF, unigrams + bigrams, top 5,000 features
5. **Train & compare 3 classifiers** on an 80/20 stratified split:
   - Logistic Regression
   - Multinomial Naive Bayes
   - Linear SVM
6. **Analyze trends** — sentiment split overall and per candidate
   (voter behavior pattern), top TF-IDF terms per sentiment class
7. **Save** all charts + `summary.json` to `outputs/`

## Setup & Run

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt'); nltk.download('punkt_tab')"
python sentiment_analysis.py
```

Outputs land in `outputs/`.

## Results (this run)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | 81.4% | 81.4% |
| Naive Bayes | 75.7% | 75.5% |
| **SVM (Linear)** | **81.8%** | **81.8%** |

**Overall sentiment:** 34.6% Positive · 30.4% Neutral · 35.0% Negative

**By candidate:**
- Donald Trump: 31.6% Positive, 28.0% Neutral, 40.4% Negative
- Joe Biden: 34.5% Positive, 33.4% Neutral, 32.1% Negative

Joe Biden tweets skewed slightly more positive/neutral; Trump tweets
had the highest negative share of the two, consistent with a more
polarized/critical discourse in this dataset.

## Interactive app (`app.py`)

A Streamlit app that lets anyone type text and get a live sentiment
prediction, plus browse the dataset charts and model comparison —
built on the same trained models (saved to `models/` via `joblib`).

```bash
streamlit run app.py
```

### Deploy for free (Streamlit Community Cloud)

1. Push this project to a GitHub repo (must include `models/*.joblib`
   and `outputs/*.png` — commit them, don't gitignore them).
2. Go to https://share.streamlit.io → sign in with GitHub.
3. "New app" → pick your repo/branch → set **Main file path** to `app.py`.
4. Deploy. You'll get a free link like `https://your-app.streamlit.app`.

No separate database or backend needed — everything the app needs
(trained models, charts) is loaded from files already in the repo.

## Files generated in `outputs/`

- `sentiment_distribution.png` — overall Positive/Neutral/Negative split
- `sentiment_by_candidate.png` — stacked % sentiment per candidate
- `model_comparison.png` — accuracy & F1 across the 3 models
- `*_confusion.png` — confusion matrix per model
- `top_words_positive.png` / `_negative.png` / `_neutral.png` — top TF-IDF
  terms per sentiment class
- `summary.json` — all numeric results in one file
