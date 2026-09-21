"""
Text preprocessing utilities for tweet sentiment analysis.
Cleans raw tweet text: removes URLs, mentions, hashtags symbols,
punctuation/numbers, lowercases, tokenizes, removes stopwords,
and lemmatizes using NLTK.
"""

import re
import string
import nltk

# Ensure required NLTK corpora are available (downloads once on a fresh
# environment like Streamlit Cloud; no-op if already present locally).
for pkg, path in [
    ("stopwords", "corpora/stopwords"),
    ("wordnet", "corpora/wordnet"),
    ("omw-1.4", "corpora/omw-1.4"),
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("vader_lexicon", "sentiment/vader_lexicon"),
]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_SYMBOL_RE = re.compile(r"#")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase + strip URLs/mentions/hashtag-symbols/punctuation/numbers."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_SYMBOL_RE.sub(" ", text)  # keep the word, drop the '#'
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def tokenize_and_lemmatize(text: str) -> list:
    """Tokenize, drop stopwords/short tokens, lemmatize."""
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return tokens


def preprocess(text: str) -> str:
    """Full pipeline: clean -> tokenize -> lemmatize -> rejoin as a string.

    Returns a space-joined string (ready for TF-IDF vectorization).
    """
    cleaned = clean_text(text)
    tokens = tokenize_and_lemmatize(cleaned)
    return " ".join(tokens)
