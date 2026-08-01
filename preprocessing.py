
from __future__ import annotations

import re
from functools import lru_cache
from bs4 import BeautifulSoup
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import nltk

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

@lru_cache(maxsize=1)
def _lemmatizer() -> WordNetLemmatizer:
    return WordNetLemmatizer()

def _safe_lemmatize(text: str) -> str:
    try:
        lemmatizer = _lemmatizer()
        return " ".join(lemmatizer.lemmatize(word, pos=wordnet.VERB) for word in text.split())
    except LookupError:
        return text


def preprocess_text(text: str, remove_stopwords: bool = False, lemmatize: bool = True) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = BeautifulSoup(text, "html.parser").get_text(" ")
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)
    cleaned = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", " ", cleaned)
    cleaned = re.sub(r"(?:\+?\d{1,3}[ .-]?)?(?:\(?\d{2,4}\)?[ .-]?)?\d{3}[ .-]?\d{3,4}\b", " ", cleaned)
    cleaned = cleaned.lower()
    cleaned = cleaned.replace("c++", "cplusplus").replace("c#", "csharp")
    cleaned = re.sub(r"[^a-z0-9+#.\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if remove_stopwords:
        try:
            stop_words = set(stopwords.words("english"))
            cleaned = " ".join(word for word in cleaned.split() if word not in stop_words)
        except LookupError:
            pass
    return _safe_lemmatize(cleaned) if lemmatize else cleaned
