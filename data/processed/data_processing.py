import re
import nltk
from nltk.corpus import stopwords


stop_words = set(stopwords.words("english"))

def clean_text(text:str) -> str:
    if not isinstance(text, str): #typeGuarda para garantir que o text é string
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove HTML
    text = re.sub(r"<.*?>", " ", text)

    # Remove unicode escape sequences (\x, \u, etc.)
    text = re.sub(r"\\x[a-f0-9]{0,2}", " ", text)
    text = re.sub(r"\\u[a-f0-9]{0,4}", " ", text)

    # Remove leftover hex patterns
    text = re.sub(r"\bx[a-f0-9]{1,2}\b", " ", text)

    # Remove non-letter characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Tokenize
    tokens = text.split()

    # Remove stopwords
    tokens = [t for t in tokens if t not in stop_words]

    # Remove short tokens (CRITICAL)
    tokens = [t for t in tokens if len(t) > 2]

    # Normaliza spacos
    return " ".join(tokens)