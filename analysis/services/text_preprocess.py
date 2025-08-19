import re

URL_RE = re.compile(r'https?://\S+')
MENTION_RE = re.compile(r'@\w+')


def preprocess_text(text: str) -> str:
    """Lowercase, strip URLs/mentions, normalize whitespace."""
    t = text.strip().lower()
    t = URL_RE.sub(' ', t)
    t = MENTION_RE.sub(' ', t)
    t = re.sub(r'\s+', ' ', t)
    return t


def preprocess_batch(texts):
    return [preprocess_text(t) for t in texts]
