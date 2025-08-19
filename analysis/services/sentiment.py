from typing import List, Dict
import math

# Lightweight fallback: VADER
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _HAS_VADER = True
except Exception:
    _HAS_VADER = False

# Optional: Transformers sentiment
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False


class SentimentAnalyzer:
    """Abstraction over sentiment backends.
    - Tries Transformers if requested (and libs/models available).
    - Falls back to VADER (lexicon-based).
    - If neither available, returns neutral.
    """
    def __init__(self, use_transformers: bool = True, model_name: str = None):
        self.use_transformers = use_transformers and _HAS_TRANSFORMERS
        self._transformers_ready = False
        self.model_name = model_name or 'cardiffnlp/twitter-xlm-roberta-base-sentiment'

        if self.use_transformers:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self.model.eval()
                self._transformers_ready = True
            except Exception:
                self.use_transformers = False

        if not self.use_transformers:
            if _HAS_VADER:
                self.vader = SentimentIntensityAnalyzer()
            else:
                self.vader = None

    def _softmax(self, x):
        m = max(x)
        exps = [math.exp(v - m) for v in x]
        s = sum(exps)
        return [v/s for v in exps]

    def predict_one(self, text: str) -> Dict:
        # 1) Transformers (if available)
        if self.use_transformers and self._transformers_ready:
            with torch.no_grad():
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
                logits = self.model(**inputs).logits[0].tolist()  # [neg, neu, pos] (CardiffNLP)
                probs = self._softmax(logits)
                labels = ['negative','neutral','positive']
                idx = int(max(range(len(probs)), key=lambda i: probs[i]))
                return {"label": labels[idx], "score": float(probs[idx])}

        # 2) VADER fallback
        if self.vader is not None:
            scores = self.vader.polarity_scores(text)
            c = scores['compound']  # [-1,1]
            if c >= 0.05:
                return {"label": "positive", "score": float((c+1)/2)}
            elif c <= -0.05:
                return {"label": "negative", "score": float((1-c)/2)}
            else:
                return {"label": "neutral", "score": float(1-abs(c))}

        # 3) Last resort
        return {"label": "neutral", "score": 0.5}

    def batch_predict(self, texts: List[str]) -> List[Dict]:
        return [self.predict_one(t) for t in texts]
