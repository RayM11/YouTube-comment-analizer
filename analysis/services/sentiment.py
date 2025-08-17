from typing import List, Dict
import math

# Fallback VADER (rápido). Si está, lo usamos; si no, hacemos regla trivial.
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _HAS_VADER = True
except Exception:
    _HAS_VADER = False

# Opcional Transformers (CardiffNLP). Solo se usa si settings.USE_TRANSFORMERS = True y está instalado.
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False


class SentimentAnalyzer:
    def __init__(self, use_transformers: bool = False):
        self.use_transformers = use_transformers and _HAS_TRANSFORMERS
        self._transformers_ready = False
        if self.use_transformers:
            try:
                # Modelo multilingüe recomendado (CardiffNLP). Requiere pesos en caché o internet previo.
                self.tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-xlm-roberta-base-sentiment')
                self.model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-xlm-roberta-base-sentiment')
                self.model.eval()
                self._transformers_ready = True
            except Exception:
                # Si falla la carga, caemos a VADER
                self.use_transformers = False

        if not self.use_transformers:
            if _HAS_VADER:
                self.vader = SentimentIntensityAnalyzer()
            else:
                self.vader = None  # fallback trivial si no hay VADER

    def _softmax(self, x):
        m = max(x)
        exps = [math.exp(v - m) for v in x]
        s = sum(exps)
        return [v/s for v in exps]

    def predict_one(self, text: str) -> Dict:
        if self.use_transformers and self._transformers_ready:
            with torch.no_grad():
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
                logits = self.model(**inputs).logits[0].tolist()  # [neg, neu, pos] en este modelo
                probs = self._softmax(logits)
                labels = ['negative','neutral','positive']
                idx = int(max(range(len(probs)), key=lambda i: probs[i]))
                return {"label": labels[idx], "score": float(probs[idx])}

        if self.vader is not None:
            scores = self.vader.polarity_scores(text)
            # compound en [-1,1]
            c = scores['compound']
            if c >= 0.05:
                return {"label": "positive", "score": float((c+1)/2)}
            elif c <= -0.05:
                return {"label": "negative", "score": float((1-c)/2)}  # escala simple
            else:
                return {"label": "neutral", "score": float(1-abs(c))}

        # Fallback trivial si nada está instalado
        return {"label": "neutral", "score": 0.5}

    def batch_predict(self, texts: List[str]) -> List[Dict]:
        return [self.predict_one(t) for t in texts]
