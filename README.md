# Aspect-Based Sentiment Analysis API for YouTube comments

## Project Goal

Development of a modular REST API in Python to analyze textual comments (primarily from YouTube) to identify common aspects and evaluate the sentiment polarity associated with each one. The solution will allow for understanding public opinion on different elements of a piece of content (audio, script, entertainment, etc.) without persistently storing sensitive information.

## Main Features

- **Aspect-based analysis**: Automatic identification of recurring themes and specific sentiment analysis for each aspect.
- **Overall analysis**: Evaluation of the global polarity of a set of comments.
- **Aspect extraction**: Identification and grouping of themes without sentiment analysis.
- **YouTube integration**: Automatic retrieval of comments from video URLs.
- **Direct analysis**: Processing of comment lists provided directly.

## Actual status
This project has just started and all bellow this is the planned structure and functions. It may change in the future

---

This project delivers an **async-ready MVP API** (Django + DRF) with:
- `POST /api/analysis/comments/`: analyze a list of comments.
- **Sentiment**: uses **Transformers by default** (`TRANSFORMER_MODEL` in `.env`), with **VADER fallback** if libs/models are missing.
- **Aspects**: default **KMeans** with TF-IDF; **algorithm is selectable** per request: `kmeans` (default), `fcm` (requires `scikit-fuzzy`), `bertopic` (requires `bertopic` + deps).
- **Representatives**: returns the **N most representative comments** per cluster (configurable via `.env`: `CLUSTER_REPRESENTATIVES`).
- **Validations**: enforces `MAX_COMMENTS` and `MAX_COMMENT_LENGTH`. Overlong comments are **truncated** and a warning is included in the response.

## Environment variables

Create `.env` from the example below:

```
SECRET_KEY=change-me
DEBUG=true
ALLOWED_HOSTS=127.0.0.1,localhost
MAX_COMMENTS=100
MAX_COMMENT_LENGTH=500
CLUSTER_REPRESENTATIVES=2
SENTIMENT_MODEL=cardiffnlp/twitter-xlm-roberta-base-sentiment
```

## Install

Python 3.10+ recommended.

```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```

## Request payload example

```json
{
  "comments": [
    "Great video, very useful!",
    "Too long, I got bored halfway.",
    "Good editing but the audio was low."
  ],
  "options": {
    "num_aspect_clusters": 3,
    "use_transformers": true,
    "clustering_algorithm": "kmeans"
  }
}
```

## Response example (abridged)

```json
{
  "summary": {
    "n_comments": 3,
    "n_clusters": 3,
    "aspects": [
      {
        "cluster_id": 0,
        "keywords": ["editing","audio","low"],
        "size": 1,
        "representatives": ["Good editing but the audio was low"]
      }
    ]
  },
  "items": [
    {
      "text": "Great video, very useful!",
      "sentiment": {"label": "positive", "score": 0.92},
      "cluster_id": 2
    }
  ],
  "warnings": []
}
```

## Notes

- If `transformers` or the model weights are not available, the API **falls back to VADER** automatically.

