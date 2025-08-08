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

## Project Structure

```
sentiment_analysis_api/
├── manage.py
├── requirements.txt
├── .env.example
├── config/
│   ├── settings.py          # Django settings
│   ├── urls.py             # Main URLs
│   └── wsgi.py             # WSGI server
├── apps/
│   ├── analysis/
│   │   ├── views.py        # API Endpoints
│   │   ├── serializers.py  # Data validation
│   │   └── urls.py         # Analysis routes
│   └── core/
│       ├── controllers/
│       │   └── analysis_controller.py    # Main orchestrator
│       ├── analyzers/
│       │   ├── sentiment_analyzer.py     # Polarity analysis
│       │   ├── aspect_analyzer.py        # Aspect extraction
│       │   └── text_preprocessor.py      # Text cleaning
│       ├── scrapers/
│       │   └── youtube_scraper.py        # YT comment extraction
│       ├── utils/
│       │   ├── text_utils.py            # Text utilities
│       │   ├── validators.py            # Validations
│       │   └── url_utils.py             # URL handling
│       └── config/
│           └── analyzer_config.py        # Analysis configuration
└── tests/                               # Unit tests
```

---

## Main Components

### 1. **AnalysisController**
**File**: `core/controllers/analysis_controller.py`

**Responsibilities**:
- Orchestration of all analysis types
- Coordination between components (scraper → preprocessor → analyzers)
- Aggregation and formatting of final results
- Handling of business errors and validations

**Main Functions**:
- `analyze_comments_direct()`: Direct analysis of a comment list
- `analyze_youtube_video()`: Complete analysis from a YouTube URL
- `_orchestrate_analysis()`: Common analysis logic

### 2. **SentimentAnalyzer**
**File**: `core/analyzers/sentiment_analyzer.py`

**Responsibilities**:
- Polarity analysis using Transformer models
- Batch and individual comment processing
- Calculation of confidence scores
- Calibration of classification thresholds

**Main Functions**:
- `analyze_batch()`: Batch analysis of comments
- `analyze_single()`: Single analysis
- `get_confidence_scores()`: Confidence metrics

### 3. **AspectAnalyzer**
**File**: `core/analyzers/aspect_analyzer.py`

**Responsibilities**:
- Automatic identification of recurring aspects/themes
- Semantic clustering of comments
- Classification of comments by identified aspect
- Extraction of representative keywords per aspect

**Main Functions**:
- `extract_aspects()`: Identification of main themes
- `cluster_by_aspects()`: Grouping of comments
- `identify_aspect_keywords()`: Keywords per aspect

### 4. **YouTubeScraper**
**File**: `core/scrapers/youtube_scraper.py`

**Responsibilities**:
- Extraction of comments without an API key
- Validation and processing of YouTube URLs
- Handling of pagination and comment limits
- Management of scraping-specific errors

**Main Functions**:
- `extract_video_id()`: Parsing of URL to video ID
- `get_comments()`: Downloading comments
- `validate_youtube_url()`: Validation of URLs

### 5. **TextPreprocessor**
**File**: `core/analyzers/text_preprocessor.py`

**Responsibilities**:
- Text cleaning and normalization
- Filtering of spam and irrelevant comments
- Automatic language detection
- Specialized tokenization for analysis

**Main Functions**:
- `clean_text()`: Basic cleaning
- `normalize_text()`: Advanced normalization
- `remove_spam_comments()`: Spam filtering

---

## Libraries and Dependencies

### **Web Framework**
- **django**: Main web framework
- **djangorestframework**: REST API construction
- **django-cors-headers**: CORS handling for frontend
- **drf-spectacular**: Automatic API documentation

### **Natural Language Processing**
- **transformers**: BERT/RoBERTa models for sentiment analysis
- **torch**: Backend for Transformer models
- **sentence-transformers**: Semantic embeddings for clustering
- **scikit-learn**: Clustering and classification algorithms
- **spacy**: Advanced text processing and tokenization
- **nltk**: Complementary NLP utilities

### **YouTube Scraping**
- **youtube-comment-downloader**: Comment extraction without an API key
- **requests**: Additional HTTP requests
- **beautifulsoup4**: HTML parsing if necessary

### **Utilities and Processing**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical operations and arrays
- **python-dotenv**: Environment variable management
- **pydantic**: Robust data validation

### **Development and Testing**
- **pytest**: Testing framework
- **pytest-django**: Pytest integration with Django
- **black**: Automatic code formatting
- **flake8**: Code linting

---

## API Endpoints

### **Direct Analysis**
```
POST /api/analysis/comments/
Content-Type: application/json

Body: {
    "comments": ["comment 1", "comment 2", ...],
    "analysis_type": "sentiment_by_aspects" | "overall_sentiment" | "aspects_only",
    "max_comments": 500  // optional
}
```

### **YouTube Analysis**
```
POST /api/analysis/youtube/
Content-Type: application/json

Body: {
    "youtube_url": "https://www.youtube.com/watch?v=VIDEO_ID",
    "analysis_type": "sentiment_by_aspects" | "overall_sentiment" | "aspects_only",
    "max_comments": 500  // optional
}
```

### **Example Response**
```json
{
    "video_info": {
        "title": "Video Title",
        "url": "https://youtube.com/...",
        "comments_processed": 245
    },
    "overall_sentiment": {
        "positive": 0.65,
        "negative": 0.25,
        "neutral": 0.10
    },
    "aspects": {
        "audio": {
            "sentiment": {"positive": 0.80, "negative": 0.15, "neutral": 0.05},
            "keywords": ["sound", "quality", "music"],
            "comment_count": 45
        },
        "content": {
            "sentiment": {"positive": 0.70, "negative": 0.20, "neutral": 0.10},
            "keywords": ["information", "explanation", "topic"],
            "comment_count": 89
        }
    }
}
```