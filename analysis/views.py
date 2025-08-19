from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings

from .serializers import CommentsAnalysisRequestSerializer
from .services.text_preprocess import preprocess_batch
from .services.sentiment import SentimentAnalyzer
from .services.aspects import AspectClusterer


class CommentsAnalysisView(APIView):
    """Analyze a list of comments.
    POST /api/analysis/comments/
    Body:
      {
        "comments": [...],
        "options": {
          "num_aspect_clusters": 5,
          "use_transformers": true,
          "clustering_algorithm": "kmeans"
        }
      }
    Response includes summary, items, and warnings.
    """
    def post(self, request):
        serializer = CommentsAnalysisRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        payload = serializer.validated_data

        comments = payload['comments']
        options = payload.get('options', {})

        warnings = []

        # Validate number of comments
        if len(comments) > settings.MAX_COMMENTS:
            comments = comments[:settings.MAX_COMMENTS]
            warnings.append("Too many comments provided; truncated to MAX_COMMENTS.")

        # Validate per-comment length (truncate, do not reject)
        limited_comments = []
        truncated_count = 0
        for c in comments:
            if len(c) > settings.MAX_COMMENT_LENGTH:
                limited_comments.append(c[:settings.MAX_COMMENT_LENGTH])
                truncated_count += 1
            else:
                limited_comments.append(c)
        if truncated_count:
            warnings.append(f"{truncated_count} comments were truncated due to MAX_COMMENT_LENGTH.")

        # 1) Preprocess
        cleaned = preprocess_batch(limited_comments)

        # 2) Sentiment (default try Transformers)
        use_transformers = options.get('use_transformers', True)
        sentiment_analyzer = SentimentAnalyzer(
            use_transformers=use_transformers,
            model_name=settings.SENTIMENT_MODEL
        )
        sentiments = sentiment_analyzer.batch_predict(cleaned)

        # 3) Aspects (algorithm param)
        clustering_algorithm = options.get('clustering_algorithm', 'kmeans')
        clusterer = AspectClusterer(algorithm=clustering_algorithm)
        clusters = clusterer.cluster(
            cleaned,
            n_clusters=options.get('num_aspect_clusters', 3)
        )

        # Compose response
        items = []
        for i, original_text in enumerate(limited_comments):
            label = clusters['labels'][i]
            items.append({
                "text": original_text,
                "sentiment": sentiments[i],
                "cluster_id": int(label) if label is not None else None
            })

        summary = {
            "n_comments": len(limited_comments),
            "n_clusters": clusters.get('n_clusters', 0),
            "aspects": [
                {
                    "cluster_id": int(cid),
                    "keywords": kw,
                    "size": int(sz),
                    "representatives": reps
                }
                for cid, kw, sz, reps in zip(clusters.get('cluster_ids', []),
                                             clusters.get('keywords', []),
                                             clusters.get('sizes', []),
                                             clusters.get('representatives', []))
            ]
        }

        response = {"summary": summary, "items": items, "warnings": warnings}
        return Response(response, status=status.HTTP_200_OK)
