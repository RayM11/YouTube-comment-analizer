from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings

from .serializers import (
    CommentsAnalysisRequestSerializer,
    AnalysisResponseSerializer
)
from .services.text_preprocess import preprocess_batch
from .services.sentiment import SentimentAnalyzer
from .services.aspects import AspectClusterer


class CommentsAnalysisView(APIView):
    """POST /api/analysis/comments/

    Cuerpo:

    {

      "comments": ["texto1", "texto2", ...],

      "options": {"num_aspect_clusters": 5}

    }

    Respuesta: ver AnalysisResponseSerializer

    """
    def post(self, request):
        serializer = CommentsAnalysisRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        payload = serializer.validated_data

        comments = payload['comments']
        options = payload.get('options', {})

        cleaned = preprocess_batch(comments)

        sentiment_analyzer = SentimentAnalyzer(use_transformers=settings.USE_TRANSFORMERS)
        sentiments = sentiment_analyzer.batch_predict(cleaned)

        clusterer = AspectClusterer()
        clusters = clusterer.cluster(cleaned, n_clusters=options.get('num_aspect_clusters', 5))
        # clusters: dict with 'labels', 'keywords', 'sizes'

        items = []
        for i, original_text in enumerate(comments):
            items.append({
                "text": original_text,
                "sentiment": sentiments[i],
                "cluster_id": int(clusters['labels'][i]) if clusters['labels'] is not None else None
            })

        summary = {
            "n_comments": len(comments),
            "n_clusters": clusters.get('n_clusters', 0),
            "aspects": [
                {"cluster_id": int(cid), "keywords": kw, "size": int(sz)}
                for cid, kw, sz in zip(clusters.get('cluster_ids', []),
                                       clusters.get('keywords', []),
                                       clusters.get('sizes', []))
            ]
        }

        response = {"summary": summary, "items": items}
        out = AnalysisResponseSerializer(response)
        return Response(out.data, status=status.HTTP_200_OK)
