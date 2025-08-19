from rest_framework import serializers


class AnalysisOptionsSerializer(serializers.Serializer):
    # Number of aspect clusters (only used by some algorithms)
    num_aspect_clusters = serializers.IntegerField(required=False, min_value=2, max_value=50, default=5)
    # Whether to try Transformers in this request (defaults to True)
    use_transformers = serializers.BooleanField(required=False, default=True)
    # Clustering algorithm choice
    clustering_algorithm = serializers.ChoiceField(
        choices=['kmeans', 'fcm', 'bertopic'],
        required=False,
        default='kmeans'
    )


class CommentsAnalysisRequestSerializer(serializers.Serializer):
    comments = serializers.ListField(
        child=serializers.CharField(allow_blank=False, trim_whitespace=True),
        allow_empty=False
    )
    options = AnalysisOptionsSerializer(required=False)
