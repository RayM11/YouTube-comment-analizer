from rest_framework import serializers


class AnalysisOptionsSerializer(serializers.Serializer):
    num_aspect_clusters = serializers.IntegerField(required=False, min_value=2, max_value=20, default=5)


class CommentsAnalysisRequestSerializer(serializers.Serializer):
    comments = serializers.ListField(
        child=serializers.CharField(allow_blank=False, trim_whitespace=True),
        allow_empty=False
    )
    options = AnalysisOptionsSerializer(required=False)


class SentimentSerializer(serializers.Serializer):
    label = serializers.ChoiceField(choices=['positive','neutral','negative'])
    score = serializers.FloatField()


class ItemResultSerializer(serializers.Serializer):
    text = serializers.CharField()
    sentiment = SentimentSerializer()
    cluster_id = serializers.IntegerField(allow_null=True)


class AspectSummarySerializer(serializers.Serializer):
    cluster_id = serializers.IntegerField()
    keywords = serializers.ListField(child=serializers.CharField())
    size = serializers.IntegerField()


class AnalysisResponseSerializer(serializers.Serializer):
    summary = serializers.DictField()
    items = ItemResultSerializer(many=True)
