from django.urls import path
from .views import CommentsAnalysisView


urlpatterns = [
    path('analysis/comments/', CommentsAnalysisView.as_view(), name='analysis-comments'),
]
