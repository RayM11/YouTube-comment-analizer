import json
from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient

class CommentsAnalysisAPITests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.url = reverse('analysis-comments')

    def test_basic_request(self):
        payload = {
            "comments": [
                "Excelente contenido, aprendí mucho",
                "Muy largo y aburrido",
                "El audio estaba bajo pero la edición bien"
            ],
            "options": {"num_aspect_clusters": 3}
        }
        resp = self.client.post(self.url, data=json.dumps(payload), content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn('summary', data)
        self.assertIn('items', data)
        self.assertEqual(len(data['items']), 3)
        # Cada item debe tener sentiment y cluster_id (puede ser None si no hay sklearn)
        for it in data['items']:
            self.assertIn('sentiment', it)
            self.assertIn('label', it['sentiment'])
