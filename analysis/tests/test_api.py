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
                "Great content, learned a lot",
                "Too long and boring",
                "Audio was low but editing was good"
            ],
            "options": {"num_aspect_clusters": 3}
        }
        resp = self.client.post(self.url, data=json.dumps(payload), content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn('summary', data)
        self.assertIn('items', data)
        self.assertIn('warnings', data)
        self.assertEqual(len(data['items']), 3)

    def test_truncation_warning(self):
        long_comment = "x" * 10000  # should exceed MAX_COMMENT_LENGTH default
        payload = {"comments": [long_comment, "short"], "options": {}}
        resp = self.client.post(self.url, data=json.dumps(payload), content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(any("truncated" in w.lower() for w in data.get("warnings", [])))

    def test_algorithm_param_fcm(self):
        payload = {
            "comments": ["a good video", "bad audio", "excellent tutorial", "boring and slow"],
            "options": {"num_aspect_clusters": 2, "clustering_algorithm": "fcm"}
        }
        resp = self.client.post(self.url, data=json.dumps(payload), content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn('summary', data)
