from typing import List, Dict
# Intentamos usar scikit-learn; si no está, el clusterer devolverá estructura vacía.
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    import numpy as np
    _HAS_SK = True
except Exception:
    _HAS_SK = False

def _top_terms_per_cluster(vectorizer, kmeans, top_n=5):
    terms = vectorizer.get_feature_names_out()
    # centroides -> escoger términos de mayor peso
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    keywords = []
    for i in range(kmeans.n_clusters):
        top = [terms[ind] for ind in order_centroids[i, :top_n]]
        keywords.append(top)
    return keywords

class AspectClusterer:
    def cluster(self, texts: List[str], n_clusters: int = 5) -> Dict:
        if not _HAS_SK or len(texts) < 2:
            return {
                "labels": [None]*len(texts),
                "cluster_ids": [],
                "keywords": [],
                "sizes": [],
                "n_clusters": 0
            }
        try:
            vectorizer = TfidfVectorizer(min_df=1, max_df=0.9, ngram_range=(1,2))
            X = vectorizer.fit_transform(texts)
            k = max(2, min(n_clusters, X.shape[0]))  # no más clusters que documentos
            kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
            labels = kmeans.fit_predict(X)

            keywords = _top_terms_per_cluster(vectorizer, kmeans, top_n=5)
            sizes = [int((labels == i).sum()) for i in range(k)]
            return {
                "labels": labels.tolist(),
                "cluster_ids": list(range(k)),
                "keywords": keywords,
                "sizes": sizes,
                "n_clusters": k
            }
        except Exception:
            return {
                "labels": [None]*len(texts),
                "cluster_ids": [],
                "keywords": [],
                "sizes": [],
                "n_clusters": 0
            }
