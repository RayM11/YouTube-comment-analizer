from typing import List, Dict
from django.conf import settings

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics import pairwise_distances
    import numpy as np
    _HAS_SK = True
except Exception:
    _HAS_SK = False

# Optional fuzzy C-means
try:
    import skfuzzy as fuzz
    _HAS_FCM = True
except Exception:
    _HAS_FCM = False

# Optional BERTopic
try:
    from bertopic import BERTopic
    _HAS_BERTOPIC = True
except Exception:
    _HAS_BERTOPIC = False


def _top_terms_from_centers(feature_names, centers, top_n=5):
    """Pick top-n weighted terms per center row."""
    keywords = []
    for i in range(centers.shape[0]):
        row = centers[i]
        top_idx = np.argsort(row)[::-1][:top_n]
        keywords.append([feature_names[j] for j in top_idx if row[j] > 0])
    return keywords


def _kmeans_cluster(texts: List[str], n_clusters: int) -> Dict:
    vectorizer = TfidfVectorizer(min_df=1, max_df=0.9, ngram_range=(1,2))
    X = vectorizer.fit_transform(texts)
    k = max(2, min(n_clusters, X.shape[0]))
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    labels = kmeans.fit_predict(X)

    feature_names = vectorizer.get_feature_names_out()
    keywords = _top_terms_from_centers(feature_names, kmeans.cluster_centers_, top_n=5)
    sizes = [int((labels == i).sum()) for i in range(k)]

    # Representatives: closest to centroids (Euclidean in TF-IDF space)
    representatives = []
    for i in range(k):
        idxs = np.where(labels == i)[0]
        if len(idxs) == 0:
            representatives.append([])
            continue
        center = kmeans.cluster_centers_[i].reshape(1, -1)
        dists = pairwise_distances(X[idxs], center, metric='euclidean').ravel()
        order = np.argsort(dists)[:settings.CLUSTER_REPRESENTATIVES]
        reps = [texts[int(idxs[j])] for j in order]
        representatives.append(reps)

    return {
        "labels": labels.tolist(),
        "cluster_ids": list(range(k)),
        "keywords": keywords,
        "sizes": sizes,
        "n_clusters": k,
        "representatives": representatives
    }


def _fcm_cluster(texts: List[str], n_clusters: int) -> Dict:
    if not (_HAS_SK and _HAS_FCM) or len(texts) < 2:
        # Fallback to KMeans if libs missing
        return _kmeans_cluster(texts, n_clusters)

    # TF-IDF + dimensionality reduction for FCM stability
    vectorizer = TfidfVectorizer(min_df=1, max_df=0.9, ngram_range=(1,2))
    x = vectorizer.fit_transform(texts)
    n = x.shape[0]
    if n < 2:
        return {
            "labels": [0]*n,
            "cluster_ids": [0],
            "keywords": [[]],
            "sizes": [n],
            "n_clusters": 1,
            "representatives": [texts]
        }
    k = max(2, min(n_clusters, n))

    dim = min(100, max(2, min(x.shape[0]-1, x.shape[1]-1)))
    svd = TruncatedSVD(n_components=dim, random_state=42)
    x_red = svd.fit_transform(x)             # shape: (n_samples, dim)
    data = x_red.T                           # shape: (dim, n_samples) for skfuzzy

    # Run FCM
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        data, c=k, m=2.0, error=0.005, maxiter=1000, init=None
    )
    # Hard labels from memberships
    labels = np.argmax(u, axis=0)            # shape: (n_samples,)

    # Keywords via weighted centroids in TF-IDF space
    # centroid_j = (sum_i u[j,i] * X[i]) / sum_i u[j,i]
    weights_sum = u.sum(axis=1) + 1e-9       # (k,)
    centers_tfidf = []
    for j in range(k):
        w = u[j, :].reshape(-1, 1)           # (n_samples, 1)
        # Weighted sum in TF-IDF space using sparse ops
        weighted = (w.T @ x).A1 / float(weights_sum[j])  # dense 1D
        centers_tfidf.append(weighted)
    centers_tfidf = np.vstack(centers_tfidf) # (k, n_features)

    feature_names = vectorizer.get_feature_names_out()
    keywords = _top_terms_from_centers(feature_names, centers_tfidf, top_n=5)

    sizes = [int((labels == i).sum()) for i in range(k)]

    # Representatives: closest to FCM centers in reduced space
    representatives = []
    for i in range(k):
        idxs = np.where(labels == i)[0]
        if len(idxs) == 0:
            representatives.append([])
            continue
        center = cntr[i].reshape(1, -1)      # (1, dim)
        dists = np.linalg.norm(x_red[idxs] - center, axis=1)
        order = np.argsort(dists)[:settings.CLUSTER_REPRESENTATIVES]
        reps = [texts[int(idxs[j])] for j in order]
        representatives.append(reps)

    return {
        "labels": labels.tolist(),
        "cluster_ids": list(range(k)),
        "keywords": keywords,
        "sizes": sizes,
        "n_clusters": k,
        "representatives": representatives
    }


def _bertopic_cluster(texts: List[str]) -> Dict:
    if not _HAS_BERTOPIC or len(texts) < 2:
        # Fallback to KMeans
        return _kmeans_cluster(texts, n_clusters=5)

    # Fit BERTopic (default multilingual config)
    topic_model = BERTopic(verbose=False)
    topics, _ = topic_model.fit_transform(texts)

    # Map arbitrary topic IDs to consecutive cluster_ids
    unique_topics = [t for t in sorted(set(topics)) if t != -1]
    if not unique_topics:
        return {
            "labels": [None]*len(texts),
            "cluster_ids": [],
            "keywords": [],
            "sizes": [],
            "n_clusters": 0,
            "representatives": []
        }
    id_map = {t: i for i, t in enumerate(unique_topics)}
    labels = [id_map.get(t, None) for t in topics]

    # Keywords per topic
    keywords = []
    sizes = []
    representatives = []
    for t in unique_topics:
        # Top terms
        topic_terms = topic_model.get_topic(t) or []
        kw = [term for term, _ in topic_terms[:5]]
        keywords.append(kw)

        # Size
        idxs = [i for i, tt in enumerate(topics) if tt == t]
        sizes.append(len(idxs))

        # Representatives: use representative docs if available
        reps = []
        try:
            reps = topic_model.get_representative_docs(t)[:settings.CLUSTER_REPRESENTATIVES]
        except Exception:
            reps = [texts[i] for i in idxs[:settings.CLUSTER_REPRESENTATIVES]]
        representatives.append(reps)

    return {
        "labels": labels,
        "cluster_ids": list(range(len(unique_topics))),
        "keywords": keywords,
        "sizes": sizes,
        "n_clusters": len(unique_topics),
        "representatives": representatives
    }


class AspectClusterer:
    """Cluster comments into aspects with selectable algorithm.
    - 'kmeans' (default): TF-IDF + KMeans
    - 'fcm'            : TF-IDF (+ SVD) + Fuzzy C-Means (if scikit-fuzzy available)
    - 'bertopic'       : BERTopic (if installed)
    Returns labels, keywords, sizes, representatives.
    """
    def __init__(self, algorithm='kmeans'):
        self.algorithm = algorithm

    def cluster(self, texts: List[str], n_clusters: int = 5) -> Dict:
        if not _HAS_SK or len(texts) < 2:
            return {
                "labels": [None]*len(texts),
                "cluster_ids": [],
                "keywords": [],
                "sizes": [],
                "n_clusters": 0,
                "representatives": []
            }
        try:
            if self.algorithm == 'kmeans':
                return _kmeans_cluster(texts, n_clusters=n_clusters)
            elif self.algorithm == 'fcm':
                return _fcm_cluster(texts, n_clusters=n_clusters)
            elif self.algorithm == 'bertopic':
                # n_clusters is ignored by BERTopic (determines topics automatically)
                return _bertopic_cluster(texts)
            else:
                # Unknown algorithm -> fallback
                return _kmeans_cluster(texts, n_clusters=n_clusters)
        except Exception:
            return {
                "labels": [None]*len(texts),
                "cluster_ids": [],
                "keywords": [],
                "sizes": [],
                "n_clusters": 0,
                "representatives": []
            }
