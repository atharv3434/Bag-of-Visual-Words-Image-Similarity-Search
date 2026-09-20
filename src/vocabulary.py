"""Build a visual vocabulary from pooled ORB descriptors (K-Means), and
represent each image as a histogram over that vocabulary — the core of
the Bag-of-Visual-Words (BoVW) pipeline.

Design note: ORB descriptors are binary (32 bytes, meant to be compared
with Hamming distance), but this project clusters them with ordinary
K-Means (Euclidean distance) after casting to float. This is a common,
practical simplification in BoVW tutorials/implementations — it's not the
theoretically "correct" distance for binary descriptors (K-Medoids or
K-Means with true Hamming distance would be more principled), but it works
reasonably well in practice and keeps the pipeline simple and fast with
scikit-learn's off-the-shelf KMeans. Worth knowing if you adapt this for
more demanding real-world use.
"""

import numpy as np
from sklearn.cluster import KMeans


def build_vocabulary(all_descriptors, vocabulary_size, random_state=42):
    """Pool descriptors from every training image and cluster them into
    `vocabulary_size` "visual words". Returns the fitted KMeans model.
    """
    pooled = np.vstack(all_descriptors).astype(np.float64)
    kmeans = KMeans(n_clusters=vocabulary_size, random_state=random_state, n_init=10)
    kmeans.fit(pooled)
    return kmeans


def compute_histogram(descriptors, vocabulary, vocabulary_size):
    """Assign each descriptor to its nearest visual word and build a
    normalized (term-frequency) histogram over the vocabulary.
    """
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(vocabulary_size, dtype=np.float64)

    word_ids = vocabulary.predict(descriptors.astype(np.float64))
    histogram, _ = np.histogram(word_ids, bins=np.arange(vocabulary_size + 1))
    total = histogram.sum()
    return histogram / total if total > 0 else histogram.astype(np.float64)


def compute_tfidf(histogram_matrix):
    """Apply TF-IDF weighting across a matrix of per-image term-frequency
    histograms (rows = images, columns = visual words), down-weighting
    visual words that appear in nearly every image and up-weighting ones
    that are more distinctive — exactly the same idea as TF-IDF for text,
    just with "visual words" standing in for vocabulary terms.

    Returns the TF-IDF-weighted matrix, with each row L2-normalized so
    cosine similarity between rows is well-defined.
    """
    n_images = histogram_matrix.shape[0]
    doc_freq = np.sum(histogram_matrix > 0, axis=0)
    idf = np.log((n_images + 1) / (doc_freq + 1)) + 1  # smoothed, like sklearn's TfidfVectorizer

    weighted = histogram_matrix * idf
    norms = np.linalg.norm(weighted, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return weighted / norms
