"""Find the most similar images to a query image using its Bag-of-Visual-
Words histogram.

Usage:
    python src/query.py --image data/images/dots_022.png [--top-k 5] [--visualize]
"""

import argparse
import os
import sys

import cv2
import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors

sys.path.append(os.path.dirname(__file__))
from utils import load_config
from features import extract_descriptors
from vocabulary import compute_histogram


def load_index(index_path):
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"No BoVW index found at '{index_path}'. Run `python src/build_index.py` first."
        )
    return joblib.load(index_path)


def image_to_histogram(image_bgr, index):
    config = index["config"]
    descriptors = extract_descriptors(image_bgr, config["orb_n_features"])
    histogram = compute_histogram(descriptors, index["vocabulary"], config["vocabulary_size"])

    # Match the same weighting used to build the index: applying the
    # dataset's IDF vector to a new query would need the fitted IDF stored
    # separately; for simplicity (and since queries here are drawn from the
    # same kind of images) this project L2-normalizes the raw term
    # frequencies for queries, which is a very close approximation when the
    # dataset is fairly homogeneous like this one.
    norm = np.linalg.norm(histogram)
    return histogram / norm if norm > 0 else histogram


def find_similar(query_hist, index, top_k, exclude_file=None):
    matrix = index["histograms"]
    records = index["records"]
    metric = index["config"].get("distance_metric", "cosine")

    n_neighbors = min(top_k + 1, len(records))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    nn.fit(matrix)
    distances, indices = nn.kneighbors(query_hist.reshape(1, -1))

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        record = records[idx]
        if exclude_file is not None and record["file"] == exclude_file:
            continue
        similarity = 1 - dist if metric == "cosine" else -dist
        results.append({**record, "similarity": round(float(similarity), 4)})
        if len(results) == top_k:
            break
    return results


def build_montage(query_path, results, images_dir, cell_size=140):
    query_img = cv2.resize(cv2.imread(query_path), (cell_size, cell_size))
    cv2.putText(query_img, "QUERY", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    tiles = [query_img]
    for r in results:
        img = cv2.resize(cv2.imread(os.path.join(images_dir, r["file"])), (cell_size, cell_size))
        label = f"{r['category']} {r['similarity']:.2f}"
        cv2.putText(img, label, (5, cell_size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(img)

    separator = np.full((cell_size, 4, 3), 0, dtype=np.uint8)
    spaced = []
    for i, t in enumerate(tiles):
        if i > 0:
            spaced.append(separator)
        spaced.append(t)
    return np.hstack(spaced)


def main():
    parser = argparse.ArgumentParser(description="Find similar images via BoVW.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--image", required=True, help="Path to the query image")
    parser.add_argument("--top-k", type=int, help="Number of results (overrides config)")
    parser.add_argument("--visualize", action="store_true", help="Save a montage image of the results")
    args = parser.parse_args()

    config = load_config(args.config)
    top_k = args.top_k or config.get("top_k", 5)
    index = load_index(config["index_path"])

    query_img = cv2.imread(args.image)
    if query_img is None:
        raise FileNotFoundError(f"Could not read image at '{args.image}'")

    query_hist = image_to_histogram(query_img, index)
    query_filename = os.path.basename(args.image)
    results = find_similar(query_hist, index, top_k, exclude_file=query_filename)

    print(f"Top {len(results)} matches for {args.image}:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['file']:24s} category={r['category']:14s} similarity={r['similarity']:.3f}")

    if args.visualize:
        os.makedirs(config["output_dir"], exist_ok=True)
        montage = build_montage(args.image, results, config["images_dir"])
        out_path = os.path.join(config["output_dir"], f"query_{query_filename}")
        cv2.imwrite(out_path, montage)
        print(f"\nSaved visualization to {out_path}")


if __name__ == "__main__":
    main()
