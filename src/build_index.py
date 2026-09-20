"""Build the full BoVW index: extract ORB descriptors for every image,
cluster them into a visual vocabulary, compute each image's (optionally
TF-IDF-weighted) visual-word histogram, and save everything to disk.

Usage:
    python src/build_index.py [--config config.yaml]
"""

import argparse
import os
import sys

import cv2
import joblib
import numpy as np

sys.path.append(os.path.dirname(__file__))
from utils import load_config, load_metadata
from features import extract_descriptors
from vocabulary import build_vocabulary, compute_histogram, compute_tfidf


def main():
    parser = argparse.ArgumentParser(description="Build the BoVW similarity search index.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    metadata = load_metadata(config["metadata_path"])
    records = metadata["images"]

    print(f"Extracting ORB descriptors for {len(records)} images...")
    all_descriptors = []
    per_image_descriptors = []
    kept_records = []

    for record in records:
        path = os.path.join(config["images_dir"], record["file"])
        image = cv2.imread(path)
        if image is None:
            print(f"  Warning: could not read {path}, skipping.")
            continue
        descriptors = extract_descriptors(image, config["orb_n_features"])
        if descriptors is None or len(descriptors) == 0:
            print(f"  Warning: no keypoints found in {record['file']}, skipping.")
            continue
        all_descriptors.append(descriptors)
        per_image_descriptors.append(descriptors)
        kept_records.append(record)

    total_descriptors = sum(len(d) for d in all_descriptors)
    print(f"Pooled {total_descriptors} descriptors from {len(kept_records)} images.")

    print(f"Building visual vocabulary (K={config['vocabulary_size']}) via K-Means...")
    vocabulary = build_vocabulary(all_descriptors, config["vocabulary_size"], config["vocabulary_random_state"])

    print("Computing per-image visual-word histograms...")
    histograms = np.vstack([
        compute_histogram(desc, vocabulary, config["vocabulary_size"])
        for desc in per_image_descriptors
    ])

    if config.get("use_tfidf", True):
        print("Applying TF-IDF weighting...")
        histograms = compute_tfidf(histograms)
    else:
        norms = np.linalg.norm(histograms, axis=1, keepdims=True)
        norms[norms == 0] = 1
        histograms = histograms / norms

    os.makedirs(os.path.dirname(config["index_path"]), exist_ok=True)
    joblib.dump({
        "vocabulary": vocabulary,
        "histograms": histograms,
        "records": kept_records,
        "config": config,
    }, config["index_path"])

    print(f"\nIndex built: {len(kept_records)} images, {config['vocabulary_size']}-dim BoVW histograms.")
    print(f"Saved to {config['index_path']}")


if __name__ == "__main__":
    main()
