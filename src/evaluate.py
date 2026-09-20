"""
Evaluate BoVW retrieval quality with leave-one-out precision@K.

Usage:
    python src/evaluate.py [--config config.yaml]
"""

import argparse
import os
import sys

import cv2

sys.path.append(os.path.dirname(__file__))
from utils import load_config
from query import load_index, find_similar, build_montage


def precision_at_k(index, k):
    matrix = index["histograms"]
    records = index["records"]

    hits = total = 0
    for i, record in enumerate(records):
        results = find_similar(matrix[i], index, top_k=k, exclude_file=record["file"])
        hits += sum(1 for r in results if r["category"] == record["category"])
        total += len(results)
    return hits / total if total else 0.0


def per_category_precision(index, k):
    matrix = index["histograms"]
    records = index["records"]
    categories = sorted(set(r["category"] for r in records))

    scores = {}
    for cat in categories:
        hits = total = 0
        for i, record in enumerate(records):
            if record["category"] != cat:
                continue
            results = find_similar(matrix[i], index, top_k=k, exclude_file=record["file"])
            hits += sum(1 for r in results if r["category"] == record["category"])
            total += len(results)
        scores[cat] = round(hits / total, 3) if total else 0.0
    return scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate BoVW retrieval quality.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    index = load_index(config["index_path"])
    records = index["records"]
    print(f"Loaded index with {len(records)} images, vocabulary size {config['vocabulary_size']}.\n")

    ks = config.get("eval_top_k", [1, 3, 5])
    overall = {}
    for k in ks:
        overall[k] = precision_at_k(index, k)
        print(f"Precision@{k}: {overall[k]:.3f}")

    best_k = ks[len(ks) // 2]
    print(f"\nPer-category precision@{best_k}:")
    per_cat = per_category_precision(index, best_k)
    for cat, score in per_cat.items():
        print(f"  {cat:14s} {score:.3f}")

    n_examples = config.get("n_example_queries", 4)
    os.makedirs(config["output_dir"], exist_ok=True)
    print(f"\nSaving {n_examples} example query visualizations...")
    for record in records[:n_examples]:
        query_path = os.path.join(config["images_dir"], record["file"])
        i = records.index(record)
        results = find_similar(index["histograms"][i], index, top_k=config.get("top_k", 5), exclude_file=record["file"])
        montage = build_montage(query_path, results, config["images_dir"])
        cv2.imwrite(os.path.join(config["output_dir"], f"example_{record['file']}"), montage)

    report_lines = [
        "# BoVW Image Similarity Search — Evaluation Report",
        "",
        "Leave-one-out evaluation: for every image, query the index (excluding "
        "itself) and check whether the retrieved neighbors share its texture "
        "category (stripes / dots / checkerboard / crosshatch).",
        "",
        "## Precision@K (overall)",
        "",
        "| K | Precision |",
        "|---|---|",
    ]
    for k in ks:
        report_lines.append(f"| {k} | {overall[k]:.3f} |")

    report_lines += ["", f"## Per-category precision@{best_k}", "", "| Category | Precision |", "|---|---|"]
    for cat, score in per_cat.items():
        report_lines.append(f"| {cat} | {score:.3f} |")

    report_lines += ["", "## Example queries", ""]
    for record in records[:n_examples]:
        report_lines.append(f"**Query category: {record['category']}**")
        report_lines.append(f"![{record['file']}](example_{record['file']})")
        report_lines.append("")

    report_path = os.path.join(config["output_dir"], "evaluation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
