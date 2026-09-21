# Bag-of-Visual-Words Image Similarity Search

A content-based image retrieval project implementing the classic
**Bag-of-Visual-Words (BoVW)** 
pipeline:

- Sivic & Zisserman, *"Video Google: A Text Retrieval Approach to Object
  Matching in Videos"* (ICCV 2003)
- Csurka, Dance, Fan, Willamowski, Bray, *"Visual Categorization with Bags
  of Keypoints"* (ECCV Workshop 2004)

The idea: detect many small local features in each image (keypoints),
cluster those features across the whole dataset into a "visual
vocabulary," and represent each image as a histogram of which visual
words it contains — the same way a text document is represented as a
bag-of-words over a vocabulary of terms. This is a different (and
complementary) approach from global descriptors like a color histogram:
it captures local structure and is far less sensitive to overall image
color.

## Pipeline

1. **Local feature detection (ORB)** — find keypoints (corners, distinctive
   points) in each image and describe each with a 32-byte binary
   descriptor.
2. **Visual vocabulary (K-Means)** — pool descriptors from every image and
   cluster them into K "visual words."
3. **BoVW histogram** — for each image, assign every descriptor to its
   nearest visual word and count occurrences, producing a K-dimensional
   histogram.
4. **TF-IDF weighting** — down-weight visual words that appear in nearly
   every image, up-weight distinctive ones (exactly TF-IDF from text
   retrieval, applied to visual words).
5. **Nearest-neighbor search** — find similar images via cosine similarity
   between histograms.

## Project structure

```
bovw-image-search/
├── config.yaml                  # ORB, vocabulary, and search settings
├── requirements.txt
├── data/
│   ├── generate_data.py         # (re)generates the synthetic textured images
│   ├── images/                  # pre-generated sample images
│   └── metadata.json            # category labels (evaluation only)
├── src/
│   ├── utils.py                 # config + metadata loading
│   ├── features.py              # ORB keypoint/descriptor extraction
│   ├── vocabulary.py            # K-Means vocabulary + BoVW histograms + TF-IDF
│   ├── build_index.py           # builds the vocabulary and index over the dataset
│   ├── query.py                 # find images similar to a query image
│   └── evaluate.py              # leave-one-out precision@K evaluation
├── models/
│   └── bovw_index.joblib         # saved vocabulary + histograms
├── output/                       # query visualizations + evaluation report
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## 1. Build the index

```bash
python src/build_index.py
```

Extracts ORB descriptors for every image, builds a 60-word visual
vocabulary via K-Means, computes each image's TF-IDF-weighted histogram,
and saves everything to `models/bovw_index.joblib`.

## 2. Search

```bash
python src/query.py --image data/images/dots_022.png --top-k 5 --visualize
```

```
Top 5 matches for data/images/dots_022.png:
  1. dots_018.png             category=dots           similarity=0.924
  2. dots_015.png             category=dots           similarity=0.910
  3. dots_017.png             category=dots           similarity=0.736
  4. dots_021.png             category=dots           similarity=0.710
  5. dots_026.png             category=dots           similarity=0.692
```

With `--visualize`, saves a montage to `output/query_<file>.png`. Notice
the matches are all correctly "dots" *regardless of color* — a good sign
the vocabulary is capturing texture/structure, not just color, since ORB
descriptors only see local grayscale patches.

## 3. Evaluate

```bash
python src/evaluate.py
```

Leave-one-out precision@K across the whole dataset:

```
Precision@1: 0.967
Precision@3: 0.944
Precision@5: 0.923

Per-category precision@3:
  checkerboard   1.000
  crosshatch     1.000
  dots           0.978
  stripes        0.800
```

Stripes score lowest — makes sense, since dashed diagonal stripes at
varying angles and spacing are the most geometrically variable pattern in
the dataset, giving K-Means the least consistent local structure to
cluster.

## A real bug this project caught (and the fix)

The first version of the synthetic data generator drew stripes as
**continuous, unbroken lines**. Testing ORB on that pattern directly found
**zero keypoints** — a straight, uninterrupted line has no corners for a
corner-based detector like ORB (which relies on the FAST detector under
the hood) to find, so the entire "stripes" category would have been
unusable for BoVW. The fix was to draw stripes as short **dashes** instead
of continuous lines: every dash has two endpoints (corners), which gave
ORB real structure to detect (65–216 keypoints per image after the fix).
This is a good illustration of a broader point about keypoint-based
methods: they need actual corner/edge structure, and it's worth checking
keypoint counts on your real data before trusting a pipeline built on top
of them.

## Design notes / limitations

- **K-Means on binary descriptors**: ORB descriptors are meant for Hamming
  distance, but this project clusters them with ordinary (Euclidean)
  K-Means after casting to float — a common practical simplification, not
  the theoretically ideal approach. See the note in `vocabulary.py`.
- **Query-side normalization**: histograms in the index are TF-IDF
  weighted using statistics computed once from the training/index set.
  New query images are L2-normalized directly (an approximation) rather
  than re-weighted with the stored IDF vector — reasonable when queries
  come from a similar distribution to the indexed images, as they do here.

## Extending this project

- **SIFT instead of ORB**: swap the detector in `features.py` (SIFT is
  patent-free as of 2020 and available in modern OpenCV) for descriptors
  designed for Euclidean distance, making the K-Means clustering step more
  theoretically sound.
- **Larger/adaptive vocabulary**: tune `vocabulary_size` in `config.yaml`,
  or pick it automatically the same way `image-segmentation`'s K-Means
  step does (via silhouette score) if you adapt this to a much larger,
  more varied dataset.
- **Spatial verification**: for real applications, add a second stage that
  re-ranks the top candidates using actual keypoint geometric matching
  (e.g. RANSAC-fitted homography) rather than trusting histogram
  similarity alone — standard practice in real BoVW retrieval systems.
