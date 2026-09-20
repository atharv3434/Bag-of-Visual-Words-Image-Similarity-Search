"""
Local feature extraction with ORB (Oriented FAST and Rotated BRIEF).

Unlike a global descriptor (a single vector summarizing the whole image),
ORB finds many local keypoints — corners and other distinctive points —
and describes each with a small binary descriptor. This gives Bag-of-
Visual-Words something to build a vocabulary out of: many small, reusable
"visual words" that recur across different images, rather than one
monolithic per-image fingerprint.
"""

import cv2


def extract_descriptors(image_bgr, n_features=300):
    """Returns an (n_keypoints, 32) array of ORB descriptors, or None if no
    keypoints were found (e.g. a completely blank image).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return descriptors


def extract_keypoints_and_descriptors(image_bgr, n_features=300):
    """Like extract_descriptors, but also returns the cv2.KeyPoint objects
    (needed for visualization).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors
