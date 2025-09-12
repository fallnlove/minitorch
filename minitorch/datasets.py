"""
Dataset generation functions for machine learning experiments.

This module provides various 2D binary classification datasets commonly used
for testing and demonstrating machine learning algorithms. Each dataset function
generates N random points in the unit square [0,1] x [0,1] and assigns binary
labels according to different geometric patterns.
"""
import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """
    Generate N random points in the unit square [0,1] x [0,1].

    Args:
        N: Number of points to generate

    Returns:
        List of (x, y) coordinate tuples where both x and y are in [0,1]
    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    """
    Container for a 2D binary classification dataset.

    Attributes:
        N: Number of data points
        X: List of (x, y) coordinate pairs representing input features
        y: List of binary labels (0 or 1) corresponding to each point in X
    """
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """
    Generate a simple linearly separable dataset.

    Creates a dataset where points are labeled based on their x-coordinate:
    - Label 1 if x < 0.5
    - Label 0 if x >= 0.5

    This creates a vertical decision boundary at x = 0.5, making it easily
    separable by a linear classifier.

    Args:
        N: Number of data points to generate

    Returns:
        Graph object containing the dataset with binary labels
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """
    Generate a diagonally separable dataset.

    Creates a dataset where points are labeled based on the sum of their coordinates:
    - Label 1 if x + y < 0.5
    - Label 0 if x + y >= 0.5

    This creates a diagonal decision boundary (line x + y = 0.5) from bottom-left
    to top-right, still linearly separable but requiring both features.

    Args:
        N: Number of data points to generate

    Returns:
        Graph object containing the dataset with binary labels
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """
    Generate a split dataset with two separated regions.

    Creates a dataset where points are labeled based on their x-coordinate:
    - Label 1 if x < 0.2 OR x > 0.8 (left and right regions)
    - Label 0 if 0.2 <= x <= 0.8 (middle region)

    This creates two separate regions of positive examples separated by a region
    of negative examples. Not linearly separable with a single hyperplane.

    Args:
        N: Number of data points to generate

    Returns:
        Graph object containing the dataset with binary labels
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """
    Generate an XOR (exclusive OR) dataset.

    Creates a dataset where points are labeled based on XOR logic:
    - Label 1 if (x < 0.5 AND y > 0.5) OR (x > 0.5 AND y < 0.5)
    - Label 0 otherwise

    This creates a checkerboard pattern with positive examples in the top-left
    and bottom-right quadrants. This is the classic non-linearly separable
    dataset that cannot be solved by linear classifiers.

    Args:
        N: Number of data points to generate

    Returns:
        Graph object containing the dataset with binary labels
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if ((x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5)) else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """
    Generate a circular dataset with ring-shaped decision boundary.

    Creates a dataset where points are labeled based on their distance from center:
    - Label 1 if distance from center (0.5, 0.5) > sqrt(0.1) ≈ 0.316
    - Label 0 if distance from center <= sqrt(0.1)

    This creates a circular decision boundary centered at (0.5, 0.5) with
    positive examples forming a ring around negative examples in the center.
    Requires non-linear classification.

    Args:
        N: Number of data points to generate

    Returns:
        Graph object containing the dataset with binary labels
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = (x_1 - 0.5, x_2 - 0.5)
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """
    Generate a two-spiral dataset.

    Creates a dataset with two interleaved spirals:
    - One spiral (label 0) starts from center and spirals outward
    - Another spiral (label 1) follows a similar but offset pattern

    The spirals are generated using parametric equations:
    - x(t) = t * cos(t) / 20.0
    - y(t) = t * sin(t) / 20.0

    This creates a highly non-linear decision boundary requiring sophisticated
    classifiers to separate the two classes.

    Args:
        N: Number of data points to generate (should be even for balanced classes)

    Returns:
        Graph object containing the dataset with binary labels
    """
    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
"""
Dictionary mapping dataset names to their corresponding generator functions.

Available datasets:
- "Simple": Linearly separable with vertical boundary
- "Diag": Linearly separable with diagonal boundary
- "Split": Two separated regions (not linearly separable)
- "Xor": Classic XOR pattern (not linearly separable)
- "Circle": Ring-shaped boundary (not linearly separable)
- "Spiral": Two interleaved spirals (highly non-linear)
"""
