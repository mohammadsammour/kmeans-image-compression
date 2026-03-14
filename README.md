# kmeans-image-compression
From-scratch K-Means clustering implementation in Python using only NumPy. Covers centroid assignment, mean recomputation, and random initialization. Applied to real-world image compression, reducing a 24-bit RGB image to 16 colors and achieving a ~6× reduction in storage size.

# K-Means Clustering & Image Compression

A from-scratch implementation of the K-Means clustering algorithm in Python, with a practical application: **compressing images by reducing their color palette using unsupervised learning**.

---
Project Ended: 14/3/2026
## Table of Contents

- [Overview](#overview)
- [How K-Means Works](#how-k-means-works)
- [Project Structure](#project-structure)
- [Implementation Details](#implementation-details)
  - [1. Finding Closest Centroids](#1-finding-closest-centroids)
  - [2. Computing Centroid Means](#2-computing-centroid-means)
  - [3. Random Initialization](#3-random-initialization)
  - [4. Image Compression](#4-image-compression)
- [Results](#results)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)

---

## Overview

This project implements the K-Means algorithm from the ground up using only NumPy, without relying on any machine learning libraries. The algorithm is first validated on a synthetic dataset, then applied to a real-world task: compressing a bird image from **24-bit RGB** down to a **4-bit indexed** representation using only 16 colors — achieving roughly a **6× compression ratio**.

---

## How K-Means Works

K-Means is an unsupervised learning algorithm that groups data into *K* cohesive clusters. Starting from randomly initialized centroids, it repeatedly alternates between two steps:

1. **Cluster Assignment** — assign each data point to its nearest centroid.
2. **Centroid Update** — recompute each centroid as the mean of all points assigned to it.

These steps repeat until convergence (centroids stop moving).

$$c^{(i)} := \arg\min_j \| x^{(i)} - \mu_j \|^2$$

$$\mu_k = \frac{1}{|C_k|} \sum_{i \in C_k} x^{(i)}$$

---

## Project Structure

```
.
├── Kmeans.ipynb       # Main notebook with full implementation and experiments
├── bird.webp          # Sample image used for compression
└── README.md
```

---

## Implementation Details

### 1. Finding Closest Centroids

```python
def find_closest_centroids(X, centroids):
```

For each data point in `X`, computes the squared L2 distance to every centroid and returns the index of the nearest one. Returns a 1D array `idx` where `idx[i]` holds the centroid index for example `i`.

---

### 2. Computing Centroid Means

```python
def compute_centroids(X, idx, K):
```

Recomputes the position of each centroid as the mean of all points currently assigned to it. Handles empty clusters gracefully by leaving the centroid unchanged.

---

### 3. Random Initialization

```python
def kMeans_init_centroids(X, K):
```

Initializes `K` centroids by randomly selecting `K` distinct data points from the dataset (without replacement). Running the algorithm multiple times with different initializations helps avoid poor local optima.

---

### 4. Image Compression

The compression pipeline works as follows:

| Step | Description |
|------|-------------|
| Load image | Read `bird.webp` as a NumPy array of shape `(H, W, 3)` |
| Normalize | Scale pixel values to the range `[0, 1]` |
| Reshape | Flatten to an `(H×W, 3)` matrix — one row per pixel |
| Cluster | Run K-Means with `K=16` to find 16 representative colors |
| Compress | Replace each pixel with its nearest centroid color |
| Reconstruct | Reshape back to `(H, W, 3)` for display |

**Storage comparison:**

| Representation | Bits per pixel | Total bits |
|---|---|---|
| Original (24-bit RGB) | 24 | 480 × 640 × 24 = **7,372,800** |
| Compressed (16 colors) | 4 + dictionary overhead | 16×24 + 480×640×4 = **1,229,184** |
| **Compression ratio** | | **~6×** |

---

## Results

After running K-Means with `K=16` and `max_iters=10`, the algorithm learns a 16-color palette from the image pixels. The compressed image is visually close to the original while using a fraction of the storage.

> Side-by-side comparison is generated at the end of the notebook.

---

## Getting Started

1. **Clone the repository**

```bash
git clone https://github.com/your-username/kmeans-image-compression.git
cd kmeans-image-compression
```

2. **Install dependencies**

```bash
pip install numpy matplotlib
```

3. **Run the notebook**

```bash
jupyter notebook Kmeans.ipynb
```

Make sure `bird.webp` is in the same directory as the notebook before running the image compression section.

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `numpy` | Array operations and linear algebra |
| `matplotlib` | Visualization of clusters and images |
| `jupyter` | Interactive notebook environment |

---

## Key Concepts Demonstrated

- Unsupervised learning with K-Means
- Vectorized distance computation using NumPy
- Effect of random centroid initialization on clustering results
- Practical application of clustering to image quantization and compression
