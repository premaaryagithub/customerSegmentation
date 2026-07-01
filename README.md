# Customer Segmentation via Unsupervised Learning

K-Means clustering pipeline segmenting customers by purchase behavior and demographics, with cluster count chosen through a validated method rather than an arbitrary guess.

## Dataset

iFood marketing dataset — 2,205 customer records with income, household composition, purchase channel usage (web/catalog/store), spending by product category, and campaign response history.

## Approach

1. **Data cleaning** — dropped constant columns (`Z_CostContact`, `Z_Revenue`), checked for nulls and duplicates.
2. **Outlier handling** — IQR-based outlier detection and removal, applied only to genuinely continuous features (Income, Recency, spend totals, purchase counts, Age). Retained **2,095 of 2,205 records** (removed 110, ~5%).

   > **Fix applied vs. the original notebook:** the original outlier-removal loop ran across *all* numeric columns, including one-hot-encoded binary flags (`marital_*`, `education_*`, `AcceptedCmp*`). IQR-based outlier detection doesn't make sense on 0/1 flag columns, and applying it there compounded across features to destroy 85% of the dataset (2,205 → 321 rows) with no signal gained. Restricting outlier removal to actual continuous columns is the statistically correct approach and is what's reflected in the results below.

3. **Feature engineering** — combined `Kidhome` + `Teenhome` into a single `Child` count feature; scaled all clustering features with `StandardScaler`.
4. **Choosing k** — ran K-Means for k=1 to 10, evaluating both the elbow method (WCSS) and **silhouette score** rather than picking a number by eye alone. Selected **k=4** via elbow inspection.

## Results (actual run output)

| k | Silhouette Score |
|---|---|
| 2 | 0.318 |
| 3 | 0.264 |
| **4 (selected)** | **0.194** |
| 5 | 0.176 |
| 6 | 0.181 |

> Note: k=2 technically scores highest on silhouette alone, but k=4 was selected via the elbow method because it produces more actionable, business-interpretable segments than a 2-way split. This tradeoff — statistical optimum vs. business utility — is worth being able to explain in an interview.

**Cluster sizes:** 629 / 547 / 503 / 416

**Cluster profiles (mean values):**

| Cluster | Avg Income | Avg Total Spend | Avg Age | Profile |
|---|---|---|---|---|
| 1 | $75,728 | $1,268.70 | 51.9 | High-value customers |
| 2 | $58,954 | $742.20 | 54.5 | Mid-high spenders |
| 3 | $41,470 | $102.70 | 56.3 | Older, low-spend |
| 0 | $30,952 | $79.60 | 43.8 | Budget-conscious, younger |

## Tech Stack

Python, pandas, NumPy, scikit-learn (KMeans, StandardScaler, silhouette_score)

## How to Run

```bash
pip install pandas numpy scikit-learn
python customersegmentation.py
```

The dataset (`ifood_df.csv`) is already committed in this repo, so this runs out of the box with no external downloads.

## Possible Extensions

- Try `GaussianMixture` for soft clustering and compare against K-Means's hard assignments.
- Add PCA for 2D visualization of clusters.
- Build a simple lookup function that assigns a new customer record to a cluster in real time.
