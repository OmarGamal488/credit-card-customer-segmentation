# Customer Segmentation for Credit Card Customers

Unsupervised machine learning project that segments 8,950 credit card customers into 7 behavioural groups using K-Means clustering.

---

## Dataset

**CC GENERAL.csv** — 8,950 customers × 18 columns (1 ID + 17 numeric behavioural features) sourced from the [Kaggle Credit Card Customer Segmentation dataset](https://www.kaggle.com/code/des137/customer-segmentation-credit-cards).

Key feature groups: balance behaviour, purchase behaviour (one-off & instalment), cash-advance behaviour, credit limits, payments, and tenure.

---

## Project Structure

```
.
├── CC GENERAL.csv                  # Raw dataset
├── customer_segmentation.ipynb     # Main analysis notebook
├── OmarGamalElKady_MachineLearning2_Report.tex   # Full LaTeX report source
├── OmarGamalElKady_MachineLearning2_Report.pdf   # Compiled report (16 pages)
├── figures/                        # Generated PNG figures (fig01–fig15)
├── pyproject.toml                  # Python dependencies (managed by uv)
└── uv.lock                         # Locked dependency versions
```

---

## Pipeline Overview

| Phase | Description |
|-------|-------------|
| 1 | **Data Exploration & Preprocessing** — missing value imputation (median), log₁p transformation on 10 skewed features, PCA to 6 components (95% variance retained) |
| 2 | **Optimal k Selection** — Elbow method (k=4) + Silhouette score (k=7); k=7 chosen |
| 3 | **Segmentation** — K-Means (silhouette 0.4477) vs GMM (0.3259); K-Means selected |
| 4 | **Visualisation** — t-SNE projection, cluster heatmap, radar charts, box plots |
| 5 | **Business Recommendations** — actionable strategies per segment |

---

## Results — 7 Customer Segments

| Cluster | Segment Name | Size | % |
|---------|-------------|------|---|
| C0 | Big Spenders (VIP) | 1,633 | 18.2% |
| C1 | Cash-Only Dependents | 2,068 | 23.1% |
| C2 | Installment Shoppers | 1,895 | 21.2% |
| C3 | High-Risk Heavy Users | 938 | 10.5% |
| C4 | One-off & Cash Advance Revolvers | 798 | 8.9% |
| C5 | One-off Shoppers | 1,138 | 12.7% |
| C6 | Installment & Cash Advance Revolvers | 480 | 5.4% |

**Final silhouette score: 0.4477**

---

## Setup & Usage

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

**Install dependencies:**
```bash
uv sync
```

**Run the notebook:**
```bash
uv run jupyter notebook customer_segmentation.ipynb
```

**Regenerate all figures:**
```bash
uv run python generate_figures.py
```

---

## Dependencies

- Python 3.11
- pandas, numpy, scikit-learn, matplotlib, seaborn, scipy, plotly

