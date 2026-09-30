# Customer Segmentation Project

## Objective
Segment customers using demographic and behavioral characteristics.

## Features
- Age
- Income
- Purchase frequency
- Average order value
- Recency of purchase
- Engagement score

## Run the project

Install dependencies:

    pip install pandas scikit-learn matplotlib

Run:

    python customer_segmentation.py

## Outputs
- `customers_segmented.csv` — customers with assigned cluster/segment
- `customer_segments.png` — PCA visualization of the segments

The project uses K-Means clustering after standardizing the numerical features.
