
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

df = pd.read_csv("customers.csv")

features = [
    "age", "income", "purchase_frequency",
    "avg_order_value", "recency_days", "engagement_score"
]

X = df[features]
X_scaled = StandardScaler().fit_transform(X)

# Create 4 customer segments
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["segment"] = kmeans.fit_predict(X_scaled)

# Segment profile
profile = df.groupby("segment")[features].mean().round(2)
print("\nCustomer Segment Profile:\n")
print(profile)

# PCA visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["segment"], s=80)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Customer Segments")
plt.colorbar(label="Segment")
plt.tight_layout()
plt.savefig("customer_segments.png", dpi=150)
plt.show()

df.to_csv("customers_segmented.csv", index=False)
print("\nSaved: customers_segmented.csv and customer_segments.png")
