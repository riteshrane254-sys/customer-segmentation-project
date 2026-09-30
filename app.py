import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("Customer Segmentation Dashboard")

df = pd.read_csv("customers.csv")

features = [
    "age",
    "income",
    "purchase_frequency",
    "avg_order_value",
    "recency_days",
    "engagement_score"
]

n_clusters = st.sidebar.slider(
    "Number of customer segments",
    min_value=2,
    max_value=8,
    value=4
)

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=10
)

df["segment"] = kmeans.fit_predict(X_scaled)

st.subheader("Customer Data")
st.dataframe(df)

st.subheader("Segment Profiles")

profile = (
    df.groupby("segment")[features]
    .mean()
    .round(2)
)

st.dataframe(profile)

st.subheader("Customer Segment Visualization")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots()

ax.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=df["segment"],
    s=80
)

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_title("Customer Segments")

st.pyplot(fig)

st.subheader("Customers per Segment")

st.bar_chart(
    df["segment"].value_counts().sort_index()
)