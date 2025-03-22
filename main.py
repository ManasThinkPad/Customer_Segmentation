import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load Dataset
file_path = "/Users/manasmulchandani/Desktop/M/Customer Segmentation/Online Retail.xlsx"
df = pd.read_excel(file_path, engine='openpyxl')

# Data Cleaning
df = df.dropna()
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Feature Engineering: RFM (Recency, Frequency, Monetary)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
latest_date = df['InvoiceDate'].max()
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (latest_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',  # Frequency
    'TotalPrice': 'sum'  # Monetary
}).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'})

# Standardizing Data
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(rfm_scaled)
rfm['Cluster_KMeans'] = kmeans_labels

# DBSCAN Clustering
dbscan = DBSCAN(eps=1.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(rfm_scaled)
rfm['Cluster_DBSCAN'] = dbscan_labels

# PCA for Visualization
pca = PCA(n_components=2)
rfm_pca = pca.fit_transform(rfm_scaled)
rfm['PCA1'] = rfm_pca[:, 0]
rfm['PCA2'] = rfm_pca[:, 1]

# Visualization
plt.figure(figsize=(10, 5))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster_KMeans', data=rfm, palette='viridis', s=100)
plt.title('K-Means Clustering Results')
plt.show()

plt.figure(figsize=(10, 5))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster_DBSCAN', data=rfm, palette='coolwarm', s=100)
plt.title('DBSCAN Clustering Results')
plt.show()

# Model Evaluation
sil_score = silhouette_score(rfm_scaled, kmeans.labels_)
print(f'Silhouette Score: {sil_score:.4f}')

wcss = kmeans.inertia_
print(f'WCSS: {wcss:.4f}')

print("Cluster Distribution:")
print(rfm['Cluster_KMeans'].value_counts())

# Elbow Method for Optimal K
wcss_values = []
for i in range(2, 10):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    wcss_values.append(kmeans.inertia_)

plt.plot(range(2, 10), wcss_values, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# Business Insights and Recommendations
cluster_descriptions = {
    0: "High-value customers: Frequent buyers with high spending.",
    1: "Recent buyers with moderate spending.",
    2: "Irregular customers: Occasional buyers with average spending.",
    3: "Inactive customers: Low engagement and spending."
}

marketing_strategies = {
    0: "Offer exclusive discounts, personalized recommendations, and loyalty rewards.",
    1: "Send follow-up emails with promotional offers to encourage repeat purchases.",
    2: "Target with seasonal sales campaigns and engagement-driven content.",
    3: "Reactivate with special offers, reminders, and re-engagement campaigns."
}

final_results = []
for cluster in sorted(rfm['Cluster_KMeans'].unique()):
    cluster_summary = {
        "Cluster": int(cluster),
        "Description": cluster_descriptions.get(cluster, "Unknown Segment"),
        "Statistics": rfm[rfm['Cluster_KMeans'] == cluster].describe().to_dict(),
        "Recommended Strategy": marketing_strategies.get(cluster, "No specific strategy.")
    }
    final_results.append(cluster_summary)
    print(f"Cluster {cluster} Summary: {cluster_summary['Description']}")
    print(rfm[rfm['Cluster_KMeans'] == cluster].describe())
    print(f"Recommended Strategy: {cluster_summary['Recommended Strategy']}")
    print("\n")

# Save results to a JSON file
with open("customer_segmentation_results.json", "w") as json_file:
    json.dump(final_results, json_file, indent=4)
