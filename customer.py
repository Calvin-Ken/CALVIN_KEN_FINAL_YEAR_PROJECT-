import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    return pd.DataFrame()

def preprocess_data(data):
    data['CustomerID'] = data['CustomerID'].astype(str)
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    data['Monetary'] = data['Quantity'] * data['UnitPrice']
    data.dropna(subset=['InvoiceDate', 'CustomerID', 'Monetary'], inplace=True)
    return data

def calculate_rfm(data):
    latest_date = data['InvoiceDate'].max() + pd.DateOffset(days=1)
    rfm = data.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (latest_date - x.max()).days,
        'InvoiceNo': 'count',
        'Monetary': 'sum'
    }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency'})
    return rfm

def apply_kmeans(data, n_clusters=3):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(data[['Recency', 'Frequency', 'Monetary']])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(rfm_scaled)
    return data

def plot_clusters(data):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    sns.scatterplot(data=data, x='Recency', y='Frequency', hue='Cluster', ax=ax[0], palette='viridis')
    sns.scatterplot(data=data, x='Frequency', y='Monetary', hue='Cluster', ax=ax[1], palette='viridis')
    sns.scatterplot(data=data, x='Recency', y='Monetary', hue='Cluster', ax=ax[2], palette='viridis')
    st.pyplot(fig)

def display_segment_insights(data):
    st.write("Segment Insights")
    for cluster in sorted(data['Cluster'].unique()):
        st.write(f"**Cluster {cluster} Insights**")
        cluster_data = data[data['Cluster'] == cluster]
        st.write(f"Average Recency: {cluster_data['Recency'].mean():.2f} days")
        st.write(f"Average Frequency: {cluster_data['Frequency'].mean():.2f}")
        st.write(f"Average Monetary: ${cluster_data['Monetary'].mean():.2f}")

def main():
    st.title('Customer Segmentation Dashboard')
    data = load_data()
    if not data.empty:
        data = preprocess_data(data)
        rfm_data = calculate_rfm(data)
        rfm_data = apply_kmeans(rfm_data)
        st.write("Data Overview", rfm_data.head())
        plot_clusters(rfm_data)
        display_segment_insights(rfm_data)
    else:
        st.info("Please upload data to begin analysis.")

if __name__ == "__main__":
    main()
