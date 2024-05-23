import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import base64

def load_data():
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    return pd.DataFrame()

def preprocess_data(data):
    try:
        data['CustomerID'] = data['CustomerID'].astype(str)
        data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
        data['Monetary'] = data['Quantity'] * data['UnitPrice']
        data.dropna(subset=['InvoiceDate', 'CustomerID', 'Monetary'], inplace=True)
        return data
    except KeyError as e:
        st.error(f"Missing required column: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return pd.DataFrame()

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

def plot_pie_chart(data):
    cluster_counts = data['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    fig = px.pie(cluster_counts, names='Cluster', values='Count', title='Cluster Distribution')
    st.plotly_chart(fig)

def plot_bar_chart(data):
    cluster_means = data.groupby('Cluster').mean().reset_index()
    fig = px.bar(cluster_means, x='Cluster', y=['Recency', 'Frequency', 'Monetary'], title='Average RFM Values per Cluster')
    st.plotly_chart(fig)

def display_segment_insights(data):
    st.write("Segment Insights")
    for cluster in sorted(data['Cluster'].unique()):
        st.write(f"**Cluster {cluster} Insights**")
        cluster_data = data[data['Cluster'] == cluster]
        st.write(f"Average Recency: {cluster_data['Recency'].mean():.2f} days")
        st.write(f"Average Frequency: {cluster_data['Frequency'].mean():.2f}")
        st.write(f"Average Monetary: ${cluster_data['Monetary'].mean():.2f}")

def download_results(data):
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="customer_segments.csv">Download CSV file</a>'
    st.markdown(href, unsafe_allow_html=True)

def main():
    st.title('Customer Segmentation Dashboard')
    data = load_data()
    if not data.empty:
        if set(['CustomerID', 'InvoiceDate', 'Quantity', 'UnitPrice', 'InvoiceNo']).issubset(data.columns):
            data = preprocess_data(data)
            if not data.empty:
                rfm_data = calculate_rfm(data)
                st.sidebar.subheader('Clustering Options')
                n_clusters = st.sidebar.slider('Number of clusters', 2, 10, 3)
                rfm_data = apply_kmeans(rfm_data, n_clusters)
                st.write("Data Overview", rfm_data.head())
                plot_clusters(rfm_data)
                plot_pie_chart(rfm_data)
                plot_bar_chart(rfm_data)
                display_segment_insights(rfm_data)  # Display segment insights here
                st.sidebar.subheader('Download Results')
                if st.sidebar.button('Download CSV'):
                    download_results(rfm_data)
        else:
            st.error("Uploaded data does not contain required columns: CustomerID, InvoiceDate, Quantity, UnitPrice, InvoiceNo")
    else:
        st.info("Please upload data to begin analysis.")

if __name__ == "__main__":
    main()
