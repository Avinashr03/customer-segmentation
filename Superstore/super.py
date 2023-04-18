import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hc
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pylab import rcParams

# Set up page
st.set_page_config(page_title="Customer Segmentation App")

# Load data
@st.cache
def load_data():
    data = pd.read_csv('Mall_Customers.csv')
    return data

df = load_data()

# Preprocess data
scaler = StandardScaler()
df.iloc[:,2:] = scaler.fit_transform(df.iloc[:,2:])
X = df.iloc[:,[3,4]].values

# Sidebar
st.sidebar.title("Customer Segmentation")

# Show raw data
if st.sidebar.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(df)

# Exploratory data analysis
st.sidebar.subheader('Exploratory Data Analysis')
# Boxplot
st.sidebar.subheader('Boxplot')
x_axis = st.sidebar.selectbox('Select feature for x-axis', df.columns[1:4])
y_axis = st.sidebar.selectbox('Select feature for y-axis', df.columns[2:5])
sns.boxplot(x=x_axis, y=y_axis, data=df)
st.pyplot()

# Violinplot
st.sidebar.subheader('Violinplot')
sns.violinplot(x=x_axis, y=y_axis, data=df)
st.pyplot()

# Clustering
st.sidebar.subheader('Clustering')

# Elbow method
st.sidebar.subheader('Elbow method')
wcss = []
for i in range(1, 11):
    # kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
fig, ax = plt.subplots()
ax.plot(range(1, 11), wcss)
ax.set_title('The Elbow Point Graph')
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('WCSS')
st.pyplot(fig)
# K-means clustering
st.sidebar.subheader('K-means clustering')
n_clusters = st.sidebar.slider('Select number of clusters', 2, 10, 5)
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0)
Y = kmeans.fit_predict(X)
fig, ax = plt.subplots(figsize=(15, 10))
colors = ['green', 'red', 'yellow', 'violet', 'blue', 'orange', 'pink', 'brown', 'gray', 'olive']
for i in range(n_clusters):
    ax.scatter(X[Y==i,0], X[Y==i,1], s=50, c=colors[i], label=f'Cluster {i+1}')
ax.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='cyan', label='Centroids')
ax.set_title('Customer Groups')
ax.set_xlabel('Annual Income')
ax.set_ylabel('Spending Score')
ax.legend()
st.pyplot(fig)

import warnings
warnings.filterwarnings("ignore", message="The default value of `n_init` will change from 10 to 'auto' in 1.4.")

# Hierarchical clustering
st.sidebar.subheader('Hierarchical clustering')
fig, ax = plt.subplots()
dendrogram = hc.dendrogram(hc.linkage(X, method='ward'))
ax.set_title('Dendrogram')
ax.set_xlabel('Customers')
ax.set_ylabel('Euclidean Distances')
ax.axhline(5, c='r', linestyle='--')
st.pyplot(fig)
