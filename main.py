import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import tensorflow as tf

# Importing the Mall dataset with pandas
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Using the dendrogram to find the optimal number of clusters
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Fitting the hierarchical clustering to the mall dataset
n_clusters = 5
hc = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
Y_hc = hc.fit_predict(X)

# Using TensorFlow for GPU acceleration
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("GPU available for TensorFlow!")

# Visualizing the clusters using Seaborn
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=Y_hc, palette='viridis', s=100)
plt.title(f'Hierarchical Clustering - {n_clusters} Clusters')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()
