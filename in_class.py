# %%
# load libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# %%
# load data
df = pd.read_csv('house_votes_Dem.csv', encoding='latin-1')
df.head()

# %%
# take a look at the data
df.info()

# %%
# separate out the numeric features
c_num = df[["aye", "nay", "other"]]
c_num.head()

# %%
# documentation for kmeans in sklearn
help(KMeans)


# %% build a kmeans model
kmeans = KMeans(n_clusters=3, random_state=42, verbose=1)
kmeans.fit(c_num)


# %% look at the information in the model
print("Intertia: ", kmeans.inertia_)
print("Cluster centers: ", kmeans.cluster_centers_)


# %%
# add the cluster labels to the original data frame
print("Labels: ", kmeans.labels_)

# %%
df['cluster'] = kmeans.labels_
df.head() 

# %%
# use a loop to check for different clusters numbers
# and see how their inertia changes

inertia = []
k_values = range(1, 10)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, verbose=0)
    kmeans.fit(c_num)
    inertia.append(kmeans.inertia_)

# %%
# plot the inertia values to find the elbow point
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.xticks(k_values)
plt.grid()
plt.show()

# %% simple plot of the clusters
plt.scatter(df['aye'], df['nay'], c=df['cluster'])
plt.xlabel('aye')
plt.ylabel('nay')
plt.title('KMeans Clusters of House Votes')
plt.legend(*plt.gca().get_legend_handles_labels(), title="Cluster") 
plt.show()


# %%
