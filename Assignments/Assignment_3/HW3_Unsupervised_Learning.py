#!/usr/bin/env python
# coding: utf-8

# # Homework 3: Unsupervised Learning

# Due Wednesday 11/24 at 11:59 pm EST
# 
# In this notebook, we will be applying unsupervised learning approaches to a problem in computational biology. Specifically, we will be analyzing single-cell genomic sequencing data. Single-cell genomics is a set of revolutionary new technologies which can profile the genome of a specimen (tissue, blood, etc.) at the resolution of individual cells. This increased granularity can help capture intercellular heterogeneity, key to better understanding and treating complex genetic diseases such as cancer and Alzheimer's. 
# 
# <img src="https://cdn.10xgenomics.com/image/upload/v1574196658/blog/singlecell-v.-bulk-image.png" width="800px"/>
# 
# <center>Source: 10xgenomics.com/blog/single-cell-rna-seq-an-introductory-overview-and-tools-for-getting-started</center>
# 
# A common challenge of genomic datasets is their high-dimensionality: a single observation (a cell, in the case of single-cell data) may have tens of thousands of gene expression features. Fortunately, biology offers a lot of structure - different genes work together in pathways and are co-regulated by gene regulatory networks. Unsupervised learning is widely used to discover this intrinsic structure and prepare the data for further analysis.

# In[154]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ## Dataset: single-cell RNASeq of mouse brain cells

# We will be working with a single-cell RNASeq dataset of mouse brain cells. In the following gene expression matrix, each row represents a cell and each column represents a gene. Each entry in the matrix is a normalized gene expression count - a higher value means that the gene is expressed more in that cell. The dataset has been pre-processed using various quality control and normalization methods for single-cell data. 
# 
# Data source: https://chanzuckerberg.github.io/scRNA-python-workshop/preprocessing/00-tabula-muris.html

# In[156]:


cell_gene_counts_df = pd.read_csv('data/mouse_brain_cells_gene_counts.csv', index_col='cell')
cell_gene_counts_df


# Note the dimensionality - we have 1000 cells (observations) and 18,585 genes (features)!
# 
# We are also provided a metadata file with annotations for each cell (e.g. cell type, subtissue, mouse sex, etc.)

# In[157]:


cell_metadata_df = pd.read_csv('data/mouse_brain_cells_metadata.csv')
cell_metadata_df


# Different cell types

# In[158]:


cell_metadata_df['cell_ontology_class'].value_counts()


# Different subtissue types (parts of the brain)

# In[159]:


cell_metadata_df['subtissue'].value_counts()


# In[160]:


cell_metadata_df['mouse.id'].value_counts()


# Our goal in this exercise is to use dimensionality reduction and clustering to visualize and better understand the high-dimensional gene expression matrix. We will use the following pipeline, which is common in single-cell analysis:
# 1. Use PCA to project the gene expression matrix to a lower-dimensional linear subspace.
# 2. Cluster the data using K-means on the first 20 principal components.
# 3. Use t-SNE to project the first 20 principal components onto two dimensions. Visualize the points and color by their clusters from (2).

# ## Part 1: PCA

# **Perform PCA and project the gene expression matrix onto its first 50 principal components. You may use `sklearn.decomposition.PCA`.**

# In[161]:


pca = PCA(n_components=50)
principalComponents = pca.fit_transform(cell_gene_counts_df)
principalDf = pd.DataFrame(data = principalComponents)
principalDf.head()


# **Plot the cumulative proportion of variance explained as a function of the number of principal components. How much of the total variance in the dataset is explained by the first 20 principal components?**

# In[162]:


fig = plt.figure(figsize=(12,8))
top_20_pca_var = pca.explained_variance_ratio_[:20]
ax = fig.add_subplot(1,1,1)
plt.plot(np.arange(1,21), top_20_pca_var.cumsum()*100)
ax.set_xlabel("# of PCs")
ax.set_ylabel("% of variance explained")

#A little over 20% of the data is explained by the first 20 components 


# **For the first principal component, report the top 10 loadings (weights) and their corresponding gene names.** In other words, which 10 genes are weighted the most in the first principal component?

# In[163]:




weights = pca.components_
first_component = abs(weights[0])
ind = np.argpartition(first_component, -10)[-10:]

col_names = cell_gene_counts_df.columns[ind]
col_names
print(col_names)
#'Erc2', 'Cpne5', 'Hpca', 'Nrsn2', 'Camkv', 'Nsg2', 'Rasgef1a', 'Kcnj4',
#       'Ptpn5', 'St8sia3'


# **Plot the projection of the data onto the first two principal components using a scatter plot.**

# In[165]:



plt.scatter(x=principalDf[0], y=principalDf[1])
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title('First two PCAs for gene expression data')


# **Now, use a small multiple of four scatter plots to make the same plot as above, but colored by four annotations in the metadata: cell_ontology_class, subtissue, mouse.sex, mouse.id. Include a legend for the labels.** For example, one of the plots should have points projected onto PC 1 and PC 2, colored by their cell_ontology_class.

# In[166]:


fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(20,45))
cell_ontology_colors = ['red', 'blue', 'yellow', 'purple', 'black', 'green', 'orange']
cell_ontology_colors_map = {
  "oligodendrocyte": "red",
  "endothelial cell": "blue",
  "astrocyte": "purple",
  "neuron": "black",
  "brain pericyte": "green",
  "oligodendrocyte precursor cell": "orange",
  "Bergmann glial cell": "yellow",
}

mouse_sex_colors_map = {
  "F": "red",
  "M": "blue"
}

mouse_id_colors_map = {
  "3_10_M": "red",
  "3_9_M": "blue",
  "3_38_F": "purple",
  "3_8_M": "black",
  "3_11_M": "green",
  "3_39_F": "orange",
  "3_56_F": "yellow",
}

cell_subtissue_colors_map = {
  "Cortex": "red",
  "Hippocampus": "blue",
  "Striatum": "purple",
  "Cerebellum": "black",
}

cell_metadata_df['ontology_color'] = cell_metadata_df.apply(lambda row : cell_ontology_colors_map[row['cell_ontology_class']],axis=1)
cell_metadata_df['sex_color'] = cell_metadata_df.apply(lambda row : mouse_sex_colors_map[row['mouse.sex']],axis=1)
cell_metadata_df['id_color'] = cell_metadata_df.apply(lambda row : mouse_id_colors_map[row['mouse.id']],axis=1)
cell_metadata_df['subtissue_color'] = cell_metadata_df.apply(lambda row : cell_subtissue_colors_map[row['subtissue']],axis=1)

ax1.scatter(x=principalDf[0], y=principalDf[1], c=cell_metadata_df['ontology_color'])
ax1.set_xlabel("PC 1")
ax1.set_ylabel("PC 2")
ax1.set_title("Cell ontology type and first two PCs")
ax2.scatter(x=principalDf[0], y=principalDf[1], c=cell_metadata_df['sex_color'])
ax2.set_xlabel("PC 1")
ax2.set_ylabel("PC 2")
ax2.set_title("Mouse.sex and first two PCs")
ax3.scatter(x=principalDf[0], y=principalDf[1], c=cell_metadata_df['id_color'])
ax3.set_xlabel("PC 1")
ax3.set_ylabel("PC 2")
ax3.set_title("Mouse.id and first two PCs")
ax4.scatter(x=principalDf[0], y=principalDf[1], c=cell_metadata_df['subtissue_color'])
ax4.set_xlabel("PC 1")
ax4.set_ylabel("PC 2")
ax3.set_title("subtissue and first two PCs")


fig.tight_layout()


#plt.xlabel("PC 1")
#plt.ylabel("PC 2")


# # **Based on the plots above, the first two principal components correspond to which aspect of the cells? What is the intrinsic dimension that they are describing?**

# In[167]:


#PC1 and PC2 are able to distinguish well between cell ontology type 
#and M vs Female best, so I think that these two components correpond to 
#these aspects of the cells



# ## Part 2: K-means

# While the annotations provide high-level information on cell type (e.g. cell_ontology_class has 7 categories), we may also be interested in finding more granular subtypes of cells. To achieve this, we will use K-means clustering to find a large number of clusters in the gene expression dataset. Note that the original gene expression matrix had over 18,000 noisy features, which is not ideal for clustering. So, we will perform K-means clustering on the first 20 principal components of the dataset.

# **Implement a `kmeans` function which takes in a dataset `X` and a number of clusters `k`, and returns the cluster assignment for each point in `X`. You may NOT use sklearn for this implementation. Use lecture 6, slide 14 as a reference.**

# In[133]:


import random
from scipy.spatial import distance

def kmeans(X, k, iters=10):
    '''Groups the points in X into k clusters using the K-means algorithm.

    Parameters
    ----------
    X : (m x n) data matrix
    k: number of clusters
    iters: number of iterations to run k-means loop

    Returns
    -------
    y: (m x 1) cluster assignment for each point in X
    '''
    #random var between min and max of each col
    #k_vals: (k x n)
    

    #print(n)
    #should be 20 

    
    #k_vals = np.zeros((k, n))
    #for i in range(0, k):
    #    for j in range (0, n):
    #        min_val_col = int(round(np.min(X[:, j])))
    #        max_val_col = int(round(np.max(X[:, j])))
    #        k_vals[i][j] = random.randint(min_val_col, max_val_col)
            
    count = 0  
    m = len(X)
    idx = np.random.choice(m, k, replace=False)
    n = len(X[0])
    centroids = X[idx, :]
    distances = distance.cdist(X, centroids, 'euclidean')
    min_ks = np.array([np.argmin(i) for i in distances])
    
    while count < iters: 
        centroids = []
        for idx in range(k):
            centroids.append(X[min_ks==idx].mean(axis=0))

        centroids = np.vstack(centroids)
        distances = distance.cdist(X, centroids ,'euclidean')
        min_ks = np.array([np.argmin(i) for i in distances])
        
        count = count +1
        
    return min_ks
    
    


# Before applying K-means on the gene expression data, we will test it on the following synthetic dataset to make sure that the implementation is working.

# In[134]:


np.random.seed(0)
x_1 = np.random.multivariate_normal(mean=[1, 2], cov=np.array([[0.8, 0.6], [0.6, 0.8]]), size=100)
x_2 = np.random.multivariate_normal(mean=[-2, -2], cov=np.array([[0.8, -0.4], [-0.4, 0.8]]), size=100)
x_3 = np.random.multivariate_normal(mean=[2, -2], cov=np.array([[0.4, 0], [0, 0.4]]), size=100)
X = np.vstack([x_1, x_2, x_3])
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], s=10)
plt.xlabel('$x_1$', fontsize=15)
plt.ylabel('$x_2$', fontsize=15)


# **Apply K-means with k=3 to the synthetic dataset above. Plot the points colored by their K-means cluster assignments to verify that your implementation is working.**

# In[168]:


label = kmeans(X, 3, 25)

for i in np.unique(label):
    plt.scatter(X[label == i , 0] , X[label == i , 1] , label = i)
plt.legend()
plt.title('synthetic data kmeans = 3')
plt.show()


# **Use K-means with k=20 to cluster the first 20 principal components of the gene expression data.**

# In[145]:


pca = PCA(n_components=20)
principalComponents = pca.fit_transform(cell_gene_counts_df)
principalDf_20 = pd.DataFrame(data = principalComponents)

principalDf_20.head()


# In[148]:


principalDf_20_numpy = principalDf_20.to_numpy()

pca_labels = kmeans(principalDf_20_numpy, 20, 15)


# In[150]:


for i in np.unique(pca_labels):
    plt.scatter(principalDf_20_numpy[pca_labels == i , 0] , principalDf_20_numpy[pca_labels == i , 1] , label = i)

plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('kmeans on top 20 on top 20 pca components of gene expression data')
plt.legend()
plt.show()


# ## Part 3: t-SNE

# In this final section, we will visualize the data again using t-SNE - a non-linear dimensionality reduction algorithm. You can learn more about t-SNE in this interactive tutorial: https://distill.pub/2016/misread-tsne/.

# **Use t-SNE to reduce the first 20 principal components of the gene expression dataset to two dimensions. You may use `sklearn.manifold.TSNE`.** Note that it is recommended to first perform PCA before applying t-SNE to suppress noise and speed up computation.

# In[113]:


tsne = TSNE()
tsne_pca_results = tsne.fit_transform(principalDf_20)


# **Plot the data (first 20 principal components) projected onto the first two t-SNE dimensions.**

# In[169]:


plt.figure(figsize=(8, 5))
plt.scatter(tsne_pca_results[:, 0], tsne_pca_results[:, 1], s=10)
plt.xlabel('$TSNE_1$', fontsize=15)
plt.ylabel('$TSNE_2$', fontsize=15)
plt.title('TSNE1 and TSNE2 for PCA20 dimensions from gene expression data')


# **Plot the data (first 20 principal components) projected onto the first two t-SNE dimensions, with points colored by their cluster assignments from part 2.**

# In[170]:


for i in np.unique(pca_labels):
    plt.scatter(tsne_pca_results[pca_labels == i , 0] , tsne_pca_results[pca_labels == i , 1] , label = i)
plt.xlabel('$TSNE_1$', fontsize=15)
plt.ylabel('$TSNE_2$', fontsize=15)
plt.title('kmeans on top 20 on top 20 pca components of gene expression data then tSNE')
plt.legend()
plt.show()


# **Why is there overlap between points in different clusters in the t-SNE plot above?**

# In[19]:


### There is overlap because we are reducing an already reduced dimensionality of PCA further, so we 
### are unable to see the 'depth' or third/more dimensions which may be separating the dataset. Also, 
### tSNE is a probabilisitic algorithm so it's possible that the overlap is due to the probability based nature -
### which lends itself to non-clear cut /black and white slices.


# These 20 clusters may correspond to various cell subtypes or cell states. They can be further investigated and mapped to known cell types based on their gene expressions (e.g. using the K-means cluster centers). The clusters may also be used in downstream analysis. For instance, we can monitor how the clusters evolve and interact with each other over time in response to a treatment.
