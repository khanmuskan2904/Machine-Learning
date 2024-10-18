#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# customer segmentation using k means clustering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df= pd.read_csv('Mall_Customers.csv')

df.head()

df.info()

x= df.iloc[:,[3,4]]

x.head()

from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans= KMeans(n_clusters=i, random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('number of clusters')
plt.ylabel('wcss')
plt.show()

