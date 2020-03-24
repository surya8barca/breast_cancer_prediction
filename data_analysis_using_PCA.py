#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


from sklearn.datasets import load_breast_cancer


# In[5]:


ds=load_breast_cancer()


# In[8]:


print(ds.keys())


# In[10]:


df=pd.DataFrame(ds['data'],columns=ds['feature_names'])


# In[11]:


df.head()


# In[12]:


#scaling the data


# In[13]:


from sklearn.preprocessing import StandardScaler


# In[14]:


ss=StandardScaler()


# In[15]:


ss.fit(df)


# In[29]:


scaled_ds=ss.transform(df)


# In[30]:


scaled_ds.shape


# In[21]:


#PCA


# In[22]:


from sklearn.decomposition import PCA


# In[23]:


pc=PCA(n_components=2)


# In[24]:


pc.fit(scaled_ds)


# In[25]:


pca_ds=pc.transform(scaled_ds)


# In[28]:


pca_ds.shape


# In[31]:


#visualization


# In[48]:


plt.figure(figsize=(10,6))
plt.scatter(pca_ds[:,0],pca_ds[:,1],c=ds['target'])


# In[49]:


pc.components_


# In[51]:


df2=pd.DataFrame(pc.components_,columns=ds['feature_names'])


# In[52]:


df2


# In[53]:


sns.heatmap(df2)


# In[ ]:




