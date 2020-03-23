#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#inbuilt breast cancer dataset


# In[2]:


from sklearn.datasets import load_breast_cancer


# In[3]:


cancer=load_breast_cancer()


# In[4]:


cancer.keys()


# In[5]:


print(cancer['DESCR'])


# In[7]:


df_features=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df_features.head()


# In[11]:


df_features.columns


# In[21]:


cancer['target']


# In[22]:


cancer['target_names']


# In[13]:


fig=sns.pairplot(df_features)
fig.savefig('features pairplot.jpg')


# In[15]:


#train test split the data


# In[16]:


x=df_features
y=cancer['target']


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4)


# In[23]:


#perform fit and predict using SVM


# In[24]:


from sklearn.svm import SVC


# In[25]:


svcm=SVC()


# In[26]:


svcm.fit(x_train,y_train)


# In[27]:


pred1=svcm.predict(x_test)


# In[30]:


pred1


# In[32]:


#evaluate model


# In[33]:


from sklearn.metrics import confusion_matrix,classification_report


# In[35]:


print(confusion_matrix(y_test,pred1))
print()
print(classification_report(y_test,pred1))


# In[29]:


sns.lineplot(y_test,pred1)


# In[36]:


# now using data grid


# In[38]:


from sklearn.model_selection import GridSearchCV


# In[39]:


d={'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001]}


# In[40]:


gscv=GridSearchCV(SVC(),d,verbose=3)


# In[41]:


gscv.fit(x_train,y_train)


# In[42]:


pred2=gscv.predict(x_test)


# In[43]:


pred2


# In[44]:


#evaluate model


# In[45]:


print(confusion_matrix(y_test,pred2))
print()
print(classification_report(y_test,pred2))


# In[46]:


sns.lineplot(y_test,pred2)


# In[47]:


#comparison


# In[49]:


sns.lineplot(pred1,pred2)


# In[ ]:




