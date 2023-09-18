#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

pd.__version__


# In[13]:


df = pd.read_csv('C:/Users/790041733/OneDrive - EVOLUTIO CLOUD ENABLER, S.A/ZOOMCAMP/Week1/housing.csv')
df.describe()


# In[14]:


df.info()


# In[17]:


print(df.isnull().any())


# In[22]:


print(df['ocean_proximity'].unique())


# In[27]:


print(df.loc[df['ocean_proximity'] =='NEAR BAY', 'median_house_value'].mean())


# In[28]:


print(df['total_bedrooms'].mean())


# In[30]:


df['total_bedrooms'] = df['total_bedrooms'].fillna(537.8705525375618)


# In[32]:


df.describe()


# In[36]:


print(df.loc[df['ocean_proximity'] =='ISLAND', ('housing_median_age', 'total_rooms', 'total_bedrooms')])


# In[37]:


import numpy as np


# In[42]:


X = (df.loc[df['ocean_proximity'] =='ISLAND', ('housing_median_age', 'total_rooms', 'total_bedrooms')]).to_numpy()


# In[68]:


print(X)


# In[69]:


XTX = X.T


# In[70]:


print(XTX)


# In[71]:


XTX= np.dot(XTX,X)


# In[72]:


print(XTX)


# In[73]:


print(np.linalg.inv(XTX))


# In[74]:


Y = np.array([950, 1300, 800, 1000, 1300])


# In[75]:


print(Y)


# In[76]:


W = np.dot(np.dot(np.linalg.inv(XTX),X.T),Y)


# In[77]:


print(W)


# In[ ]:




