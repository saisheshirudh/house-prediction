#!/usr/bin/env python
# coding: utf-8

# In[6]:


#importing all the necessary modules
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split


# In[9]:


#loading the california housing price details
from sklearn.datasets import fetch_california_housing
california = fetch_california_housing()
print(california)


# In[10]:


#assigning variables to the rows and columns of the considered data set
df_x = pd.DataFrame(california.data, columns=california.feature_names)
df_y = pd.DataFrame(california.target)


# In[11]:


#showing the data consided in each attribute
df_x.describe()


# In[12]:


df_y.describe()


# In[13]:


#initialsing a machine learning model
reg = linear_model.LinearRegression()


# In[35]:


#splitting the data set
x_train,x_test,y_train,y_test = train_test_split(df_x, df_y, test_size= 0.33, random_state= 42)


# In[36]:


#training the model
reg.fit(x_train, y_train)


# In[37]:


#print the coefficients/weights for each column
print(reg.coef_)


# In[38]:


#print prdictions
y_pred = reg.predict(x_test)
print(y_pred)


# In[39]:


print(y_test)


# In[40]:


print( np.mean( (y_pred - y_test)**2))


# In[41]:


from sklearn.metrics import mean_squared_error
print( mean_squared_error(y_test, y_pred))


# In[ ]:




