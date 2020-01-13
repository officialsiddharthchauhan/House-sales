#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


file_name='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/coursera/project/kc_house_data_NaN.csv'
df=pd.read_csv(file_name)


# In[3]:


df.dtypes


# In[4]:


df.describe()


# In[5]:



df.drop(['id', 'Unnamed: 0'], axis=1, inplace = True)
df.describe()


# In[6]:


mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)


# In[7]:



mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)


# In[8]:



df["floors"].value_counts().to_frame()


# In[9]:


sns.boxplot(x="waterfront", y="price", data=df)


# In[10]:


sns.regplot(x="sqft_above", y="price", data=df)


# In[11]:


df.corr()['price'].sort_values()


# In[12]:


import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[13]:



X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm
lm.fit(X,Y)
lm.score(X, Y)


# In[14]:



X = df[['sqft_living']]
Y = df['price']
lm1 = LinearRegression()
lm1
lm1.fit(X,Y)
print('The R-square is: ', lm1.score(X, Y))


# In[15]:



features =df[["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]]


# In[16]:


lm1.fit(features, df['price'])
print('The R-square is: ', lm1.score(features,df['price']))


# In[17]:


Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]


# In[18]:


pipe=Pipeline(Input)
pipe


# In[19]:


pipe.fit(X,Y)


# In[20]:


pipe.score(X,Y)


# In[21]:



pipe.fit(features,Y)
ypipe=pipe.predict(features)
lm1.fit(features, Y)
print('The R-square is: ', lm.score(features, Y))


# In[22]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
print("done")


# In[23]:



features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features ]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


# In[ ]:





# In[24]:


from sklearn.linear_model import Ridge


# In[25]:


RigeModel = Ridge(alpha=0.1)
RigeModel.fit(x_train, y_train)
RigeModel.score(x_test, y_test)


# In[27]:



poly1 = LinearRegression()
poly1.fit(x_train, y_train)
pr1=PolynomialFeatures(degree=2)
x_train_pr=pr1.fit_transform(x_train[features])
x_test_pr=pr1.fit_transform(x_test[features])

RigeModel=Ridge(alpha=0.1)
RigeModel.fit(x_train_pr, y_train)
RigeModel.score(x_test_pr, y_test)

