#!/usr/bin/env python
# coding: utf-8

# ## Regression model: working with data

# In[5]:


# Importing the important libraries

import mlxtend.preprocessing
import mlxtend.frequent_patterns
# Libraries to be used
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

# Decision trees
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier


# Grid search
from sklearn.model_selection import GridSearchCV

# Regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as MSE


# In[6]:


# Loading the data
house_data = pd.read_csv('House_details.csv')
house_data.head()


# ## Exploratory data analysis: Playing with data through relation visualizaition

# In[15]:


house_data.shape


# **Dataset has 10000 observation and 12 characteristics**

# In[13]:


house_data.info()


# **Dataset has only interger values and float. No null/missing values are there**

# In[16]:


house_data.describe()


# **The mean value of most of the column are above 50%.**

# In[17]:


house_data.corr()


# **visualisation of determining column**

# In[18]:


sns.histplot(house_data['Price'], bins = 15)


# In[19]:


sns.heatmap(house_data.corr())


# **Dark shades represents positive correlation and ligher shades represents negative correlation.**

# ## Viusalising and finding relation with different columns of the house_data. Normally it can be done with factors affecting price in the data

# In[21]:


sns.jointplot(x = 'Owners', y ='Price', data = house_data)


# **visualising Area and Price.**

# In[22]:


sns.jointplot(x = 'Area', y ='Price', data = house_data)


# In[23]:


sns.boxplot(x = 'Yard', y = 'Guest', data = house_data)


# In[25]:


sns.boxplot(x = 'Storage', y = 'Guest', data = house_data)


# **In this way can do exploratory analysis**

# ## Building Regression model

# In[7]:


# Similar analysis can be done focusing on the column price
# What you can find that almost all the variables seem to behave similarly, as they either have 0 correlation, 
# or when they are categorical, mean of price across categories is the same
# The only varibale that somewhat reasonable to include is square meter
# It is okay if you included others, you cannot really get better results

X = house_data[['Area']]
y = house_data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)


# ## Building a decision tree regression model. It can be used to predict selling price for new house on the market

# In[8]:


# Build base decision tree regression model

house_tree = DecisionTreeRegressor(random_state = 42)

house_tree.fit(X_train, y_train)
house_pred = house_tree.predict(X_test)

# The MSE is around 22 million
mse_house = MSE(y_test, house_pred)
print('MSE:', mse_house)


# ## Optimization of parameter

# In[10]:


# When we take the square root of MSE, it is less than 5000
# which is not bad a mistake, considering that the average error is approx. 0.1% of the mean price
# Using only one variable
100 * mse_house**0.5 / house_data.Price.mean()

