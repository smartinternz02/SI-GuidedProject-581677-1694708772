#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[4]:


df=pd.read_csv('C:/Users/THIRU/Downloads/penguins_size.csv')


# In[5]:


df


# In[6]:


df.describe()


# In[7]:


df.info()


# In[10]:


df.isnull().any()


# In[11]:


df.isnull()


# In[ ]:


#UNIVARIATE ANALYSIS


# In[9]:


sns.distplot(df.body_mass_g)


# In[17]:


sns.barplot(x =df.sex.value_counts().index,y =df.sex.value_counts() )


# In[ ]:


#BIVARIATE ANALYSIS


# In[19]:


sns.jointplot(x='culmen_length_mm',y='culmen_depth_mm',data=df)


# In[22]:


sns.boxplot(x='sex',y='body_mass_g',data=df)


# In[ ]:


#MULTIVARIATE ANALYSIS


# In[20]:


sns.pairplot(df)


# In[24]:


sns.heatmap(df.corr(),annot=True)


# In[25]:


df.isnull().any()


# In[26]:


df.isnull().sum()


# In[27]:


df['culmen_length_mm'].fillna(df['culmen_length_mm'].median(),inplace=True)


# In[28]:


df['culmen_depth_mm'].fillna(df['culmen_depth_mm'].median(),inplace=True)


# In[29]:


df['flipper_length_mm'].fillna(df['flipper_length_mm'].median(),inplace=True)


# In[30]:


df['body_mass_g'].fillna(df['body_mass_g'].median(),inplace=True)


# In[31]:


df['sex'].fillna(df['sex'].mode().iloc[0],inplace=True)


# In[33]:


df.isnull().any()


# In[ ]:


#OUTLIERS CHECK


# In[34]:


sns.boxplot(df.culmen_length_mm)


# In[35]:


sns.boxplot(df.culmen_depth_mm)


# In[36]:


sns.boxplot(df.flipper_length_mm)


# In[37]:


sns.boxplot(df.body_mass_g)


# In[ ]:


#THUS THERE ARE NO OUTLAYERS IN THE GIVEN DATA


# In[ ]:


#ENCODING


# In[40]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['species'] = le.fit_transform(df['species'])
df['island'] = le.fit_transform(df['island'])
df.head()


# In[ ]:


#correlation of independent variables with the target


# In[41]:


df.corr().species.sort_values(ascending=False)


# In[ ]:


#Spliting the data into dependent and independent variables


# In[46]:


X=df.drop(columns='species',axis=1)
X.head()


# In[48]:


Y=df['species']
Y.head()


# In[ ]:


#Scaling the data


# In[50]:


from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler()
X_scaled=pd.DataFrame(scale.fit_transform(X),columns=X.columns)
X_scaled.head()


# In[ ]:


#Spliting the data into training and testing


# In[52]:


from sklearn.model_selection import train_test_split


# In[55]:


x_train,x_test,y_train,y_test=train_test_split(X_scaled,Y,test_size=0.3,random_state=0)


# In[ ]:


#Checking the training and testing data shape


# In[57]:


x_train.shape


# In[59]:


x_test.shape


# In[58]:


y_train.shape


# In[60]:


y_test.shape

