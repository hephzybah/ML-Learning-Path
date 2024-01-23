#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import libraries 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm


# In[4]:


#read csv and get first five rows 
df = pd.read_csv('Breast_Cancer.csv')
df.head()


# In[14]:


#get last five rows 
df.tail()


# In[16]:


#get shape 
df.shape


# In[18]:


#get size 
df.size


# In[20]:


#count number of rows 
df.count()


# In[22]:


#count the number observations in the dataset 
df.isnull().count()


# In[24]:


#describe dataset
df.describe()


# In[26]:


#count the number of null values 
df.isna().sum()


# In[30]:


#sum up the categories in the Status column
df['Status'].value_counts().to_frame()


# In[51]:


#get the correlation 
df.corr()


# In[91]:


#Extract non-numeric columns from the data set for encoding 
race = df['Race']
race
marital = df['Marital Status']
marital

#for some reason, the below commented code refused to be extracted 
'''t_stage = df['T Stage']
n_stage = df['N Stage']'''
sixth_stage = df['6th Stage']
estro = df['Estrogen Status']
progest = df['Progesterone Status']
a_stage = df['A Stage']


# In[88]:


#import Label encoder 
from sklearn.preprocessing import LabelEncoder

#function definition 
le = LabelEncoder()

#encode the extracted columns
e_race = le.fit_transform(race)
e_race
e_marital = le.fit_transform(marital)
e_sixth_stage = le.fit_transform(sixth_stage)
e_estro = le.fit_transform(estro)
e_progest = le.fit_transform(progest)
e_a_stage = le.fit_transform(a_stage)


# In[92]:


#replace columns back into the datset 
df['Race'] = e_race
df['Marital Status'] = e_marital
df['6th Stage'] = e_sixth_stage
df['Estrogen Status'] = e_estro
df['Progesterone Status'] = e_progest
df['A Stage'] = e_a_stage

#show current dataset 
df


# In[95]:


'''#drop coloumns
df.drop('T Stage', axis=1)'''


# In[5]:


#plot cluster map to find any correlation 
sns.clustermap(data=df.corr())


# In[96]:


x=df.drop(['Status'],axis=1)
y=df['Status']


# In[97]:


X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=42)


# In[98]:


model=svm.SVC()


# In[99]:


model.fit(X_train, y_train)

