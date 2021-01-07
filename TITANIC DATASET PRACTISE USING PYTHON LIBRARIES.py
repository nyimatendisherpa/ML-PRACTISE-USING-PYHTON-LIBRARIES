#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import math
df=pd.read_csv("titanic.csv")
df.head(10)
df


# In[8]:


total = df.isnull().sum().sort_values(ascending=False)
percent_1 = df.isnull().sum()/df.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head(5)


# In[9]:


df.describe()


# In[16]:


len(df)


# In[15]:


df.head(10)


# In[10]:


df.count()


# In[11]:


df.shape


# In[14]:


df.Survived.value_counts()


# In[16]:


342/891.0


#  BY USING PANDAS AND EDA

# In[22]:


df.Sex.values


# In[24]:


df.Sex.value_counts()


# In[26]:


df.Sex.value_counts().plot(kind='bar')


# In[35]:


df[df.Sex=='male']


# In[1]:


df[df.Survived=='0']


# In[39]:


df[df.Sex.isnull()]


# In[42]:


df.Fare.value_counts()


# In[43]:


df.describe()


# In[44]:


df.size


# In[49]:


df.Fare.hist(bins=5)


# In[48]:


df.Pclass.hist()


# In[50]:


df.PassengerId.hist()


# In[65]:


df[df.Cabin.isnull()]


# In[66]:


df.Sex.hist()


# In[68]:


df[df.Sex=='male'].Survived.value_counts()


# In[69]:


df[df.Sex=='Female'].Survived.value_counts().hist


# In[70]:


df[df.Sex=='male'].hist


# In[81]:


df.Sex.value_counts().plot(kind='bar')


# In[79]:


df.Sex.value_counts().plot(kind='barh')


# In[82]:


df.Sex.hist


# In[83]:


df.Sex.hist()


# In[1]:


df[df.Sex=='female'].Survived.value_counts().plot(kind='bar')


# In[89]:


df[df.Sex=='female'].Age.value_counts().plot(kind='bar')


# In[92]:


df[df.Sex=='male'].Survived.value_counts().plot(kind='barh')


# In[93]:


df[df.Sex=='male'].Age.value_counts().hist()


# In[4]:


df[df.Sex=='male'].Age.value_counts().hist(),boxplot.plot(kind='bar',title='male')


# In[120]:


df.tail().hist


# rows and columns

# In[103]:


df.loc[0,:]


# In[104]:


df.loc[2,:]


# In[26]:


df.iloc[2:9,:]


# In[29]:


df.iloc[2:13,:]


# In[112]:


df.loc[[2,3,9],:]


# In[115]:


df[df.Sex=='male'].loc[2:3,:]


# In[122]:


df[df.Sex=='male'].loc[2:].hist


# In[125]:


df[df.Sex=='female'].loc[2:].plot(kind="bar")


# In[126]:


df.loc[3,'Pclass']


# In[127]:


df.loc[:,'Pclass']


# In[136]:


df.loc[:,'Survived']


# In[17]:


df[df.Sex=='female'].loc[2:].plot(kind='bar')


# In[34]:


df[df.Sex=='male'].hist()


# In[44]:


df[['PassengerId','Ticket']]


# In[5]:


df.describe()


# In[2]:



df.Age.value_counts()


# In[1]:


plt.plot(x,sin('Sex'))


# In[ ]:





# In[ ]:




