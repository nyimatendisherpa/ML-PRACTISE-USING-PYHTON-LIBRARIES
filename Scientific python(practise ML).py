#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
data=[1,2,3,4,5]
array=np.array(data)
data2=(5,6,7,8,9)
arr2=np.array(data2)
data3=[range(1,3),range(5,9)]
arr3=np.array(data3)
print(arr2.tolist())
print(arr3.tolist())


# In[24]:


np.zeros(10)
np.zeros((3,6))
np.linspace(1,8,4)


# In[26]:


int_array = np.arange(5)
float_array = int_array.astype(float)np


# In[36]:


arr=np.arange(10,dtype=int).reshape((2,5))
print(arr.shape)
print(arr.reshape(5,2))


# ADD AN AXIS

# In[55]:


a=np.array([0,1,2,3,4,5])
a_col=a[:,np.newaxis]
print(a_col)
b=np.array([2,8])
b_col=b[:,np.newaxis]
print(b_col)


# TRANSPOSE

# In[58]:


print(a_col.T)


# FLATTEN::(WHICH ALWYS RETURNS FLAT COPY OF THE ORIGINAL ONE)

# In[81]:


arr_fltt=arr.flatten()
arr_fltt[0]
print(arr_fltt)
print(arr)


# In[82]:


a=np.array([[1,2,3,4],[4,5,6,7]])
a.flatten()
print(a.flatten())


# RAVEL::which return the original array

# In[86]:


a=np.array([[1,2,3,4,5],[5,6,7,8,9]])
a.ravel()
print(a.ravel())
print(a)


# ARRAY AND MATRICES

# In[104]:


x=np.arange(1,8)
print(x)
y=np.arange(2*3*4)
print(y)


# In[148]:


b=b.reshape(2,1)
print(b)


# In[133]:


y=np.arange(1,100)
print(y)


# STACK ARRAY

# In[12]:


import numpy as np
arr1=np.arange(10,dtype=float).reshape(2,5)
print(arr1)
print(arr1[0])
print(arr1[1])
print(arr1[0,3])
print(arr1[0],[3])
print(arr1[0,:])
print(arr1[:,0])
print(arr1[])


# VECTORIZED  OPERATION

# In[24]:


num=np.arange(10)
print(num)
print(num*10)
print(np.sqrt(num))
print(np.ceil(num))
print(np.isnan(num))
num + np.arange(10)
np.maximum(num,np.array([-1,-2,-3,-4,-5,-6,-7,-8,-9,10]))
np.array([-1,-2,-3,-4,-5,-6,-7,-8,-9,-10])


# In[40]:


#euclidean
import random 
vec1=np.random.randn(10)
vec2=np.random.randn(10)
dist = np.sqrt(np.sum((vec1 - vec2) ** 2))
dist=np.sqrt(np.sum(vec1-vec2)**2)
print(dist)


# In[59]:


#math and stats
an=np.random.randn(4,3)
print(an)
print(an.mean())
print(an.std())
print(an.argmin())# index of minimum element
print(an.argmax())
print(an.sum())
print(an.sum(axis=0))
print(an.sum(axis=1))
print(an.sum(axis=1))#sum of rows
print(an.sum(axis=0))#sum of columns
print((an>0).sum())
print((an<2).sum())
print((an>0).any())#check if any value is true
print((an>0).sum())#check any value grater than 0
print((an>0).all())# checks if all values are True


# In[66]:


print(np.random.seed(1234))#set the seed
np.random.rand(2,1)
np.random.randint(0,2,10) #10 randomly picked 0 or 1


# In[67]:


names = np.array(['nyima', 'passang', 'tashi', 'lhakpa'])
names == 'Bob'    # returns a boolean array
names[names != 'Bob'] # logical selection
(names == 'Bob') | (names == 'Will') # keywords "and/or" don't work with boolean arrays
names[names != 'Bob'] = 'Joe' # assign based on a logical selection
np.unique(names) # set function


# In[70]:


names = np.array(['nyima', 'passang', 'tashi', 'lhakpa'])
names =='nyima'   # returns a boolean array


# In[73]:


names[names!='nyima']
(names=='nyima')|(names=='tashi')


# In[77]:


names[names!='nyima']='tashi'


# In[78]:


np.unique(names)


# PANDAS:: DATA MANUPULATION

# In[47]:


import pandas as pd
nyima=['Geeks', 'For', 'Geeks', 'is', 
            'portal', 'for', 'Geeks']
df=pd.DataFrame(nyima)
print(df)


# In[43]:


data = {'Name':['Tom', 'nick', 'krish', 'jack'],
    'Age':[20, 21, 19, 18]}
 
# Create DataFrame
df = pd.DataFrame(data)
 
# Print the output.
print(df)


# In[39]:


data = {'Name':['Tom', 'nick', 'krish', 'jack'],
        'Age':[20, 21, 19, 18]}
 
# Create DataFrame
df = pd.DataFrame(data)
 
# Print the output.
print(df)


# In[5]:


columns = ['name', 'age', 'gender', 'job']
user1 = pd.DataFrame([['alice', 19, "F", "student"],
['john', 26, "M", "student"]],
columns=columns)
user2 = pd.DataFrame([['eric', 22, "M", "student"],
['paul', 58, "F", "manager"]],
columns=columns)
user3 = pd.DataFrame(dict(name=['peter', 'julie'],
age=[33, 44], gender=['M', 'F'],
job=['engineer', 'scientist']))
print(user1)
print(user1)
print(user3)
row2=user3.iloc()
row3=user3.loc()
df.dropna()


# concatenate Data Frame

# In[9]:


print(user1.append(user2))
print(user2.append(user3))


# In[12]:


users=pd.concat([user1,user2,user3])
print(users)


# In[17]:


user=pd.DataFrame(dict(name=['nyima','tashi','passang','tsering'],
                       age=['23','34','45','23'],
                       job=['pilot','engineer','doctor','scientist']))
print(user)


# In[20]:


merge_inter=pd.merge(users,user,on="name")
print(merge_inter)


# In[25]:


merge=pd.merge(users,user,on="name",how="outer")
print(merge)


# In[29]:


merge=pd.merge(users,user,on='name',how='inner')
print(merge)


# In[37]:


merge=pd.merge(users,user,on='name',how='left')
print(merge)


# In[35]:


merge=pd.merge(users,user,on='name',how='right')
print(merge)


# unpivot

# In[52]:


staked=pd.melt(user,id_vars='name',var_name='age',value_name='value')
print(staked)


# In[58]:


name=staked.pivot(index='name',columns='age',values='value')
print(name)


# In[67]:


staked=pd.melt(users,id_vars='name',var_name='age',value_name='value')
print(staked)


# In[65]:


pivot2=staked.pivot(index='name',columns='age',values='value')
print(pivot2)


# In[73]:


users


# In[74]:


user


# In[75]:


print(user.tail())
print(user.head())


# In[78]:


user.index


# In[79]:


user.columns


# In[80]:


user.dtypes


# In[81]:


user.info()


# In[82]:


user.describe()


# In[88]:


users.shape


# In[16]:


user.values


# In[90]:


users.values


# In[91]:


users.gender


# In[92]:


users['gender']


# In[99]:


users[['age','name','job']]


# In[9]:


my_cols=['age','name']
print(my_cols)

ROWS SELECTION BASED
# In[43]:



import pandas as pd
example=['name','age','job','place']
user1=pd.DataFrame({'name':['nyima','tashi'],
      'age':['24','34'],
      'job':['pilot','scientist'],
      'place':['nepal','kathmandu']})
user2=pd.DataFrame({'name':['passang','namgyal'],
      'age':['34','45'],
      'job':['CA','doctor'],
      'place':['sikkim','usa']})
user3=pd.DataFrame(dict(name=['tsewang','ljakpa'],
      age=['23','34'],
      job=['receiptionist','tourist'],
      place=['china','spain']))
print(user3)


# In[44]:


print(user1)
print(user2)
print(user3)


# In[45]:


print(user1.append(user2))
print(user2.append(user3))


# In[46]:


users=pd.concat([user1,user2,user3])
print(users)


# In[55]:


df=users.iloc(0)
print(df)
print(users.loc())
print(users.iloc(0))


# In[5]:


my_cols=['age','job']
print(my_cols)


# In[11]:


users1=['name','place','age','job']
list1=pd.DataFrame({'name':['nyima','dolma','tashi','sangpo'],
        'age':['23','24','25','26'],
        'place':['ktm','pokhara','chitwan','biratnagar'],
        'job':['passenger','pilot','scientist','doctor']})
list2=pd.DataFrame({'name':['karfan','kaji','rinji','lhaka'],
        'age':['23','34','45','56'],
        'place':['bouddha','birstnagat','birgumj','lumbini'],
        'job':['singer','dancer','writer','a12']})
print(list1)


# In[12]:


users1


# In[35]:


df=users1.copy
print(df)


# ROW SELECTION (FILTERING)

# In[5]:


import pandas as pd
users1=['name','place','age','job']
list1=pd.DataFrame({'name':['nyima','dolma','tashi','sangpo'],
        'age':['23','24','25','26'],
        'place':['ktm','pokhara','chitwan','biratnagar'],
        'job':['passenger','pilot','scientist','doctor']})
list2=pd.DataFrame({'name':['karfan','kaji','rinji','lhaka'],
        'age':['23','34','45','56'],
        'place':['bouddha','birstnagat','birgumj','lumbini'],
        'job':['singer','dancer','writer','a12']})
print(list1)
print(list2)


# In[14]:


df=users1.copy()
df.iloc[0]
df.iloc[0,0]
df.iloc[0,0]=55
for i in range(users1.shape[0]):
    row=df.iloc[i]


# ROW SELECTION(FILTERING

# In[9]:


users1[users1.age<20]
youmg_bool=users1.age<20;
young=users1[young_bool]
users1[users1.age<20].job
users1[users.age<20]
users1[users1.age<20].job
print(young)


# SORTING

# In[17]:


import pandas as pd
columns = ['name', 'age', 'gender', 'job']
user1 = pd.DataFrame([['alice', 19, "F", "student"],
['john', 26, "M", "student"]],
columns=columns)
user2 = pd.DataFrame([['eric', 22, "M", "student"],
['paul', 58, "F", "manager"]],
columns=columns)
user3 = pd.DataFrame(dict(name=['peter', 'julie'],
age=[33, 44], gender=['M', 'F'],
job=['engineer', 'scientist']))
print(user3)
users=pd.concat([user1,user2,user3])
print(users)


# In[46]:


df=users.copy()
print(df.age.sort_values())
print(df.sort_values(by='age'))
print(df.sort_values(by='job'))
print(df.sort_values(by='age',ascending='false'))


# In[43]:


print(df.sort_values(by=['age','gender']))
print(df.sort_values(by=['age','job'],inplace=False))
print(df)

import pandas as pd
columns = ['name', 'age', 'gender', 'job']
user1 = pd.DataFrame([['alice', 19, "F", "student"],
['john', 26, "M", "student"]],
columns=columns)
user2 = pd.DataFrame([['eric', 22, "M", "student"],
['paul', 58, "F", "manager"]],
columns=columns)
user3 = pd.DataFrame(dict(name=['peter', 'julie'],
age=[33, 44], gender=['M', 'F'],
job=['engineer', 'scientist']))
print(user3)
users=pd.concat([user1,user2,user3])
print(users)
# In[10]:


df=users.copy()


# In[4]:


users.sort_values(by=['age','gender'])

print(users.age.sort_values())
# In[11]:


df.age.sort_values()


# In[12]:


df.sort_values(by=['name','age'])


# In[16]:


df.sort_values(by=["name"])


# In[18]:


df.age.sort_values()


# In[19]:


df.name.sort_values()


# In[20]:


df.sort_values(by='name')


# In[21]:


df.loc[0]


# In[23]:


df.loc[0,'age']


# In[28]:


df.iloc[0]


# In[31]:


df.loc[1,'age']


# In[32]:


users[users.age<20]


# In[33]:


users[users.age<20][['age','name']]


# In[34]:


users[users.age>20].job


# In[35]:


users[users.age>20]


# In[38]:


df.sort_values(by=['gender','job'])


# In[40]:


df.sort_values(by='name',ascending=True)


# In[44]:


print(df.sort_values(by=['name','age'],inplace=True))


# In[43]:


print(df.sort_values(by=['age','name'],inplace=False))


# In[46]:


users[(users.age>20),(users.gender=='M')]


# DESCRITIVE

# In[47]:


users.describe()


# In[48]:


df.describe()


# In[49]:


users.describe(include='all')


# In[51]:


users.describe(include=['object'])


# In[3]:


import pandas as pd
users1=['name','place','age','job']
list1=pd.DataFrame({'name':['nyima','dolma','tashi','sangpo'],
        'age':['23','24','25','26'],
        'place':['ktm','pokhara','chitwan','biratnagar'],
        'job':['passenger','pilot','scientist','doctor']})
list2=pd.DataFrame({'name':['karfan','kaji','rinji','lhaka'],
        'age':['23','34','45','56'],
        'place':['bouddha','birstnagat','birgumj','lumbini'],
        'job':['singer','dancer','writer','a12']})
list3=pd.DataFrame({'name':['john','abhram','messi','ronaldo'],
                    'place':['spain','switz','nepal','cyprus'],
                   'age':['23','45','34','26'],
                    'job':['conductor','doctor','pilot','player']
                   })
print(list1)
print(list2)
print(list3)


# In[7]:


users=pd.concat([list1,list2,list3])
print(users)


# In[11]:


df=users.copy()
users.describe()


# In[16]:


print(users.describe(include='all'))


# In[13]:


print(users.describe(include=['object']))


# In[18]:


print(df.groupby("job").mean())


# In[19]:


print(df.groupby('job')["age"].mean())


# In[24]:


print(df.groupby("job").describe(include='all'))
print(df.groupby("job"))


# In[28]:


print(df.groupby('job'))


# In[31]:


for grp, data in df.groupby("job"):
print(grp, data)


# MISSING DATA
# 
df=users.copy(
print(df.drop_duplicates(subset="first-name",keep=False,inplace=False))
print(df)
# In[1]:


import pandas as pd
users1=['name','place','age','job']
list1=pd.DataFrame({'name':['nyima','dolma','tashi','sangpo'],
        'age':['23','24','25','26'],
        'place':['ktm','pokhara','chitwan','biratnagar'],
        'job':['passenger','pilot','scientist','doctor']})
list2=pd.DataFrame({'name':['karfan','kaji','rinji','lhaka'],
        'age':['23','34','45','56'],
        'place':['bouddha','birstnagat','birgumj','lumbini'],
        'job':['singer','dancer','writer','a12']})
list3=pd.DataFrame({'name':['john','abhram','messi','ronaldo'],
                    'place':['spain','switz','nepal','cyprus'],
                   'age':['23','45','34','26'],
                    'job':['conductor','doctor','pilot','player']
                   })
print(list1)
print(list2)
print(list3)


# In[9]:


users=pd.concat([list1,list2,list3])
users


# In[17]:


df=users.copy()
print(df.describe(include='all'))
print(df.describe())


# In[19]:


print(df.age.isnull())


# In[21]:


print(df.age.notnull())


# In[25]:


print(df.describe())
print(users)
df[df.age.notnull()]


# In[27]:


df.age.notnull().sum()


# In[29]:


print(df.age.isnull().sum())


# In[33]:


df.notnull().sum()


# Drop missing value

# In[34]:


df.dropna()


# In[43]:


df.dropna()


# In[44]:


df.dropna(how='all')


# In[45]:


df.age.mean()


# In[53]:


user4=pd.DataFrame(dict(name=['nyima','passang','kelsang','tash'],
                    age=['23','45','56',67],
                place=['ktm','pkr','tibt','amer'],
                job=['analy','math','senio','doc']))
user4


# In[55]:


df=pd.concat([users,user4])
print(df)


# In[60]:


df=users.copy()
print(df)


# In[61]:


df.age.mean()


# In[67]:



df.loc[0]


# In[68]:


df.iloc[0]


# In[69]:


df.iloc[1]


# In[71]:


df.loc[1]


# In[83]:


df.age.mean()
df=users.copy()
print(df)
df.loc[df.age.isnull()]


# RENAMe VALUES

# In[103]:


df=users.copy()
a=pd.Series(df['age'])
print(a)


# In[106]:


a=pd.Series(df['name'])
print(a)


# In[1]:


import pandas as pd
users1=['name','place','age','job']
list1=pd.DataFrame({'name':['nyima','dolma','tashi','sangpo'],
        'age':['23','24','25','26'],
        'place':['ktm','pokhara','chitwan','biratnagar'],
        'job':['passenger','pilot','scientist','doctor']})
list2=pd.DataFrame({'name':['karfan','kaji','rinji','lhaka'],
        'age':['23','34','45','56'],
        'place':['bouddha','birstnagat','birgumj','lumbini'],
        'job':['singer','dancer','writer','a12']})
list3=pd.DataFrame({'name':['john','abhram','messi','ronaldo'],
                    'place':['spain','switz','nepal','cyprus'],
                   'age':['23','45','34','26'],
                    'job':['conductor','doctor','pilot','player']
                   })
print(list1)
print(list2)
print(list3)


# In[5]:


users=pd.concat([list1,list2,list3])
print(users)


# DEALING WITH OUTLIERS

# df=users.copy()
# print(df)
# print(df.dropna())

# In[13]:


df['place']


# In[15]:


df.age.mean()


# In[18]:


import numpy as np
size=pd.Series(np.random.normal(loc=175,size=20,scale=10))
print(size)


# In[22]:


size_out_mean=size.copy()

