#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import matplotlib.pyplot as plt 
fig=plt.figure()
ax1=fig.add_subplots(2,4,5)
plt.style.use('fivethirtyeight')
x=[2,3,4,6,7]
y=[3,4,5,7,7]
plt.plot(x,y,label='boys_age',color='k',linestyle='--')
x=[5,6,7,8,9]
y=[7,8,3,4,5]
plt.plot(x,y,label='girls_age',color='b',linestyle='--')
plt.legend()
plt.xlabel('boys')
plt.ylabel('girls')
plt.title('hostel')
plt.grid('True')
plt.tight_layout()


# In[39]:


import numpy as np
import matplotlib.pyplot  as plt
plt.style.use('fivethirtyeight')
x=[2,3,4,6,7]
y=[3,4,5,7,7]
x_indexes=np.arange(len(x))
width=0.25
plt.bar(x_indexes - width,y,width=width,label='boys_age',color='k')
x=[3,4,5,8]
y=[4,3,7,9]
plt.bar(x_indexes,y,width=width,label='gay',color='g')
x=[5,6,7,8,9]
y=[7,8,3,4,5]
plt.bar(x_indexes + width,y,width=width,label='girls_age',color='k')
plt.xlabel('boys')
plt.ylabel('girls')
plt.legend()


# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import panda as pd
import requests
result=requests
result=requests.get("https://www.trackcorona.live/api/countries")
result


# In[8]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
result=requests
result=requests.get("https://www.trackcorona.live/api/countries")
result


# In[15]:


result.headers


# In[16]:


result.text


# In[18]:


result.json()


# In[22]:


df=pd.DataFrame(result)
print(df)


# In[24]:


df=pd.DataFrame(result)
print(df.head(2))
print(df.tail(2))


# In[39]:


import matplotlib.pyplot as plt
fig=plt.figure()
fig=plt.figure(figsize=plt.figaspect(2.0))
fig.add_axes()
ax1 = fig.add_subplot(221)
ax3 = fig.add_subplot(212)
fig3, axes = plt.subplots(nrows=2,ncols=2)
fig4, axes2 = plt.subplots(ncols=3)
ax1.scatter([2,3,4],[4,5,6],[5,4,3],color='darkgreen',marker='.')
x_label='ball'
ylabelb='cat'
titles='dog'


# In[66]:


import numpy as npn
import matplotlib.pyplot as plt
fig=plt.figure()
plt.style.use("fivethi")
x=[2,3,4,6,7,8]
y=[3,2,4,5,6,7]
plt.plot(x,y,color='darkgreen',linewidth='4',label='ball')
xlabel=('dog')
ylabel=('cat')
titles=('bash')
x=[3,5,6,4,3,5]
y=[4,3,5,7,8,9]
plt.scatter(x,y,color='blue',marker='*',label='cat')
plt.bar(x,y,color='b')
plt.grid('true')
plt.legend()
plt.show()


# In[35]:


import matplotlib.pyplot as plt
fig=plt.figure()
plt.style.use('fivethirtyeight')
plt.style.use('ggplot')
x=range(2,60)
y=[value*3 for value in x]
plt.plot(x,y)
plt.xlabel('x-axis')
plt.ylabel('y-axis') 
plt.title('population of kathmandu')
plt.grid('true')
plt.show()


# In[23]:


import matplotlib.pyplot as plt
x=[1,2,3]
y=[2,3,4]
plt.plot(x,y)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('population')
plt.grid('true')
plt.show()


# In[34]:


import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv("titanic.csv",parse_dates=True,index_col=0,sep=',')
plt.plot()
df.plot()
df.title('titanic')
df.xlabel('passengerid')
plt.show()


# In[53]:


import matplotlib.pyplot as plt
x1=[10,20,30]
y1=[20,40,10]
plt.plot('x1,y1',label="line 1",linewidth='4')
x2=[10,20,30]
y2=[20,40,10]
plt.plot('x2,y2',label="line 2",linewidth='4')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('population')
plt.legend()
plt.grid('true')
plt.show()


# In[48]:


import matplotlib.pyplot as plt
# line 1 points
x1 = [10,20,30]
y1 = [20,40,10]
# plotting the line 1 points 
plt.plot(x1, y1, label = "line 1")
# line 2 points
x2 = [10,20,30]
y2 = [40,10,30]
# plotting the line 2 points 
plt.plot(x2, y2, label = "line 2")
plt.xlabel('x - axis')
# Set the y axis label of the current axis.
plt.ylabel('y - axis')
# Set a title of the current axes.
plt.title('Two or more lines on same plot with suitable legends ')
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()


# In[32]:


import numpy as np
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
ages_x=(23,34,56,67,78,67,89,90)
x_indexes=np.arange(len(ages_x))
width=0.25
workers_y=(345,456,567,456,897,789,789,890)
plt.bar(x_indexes-width,workers_y,label='workers',color='k',width=width,linewidth='0.34')
staff_y=(345,456,567,678,345,789,785,567)
plt.bar(x_indexes,staff_y,label='staff',color='g',width=width,linewidth='2.0')
student_y=(234,456,567,678,789,890.901,902)
plt.bar(x_indexes+width,staff_y,label='student',color='r',width=width,linewidth='7')
plt.legend()
plt.xticks(ticks=x_indexes,labels=ages_x)
plt.title("information about salary")
plt.xlabel("ages")
plt.ylabel("salary")
plt.tight_layout()
plt.show()


# In[12]:


import numpy as np
import matplotlib.pyplot as plt
data = [[30, 25, 50, 20],
[40, 23, 51, 17],
[35, 22, 45, 19]]
x=np.arange(4)
fig=plt.figure(figsize=(10,5))
fig=plt.figure()


# import matplotlib.pyplot as plt
# y = [1, 4, 9, 16, 25,36,49, 64]
# x1 = [1, 16, 30, 42,55, 68, 77,88]
# x2 = [1,6,12,18,28, 40, 52, 65]
# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# l1 = ax.plot(x1,y,'bD-') # solid line with yellow colour and square marker
# l2 = ax.plot(x2,y,'go--') # dash line with green colour and circle marker
# ax.legend(labels = ('tv', 'Smartphone'), loc = 'lower right') # legend placed at lower right
# ax.set_title("Advertisement effect on sales")
# ax.set_xlabel('medium')
# ax.set_ylabel('sales')
# plt.show()

# In[11]:


import matplotlib.pyplot as plt
fig=plt.figure()
ax1=fig.add_subplot(111)
ax1.plot([1,2,3])
ax2=fig.add_subplot(111,facecolor='b')
ax2.plot([1,2,7])


# In[33]:


import matplotlib.pyplot as plt
fig,a =  plt.subplots(2,2)
import numpy as np
x = np.arange(1,5)
a[0][0].plot(x,x*x)
a[0][1].set_title('square')
a[0][0].plot(x,np.sqrt(x))
a[0][1].set_title('square root')
a[1][0].plot(x,np.exp(x))
a[1][0].set_title('exp')
a[1][1].plot(x,np.log10(x))
a[1][1].set_title('log')
plt.show()


# In[30]:


import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure()
ax1=fig.add_subplots(1,3,figsize=(12,4))
x=np.arange(1,9)
ax1=plt.plot(x,x**3,'g',lw=2)
ax1.grid(True)


# In[ ]:





# In[ ]:




