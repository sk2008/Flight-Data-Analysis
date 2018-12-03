
# coding: utf-8

# In[328]:


import requests
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib 
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model


# In[329]:


response = requests.get("https://opensky-network.org/api/states/all")
#response.content


# In[330]:


json_data = response.json()
#print(json_data)


# In[331]:


a =json_data["states"]
#a


# In[332]:


df = pd.DataFrame(a)
df


# In[333]:



df.rename(columns={0 : "icao24",1 : "callsign",2 : "origin_country",3 : "time_position",
                   4 : "last_contact",5 : "longitude",6 : "latitude",7 : "geo_altitude",
                   8 : "on_ground",9 : "velocity",10 : "heading",11 : "vertical_rate",
                   12 : "sensors",13 : "baro_altitude",14 : "squawk",15 : "spi",
                   16 : "position_source"}, inplace=True)


# In[334]:


df.columns


# In[335]:


#Keep only the rows with at least 15 non-na values:
df = df.dropna(thresh=15)


# In[336]:


df.drop(df.columns[[0, 1, 2, 3, 4, 12, 16]], axis=1, inplace=True)
df = df.dropna(thresh=10)
df


# In[337]:


df.spi.unique()


# In[338]:


y = df.spi


# In[339]:


from sklearn.cross_validation import train_test_split


# In[340]:


X_train, X_test, y_train, y_test = train_test_split(df, y, 
                                                    test_size=0.2, 
                                                    random_state=0)


# In[341]:


print(X_train.shape, y_train.shape)


# In[342]:


print(X_test.shape, y_test.shape)


# In[343]:


from sklearn import tree
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[344]:


model = LogisticRegression()


# In[345]:


model.fit(X_train, y_train)


# In[346]:


model.score(X_train, y_train)


# In[347]:


#Predict Output
predicted= model.predict(X_test)


# In[348]:


predicted


# In[349]:


# calculate accuracy
from sklearn import metrics


# In[350]:


print(metrics.accuracy_score(y_test, predicted))


# In[351]:


s = pd.Series(df.spi)


# In[352]:


finalCount = s.value_counts()


# In[353]:


finalCountDataframe = pd.DataFrame(finalCount)


# In[354]:


finalCountDataframe


# In[355]:


finalCountDataframe.rename(columns={'spi' : "count"}, inplace=True)


# In[356]:


finalCountDataframe


# In[357]:


finalCountDataframe = pd.DataFrame.transpose(finalCountDataframe)


# In[358]:


finalCountDataframe


# In[359]:


import matplotlib.pyplot as plt



# In[360]:


FinalBarPlot = finalCountDataframe.plot(kind='bar', title ="V comp", figsize=(15, 10), legend=True, fontsize=12)


# In[361]:


import numpy as np
import matplotlib.pyplot as plt



# In[366]:



N = len(df.index)



# In[367]:


N


# In[368]:


x = df.velocity
y = df.heading


# In[369]:


colors = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radii

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()

