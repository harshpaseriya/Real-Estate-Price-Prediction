#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[46]:


ds = pd.read_csv(r"A:\ML Work\Real Estate Price Pridicting Project\dataset.csv")
ds.head()


# In[47]:


ds.groupby('area_type')['area_type'].agg('count')


# In[48]:


ds1 = ds.drop(['area_type','availability','society'], axis = 'columns')
ds1.head()


# In[49]:


ds1.shape


# In[50]:


ds1.isnull().sum()


# In[51]:


ds2 = ds1.dropna()
ds2.isnull().sum()


# In[52]:


ds2.shape


# In[53]:


ds2['bhk'] = ds2['size'].apply(lambda x: int(x.split(' ')[0]))
ds2.head()


# In[54]:


ds2['total_sqft'].unique()


# In[55]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[56]:


ds2[~ds2['total_sqft'].apply(is_float)].head()


# In[57]:


ds2['total_sqft'].unique()
ds2.head()


# In[58]:


def total_sqft_con(x):
    tokens = x.split('-')
    if len(tokens)==2:
        return(float(tokens[0]) + float(tokens[1]))/2 
    try: 
        return float(x)
    except: 
        return None


# In[59]:


ds2.head()
ds2.shape


# In[60]:


ds3= ds2.copy()
ds3['total_sqft'] = ds3['total_sqft'].apply(total_sqft_con)
ds3.head()


# In[61]:


ds3['total_sqft'].unique()


# In[80]:


ds4 = ds3.drop(['size'],axis = 'columns')
ds4['price_per_sqft'] = ds4['price']*100000 / ds4['total_sqft']
ds4.head()


# In[81]:


ds5 = ds4.copy()
len(ds5.location.unique())
location_stats = ds5.groupby('location')['location'].agg('count').sort_values(ascending = False)


# In[82]:


len(location_stats[location_stats<=10])


# In[83]:


location_stats_less_than_10 = location_stats[location_stats<=10]
ds5.location = ds5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(ds5.location.unique())


# In[84]:


ds6 = ds5[~(ds5.total_sqft/ds5.bhk<300)]
ds6.shape


# In[98]:


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index = True)
    return df_out


# In[99]:


ds7 = remove_pps_outliers(ds6)


# In[100]:


ds7.shape


# In[104]:


def scatter_fun(df,location):
    bhk2 = df[(df.location == location) & (df.bhk == 2)]
    bhk3 = df[(df.location == location) & (df.bhk == 3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft, bhk2.price, color = 'blue', label = '2 BHK', s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price, marker = '+', color = 'green', label = '2 BHK', s=50)
    plt.xlabel('Total Area')
    plt.ylabel('price')
    plt.title(location)
    plt.legend


# In[105]:


scatter_fun(ds7,'Rajaji Nagar')

