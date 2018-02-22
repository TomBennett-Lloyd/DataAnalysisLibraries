
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___
# # Logistic Regression Project 
# 
# In this project we will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.
# 
# This data set contains the following features:
# 
# * 'Daily Time Spent on Site': consumer time on site in minutes
# * 'Age': cutomer age in years
# * 'Area Income': Avg. Income of geographical area of consumer
# * 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
# * 'Ad Topic Line': Headline of the advertisement
# * 'City': City of consumer
# * 'Male': Whether or not consumer was male
# * 'Country': Country of consumer
# * 'Timestamp': Time at which consumer clicked on Ad or closed window
# * 'Clicked on Ad': 0 or 1 indicated clicking on Ad
# 
# ## Import Libraries
# 
# **Import a few libraries you think you'll need (Or just import them as you go along!)**

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# ## Get the Data
# **Read in the advertising.csv file and set it to a data frame called ad_data.**

# In[3]:


ad_data = pd.read_csv('advertising.csv')


# **Check the head of ad_data**

# In[10]:


ad_data.head()


# ** Use info and describe() on ad_data**

# In[15]:


ad_data['Time'] = ad_data['Timestamp'].apply(pd.to_datetime)


# In[16]:


ad_data['Time'] = ad_data['Time'].dt.hour


# In[17]:


def timeCorr(hour):
    out=hour-7
    if out<1:
        out+=24
    return float(out)


# In[18]:


ad_data['TimeCorr'] = ad_data['Time'].apply(timeCorr)


# In[20]:


ad_data['NameLen'] = ad_data['Ad Topic Line'].apply(len)
ad_data['NameLen'] = ad_data['NameLen'].apply(float)


# In[24]:


ad_data['Age'] = ad_data['Age'].apply(float)


# In[25]:


ad_data.info()


# In[11]:


ad_data.describe()


# ## Exploratory Data Analysis
# 
# Let's use seaborn to explore the data!
# 
# Try recreating the plots shown below!
# 
# ** Create a histogram of the Age**

# In[26]:


sns.distplot(ad_data['Age'])


# **Create a jointplot showing Area Income versus Age.**

# In[12]:


sns.lmplot(data=ad_data,x='Age',y='Area Income',hue='Clicked on Ad')


# **Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age.**

# In[13]:


sns.jointplot(data=ad_data,x='Age',y='Area Income',kind='kde')


# ** Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'**

# In[12]:


sns.jointplot(data=ad_data,x='Daily Time Spent on Site',y='Daily Internet Usage',kind='kde')


# ** Finally, create a pairplot with the hue defined by the 'Clicked on Ad' column feature.**

# In[38]:


grid=sns.PairGrid(data=ad_data,hue='Clicked on Ad',vars=['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','TimeCorr','NameLen'], diag_sharey=False)
grid.map_diag(plt.hist)
grid.map_lower(plt.scatter, s=0.3)
grid.map_upper(sns.kdeplot, cmap="Blues")
#grid.data=ad_data[ad_data['Clicked on Ad']==0]
#grid.map_diag(sns.distplot, color='r')
#grid.map_lower(plt.scatter, color='r')
#grid.map_upper(sns.kdeplot, shade=True, cmap="Reds")


# # Logistic Regression
# 
# Now it's time to do a train test split, and train our model!
# 
# You'll have the freedom here to choose columns that you want to train on!

# ** Split the data into training set and testing set using train_test_split**

# In[39]:


from sklearn.model_selection import train_test_split


# In[95]:


X_train, X_test, y_train, y_test = train_test_split(ad_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','TimeCorr','NameLen']], ad_data['Clicked on Ad'], test_size=0.30, random_state=101)


# ** Train and fit a logistic regression model on the training set.**

# In[42]:


from sklearn.linear_model import LogisticRegression


# In[96]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# ## Predictions and Evaluations
# ** Now predict values for the testing data.**

# In[97]:


predictions = logmodel.predict(X_test)


# ** Create a classification report for the model.**

# In[98]:


from sklearn.metrics import classification_report


# In[99]:


print(classification_report(y_test,predictions))


# In[100]:


from sklearn.metrics import confusion_matrix


# In[101]:


print(confusion_matrix(y_test,predictions))


# ## Great Job!
