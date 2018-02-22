
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___
# # K Means Clustering Project 
# 
# For this project we will attempt to use KMeans Clustering to cluster Universities into to two groups, Private and Public.
# 
# ___
# It is **very important to note, we actually have the labels for this data set, but we will NOT use them for the KMeans clustering algorithm, since that is an unsupervised learning algorithm.** 
# 
# When using the Kmeans algorithm under normal circumstances, it is because you don't have labels. In this case we will use the labels to try to get an idea of how well the algorithm performed, but you won't usually do this for Kmeans, so the classification report and confusion matrix at the end of this project, don't truly make sense in a real world setting!.
# ___
# 
# ## The Data
# 
# We will use a data frame with 777 observations on the following 18 variables.
# * Private A factor with levels No and Yes indicating private or public university
# * Apps Number of applications received
# * Accept Number of applications accepted
# * Enroll Number of new students enrolled
# * Top10perc Pct. new students from top 10% of H.S. class
# * Top25perc Pct. new students from top 25% of H.S. class
# * F.Undergrad Number of fulltime undergraduates
# * P.Undergrad Number of parttime undergraduates
# * Outstate Out-of-state tuition
# * Room.Board Room and board costs
# * Books Estimated book costs
# * Personal Estimated personal spending
# * PhD Pct. of faculty with Ph.D.’s
# * Terminal Pct. of faculty with terminal degree
# * S.F.Ratio Student/faculty ratio
# * perc.alumni Pct. alumni who donate
# * Expend Instructional expenditure per student
# * Grad.Rate Graduation rate

# ## Import Libraries
# 
# ** Import the libraries you usually use for data analysis.**

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# ## Get the Data

# ** Read in the College_Data file using read_csv. Figure out how to set the first column as the index.**

# In[8]:


College_Data=pd.read_csv('College_Data',index_col=0)


# **Check the head of the data**

# In[10]:


College_Data.head()


# ** Check the info() and describe() methods on the data.**

# In[11]:


College_Data.info()


# In[12]:


College_Data.describe()


# ## EDA
# 
# It's time to create some data visualizations!
# 
# ** Create a scatterplot of Grad.Rate versus Room.Board where the points are colored by the Private column. **

# In[45]:


sns.lmplot(data=College_Data,x='Room.Board',y='Grad.Rate',hue='Private',palette='coolwarm',fit_reg=False)


# **Create a scatterplot of F.Undergrad versus Outstate where the points are colored by the Private column.**

# In[20]:


sns.lmplot(data=College_Data,x='Outstate',y='F.Undergrad',hue='Private',palette='coolwarm',fit_reg=False)


# ** Create a stacked histogram showing Out of State Tuition based on the Private column. Try doing this using [sns.FacetGrid](https://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.FacetGrid.html). If that is too tricky, see if you can do it just by using two instances of pandas.plot(kind='hist'). **

# In[32]:


sns.set_style('darkgrid')
hplot=sns.FacetGrid(data=College_Data,hue='Private',palette='coolwarm',size=6,aspect=2)
hplot.map(plt.hist,'Outstate',bins=20,alpha=0.7)


# **Create a similar histogram for the Grad.Rate column.**

# In[44]:


sns.set_style('darkgrid')
hplot=sns.FacetGrid(data=College_Data,hue='Private',palette='coolwarm',size=6,aspect=2)
hplot.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)


# ** Notice how there seems to be a private school with a graduation rate of higher than 100%.What is the name of that school?**

# In[43]:


College_Data.loc[College_Data['Grad.Rate'].idxmax,'Grad.Rate']=100


# ** Set that school's graduation rate to 100 so it makes sense. You may get a warning not an error) when doing this operation, so use dataframe operations or just re-do the histogram visualization to make sure it actually went through.**

# In[93]:





# In[94]:





# In[95]:





# ## K Means Cluster Creation
# 
# Now it is time to create the Cluster labels!
# 
# ** Import KMeans from SciKit Learn.**

# In[46]:


from sklearn.cluster import KMeans


# ** Create an instance of a K Means model with 2 clusters.**

# In[47]:


clusterModel=KMeans(n_clusters=2)


# **Fit the model to all the data except for the Private label.**

# In[49]:


clusterModel.fit(College_Data.drop('Private',axis=1))


# ** What are the cluster center vectors?**

# In[50]:


clusterModel.cluster_centers_


# ## Evaluation
# 
# There is no perfect way to evaluate clustering if you don't have the labels, however since this is just an exercise, we do have the labels, so we take advantage of this to evaluate our clusters, keep in mind, you usually won't have this luxury in the real world.
# 
# ** Create a new column for df called 'Cluster', which is a 1 for a Private school, and a 0 for a public school.**

# In[59]:


def clustConverter(private):
    out=-1
    if private=='Yes':
        out=0
    else:
        out=1
    return out


# In[60]:


College_Data['Cluster']=College_Data['Private'].apply(clustConverter)


# In[61]:


College_Data.head()


# ** Create a confusion matrix and classification report to see how well the Kmeans clustering worked without being given any labels.**

# In[62]:


from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(College_Data['Cluster'],clusterModel.labels_))
print('n/')
print(classification_report(College_Data['Cluster'],clusterModel.labels_))


# Not so bad considering the algorithm is purely using the features to cluster the universities into 2 distinct groups! Hopefully you can begin to see how K Means is useful for clustering un-labeled data!
# 
# ## Great Job!
