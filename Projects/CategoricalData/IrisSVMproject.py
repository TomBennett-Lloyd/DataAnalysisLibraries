
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___
# # Support Vector Machines Project 
# 
# Welcome to your Support Vector Machine Project! Just follow along with the notebook and instructions below. We will be analyzing the famous iris data set!
# 
# ## The Data
# For this series of lectures, we will be using the famous [Iris flower data set](http://en.wikipedia.org/wiki/Iris_flower_data_set). 
# 
# The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by Sir Ronald Fisher in the 1936 as an example of discriminant analysis. 
# 
# The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor), so 150 total samples. Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.
# 
# Here's a picture of the three different Iris types:

# In[1]:


# The Iris Setosa
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)


# In[2]:


# The Iris Versicolor
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
Image(url,width=300, height=300)


# In[3]:


# The Iris Virginica
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
Image(url,width=300, height=300)


# The iris dataset contains measurements for 150 iris flowers from three different species.
# 
# The three classes in the Iris dataset:
# 
#     Iris-setosa (n=50)
#     Iris-versicolor (n=50)
#     Iris-virginica (n=50)
# 
# The four features of the Iris dataset:
# 
#     sepal length in cm
#     sepal width in cm
#     petal length in cm
#     petal width in cm
# 
# ## Get the data
# 
# **Use seaborn to get the iris data by using: iris = sns.load_dataset('iris') **

# In[5]:


import seaborn as sns
iris = sns.load_dataset('iris') 


# Let's visualize the data and get you started!
# 
# ## Exploratory Data Analysis
# 
# Time to put your data viz skills to the test! Try to recreate the following plots, make sure to import the libraries you'll need!
# 
# **Import some libraries you think you'll need.**

# In[42]:


import pandas as pd
import numpy as np


# In[18]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# ** Create a pairplot of the data set. Which flower species seems to be the most separable?**

# In[13]:


iris.info()


# In[12]:


iris['species'].unique()


# In[19]:


sns.pairplot(iris,hue='species')


# **Create a kde plot of sepal_length versus sepal width for setosa species of flower.**

# In[25]:


setosa = iris[iris['species']=='setosa']
sns.kdeplot(setosa['sepal_width'],setosa['sepal_length'],shade=True, shade_lowest=False, cmap="plasma")


# # Train Test Split
# 
# ** Split your data into a training set and a testing set.**

# In[27]:


from sklearn.model_selection import train_test_split


# In[51]:


X=iris.drop('species',axis=1)
y=iris['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# # Train a Model
# 
# Now its time to train a Support Vector Machine Classifier. 
# 
# **Call the SVC() model from sklearn and fit the model to the training data.**

# In[31]:


from sklearn.svm import SVC


# In[32]:


model=SVC()


# In[35]:


model.fit(X=X_train,y=y_train)


# ## Model Evaluation
# 
# **Now get predictions from the model and create a confusion matrix and a classification report.**

# In[36]:


predictions=model.predict(X_test)


# In[37]:


from sklearn.metrics import classification_report,confusion_matrix


# In[38]:


print(confusion_matrix(predictions,y_test))


# In[39]:


print(classification_report(predictions,y_test))


# Wow! You should have noticed that your model was pretty good! Let's see if we can tune the parameters to try to get even better (unlikely, and you probably would be satisfied with these results in real like because the data set is quite small, but I just want you to practice using GridSearch.

# ## Gridsearch Practice
# 
# ** Import GridsearchCV from SciKit Learn.**

# In[40]:


from sklearn.model_selection import GridSearchCV


# **Create a dictionary called param_grid and fill out some parameters for C and gamma.**

# In[43]:


param_grid={'C':[0.1,1,5,10,20,50,100],'gamma':[1,0.1,0.01,0.001,0.0001]}


# ** Create a GridSearchCV object and fit it to the training data.**

# In[52]:


gSearcher=GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
gSearcher.fit(X_train,y_train)


# ** Now take that grid model and create some predictions using the test set and create classification reports and confusion matrices for them. Were you able to improve?**

# In[53]:


predictions2=gSearcher.predict(X_test)


# In[54]:


gSearcher.best_estimator_


# In[55]:


print(confusion_matrix(predictions2,y_test))


# In[56]:


print(classification_report(predictions2,y_test))


# You should have done about the same or exactly the same, this makes sense, there is basically just one point that is too noisey to grab, which makes sense, we don't want to have an overfit model that would be able to grab that.

# ## Great Job!
