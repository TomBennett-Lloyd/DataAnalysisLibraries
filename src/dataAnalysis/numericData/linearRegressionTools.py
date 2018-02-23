# coding: utf-8
"""
Created on Thu Feb 22 16:21:37 2018

@author: Tom

Set of tools for linear regression modelling
"""


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

class linearModelBuilder():
    
    def __init__(self,X,y):
        self.lm=self.build(X,y)
        
        

    def build(self,X,y):
        
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn import metrics
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        
        lm=LinearRegression()
        lm.fit(X_train,y_train)
        self.y_predict=lm.predict(X_test)
        
        self.coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
        print(self.coeff_df)
        
       
        
        plt.scatter(y_test,self.y_predict)
        plt.show()
        
        
        print('MAE:', metrics.mean_absolute_error(y_test, self.y_predict))
        print('MSE:', metrics.mean_squared_error(y_test, self.y_predict))
        print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, self.y_predict)))
        
        self.residuals=y_test-self.y_predict
        sns.distplot(self.residuals)
        
        
        return lm


def pairgrid(data, hue=[False]):
    if any(h!=False for h in hue):
        g = sns.PairGrid(data,hue=hue)
    else:
        g = sns.PairGrid(data)
    g = g.map_upper(sns.regplot)
    g = g.map_lower(sns.kdeplot, cmap="Blues_d")
    g = g.map_diag(sns.distplot)
    

def readData(file):
    data=pd.read_csv(file)
    printInfo(data)
    pairgrid(data)
    return (data)

def printInfo (df):
    print(df.head())
    print(df.describe())
    print(df.info())

