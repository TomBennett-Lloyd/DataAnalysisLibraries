# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 13:41:33 2018

@author: Tom
"""

import numpy as np
import seaborn as sns
import pandas as pd

class missingDataAnalyser():
    
    def __init__ (self, train, test=pd.DataFrame(), responseVar = np.nan):
        self.train = train
        self.test = test
        if len(test)>0 and np.isnan(responseVar):
            self.responseVar = self.findResponseVar()
        else:
            self.responseVar = responseVar
        self.rmMissingResponse()
        self.allDat=pd.concat([self.train,self.test])
        self.checkMissing()

    def findResponseVar (self):
        testCol=set(self.test.columns.values)
        resp=[]
        for col in set(self.train.columns.values):
            if col not in testCol or all(self.test[col].isnull()):
                resp.append(col)
        if len(resp)<0:
            resp=np.nan
            
        return resp
        
    def rmMissingResponse (self):
        if type(self.responseVar)==list:
            for resp in self.responseVar:
                self.train.drop(self.train[resp].isnull(),inplace=True)

    def checkMissing(self):
        if type(self.responseVar)!=list:
            data=self.allDat.isnull()
        else:
            data=self.allDat.drop(self.responseVar,axis=1).isnull()
        
        sns.heatmap(data,yticklabels=False,cbar=False,cmap='viridis')
    
    def imputeMissing (self,function=np.nan,columns=np.nan):
        #impute missing values from training and test
        if not (np.isnan(function) and np.isnan(columns)):
            #user has defined imputation function
            imputedData=self.allDat[columns].apply(function)
        else:
            if np.isnan(columns):
                if not np.isnan(function):
                    print('This method cannot handle a function without knowing which columns to apply it to')
                    return self.allDat
                else:
                    columns=self.getSparseCols()
                    for index, col in columns.iterrows():
                       self.imputeModel(col)
                    imputedData=self.allDat
        return imputedData
    
    def imputeModel(self,col):
        if any(col[0]==resp for resp in self.responseVar):
            pass
        elif col[1]<0.02:
            self.allDat.drop(self.allDat[col[0]].isnull(),inplace=True)
        elif col[1]>0.3:
            self.allDat.drop(col[0],axis=1,inplace=True)
        else:
            pass
        
    
    def getSparseCols(self):
        sparse=[]
        for col in set(self.allDat.columns.values):
            boolList=self.allDat[col].isnull()
            if any(boolList):
                counts=boolList.value_counts(normalize=True)
                sparse.append([col,counts.loc[True]])
        if len(sparse)<0:
            sparse=np.nan
        else:
            sparse=pd.DataFrame(sparse)
            sparse=sparse.iloc[sparse.iloc[:,1].argsort()]
        return sparse


     