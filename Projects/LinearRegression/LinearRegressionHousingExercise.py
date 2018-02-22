# coding: utf-8

from LinearRegressionTools import linearModelBuilder,readData


data=readData('USA_Housing.csv')



y=data['Price']

""" First model run with all numeric variables """

X=data.drop(['Price','Address'],axis=1)
lm=linearModelBuilder(X,y)

""" 
The results from the first model with all of the numeric variables included
showed that the Area population had relativley little impact on the value of
the houses when combined with the other variables. To improve parsimony of the
model I wondered whether droping the Area Population might improve the models 
performance on the test data. The results without this variable showed however
that the model lost predictive power without this variable.
"""

X=data.drop(['Price','Address','Area Population'],axis=1)
lm2=linearModelBuilder(X,y)


"""
We can see from the pairplot that there is understandably some corelation 
between the number of bedrooms and the total number of rooms. This colinearity,
whilst only slight could reduce model performance. Because the number of 
bedrooms is essentially a subset of total rooms, I decided that the proportion
of the rooms that are bedrooms might be a more useful description of the type 
of house than the number of bedrooms alone, with the total rooms being a good
descriptor of house size. This retains the information from both variables
but ensures that they are no longer colinear. The results showed a significant
improvement in model performance.

"""

data['rooms']=data['Avg. Area Number of Bedrooms']/data['Avg. Area Number of Rooms']
X=data.drop(['Price','Address','Avg. Area Number of Bedrooms'],axis=1)
lm2=linearModelBuilder(X,y)