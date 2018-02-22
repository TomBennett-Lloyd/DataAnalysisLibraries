
# coding: utf-8

##--Set up--##

##import libraries

# Import the built-in logging module and configure it so that Word2Vec creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from Qualitative_modelling import readData,split,stringCleaner,W2Vtrain,AveFeatureVectoriser,initModels,writeOut

##--Start of main code--##

train,ulTrain,test = readData(["labeledTrainData.tsv","unlabeledTrainData.tsv","testData.tsv"])

##initialise Bag of Words Transformer

print('initialising Bag of Words transformer')
bow_transformer = CountVectorizer(analyzer = split,tokenizer = None,preprocessor = None,stop_words = None,max_features = 5000)

print('applying cleaning algorithm to training and test data')
data=[]#empty list to store potentially multiple string cleaning methods
data.append([train["review"].apply(stringCleaner),test["review"].apply(stringCleaner)])

#train Word2Vec model (has own string cleaning)
W2Vmodel=W2Vtrain([train["review"],ulTrain["review"]])

trainTypes=['_bow_array','_W2V','_tfidf']

for datind,preDat in enumerate(data):
    print('applying Bag of Words transformation to preprocessed data '+str(datind))
    train_data_features = bow_transformer.fit_transform(preDat[0])
    train_data = [train_data_features.toarray()]

    test_data_features = bow_transformer.fit_transform(preDat[1])
    test_data = [test_data_features.toarray()]

    print('get average feature vectors from W2V for dataset '+str(datind))
    aFV = AveFeatureVectoriser(preDat[0], W2Vmodel)
    train_data.append(aFV.getAvgFeatureVecs())
    aFV = AveFeatureVectoriser(preDat[1], W2Vmodel)
    test_data.append(aFV.getAvgFeatureVecs())

    print('applying TFIDF transformation to Bag of Words '+str(datind))
    ##make tfidf transformed data
    tfidf_transformer = TfidfTransformer().fit(train_data[0])
    train_data.append(tfidf_transformer.transform(train_data_features))
    test_data.append(tfidf_transformer.transform(test_data_features))

    for testDat,trainDat,trainType in zip(test_data,train_data,trainTypes):
        ##fit models
        print('initialising the models for training data type '+trainType)
        models=initModels(trainType)
        for pModel in models:
            print('fitting the '+pModel[1]+' model')
            name=pModel[1]+trainType
            thisModel = pModel[0].fit(trainDat, train["sentiment"])
            print('predicting test data using the '+pModel[1]+' model')
            result = thisModel.predict(testDat)
            writeOut(result,name,test)
