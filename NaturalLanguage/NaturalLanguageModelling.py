
# coding: utf-8

##--Set up--##

##import libraries

import pandas as pd
import numpy as np
import re

from nltk.corpus import stopwords
import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

from bs4 import BeautifulSoup as BS

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier


def stringCleaner (string, remove_stopwords=True,join=True,regex=False):
#function to clean string data into a bag of words

    #remove html markup
    review_text=BS(string, "lxml")
    try:
        ##apply regular expressions to filter each string
        if regex==False:
            #use a basic filter to strip all non letters
            usefulChar=re.sub("[^a-zA-Z]"," ",review_text.get_text())
        elif regex:
            #get rid of non comparable punctuation
            usefulChar=re.sub("""["\)\(,&=:;\.]""","",review_text.get_text())
            #get rid of
            usefulChar=re.sub("""([a-zA-Z0-9\)"])([^?!'a-zA-Z]|s'|'s)([a-zA-Z\s"$])""","\g<1> \g<3>",usefulChar)
            #isolate potentially useful punctuation e.g. ?!... = ? ! ...
            usefulChar=re.sub("([?!]|\.\.+)"," \g<1> ",usefulChar)
        else:
            try:
                usefulChar=re.sub(regex[1],regex[2],review_text.get_text())
            except:
                print(str(regex)+" is not a valid regex input and output, reverting to default.")
                #use a basic filter to strip all non letters
                usefulChar=re.sub("[^a-zA-Z]"," ",review_text.get_text())

        ##split into words ('tokenize')
        words=usefulChar.lower().split()

        ##filter for meaningful words
        if remove_stopwords:
            stops=set(stopwords.words("english"))
            meaningful_words = [w for w in words if not w in stops]
        else:
            meaningful_words=words
        if join:
            out=" ".join( meaningful_words )
        else:
            out=meaningful_words

    except:
        print(usefulChar)

    return( out )



def corpus_to_sentences( review, tokenizer=tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    #
    # Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []

    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences+=stringCleaner(raw_sentence,remove_stopwords,join=False)

    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences

class AveFeatureVectoriser():
    """docstring for AveFeatureVectoriser.
Class for building Average Feature Vectors.
    """
    def __init__(self, corpuses, model):
        self.corpuses = corpuses
        self.model=model
        self.num_features=model.vector_size


    def makeAveFeatureVec(self, words):
        # Function to average all of the word vectors in a given paragraph
        featureVec = np.zeros((self.num_features,),dtype="float32")
        nwords = 0.
        # Index2word is a list that contains the words in the model's vocabulary.
        index2word_set = set(self.model.wv.index2word)

        for word in words:
            if word in index2word_set:
                nwords+= 1.
                featureVec = np.add(featureVec,self.model[word])

        avFeatureVec = np.divide(featureVec,nwords)
        return avFeatureVec


    def getAvgFeatureVecs(self):
        # Given a set of reviews (each one a list of words), calculate
        # the average feature vector for each one and return a 2D numpy array
        outof=len(self.corpuses)
        counter = 0
        aveFeatureVecs = np.zeros((len(self.corpuses),self.num_features),dtype="float32")

        for corpus in self.corpuses:
           # Print a status message every 1000th review
           Cstatus (counter,outof)
           # Call the makeAveFeatureVec to make average feature vector
           aveFeatureVecs[counter] = self.makeAveFeatureVec(corpus)
           counter += 1
        return aveFeatureVecs

def W2Vtrain (TrainData):
    sentences = []  # Initialize an empty list of sentences

    print ("Parsing sentences from training sets")
    for data in TrainData:
        for corpus in data:
            sentences+=[corpus_to_sentences(corpus)]


    # Set values for Word2Vec parameters
    num_features = 300    # Word vector dimensionality
    min_word_count = 40   # Minimum word count
    num_workers = 6       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    # Initialize and train Word2Vec model (this will take some time)
    from gensim.models import word2vec
    print( "Training model...")
    W2Vmodel = word2vec.Word2Vec(sentences, workers=num_workers,size=num_features, min_count = min_word_count, window = context, sample = downsampling)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    W2Vmodel.init_sims(replace=True)

    print('model shape is: '+str(W2Vmodel.wv.syn0.shape))

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = str(num_features)+"features_"+str(min_word_count)+"minwords_"+str(context)+"context"
    W2Vmodel.save(model_name)
    return(W2Vmodel)


def writeOut (result,name,test):
    print('writing results to CSV for '+name+' Model!')
    output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
    output.to_csv("results/BOW_"+name+"_model.csv",index=False,quoting=3)

def split(string):
    return string.split()

def Cstatus (counter,outof,interval=1000):
    if counter%interval == 0:
        print ("Review %d of %d" % (counter, outof))

def readData (files):
    ##Load in data and clean the text
    print('loading in datasets...')
    dataOut=[];
    for Dfile in files:
        dataOut.append(pd.read_csv("data/"+Dfile, header=0, delimiter="\t", quoting=3))

    return(dataOut)


def initModels(trainType):
    models=[]
    ##append initialised models to list
    if trainType != '_W2V':
        models.append([MultinomialNB(),'MultinomialNB'])
    models.append([RandomForestClassifier(n_estimators = 100),'RandomForest'])
    return models
