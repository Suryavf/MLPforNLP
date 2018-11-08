#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 15:51:50 2018

@author: victor
"""
import numpy  as np
import pandas as pd
import pyprind
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

print(chr(27) + "[2J")

words = np.load('words.npy').item()
Wvals = np.array([words[p] for p in words])

stop = stopwords.words('english')
porter = PorterStemmer()



"""
Fun. stream documents
---------------------
"""
def stream_docs(path):
    with open(path, 'r') as csv:
        next(csv) # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label   


"""
Fun. tokenizer
--------------
"""
def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    text = [w for w in text.split() if w not in stop]
    tokenized = [porter.stem(w) for w in text]
    return text

# Return a lower case proccesed text
def preprocessing(texto):
    import re
    REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\n)")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    texto = REPLACE_NO_SPACE.sub('', texto.lower())
    texto = REPLACE_WITH_SPACE.sub(' ', texto)
    return texto


"""
Get batch
---------
"""
def get_minibatch(doc_stream, size):
    docs, y = [], []
    for _ in range(size):
        text, label = next(doc_stream)
        docs.append(text)
        y.append(label)
    return docs, y

##  ---------------------------------------------------------------------------    



doc_stream = stream_docs(path='shuffled_movie_data.csv')
x_, y_ = get_minibatch(doc_stream, size=50000)
count_words = [len(preprocessing(w).split())  for w in x_]
import seaborn as sns
sns.distplot(count_words,bins=100)

threshold_words = np.percentile(count_words,80)




##  ---------------------------------------------------------------------------   



"""
Features extraction
-------------------
"""
def featuresExtraction(text):
    # Review to values
    review = list()
    count = 0
    
    for w in text.split():
        if w in words:
            review.append( words[w] )
            count += 1
        
        if not count<threshold_words:
            break
            
    while count<threshold_words:
        review.append( np.zeros(300) )
        count += 1
    
    # Get feature
    review = np.array(review).T
    #feature = np.dot(Wvals,review)
    
    
    return review.flatten() #np.amax(feature, axis=1) #feature.flatten()


"""
No-Lineal functions
-------------------
"""
def sigmoid(z):
    return 1/(1 + np.exp(-z))
def sigmoid_deriv(z):
    return np.multiply( sigmoid(z), 1 - sigmoid(z) )
def tanh(z):
    return np.tanh(z)
def tanh_deriv(z):
    tanhz = tanh(z)
    return 1 - tanhz*tanhz
def ReLU(z):
    return np.multiply( int(z>=0), z )
def ReLU_deriv(z):
    return int(z>=0)
def SELU(z):
    neg = np.multiply( int(z< 0), 1.758094*( np.exp(z) - 1 ) )
    pos = np.multiply( int(z>=0), 1.0507  *         z        )
    return neg + pos
def SELU_deriv(z):
    return np.multiply( int(z<0), 1.758094*np.exp(z) ) + 1.0507*int(z>=0)
def softmax(z):
    return np.exp(z)/np.sum(np.exp(z))




def initialize_he(n_input, n_output):    
    return np.random.randn(int(n_input),n_output )*np.sqrt( 2/(n_input+n_output) )


"""
Feedforward
-----------
"""
def feedforward(w, x):
    
    # Layers
    z1 = np.dot( x[np.newaxis,:], w[0] )
    z2 = np.dot( sigmoid(z1)    , w[1] ) # sigmoid
    z3 = np.dot( sigmoid(z2)    , w[2] ) # sigmoid

    out = {"z1" : z1,
           "z2" : z2,
           "z3" : z3}
    
    return out


"""
Backpropagation
---------------
"""
def backpropagation(w, x, y, z, learning_rate):

    w1 = w[0]; w2 = w[1]; w3 = w[2];
    
    a1 = sigmoid(z["z1"]).T
    a2 = sigmoid(z["z2"]).T
    a3 = sigmoid(z["z3"]).T
    
    # Layer 3
    mod_a3 = sigmoid_deriv(z["z3"]).T # np.multiply( a3, 1-a3 )
    delta3 = np.multiply( a3 - y , mod_a3 )
    gradW3 = np.dot(a2,delta3.T)
    w3 = w3 - learning_rate*gradW3
    
    # Layer 2
    mod_a2 = sigmoid_deriv(z["z2"]).T # np.multiply( a2, 1-a2 )
    delta2 = np.multiply( np.dot(w3,delta3) , mod_a2 )
    gradW2 = np.dot(a1,delta2.T)
    w2 = w2 - learning_rate*gradW2
    
    # Layer 1
    mod_a1 = sigmoid_deriv(z["z1"]).T # np.multiply( a1, 1-a1 )
    delta1 = np.multiply( np.dot(w2,delta2) , mod_a1 )
    gradW1 = np.dot(x[:,np.newaxis],delta1.T)
    w1 = w1 - learning_rate*gradW1
    
    w[0] = w1; w[1] = w2; w[2] = w3;
    
    return w



"""
Prediction
----------
"""
def predict(w, x_test):
    
    x = np.append([1], x_test)   # add bias
    z = feedforward(w, x)
    a = sigmoid(z["z3"])
    y_pred = int(a>0.5) 
    
    return y_pred



"""
Train LR
--------
"""
import random
doc_stream = stream_docs(path='shuffled_movie_data.csv')

# Parameters
n_features    = threshold_words*300#len(words)#*threshold_words
learning_rate =  0.01
porc_data     =   0.2
n_epoch       =   300
n_input       =     5
n_hidden      =     3
n_train       = 40000
n_test        = 10000

# Train/test data
x_train, y_train = get_minibatch(doc_stream, size=n_train)
x_test , y_test  = get_minibatch(doc_stream, size=n_test )

# Inicialize
w = list()
w.append( initialize_he(n_features+1,n_input ) )
w.append( initialize_he(n_input     ,n_hidden) )
w.append( initialize_he(n_hidden    ,       1) )
accuracy = list()

# Run epoch
for _ in range(n_epoch):
    
    #
    # Train one epoch
    # ---------------
    pbar = pyprind.ProgBar(n_train*porc_data)
    for __ in range(int(n_train*porc_data)):
        
        n = round(random.uniform(0, n_train-1))
        
        # Get features
        x = featuresExtraction(preprocessing(x_train[n]))
        x = np.append([1],x)   # add bias
        y = y_train[n]
        
        # Train
        z = feedforward(w, x)
        w = backpropagation(w, x, y, z, learning_rate)
        
        # Update bar
        pbar.update()


    #
    # Run test
    # --------
    acc = 0
    pbar = pyprind.ProgBar(n_test*porc_data)
    for __ in range(int(n_test*porc_data)):
        
        n = round(random.uniform(0, n_test-1))
        
        # Get features
        x = featuresExtraction(preprocessing(x_test[n]))
        
        # Prediction
        y_pred = predict(w,x)
        acc += int(y_pred == y_test[n])
        
        # Update bar
        pbar.update()
        
    acc = acc*100/(n_test*porc_data)
    print('\n')
    print('\nEpoch ',_,'\tTest Accuracy: ',acc,'%')
    accuracy.append(acc)
