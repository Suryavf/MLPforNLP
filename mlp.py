#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 15:51:50 2018

@author: victor
"""
import numpy  as np
import pandas as pd
import pyprind
import random
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


"""
doc_stream = stream_docs(path='shuffled_movie_data.csv')
x_, y_ = get_minibatch(doc_stream, size=50000)
count_words = [len(preprocessing(w).split())  for w in x_]
import seaborn as sns
sns.distplot(count_words,bins=100)

threshold_words = np.percentile(count_words,80)
"""


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
    
    return review.flatten() 


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
    return np.multiply( (z>=0), z )
def ReLU_deriv(z):
    return (z>=0)
def SELU(z):
    neg = np.multiply( (z< 0), 1.758094*( np.exp(z) - 1 ) )
    pos = np.multiply( (z>=0), 1.0507  *         z        )
    return neg + pos
def SELU_deriv(z):
    return np.multiply( (z<0), 1.758094*np.exp(z) ) + 1.0507*(z>=0)
def softmax(z):
    return np.exp(z)/np.sum(np.exp(z))

def funDeriv(fun):
    if fun == sigmoid: return sigmoid_deriv;
    if fun ==    tanh: return    tanh_deriv;
    if fun ==    ReLU: return    ReLU_deriv;
    if fun ==    SELU: return    SELU_deriv;

def initialize_he(n_input, n_output):    
    return np.random.randn(int(n_input),n_output )*np.sqrt( 2/(n_input+n_output) )


"""
Feedforward
-----------
"""
def feedforward(w, x,actFunc = sigmoid):
    z = list()
    layers = len(w)
    for n in range(layers):     
        # Activation
        if n == 0: a = x[np.newaxis,:];
        else     : a = actFunc(z[n-1]);
        
        z.append( np.dot( a , w[n] ) )
    
    return z


"""
Backpropagation
---------------
"""
def layer(w,derivLoss,a_int,z,
                learning_rate,
                   reg_lambda,
                      actFunc):
    
    # Derivada
    actFunc_deriv = funDeriv(actFunc)
    
    delta = np.multiply( derivLoss , actFunc_deriv(z).T )
    gradW = np.dot(a_int,delta.T) + reg_lambda*w
    
    if reg_lambda > 0.0:
        gradW += reg_lambda*w
    
    w = w - learning_rate*gradW
    
    return w,delta


def backpropagation(w, x, y, z, 
                    learning_rate,
                    reg_lambda = 0.00   ,
                    hiddenFunc = sigmoid,
                    outFunc    = sigmoid):
    # Layers
    layers = len(w)
    
    # Activation function
    a = list()
    a.append(   x[:,np.newaxis]   )
    for n in range(layers): 
        a.append(hiddenFunc(z[n]).T)
    
    # Train layers
    delta = 0
    for n in reversed(range(layers)):
        
        # Out/Hidden layer
        if n == (layers-1):
            actFun = outFunc
            derivLoss = a[n+1]-y
        else:
            actFun = hiddenFunc
            derivLoss = np.dot(w[n+1],delta)
        
        # Train
        w[n],delta = layer(w[n],derivLoss,a[n],z[n],learning_rate,reg_lambda,actFun)
    
    
    """
    
    # Parameters
    w1 = w[0]; w2 = w[1]; w3 = w[2];
    
    if hiddenFunc == sigmoid: hiddenFunc_deriv = sigmoid_deriv;
    if hiddenFunc ==    tanh: hiddenFunc_deriv =    tanh_deriv;
    if hiddenFunc ==    ReLU: hiddenFunc_deriv =    ReLU_deriv;
    if hiddenFunc ==    SELU: hiddenFunc_deriv =    SELU_deriv;
    
    if outFunc == sigmoid: outFunc_deriv = sigmoid_deriv;
    if outFunc ==    tanh: outFunc_deriv =    tanh_deriv;
    if outFunc ==    ReLU: outFunc_deriv =    ReLU_deriv;
    if outFunc ==    SELU: outFunc_deriv =    SELU_deriv;
    
    # Activation function
    a1 = hiddenFunc(z["z1"]).T
    a2 = hiddenFunc(z["z2"]).T
    a3 =    outFunc(z["z3"]).T
    
    # Layer 3
    mod_a3 = outFunc_deriv(z["z3"]).T 
    delta3 = np.multiply( a3 - y , mod_a3 )
    gradW3 = np.dot(a2,delta3.T) #+ reg_lambda*w3
    w3 = w3 - learning_rate*gradW3
    
    # Layer 2
    mod_a2 = hiddenFunc_deriv(z["z2"]).T 
    delta2 = np.multiply( np.dot(w3,delta3) , mod_a2 )
    gradW2 = np.dot(a1,delta2.T) #+ reg_lambda*w2
    w2 = w2 - learning_rate*gradW2
    
    # Layer 1
    mod_a1 = hiddenFunc_deriv(z["z1"]).T 
    delta1 = np.multiply( np.dot(w2,delta2) , mod_a1 )
    gradW1 = np.dot(x[:,np.newaxis],delta1.T) #+ reg_lambda*w1
    w1 = w1 - learning_rate*gradW1
    
    w[0] = w1; w[1] = w2; w[2] = w3;
    """
    return w



"""
Prediction
----------
"""
def predict(w, x_test,
            hiddenFunc = sigmoid,
            outFunc    = sigmoid):
    
    x = np.append([1], x_test)   # add bias
    z = feedforward(w, x,hiddenFunc)
    a = outFunc(z[-1])  # z["z3"]
    
    if outFunc == sigmoid: return int(a>0.5) ;
    if outFunc ==    tanh: return int(a>0.0) ;
    if outFunc ==    ReLU: return int(a>0.0) ;
    if outFunc ==    SELU: return int(a>0.0) ;
    else                 : return -1;

"""
Train MLP
---------

import random
doc_stream = stream_docs(path='shuffled_movie_data.csv')

# Parameters
n_features    = threshold_words*300#len(words)#*threshold_words
learning_rate =   0.01
reg_lambda    =    0.1
porc_data     =    0.2
n_epoch       =     30
n_input       =      5
n_hidden      =      5
n_train       =  40000
n_test        =  10000

hidFunc =    tanh
outFunc = sigmoid

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
        # Random changing
        n = round(random.uniform(0, n_train-1))
        
        # Get features
        x = featuresExtraction(preprocessing(x_train[n]))
        x = np.append([1],x)   # add bias
        y = y_train[n]
        
        # Train
        z = feedforward(w, x, hidFunc)
        w = backpropagation(w, x, y, z,
                            learning_rate, reg_lambda, 
                            hidFunc, outFunc)
        
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
        y_pred = predict(w,x, hidFunc, outFunc)
        acc += int(y_pred == y_test[n])
        
        # Update bar
        pbar.update()
        
    acc = acc*100/(n_test*porc_data)
    print('\n')
    print('\nEpoch ',_+1,'\tTest Accuracy: ',acc,'%')
    accuracy.append(acc)
"""

def trainMLP(w,learning_rate,reg_lambda,hiddenFunc,outFunc,verbose=False):
    # Parameters
    n_features    = threshold_words*300
    porc_data     =    0.2
    n_epoch       =     30
    n_train       =  40000
    n_test        =  10000
    
    doc_stream = stream_docs(path='shuffled_movie_data.csv')
    accuracy = list()
    
    # Train/test data
    x_train, y_train = get_minibatch(doc_stream, size=n_train)
    x_test , y_test  = get_minibatch(doc_stream, size=n_test )
    
    # Run epoch
    for _ in range(n_epoch):
    
        """ Train one epoch """
        if verbose: pbar = pyprind.ProgBar(n_train*porc_data)
        for __ in range(int(n_train*porc_data)):
            # Random changing
            n = round(random.uniform(0, n_train-1))

            # Get features
            x = featuresExtraction(preprocessing(x_train[n]))
            x = np.append([1],x)   # add bias
            y = y_train[n]

            # Train
            z = feedforward(w, x, hiddenFunc)
            w = backpropagation(w, x, y, z,
                                learning_rate, reg_lambda, 
                                hiddenFunc, outFunc)
            # Update bar
            if verbose: pbar.update()

        """ Run test """
        acc = 0
        if verbose: pbar = pyprind.ProgBar(n_test*porc_data)
        for __ in range(int(n_test*porc_data)):
            # Random changing
            n = round(random.uniform(0, n_test-1))

            # Get features
            x = featuresExtraction(preprocessing(x_test[n]))

            # Prediction
            y_pred = predict(w,x, hiddenFunc, outFunc)
            acc += int(y_pred == y_test[n])

            # Update bar
            if verbose: pbar.update()

        acc = acc*100/(n_test*porc_data)
        if verbose: print('\n')
        if verbose: print('\nEpoch ',_+1,'\tTest Accuracy: ',acc,'%')
        accuracy.append(acc)
        
    return accuracy

def addLayer(w,n_int,n_out):
    w.append(initialize_he(n_int,n_out))
    
    
"""z
Testing
-------
"""
n_features = threshold_words*300

# Parameters
learning_rate =  0.01
reg_lambda    =   0.00001

# Define layers
w = list()
addLayer(w,n_features+1,5)
addLayer(w,5,5); addLayer(w,5,1)

acc = trainMLP(w,learning_rate,reg_lambda,ReLU,sigmoid,verbose=True)

#ReLU
#SELU

