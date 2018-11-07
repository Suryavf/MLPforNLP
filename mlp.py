#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 15:51:50 2018

@author: victor
"""
import numpy  as np
import pandas as pd

print(chr(27) + "[2J")

words = np.load('words.npy').item()


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
Features extraction
-------------------
"""
def featuresExtraction(text):
    feature = np.zeros(300)
    count = 0
    
    for w in text.split():
        if w in words:
            feature += words[w]
            count += 1
    
    if not count:
        print("Esto es un problemas D:")
        print(text)
        
    return feature/count


"""
Generate features
-----------------
"""
import pyprind
pbar = pyprind.ProgBar(50)
doc_stream = stream_docs(path='shuffled_movie_data.csv')


print('\nGenerate Features')
x = list()
y = list()
for _ in range(50):
    # Getting
    x_raw, y_raw = get_minibatch(doc_stream, size=1000)
    
    # Update features
    features = [ featuresExtraction(preprocessing(text)) for text in x_raw ] 
    x = x + features
    
    # Update out
    y = y + y_raw
    
    # Bar
    pbar.update()

##  ---------------------------------------------------------------------------




"""
No-Lineal functions
-------------------
"""
def sigmoid(z):
    return 1/(1 + np.exp(-z))
def softmax(z):
    return np.exp(z)/np.sum(np.exp(z))
def sigmoid_deriv(z):
    return sigmoid(z)*(1-sigmoid(z))


def initialize_he(D, K):
    W = np.random.randn(K,D)*np.sqrt(2/K)
    b = np.random.randn(1,K)*np.sqrt(2/K)
    return W, b

"""
Feedforward
-----------
"""
def feedforward(w, x):
    
    # Layers
    z1 = np.multiply(         x   , w[0] )
    z2 = np.multiply( sigmoid(z1) , w[1] )
    z3 = np.multiply( sigmoid(z2) , w[2] )

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
    
    a1 = sigmoid(z["z1"])
    a2 = sigmoid(z["z2"])
    a3 = sigmoid(z["z3"])
    
    # Layer 3
    mod_a3 = np.multiply( a3, 1-a3 )
    delta3 = np.multiply( a3 - y , mod_a3 )
    gradW3 = np.dot(a2,delta3)
    w3 = w3 - learning_rate*gradW3
    
    # Layer 2
    mod_a2 = np.multiply( a2, 1-a2 )
    delta2 = np.multiply( np.dot(w3,delta3) , mod_a2 )
    gradW2 = np.dot(a1,delta2)
    w2 = w2 - learning_rate*gradW2
    
    # Layer 1
    mod_a1 = np.multiply( a1, 1-a1 )
    delta1 = np.multiply( np.dot(w2,delta2) , mod_a1 )
    gradW1 = np.dot(x[np.newaxis,:],delta1)
    w1 = w1 - learning_rate*gradW1
    
    w[0] = w1; w[1] = w2; w[2] = w3;
    
    return w


"""
Multilayer Perceptron
---------------------
"""
def MLP(x_train, y_train, num_iterations=20000, learning_rate=0.001,
                          n_input=100, n_hidden=50):
    import random
    
    # Parameters
    n_samples  = len(x_train)
    
    # Inicialize
    w = list()
    w.append( np.random.randn(n_samples+1,n_input )*np.sqrt(2/(n_samples+1+n_input )) )
    w.append( np.random.randn(n_input    ,n_hidden)*np.sqrt(2/(n_input    +n_hidden)) )
    w.append( np.random.randn(n_hidden   ,       1)*np.sqrt(2/(n_hidden   +       1)) )
    
    # Train Loop
    for _ in range(num_iterations):
        
        # Random selection
        n = round(random.uniform(0, n_samples-1))
        x = np.append([1],x_train[n])   # add bias
        y = y_train[n]
        
        # Train
        z = feedforward(w, x)
        w = backpropagation(w, x, y, z, learning_rate)
        
    return w



"""
Prediction
----------
"""
def predict(w, x):
    
    n_samples = len(x)
    y_pred    = list()
    
    for n in range(n_samples):
        z = feedforward(w, x[n])
        a = sigmoid(z["z3"])
        y_pred.append( int(a>0.5) )
    
    return y_pred



"""
Train LR
--------
"""
from sklearn.model_selection import KFold

print('\nTrain Logistic Regression')
kf = KFold(n_splits=8)  
pbar = pyprind.ProgBar(8)

prediction = list()

for train, test in kf.split(x):
    
    # Select
    x_train = [ x[i] for i in train ]
    y_train = [ y[i] for i in train ]
    
    x_test = [ x[i] for i in test ]
    y_test = [ y[i] for i in test ]
    
    # Run train
    w = MLP(x_train, y_train)
    
    # Run test
    y_pred = predict(w, x_test )
    
    prediction.append({'Real'      : y_test,
                       'Prediction': y_pred})
    
    # Bar
    pbar.update()


"""
Result Analysis
---------------
"""
from sklearn.metrics import roc_curve

print('\nResult Analysis')
accuracy   = list()
for p in prediction:
    fpr, tpr, thresholds =roc_curve(p['Real'], p['Prediction'])
    
    fpr_tpr = [ np.abs(a-b) for a,b in zip(fpr,tpr) ]
    threshold = thresholds[ fpr_tpr.index(max(fpr_tpr)) ]
    
    y_pred = [ int(score>threshold) for score in p['Prediction']]
    acc = 0
    for real,pred in zip(y_pred,p['Real']):
        acc = acc + int( real == pred )
    
    accuracy.append( acc*100/len(y_pred) )




