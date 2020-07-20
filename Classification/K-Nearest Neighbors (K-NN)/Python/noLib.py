import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import heapq
from itertools import count

def splitData(X, y, testSize):
    splitLoc = math.floor(len(dataset)*(1-testSize))

    xtrain = X[:splitLoc,:]
    ones = np.ones((len(xtrain),1))
    xtrain = np.concatenate((xtrain, ones),1)

    xtest = X[splitLoc:,:]
    ones = np.ones((len(xtest),1))
    xtest = np.concatenate((xtest, ones),1)

    ytrain = y[:splitLoc,:]
    ytest = y[splitLoc:,:]
    return xtrain,xtest,ytrain,ytest

def featureScale(X):
    valueSet = []
    for i in range(len(X[0])):
        calc = (X[:,i] - X[:,i].mean())/X[:,i].std()
        valueSet.append([X[:,i].std(),X[:,i].mean()])
        X[:, i] = calc

    return X,valueSet

def distance(features, targetFeatures):
    return math.sqrt(np.sum((features - targetFeatures) * (features - targetFeatures)))

def kNearestNeighbors(k):
    tiebreaker = count(step = 1)

    q = []
    for r in range(len(X)-1):
        if len(q) != k:
            f = X[r,:]
            d = 1/distance(f,target)
            heapq.heappush(q,(d,next(tiebreaker),f))
        else:
            f = X[r,:]
            d = distance(f,target)

            if d < 1/q[0][0]:
                heapq.heappop(q)
                d = 1/d
                heapq.heappush(q,(d,next(tiebreaker),f))
    return q

def getClassification(k,q):
    all = []
    hm = {}
    for each in q:
        all.append(1/each[0])
        print( each[-1])
        classification = each[-1][-1]
        if hm.get(classification) is not None:
            hm[classification] = hm.get(classification)+1
        else:
            hm[classification] = 1

    classification = None
    first = True
    for key in hm:
        if first:
            classification = key
            first=False
        else:
            if hm.get(classification) < hm.get(key):
                classification = key
    all.sort()
    print(all)
    return classification

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values.reshape(-1,1)
X = np.append(X,y,1)

target = np.array([45,47550,0])
X = np.vstack([X,target])
target = X[-1,:]

k = 10
q = kNearestNeighbors(k)
classification = getClassification(k,q)

print("Final Classification "+str(classification))
