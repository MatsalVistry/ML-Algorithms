import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

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

def costFunction(X,y,theta):
    inner = np.power(((X @ theta) - y), 2)
    return np.sum(inner) / (2 * len(X))  

def gradientDescent(X, y, theta, alpha, iters):
    cost_history = [0] * iters
    m = len(y)
    
    for iteration in range(iters):
        h = X @ theta
        loss = h - y
        gradient = X.T.dot(loss) / m
        if iteration&100==0:
            print(gradient)
        theta = theta - (gradient * alpha)
        cost = costFunction(X, y, theta)

        cost_history[iteration] = cost
    return theta, cost_history

def featureScale(X):
    valueSet = []
    for i in range(len(X[0])-1):
        calc = (X[:,i] - X[:,i].mean())/X[:,i].std()
        valueSet.append([X[:,i].std(),X[:,i].mean()])
        X[:, i] = calc

    calc = X[:,-1]-X[:,-1].mean()
    valueSet.append([X[:,-1].mean()])
    X[:, -1] = calc

    return X,valueSet

def inverseScale(X, valueSet):
    for i in range(len(X[0])-1):
        X[:,i] = (X[:,i] * valueSet[i][0]) + valueSet[i][-1]

    X[:,-1] = (X[:,-1] + valueSet[-1][0])

    return X

def rsquared(X,y,theta):
    predicted = X @ theta
    sstot = np.sum(pow(y-y.mean(),2))
    ssres = np.sum(pow(y-predicted,2))
    return 1-(ssres/sstot)

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:3].values
y = dataset.iloc[:,-1].values.reshape(-1,1)
ones = np.ones((len(X),1))
X = np.concatenate((X, ones),1)
# X, valueSet = featureScale(X)

alpha = .000000000001
iterations = 5000
theta = np.array([[0,0,0,.7]]).T
theta, cost = gradientDescent(X,y,theta,alpha,iterations)
print("Score: "+ str(rsquared(X,y,theta)))
print(theta)
predicted = (X @ theta)

# X = inverseScale(X, valueSet)