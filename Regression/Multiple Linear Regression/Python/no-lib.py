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

def gradientDescent(X, y, theta, alpha, iters, costs, iterations2):
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum(((X @ theta) - y) * X)
        cost = costFunction(X, y, theta)
        if i % 10 == 0:
            costs.append(cost)
            iterations2.append(i)
    return theta, cost, costs, iterations2

def featureScale(X):
    for i in range(len(X[0])-1):
        X[:, i] = (X[:, i] - X[:, i].min())/(X[:, i].max()-X[:, i].min())
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
X = featureScale(X)
y = featureScale(y)

alpha = .002
iterations = 1000
theta = np.array([[.5,.5,.5,.5]]).T
theta, cost, costs, iterations2 = gradientDescent(X,y,theta,alpha,iterations,[],[])


predicted = X @ theta
print(np.concatenate((y,predicted),1))
print("Score: "+ str(rsquared(X,y,theta)))

plt.plot(costs,iterations2)