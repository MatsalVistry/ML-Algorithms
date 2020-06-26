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
    inner = np.power((np.matmul(X, theta) - y), 2)
    return np.sum(inner) / (2 * len(X))  

def gradientDescent(X, y, theta, alpha, iters):
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum(((X @ theta) - y) * X)
        cost = costFunction(X, y, theta)
        # if i % 10 == 0:
        #     print(cost)
    return theta, cost

def featureScale(X,y):
    valueSet = []
    for i in range(len(X[0])-1):
        calc = (X[:,i] - X[:,i].mean())/X[:,i].std()
        valueSet.append([X[:,i].std(),X[:,i].mean()])
        X[:, i] = calc

    calc = X[:,-1]-X[:,-1].mean()
    valueSet.append([X[:,-1].mean()])
    X[:, -1] = calc

    calc = (y[:,0] - y[:,0].mean())/y[:,0].std()
    valueSet.append([y[:,0].std(),y[:,0].mean()])
    y[:,0] = calc

    return X,y,valueSet

def inverseScale(X, y, predicted, valueSet):
    for i in range(len(X[0])-1):
        X[:,i] = (X[:,i] * valueSet[i][0]) + valueSet[i][-1]

    X[:,-1] = (X[:,-1] + valueSet[-2][0])
    y = (y * valueSet[-1][0]) + valueSet[-1][-1]
    predicted = (predicted * valueSet[-1][0]) + valueSet[-1][-1]

    return X,y,predicted

def rsquared(X,y,theta):
    predicted = X @ theta
    sstot = np.sum(pow(y-y.mean(),2))
    ssres = np.sum(pow(y-predicted,2))
    return 1-(ssres/sstot)

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values.reshape(-1,1)
ones = np.ones((len(X),1))
X = np.concatenate((X, ones),1)
X,y, valueSet = featureScale(X,y)


alpha = .005
iterations = 1000
theta = np.array([[0,0]]).T
theta, cost = gradientDescent(X,y,theta,alpha,iterations)
print("Score: "+ str(rsquared(X,y,theta)))
predicted = (X @ theta)

X, y, predicted = inverseScale(X,y,predicted, valueSet)

plt.scatter(X[:,0],y)
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs Years')
plt.plot(X[:,0],predicted)

