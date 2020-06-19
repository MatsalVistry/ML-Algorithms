import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

dataset = pd.read_csv('Social_Network_Ads.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y)
ss = StandardScaler()
Xtrain = ss.fit_transform(Xtrain)
Xtest = ss.transform(Xtest)

nb = GaussianNB()
nb.fit(Xtrain,ytrain)

ypred = nb.predict(Xtest)

print(np.concatenate((ypred.reshape(len(ypred),1),ytest.reshape(len(ytest),1)),1))
print(nb.score(Xtest,ytest))