import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv('Social_Network_Ads.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


Xtrain,Xtest,ytrain,ytest = train_test_split(X,y, test_size = .2, random_state = 0)
sc = StandardScaler()
Xtrain = sc.fit_transform(Xtrain)#should only be based on training
Xtest = sc.transform(Xtest)

lr = LogisticRegression(random_state=0)
lr.fit(Xtrain,ytrain)

# print(lr.predict(sc.transform([[30,87000]])))
ypred = lr.predict(Xtest)
print(np.concatenate((ypred.reshape(len(ypred),1), ytest.reshape(len(ytest),1)),1))
print(lr.score(Xtest,ytest))
print(confusion_matrix(ytest,ypred))
print(accuracy_score(ypred,ytest))