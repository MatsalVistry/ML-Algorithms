import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv('Social_Network_Ads.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size = .2,random_state = 0)
ss = StandardScaler()
Xtrain = ss.fit_transform(Xtrain)
Xtest = ss.transform(Xtest)

knc = KNeighborsClassifier(n_neighbors=5)
knc.fit(Xtrain, ytrain)

ypred = knc.predict(Xtest)
print(np.concatenate((ypred.reshape(len(ypred),1),ytest.reshape(len(ytest),1)),1))
print(knc.score(Xtest,ytest))