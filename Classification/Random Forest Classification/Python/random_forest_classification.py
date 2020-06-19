import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("Social_Network_Ads.csv")

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,random_state=0,test_size = .2)

ss = StandardScaler()
Xtrain = ss.fit_transform(Xtrain)
Xtest = ss.transform(Xtest)

rfc = RandomForestClassifier(criterion='entropy',random_state=0)
rfc.fit(Xtrain,ytrain)
ypred = rfc.predict(Xtest)

print(np.concatenate((ypred.reshape(len(ypred),1),ytest.reshape(len(ytest),1)),1))
print(rfc.score(Xtest,ytest))