import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

dataset = pd.read_csv('Social_Network_Ads.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y, test_size = .2, random_state=0)
ss = StandardScaler()
Xtrain = ss.fit_transform(Xtrain)
Xtest = ss.transform(Xtest)

nb = DecisionTreeClassifier(criterion='entropy', random_state = 0)
nb.fit(Xtrain,ytrain)

ypred = nb.predict(Xtest)

print(np.concatenate((ypred.reshape(len(ypred),1),ytest.reshape(len(ytest),1)),1))
print(nb.score(Xtest,ytest))