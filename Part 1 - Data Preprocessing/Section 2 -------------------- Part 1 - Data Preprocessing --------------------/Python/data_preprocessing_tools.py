import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
X[:,1:3] = imputer.fit_transform(X[:,1:3])
# print(X)
# print(y)

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
X = np.array(ct.fit_transform(X))
le = LabelEncoder()
y = le.fit_transform(y)
# print(X)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .2,random_state = 1)
ss = StandardScaler()
X_train[:,3:] = ss.fit_transform(X_train[:,3:])
X_test[:,3:] = ss.transform(X_test[:,3:])
print(X_train)
print(X_test)
print(y_train)
print(y_test)