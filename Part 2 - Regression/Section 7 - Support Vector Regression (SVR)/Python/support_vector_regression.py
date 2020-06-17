import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values
y = y.reshape(len(y),1)

ssx = StandardScaler()
X = ssx.fit_transform(X)
ssy = StandardScaler()
y = ssy.fit_transform(y)

svr = SVR(kernel='rbf')
svr.fit(X,y)

prediction = ssy.inverse_transform(svr.predict(ssx.transform([[6.5]])))
print(prediction)

plt.scatter(ssx.inverse_transform(X),ssy.inverse_transform(y), color='red') 
plt.plot(ssx.inverse_transform(X),ssy.inverse_transform(svr.predict(X)), color='blue')
plt.title('Support Vector Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()