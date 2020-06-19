import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

lr = LinearRegression()
lr.fit(X,y)

pf = PolynomialFeatures(degree = 4)
X_poly = pf.fit_transform(X)
pr = LinearRegression()
pr.fit(X_poly,y)

plt.scatter(X,y, color='red')
plt.plot(X,lr.predict(X), color='blue')
plt.title('Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

plt.scatter(X,y, color='red')
plt.plot(X,pr.predict(X_poly), color='blue')
plt.title('Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

print(lr.predict([[6.5]]))
print(pr.predict(pf.fit_transform([[6.5]])))