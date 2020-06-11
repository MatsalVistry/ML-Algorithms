import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

lr = LinearRegression()
lr.fit(X_train,y_train)

y_predicted = lr.predict(X_test)

plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train, lr.predict(X_train),color = 'blue')
plt.title('Salary vs Experience(Training)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test,y_test, color = 'red')
plt.plot(X_test, y_predicted,color = 'blue') 
#plot all our x test values and the y values we predicted for them and draw a line through it
plt.title('Salary vs Experience(Test)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

print(lr.predict([[4],[2]]))