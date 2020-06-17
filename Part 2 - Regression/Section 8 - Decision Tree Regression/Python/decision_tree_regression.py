import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(X,y)

# inneficient for decision tree regression
Xgrid = np.arrange(min(X),max(X),.1)
Xgrid = Xgrid.reshape((len(Xgrid),1))
plt.scatter(X,y, color = 'red')
plt.plot(Xgrid,dtr.predict(Xgrid), color='blue')
plt.title('Decision Tree Regression')
plt.xlabel('Position')
plt.ylabel('Salary')
print(dtr.predict([[6.5]]))