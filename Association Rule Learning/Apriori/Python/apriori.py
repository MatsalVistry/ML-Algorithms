import pandas as pd
import numpy as np
from apyori import apriori

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0,7501):
  transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

rules = apriori(transactions = transactions, min_support = .003, min_confidence = .2, min_lift = 3, min_length = 2, max_length = 2)
result = list(rules)
for i in result:
  print(i)
  print()