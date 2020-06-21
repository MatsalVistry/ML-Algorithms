import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

N = 10000
d = 10
adsSelected = []
numbersOfSelections = [0] * d
sumsOfRewards = [0] * d
totalReward = 0

for n in range(0,N):
    ad = 0
    maxUCF = 0
    for i in range(0,d):
        if numbersOfSelections[i] > 0:
            averageReward = sumsOfRewards[i] / numbersOfSelections[i]
            deltaI = math.sqrt(3/2 * math.log(n+1)/numbersOfSelections[i])
            upperBound = averageReward + deltaI
        else:
            upperBound = 1e400
        if upperBound > maxUCF:
            maxUCF = upperBound
            ad = i
    adsSelected.append(ad)
    numbersOfSelections[ad] +=1
    reward = dataset.values[n,ad]
    sumsOfRewards[ad] +=  reward
    totalReward += reward

plt.hist(adsSelected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()