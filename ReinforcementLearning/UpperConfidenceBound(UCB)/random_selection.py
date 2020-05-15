# Random Selection
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Random Selection
import random
N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0, N):
    # randomly select ad which will have positive result between range of 0-9 
    ad = random.randrange(d)
    # save selected machine result
    ads_selected.append(ad)
    #select actual result for selected machine and that will be our result
    reward = dataset.values[n, ad]
    # add received reward which will be either 0 or 1
    total_reward = total_reward + reward

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()