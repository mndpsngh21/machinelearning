#Practice code for Thompson Sampling 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Thompson Sampling
# create initial reward matrics
import random
N = 10000
d = 10
ads_selected = []
# fill initial data with zero
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_reward = 0

for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        #implement formula for calculate beta value
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        # if value is more than max random then consider it as selection
        if random_beta > max_random:
            max_random = random_beta
            # select which is performing better
            ad = i
    #add selected ad to selection list        
    ads_selected.append(ad)
    # check reward status
    reward = dataset.values[n, ad]
    # of reward is positive add to positive otherwise to 0 reward
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    # add reward value to total reward only positive rewards will be counted in total reward        
    total_reward = total_reward + reward

# Visualising the results - Histogram
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()