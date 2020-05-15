# Upper Confidence Bound

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
import math
N = 10000
d = 10
ads_selected = []
# create vector with zero value in initial stage
# we need to follow instruction for upper_confidence_bound algo
numbers_of_selections = [0] * d
# sums for all rewards
sums_of_rewards = [0] * d
#total rewards
total_reward = 0
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    #d is number of ads versions available
    for i in range(0, d):
        # if use has clicked this ad then process for function otherwise ignore
        if (numbers_of_selections[i] > 0):
            #find average reward for total selection till now for particular ad
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            #implement formula for calculate delta value
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            #set upper bound value
            upper_bound = average_reward + delta_i
        else:
        # initial values are considered not selected for some values and machine will prepare upper_bound
            upper_bound = 1e400
        # if upper_bound value is more than ad then select ad and update upper_bound value    
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            # update selected ad
            ad = i
    #updated selected values        
    ads_selected.append(ad)
    #update total selection value
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    # check reward for selection
    reward = dataset.values[n, ad]
    #update total reward value sum for particular 
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    #total reward value
    total_reward = total_reward + reward

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()