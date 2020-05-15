# Apriori Algorithm to find a association rule

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []

# we need to prepare data to put this data in apriori algorithm, so we are creating
# vector for input information
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# As in present libraries we dont have implemention for code, so we will use our
# custom code to implement this algorithm    
# Training Apriori on the dataset
from apyori import apriori
# min_support= we are applying formula = 3 times a day * 7 days / 7500 transaction
# min_confidence= 20% as we want more pair types together
# minimum_lift =3 means at least 3 occurence of rule
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)