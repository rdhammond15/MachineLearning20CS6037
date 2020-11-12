"""
@file Naive Bayes classification algorithm implementation for 20CS6037

@authors: Grace Gamstetter, Michael Gentile, and Richard Hammond
"""
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split

import math
import pandas

class Bayes(object):

    def __init__(self, df):
        print df
        # For each feature
        for i in range(4):
            # For each result
            for j in range(2):
                pass
                
    
    def compute_probability(self, df, feature, label):
        return df.groupby(feature).count()[label] / len(df)

