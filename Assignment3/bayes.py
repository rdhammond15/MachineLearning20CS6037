"""
@file Naive Bayes classification algorithm implementation for 20CS6037

@authors: Grace Gamstetter, Michael Gentile, and Richard Hammond
"""
from sklearn.datasets import load_iris
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


if __name__ == "__main__":
    iris = load_iris()

    # we are intrested in setosa or not so modify the label
    labels = map(lambda x: 0 if not x else 1, iris.target)

    # get training and test sets
    prop_train, prop_test, label_train, label_test = train_test_split(iris.data, labels, test_size=.5, random_state=5)

    # For each bin
    for bin in range(5, 25, 5):
        # descretize the data
        training_set = prop_train
        #print(training_set)
        #for i in range(len(training_set)):
            #print training_set[i]
        discretizer = KBinsDiscretizer(n_bins=bin, encode='ordinal', strategy='kmeans')
        discretizer.fit(training_set) 

        # create panda out of features and labels
        features = iris.feature_names
        df_train = pandas.DataFrame(discretizer.transform(training_set), columns=features)
        df_train.insert(len(features), "labels", label_train, True)

        bayes = Bayes(df_train.values)
        # validate the training set
        discretizer.fit(prop_test)
        df_test = pandas.DataFrame(discretizer.transform(prop_test), columns=features)
        df_test.insert(len(features), "labels", label_test, True)

        test_res = []
        for row in df_test.iterrows():
            pass
            #test_res.append(Bayes.classify(id3.tree, row[1]) == row[1]['labels'])

        #print "ID3 Classified {} out of {} instances correctly".format(test_res.count(1), len(test_res))