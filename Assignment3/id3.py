"""
@file ID3 algorithm implementation for 20CS6037

@authors: Grace Gamstetter, Michael Gentile, and Richard Hammond
"""
from sklearn.datasets import load_iris
from sklearn.preprocessing import KBinsDiscretizer

import math
import pandas


class ID3(object):
    def __init__(self, labels, training_set, features, bins):
        """
        @param labels: The labels for the training data
        @param training_set: Training set wich is a list of lists. 
                            The list width tells us how many features and the lenght tells us how many examples.
        @param features: Names of each feature
        @param bins: How many bins to use for discretizing
        """
        self.labels = labels
        self.p = self.labels.count(1)
        self.n = self.labels.count(0)

        # descretize the data
        discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='kmeans')
        discretizer.fit(training_set)
        self.training_set = discretizer.transform(training_set)

        # get the set entropy
        self.entropy_s = self.calc_entropy(self.p, self.n)

        # create panda out of features and labels
        df = pandas.DataFrame(self.training_set, columns=features)
        df.insert(len(features), "labels", self.labels, True)

        # for each feature create a data frame containing positive and negative counts for each value
        feature_ratios = []
        for feature in features:
            value_df = pandas.DataFrame()
            for k, v in df.groupby(feature):
                value_labels = v['labels'].value_counts()
                n = value_labels.get(0,0)
                p = value_labels.get(1,0)
                entropy = self.calc_entropy(p, n)
                value_df = value_df.append({"value": k, "n": n, "p": p, "entropy": entropy}, ignore_index=True)
            feature_ratios.append(value_df)

        # calculate average average information for each feature
        avg_info = []
        for feature in feature_ratios:
            avg_info.append(self.average_info(feature))

        # use the set entropy - avg_info to get gain
        gain = []
        for info in avg_info:
            gain.append(self.entropy_s - info)

        # largest gain is root node
        root = max(enumerate(gain), key=lambda x: x[1])


    def calc_entropy(self, p, n):
        """
        Use -p/p+nlog(p/p+n) - n/p+nlog(n/p+n) to calculate entrpy

        @param p: Number of postivie examples
        @param n: Number of negative examples

        @return The entropy
        """
        if not p or not n:
            return 0
        return ((-p/float(p + n))*math.log(p/float(p+n), 2)) - ((n/float(p+n)*math.log(n/float(p+n), 2)))
    

    def average_info(self, df):
        """
        Calculate the average information sumation of (pvalue + nvalue/ p + n) * entropy(value)
        
        @param df: Dataframe with p, n, and entropy 

        @return the average infromation
        """
        info = 0
        for _, row in df.iterrows():
            info += (row['p'] + row['n'])/float(self.p + self.n) * row['entropy']

        return info

if __name__ == "__main__":
    iris = load_iris()

    # we are intrested in setosa or not so modify the label
    setosa = map(lambda x: 0 if not x else 1, iris.target)

    id3 = ID3(setosa, iris.data, iris.feature_names, 5)