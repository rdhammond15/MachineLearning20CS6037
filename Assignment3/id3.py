"""
@file ID3 algorithm implementation for 20CS6037

@authors: Grace Gamstetter, Michael Gentile, and Richard Hammond
"""
from sklearn.datasets import load_iris
from sklearn.preprocessing import KBinsDiscretizer

import math


class ID3(object):
    def __init__(self, labels, training_set, bins):
        """
        @param labels: The labels for the training data
        @param training_set: Training set wich is a list of lists. 
                            The list width tells us how many features and the lenght tells us how many examples.
        @param bins: How many bins to use for discretizing
        """
        self.labels = labels

        # descretize the data
        discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='kmeans')
        discretizer.fit(training_set)
        self.training_set = discretizer.transform(training_set)

        # get the set entropy
        self.entropy_s = self.calc_entropy(self.labels.count(1), self.labels.count(0))
        print self.entropy_s


    def calc_entropy(self, p, n):
        """
        Use -p/p+nlog(p/p+n) - n/p+nlog(n/p+n) to calculate entrpy

        @param p: Number of postivie examples
        @param n: Number of negative examples

        @return The entryopy
        """
        return ((-p/float(p +n))*math.log(p/float(p+n), 2)) - ((n/float(p+n)*math.log(n/float(p+n), 2)))


if __name__ == "__main__":
    iris = load_iris()

    # we are intrested in setosa or not so modify the label
    setosa = map(lambda x: 0 if not x else 1, iris.target)

    id3 = ID3(setosa, iris.data, 5)