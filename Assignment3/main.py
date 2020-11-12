"""
@file Naive Bayes classification algorithm implementation for 20CS6037

@authors: Grace Gamstetter, Michael Gentile, and Richard Hammond
"""

import bayes
import id3
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score
import pandas


def compute_f_measure(ground_truth, est_targets):
    """
    @param ground_truth: Ground truth values
    @param est_targets: Values returned from a classifier
    @return: f-measure for discretization
    """
    return f1_score(ground_truth, est_targets)


if __name__ == "__main__":
    iris = load_iris()

    # we are interested in setosa or not so modify the label
    labels = map(lambda x: 0 if not x else 1, iris.target)

    # get training and test sets
    prop_train, prop_test, label_train, label_test = id3.train_test_split(iris.data, labels, test_size=.5, random_state=5)

    # train and test with different bins
    for bin in range(5, 25, 5):
        # descretize the data
        training_set = prop_train
        discretizer = id3.KBinsDiscretizer(n_bins=bin, encode='ordinal', strategy='kmeans')
        discretizer.fit(training_set)

        # create panda out of features and labels
        features = iris.feature_names
        df_train = pandas.DataFrame(discretizer.transform(training_set), columns=features)
        df_train.insert(len(features), "labels", label_train, True)

        # validate the training set
        discretizer.fit(prop_test)
        df_test = pandas.DataFrame(discretizer.transform(prop_test), columns=features)
        df_test.insert(len(features), "labels", label_test, True)

        id3 = id3.ID3(df_train)
        print id3.tree

        test_res = []
        for row in df_test.iterrows():
            test_res.append(id3.classify(id3.tree, row[1]) == row[1]['labels'])

        print "ID3 Classified {} out of {} instances correctly".format(test_res.count(1), len(test_res))

        bayes = bayes.Bayes(df_train.values)

        test_res = []
        for row in df_test.iterrows():
            test_res.append(bayes.Bayes.classify(bayes.tree, row[1]) == row[1]['labels'])

        print "Bayes Classified {} out of {} instances correctly".format(test_res.count(1), len(test_res))
