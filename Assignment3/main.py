"""
@file Naive Bayes classification algorithm implementation for 20CS6037

@authors: Grace Gamstetter, Michael Gentile, and Richard Hammond
"""
from sklearn.datasets import load_iris
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import bayes
import id3
import pandas
import matplotlib.pyplot as plt


class Accuracy(object):
    def __init__(self):
        """
        Object for measuring the accuracy of classifier
        """
        self.stats = []

    def compute_f_measure(self, expected, actual):
        """
        @param expected: Ground truth values
        @param actual: Values returned from a classifier
        @return: f-measure for discretization
        """
        return f1_score(expected, actual)

    def add_results(self, expected, actual, bin):
        """
        Add test results

        @param expected: The labels for the training data
        @param actual: What the classifier classified the data as
        @param bin: The number of bins used for this data
        """
        # calculate true positives, true negatives, false positives, and false negatives
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for expt, act in zip(expected, actual):
            # test positive values
            if expt:
                if act:
                    tp += 1
                else:
                    fn += 1
            else:
                if act:
                    fp += 1
                else:
                    tn += 1

        self.stats.append((expected, actual, bin, tp, tn, fp, fn))

    def calc_stats(self, class_name):
        """
        Calculate the accuracy stats after all data has been added with add_results

        @param class_name: The name of the classifier being used (for labeling charts)
        """
        accuracies = []
        for _, _, bin, tp, tn, fp, fn in self.stats:
            accuracy = (tp + tn)/float(tp + tn + fp + fn)
            accuracies.append(accuracy)
            plt.scatter(bin, accuracy, color='red')

        plt.ylabel('Accuracy')
        plt.xlabel('Number of Bins')
        plt.suptitle('Bin Accuracy for ' + class_name)
        plt.show()

        # Compute F Measures
        fmeasures = []
        for expected, actual, bin, _, _, _, _ in self.stats:
            fmeasure = self.compute_f_measure(expected, actual)
            fmeasures.append(fmeasure)
            plt.scatter(bin, fmeasure, color='red')

        plt.ylabel('FMeasure')
        plt.xlabel('Number of Bins')
        plt.suptitle('FMeasure for ' + class_name)
        plt.show()


if __name__ == "__main__":
    iris = load_iris()

    # we are interested in setosa or not so modify the label
    labels = map(lambda x: 0 if not x else 1, iris.target)

    # get training and test sets
    prop_train, prop_test, label_train, label_test = train_test_split(iris.data, labels, test_size=.5, random_state=5)

    # train and test with different bins
    id3_accuracy = Accuracy()
    bayes_accuracy = Accuracy()
    for bin in range(5, 25, 5):
        # descretize the data
        training_set = prop_train
        discretizer = KBinsDiscretizer(n_bins=bin, encode='ordinal', strategy='kmeans')
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
        actual = []
        for row in df_test.iterrows():
            actual.append(id3.classify(id3.tree, row[1]))
        id3_accuracy.add_results(df_test['labels'], actual, bin)

        bayes = bayes.Bayes(df_train.values)

        test_res = []
        for row in df_test.iterrows():
            test_res.append(bayes.Bayes.classify(bayes.tree, row[1]) == row[1]['labels'])
        bayes_accuracy.add_results(df_test['labels'], actual, bin)

        print "Bayes Classified {} out of {} instances correctly".format(test_res.count(1), len(test_res))

    id3_accuracy.calc_stats("ID3")
    bayes_accuracy.calc_stats("Bayes")