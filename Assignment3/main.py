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

    def calc_f_measure(self, class_name, alternate_expected=None, alternate_title=None):

        # Compute F Measures
        fmeasures = []

        plot_against = self.stats if alternate_expected is None else alternate_expected.stats

        for (_, actual, bin, _, _, _, _), (expected, _, _, _, _, _, _) in zip(self.stats, plot_against):
            fmeasure = self.compute_f_measure(expected, actual)
            fmeasures.append(fmeasure)
            plt.scatter(bin, fmeasure, color='red')

        plt.ylabel('FMeasure')
        plt.xlabel('Number of Bins')
        plt.suptitle('FMeasure for ' + class_name + '' if alternate_title is None else ' against ' + alternate_title)
        plt.show()

    def show_roc(self, class_name):
        # Stats are already here, don't reinvent the wheel. Only use this after calc_stats
        # ROC stuff
        for _, _, bin, tp, tn, fp, fn in self.stats:
            tpr = tp/(tp + fn)
            fpr = fp/(fp + tn)
            plt.plot([0, fpr, 1], [0, tpr, 1], label='{} with {} bins ROC'.format(class_name, bin))
        plt.legend()
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title("ROC in relation to Bins")
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

        # It really does not like just being called ID3. Make it happy.
        id3_obj = id3.ID3(df_train)
        print id3_obj.tree

        test_res = []
        actual = []
        for row in df_test.iterrows():
            actual.append(id3_obj.classify(id3_obj.tree, row[1]))
        id3_accuracy.add_results(df_test['labels'], actual, bin)

        bayes_obj = bayes.Bayes(df_train)

        test_res = []
        for row in df_test.iterrows():
            test_res.append(bayes_obj.classify(row) == row[1]['labels'])
        bayes_accuracy.add_results(df_test['labels'], actual, bin)

        print "Bayes Classified {} out of {} instances correctly".format(test_res.count(1), len(test_res))

    id3_accuracy.calc_stats("ID3")
    id3_accuracy.calc_f_measure("ID3")
    id3_accuracy.calc_f_measure("ID3", bayes_accuracy, 'Bayes Results')
    bayes_accuracy.calc_stats("Bayes")
    id3_accuracy.calc_f_measure("Bayes")
    id3_accuracy.calc_f_measure("Bayes", id3_accuracy, 'ID3 Results')
    bayes_accuracy.show_roc("Bayes")
    id3_accuracy.show_roc("ID3")