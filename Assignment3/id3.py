"""
@file ID3 algorithm implementation for 20CS6037

@authors: Grace Gamstetter, Michael Gentile, and Richard Hammond
"""
from sklearn.datasets import load_iris
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split

import math
import pandas
import numpy as np
import matplotlib.pyplot as plt


class ID3(object):
    def __init__(self, df):
        """
        @param labels: The labels for the training data
        @param training_set: Training set wich is a list of lists. 
                            The list width tells us how many features and the lenght tells us how many examples.
        @param features: Names of each feature
        @param bins: How many bins to use for discretizing
        """
        self.tree = self.find_node(df)


    @staticmethod
    def classify(node, instance):
        """
        Given an instance tell weather it is setosa or not

        @param instance: The data to classify in form of a panda
        
        @return True if flower other than setosa or False if setosa
        """
        branch = instance[node.name]

        try:
            next_node = node.branches[branch]
        # data is binned too much and the training data is missing one of the values
        except KeyError:
            for key in node.branches.iterkeys():
                # round up to next key
                if key > branch:
                    next_node = node.branches[key]
                    break

        if not isinstance(next_node, Node):
            return next_node
        else:
            return ID3.classify(next_node, instance)


    def find_node(self, df):
        """
        Recursively build each node

        @param df: The dataframe representing the current set to caluclate entropy on

        rparam node: The current node to create
        """
        label_counts = df['labels'].value_counts()
        self.p = label_counts.get(1, 0)
        self.n = label_counts.get(0, 0)
        self.entropy = self.calc_entropy(self.p, self.n)

        # for each feature create a data frame containing positive and negative counts for each value
        features = df.columns.difference(['labels']).array
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

        # calculate average information for each feature
        avg_info = []
        for feature in feature_ratios:
            avg_info.append(self.average_info(feature))

        # use the set entropy - avg_info to get gain
        gain = []
        for info in avg_info:
            gain.append(self.entropy - info)

        # largest gain is root node
        node = max(enumerate(gain), key=lambda x: x[1])

        # recurse for each branch of this node
        branches = {}
        for _, value in feature_ratios[node[0]].iterrows():
            # able to classify at this point
            if value['entropy'] == 0:
                branches[value['value']] = 1 if value['p'] else 0
            else:
                # figure out next node with respect to current value
                df_split = df.loc[df[features[node[0]]]] == value['value']
                del df_split[features[node[0]]]
                branches[value['value']] = self.find_node(df_split)

        return Node(features[node[0]], branches)


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


class Node(object):
    def __init__(self, attribute, values):
        """
        Initialize a node of a tree

        @param attribute: The name of the node
        @param values: The values a feature can take along with a node object {value: node}
        """
        self.name = attribute
        self.branches = values
    
    def __str__(self):
        msg = []
        for k, v in self.branches.iteritems():
            msg.append(str(k) + " with node " + str(v))
        return self.name + " branches to " + " and ".join(msg)


class ROC(object):
    def __init__(self):
        """
        Object for Receiver operating characteristic to measure the accuracy of classifier
        """
        self.stats = []

    def add_results(self, expected, actual, bin):
        """
        Add test results

        @param expected: The labels for the training data
        @param actual: What the classifier classified the data as
        @param bin: The number of bins used for this data
        """
        # calculate true positives, true negatives, false positives, and flase negatives
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

        self.stats.append((bin, tp, tn, fp, fn))

    
    def calc_stats(self, class_name):
        """
        Calculate the accuracy stats after all data has been added with add_results

        @param class_name: The name of the classifier being used (for labeling charts)
        """
        accuracies = []
        for bin, tp, tn, fp, fn in self.stats:
            accuracy = (tp + tn)/float(tp + tn + fp + fn)
            accuracies.append(accuracy)
            plt.scatter(bin, accuracy, color='red')

        plt.ylabel('Accuracy')
        plt.xlabel('Number of Bins')
        plt.suptitle('Bin Accuracy for ' + class_name)
        plt.show()