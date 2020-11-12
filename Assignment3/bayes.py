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
        self.result_frame = {}
        # Feature names are: sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  labels
        # For each feature
        self.total_n = 0
        self.total_p = 0
        features = df.columns.difference(['labels']).array
        for feature in features:
            value_df = pandas.DataFrame()
            sub_list = {}
            for k, v in df.groupby(feature):
                value_labels = v['labels'].value_counts()
                #print value_labels
                n = value_labels.get(0,0) # Setosa is 0
                p = value_labels.get(1,0) # Not setosa is 1
                self.total_n = self.total_n + n
                self.total_p = self.total_p + p
                sub_list[k]= [n,p]
            self.result_frame[feature] = sub_list
        self.total_n = self.total_n / 4
        self.total_p = self.total_p / 4 
        

    def classify(self, row):
        is_setosa_chance = self.compute_chance(row, 0)
        is_not_setosa_chance = self.compute_chance(row, 1)
        print(is_setosa_chance)
        print(is_not_setosa_chance)

        if is_setosa_chance > is_not_setosa_chance:
            return 0
        else:
            return 1
    
    def compute_chance(self, row, label):
        data_names = row[1].axes[0]
        data_array = row[1].array
        values = []
        percents = [0,0,0,0]
        # Don't include the label. 
        for i in range(len(data_array) - 1):
            if data_array[i] in self.result_frame[data_names[i]]:
                values.append(self.result_frame[data_names[i]][data_array[i]])
            else:
                #TODO fix not found stuff.
                values.append([0,0])
                #print "Not found"
        # amount of 0 is first in the list, 1 is the second value in the list
        for i in range(len(values)):
            total_amt = values[i][0]+values[i][1]
            if total_amt > 0:
                if label == 0:
                    percents[i] = float(float(values[i][0])/float(total_amt))
                else:
                    percents[i] = float(float(values[i][1])/float(total_amt))
            else:
                #TODO What do we do then?
                pass
        chance = percents[0] * percents[1] * percents[2] * percents[3]
        if label == 0:
            chance = chance * (float(self.total_n) / float((self.total_n + self.total_p)))
        else:
            print chance
            chance = chance * (float(self.total_p) / float((self.total_n + self.total_p)))
            print chance
        return chance

if __name__ == "__main__":
    iris = load_iris()

    # we are intrested in setosa or not so modify the label
    # Setosa is 0, else it is not a setosa
    labels = map(lambda x: 0 if not x else 1, iris.target)

    # get training and test sets
    prop_train, prop_test, label_train, label_test = train_test_split(iris.data, labels, test_size=.5, random_state=5)

    # For each bin
    for bin in range(5, 25, 5):
        # descretize the data
        training_set = prop_train
        discretizer = KBinsDiscretizer(n_bins=bin, encode='ordinal', strategy='kmeans')
        discretizer.fit(training_set) 

        # create panda out of features and labels
        features = iris.feature_names
        df_train = pandas.DataFrame(discretizer.transform(training_set), columns=features)
        df_train.insert(len(features), "labels", label_train, True)

        bayes = Bayes(df_train)
        # validate the training set
        discretizer.fit(prop_test)
        df_test = pandas.DataFrame(discretizer.transform(prop_test), columns=features)
        df_test.insert(len(features), "labels", label_test, True)

        test_res = []
        for row in df_test.iterrows():
            test_res.append(bayes.classify(row) == row[1]['labels'])

        print "Bayes Classified {} out of {} instances correctly".format(test_res.count(1), len(test_res))