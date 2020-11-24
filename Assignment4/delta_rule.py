"""
@file Delta rule classification algorithm implementation for 20CS6037

@author: Richard Hammond
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas
import time


class Delta(object):
    NORMAL = 0
    DECAYING = 1
    ADAPTIVE = 2

    def __init__(self, train, labels, epoch, rate=None, threshold=None, divisor=0.8, multiplier=1.02):
        """
        Initialize the delta training object

        @param train: The np array training data
        @param labels: The labels for the training data
        @param epoch: The number of iteratations to use for training
        @param rate: The learning rate eta
        @param threshold: Setting this enables incremental rate change. This value determines the increase in
                          error that triggers a rate decrease.
        @param divisor: Value smaller than one that the rate gets multiplied by when the error increases by too much
        @param multiplier: Value greater than one that the rate gets mulitplied when the change in error is not > threshold
        """
        self.labels = labels
        self.data = train 
        self.w = np.random.rand(train.shape[1])
        self.epoch = epoch
        self.rate = rate
        self.df = pandas.DataFrame(self.data, columns=['bias', 'x1', 'x2'])
        self.df.insert(3, "labels", labels, True)
        self.rate_mode = self.NORMAL
        self.last_error = None
        self.threshold = threshold
        self.divisor = divisor
        self.multiplier = multiplier

        # see if rate mode should really be normal
        if rate is None:
            self.rate = 0.5
            self.rate_mode = self.ADAPTIVE if threshold is not None else self.DECAYING

    def load_test_data(self, test, labels):
        """
        Use after training so that the hypothesis can be used wth the test data
        """
        self.labels = labels
        self.data = test
        self.df = pandas.DataFrame(self.data, columns=['bias', 'x1', 'x2'])
        self.df.insert(3, "labels", labels, True)

    @property
    def hypothesis(self):
        """
        Return the hypotheses using w*x
        """
        return self.data.dot(self.w)

    @property
    def predict(self):
        """
        Return the hypothesis after sign
        """
        return np.sign(self.hypothesis)

    @property
    def error(self):
        """
        Calculate: 1/2(Summation(t - o)^2)
        
        @return the error according to the calculation above
        """
        return .5 * np.sum(np.square(self.labels - self.hypothesis))

    def train(self, incremental=False):
        """
        Use the training data to find the weights for the features and plot the error

        @param incremental: False if batch learning else True
        """
        if incremental:
            start = time.time()
            w_update = 0
            for i in range(1, self.epoch + 1):
                for i, x in enumerate(self.data):
                    for feature_num, feature in enumerate(x):
                        self.w[feature_num] += self.rate * (self.labels[i] - np.sign(self.w[feature_num] * feature)) * feature
                        w_update += 1
            end = time.time()
            print "Time for incremental {} with {} weight updates".format(end - start, w_update)
        else:
            elapsed = 0
            w_update = 0
            for i in range(1, self.epoch + 1):
                # accumulate the error
                w_update += 1
                start = time.time()
                delta_w = self.cost()
                self.update_rate(delta_w)
                end = time.time()
                elapsed += end - start
                plt.figure(1)
                plt.plot(i, self.error, 'ro')

                # plot the decision boundry peridocially
                if i in (5, 10, 50, 100):
                    self.plot_decision_surface(i)

            print "Time for batch {} with {} weight updates".format(elapsed, w_update)

            plt.figure(1)
            plt.ylabel('Error')
            plt.xlabel('Epoch')
            plt.title("Error per Epoch")
            plt.show()

    def plot_decision_surface(self, plot_num, title="Train Data"):
        """
        Plot the hypothesis line along with where the data actually lands

        @param plot_num: The plot number to use
        @param mode: Title prefix for the plot
        """
        plt.figure(plot_num)
        x_data = np.linspace(0, 1, 100)
        x_in = [[x] for x in x_data]
        x_axis = np.asarray(x_in)
        x_axis = np.concatenate((x_in, x_axis), axis=1)
        x_axis = np.concatenate((np.ones((x_axis.shape[0], 1)), x_axis), axis=1)
        y_axis = x_axis.dot(self.w)
        plt.plot(x_data, y_axis, 'b-')

        # split up the data based on label and plot where it lies
        positive = self.df.loc[self.df['labels'] == 1]
        negative = self.df.loc[self.df['labels'] == -1]
        plt.plot(positive["x2"].values, positive["x1"].values, "r+")
        plt.plot(negative["x2"].values, negative["x1"].values, "y_")
        suffix = "at Epoch %d" % plot_num if title == "Train Data" else ""
        plt.title(title + " Decision Boundry " + suffix)

    def update_rate(self, delta_w):
        """
        Update rate based on the current mode

        @param delta_w: The amount to change the weights by if learning mode and change in error allows
        """
        if self.rate_mode == self.DECAYING:
            self.rate = self.rate * .08
            print "Mode decaying: decreasing rate to {}".format(self.rate)
        elif self.rate_mode == self.ADAPTIVE:
            # check if we can increase the rate
            if self.last_error is None or (self.error - self.last_error) < self.threshold:
                self.last_error = self.error
                self.rate = self.rate * self.multiplier
                print "Mode adaptive: increasing learning rate to {}".format(self.rate)

            # Change in error is going the wrong way. Dial back the rate.
            else:
                self.rate = self.rate * self.divisor
                self.last_error = self.error
                
                print "Mode adaptive: decreasing learning rate to {}".format(self.rate)

                # don't want to update anything so return
                return
        
        # Apply the change in w. Won't get here if incremntal and change in error is too large.
        self.w += delta_w

    def test(self):
        """
        Use the trained weight vector on a new set of data
        """
        # show accuracy stats
        correct = 0
        for predicted, y in zip(self.predict, self.labels):
            if predicted == y:
                correct +=1 
        
        print "Predicted %d correctly out of %d" % (correct, self.labels.shape[0])
        self.plot_decision_surface(1, "Test Data")
        plt.show()
    
    def cost(self):
        """
        The gradient decent for the delta rule: delta_w = eta(summation((td -od)*x))

        @return the delta change for the weights
        """
        return self.rate * np.dot((self.labels - np.sign(self.hypothesis)), self.data)


def concept(x1, x2):
    """
    The concept that we are trying to fit which is 𝑥1 + 2𝑥2 − 2 > 0

    @param x1: x1 in the equation
    @param x2: x2 in the equation

    @return The equations value
    """
    return np.sign(x1 + 2*x2 - 2)


if __name__ == "__main__":
    # generate repeatable random data
    np.random.seed(139)
    training = np.random.rand(100, 2)
    test = np.random.rand(100, 2)

    # generate labels for the data
    y_train = concept(training[:,0], training[:,1])
    y_test = concept(test[:,0], test[:,1])

    # add weights for the bias
    training = np.concatenate((np.ones((100, 1)), training), axis=1)
    test = np.concatenate((np.ones((100, 1)), test), axis=1)

    # for rate in (0.5, 0.1, 0.01, 0.001, 0.0001):
    #     print "Training with learning rate {}".format(rate)
    #     batch = Delta(training, y_train, 100, rate)
    #     incremental = Delta(training, y_train, 100, rate)
    #     batch.train()
    #     incremental.train(True)
    #     batch.load_test_data(test, y_test)
    #     incremental.load_test_data(test, y_test)
    #     batch.test()
    #     incremental.test()

    # do adaptive and decaying rate modes
    decaying = Delta(training, y_train, 100)
    adaptive = Delta(training, y_train, 100, threshold=1.03)
    print "Training with decaying rate"
    decaying.train()
    print "Training with adaptive rate"
    adaptive.train()
    decaying.load_test_data(test, y_test)
    adaptive.load_test_data(test, y_test)
    decaying.test()
    adaptive.test()
