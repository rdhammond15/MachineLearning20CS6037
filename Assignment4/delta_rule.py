"""
@file Delta rule classification algorithm implementation for 20CS6037

@authors: Richard Hammond
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas


class Delta(object):
    def __init__(self, train, labels, epoch, rate):
        """
        Initialize the delta training object

        @param train: The np array training data
        @param labels: The labels for the training data
        @param epoch: The number of iteratations to use for training
        @param rate: The learning rate eta
        """
        self.labels = labels
        self.data = train 
        self.w = np.random.rand(train.shape[1])
        self.epoch = epoch
        self.rate = rate
        self.df = pandas.DataFrame(self.data, columns=['bias', 'x1', 'x2'])
        self.df.insert(3, "labels", labels, True)

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

    def train(self):
        """
        Use the training data to find the weights for the features and plot the error
        """
        for i in range(1, self.epoch + 1):
            # accumulate the error
            delta_w = self.cost()
            self.w += delta_w
            plt.figure(1)
            plt.plot(i, self.error, 'ro')

            # plot the decision boundry peridocially
            if i in (5, 10, 50, 100):
                self.plot_decision_surface(i)

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

    for rate in (0.5, 0.1, 0.01, 0.001, 0.0001):
        print "Training with learning rate {}".format(rate)
        delta_rule = Delta(training, y_train, 100, rate)
        delta_rule.train()
        delta_rule.load_test_data(test, y_test)
        delta_rule.test()