"""
@file Delta rule classification algorithm implementation for 20CS6037

@authors: Richard Hammond
"""
import numpy as np


class Delta(object):
    def __init__(self, train, labels):
        self.train = train 
        self.labels = labels
        self.w = np.random.rand(train.shape[1])

    @property
    def hypothesis(self):
        """
        Return the hypotheses using w*x
        """
        return self.train.dot(self.w)

    @property
    def error(self):
        """
        Calculate: 1/2(Summation(t - o)^2)
        
        @return the error according to the calculation above
        """
        return .5 * np.sum(np.square(self.labels - self.hypothesis))


def concept(x1, x2):
    """
    The concept that we are trying to fit which is 𝑥1 + 2𝑥2 − 2 > 0

    @param x1: x1 in the equation
    @param x2: x2 in the equation

    @return The equations value
    """
    return x1 + 2*x2 - 2 > 0


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

    delta_rule = Delta(training, y_train)
    err = delta_rule.error
    print err
