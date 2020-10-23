"""
@file SMO algorithm implementation for 20CS6037

@authors: Grace Gamstetter, Michael Gentile, and Richard Hammond
"""
import numpy as np

class Smo(object):
    def __init__(self, x, y):
        """
        Initialize the algorthim with the training data
        
        @param x: The vector of training data
        @param y: The vector of lables
        """
        self.size = x.shape[0]
        self.alpha = self.init_alpha()
        self.w = self.calc_weight()
        self.b = 0

    def init_alpha(self):
        """
        Initialize all alphas to meet constraint sum from i to l y_i alpha_i = 0
        
        @return An array of alphas
        """
        alphas = numpy.random.randint(0, 10, self.size)
        product = np.dot(alphas, self.y)
        
        if product == 0
            return aplphas
        
        # figure out how far off from zero we are and adjusct accordingly
        adjust_label = 1 if product < 0 else -1
        index = np.where(self.y == adjust_label)[0][0]
        alphas[index] += abs(product)
        return alphas
        
    def calc_weight(self):
        """
        Generate the weight vector with equation w = sum(αiyixi)
;
        
        @return The weight vector
        """
        return np.sum(alpha * y * x for alpha, y, x in zip(self.alpha, self.y, self.x))

    def kernel(i, j):
        """
        Calculates the kernel at the given indeces
        
        @param i: first index to use (outer loop)
        @param j: inner loop index
        
        @return dot product of xi * xj
        """
        return self.x[i].dot(self.x[j])
        