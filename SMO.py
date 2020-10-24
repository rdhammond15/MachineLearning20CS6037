"""
@file SMO algorithm implementation for 20CS6037

@authors: Grace Gamstetter, Michael Gentile, and Richard Hammond
"""

import numpy as np


class Smo(object):
    def __init__(self, x, y, epsilon):
        """
        Initialize the algorithm with the training data
        
        @param x: The vector of training data
        @param y: The vector of labels
        """
        self.size = np.shape[x]
        self.alpha = self.init_alpha()
        self.alpha2_old = None
        self.i1 = None
        self.i2 = None
        self.w = self.calc_weight()
        self.b = 0
        self.x = x
        self.y = y
        self.epsilon = epsilon

    def init_alpha(self):
        """
        Initialize all alphas to meet constraint sum from i to l y_i alpha_i = 0
        
        @return An array of alphas
        """
        alphas = np.random.randint(0, 10, self.size)
        product = np.dot(alphas, self.y)
        
        if product == 0:
            return alphas
        
        # figure out how far off from zero we are and adjust accordingly
        adjust_label = 1 if product < 0 else -1
        index = np.where(self.y == adjust_label)[0][0]
        alphas[index] += abs(product)
        return alphas
        
    def calc_weight(self):
        """
        Generate the weight vector with equation w = sum(aiyixi)

        @return The weight vector
        """
        return np.sum(alpha * y * x for alpha, y, x in zip(self.alpha, self.y, self.x))

    def kernel(self, i, j):
        """
        Calculates the kernel at the given indexes
        
        @param i: first index to use (outer loop)
        @param j: inner loop index
        
        @return dot product of xi * xj
        """
        return np.dot(self.x[i], self.x[j])
        
    def error(self, i):
        """
        Calculate difference from label
        
        @param i: current index
        """
        sum = 0 
        for j in self.size:
            sum += self.alpha[j] * self.y[j] * (self.kernel(self.i1, j) - self.kernel(i, j) +
                                                self.y[i] - self.y[self.i1])
        return sum
        
    def kkt(self, i):
        """
        Calculate KKT condition
        
        @param i: current index
        """
        return self.alpha[i] * (self.y[i] * (np.dot(self.w, self.x[i]) + self.b) - 1)
        
    def calc_i1(self):
        """
        Find argmax of KKT condition
        
        @return the max index
        """
        return np.argmax([self.kkt(i) for i in range(self.size)])
        
    def calc_i2(self):
        """
        Find armgax of error
        
        @return the max index
        """
        return np.argmax([self.error(i) for i in range(self.size)])
        
    def update_alpha2(self):
        """
        Update the value of alpha 2
        """
        self.alpha2_old = self.alpha[self.i2]
        self.alpha[self.i2] = self.alpha[self.i2] + (self.y[self.i2] * self.error(self.i2))/self.k
        
    def update_alpha1(self):
        """
        Update the value of alhpa1 based on alpha 2
        """
        self.alpha[self.i1] = self.alpha[self.i1] + self.y[self.i1] * self.y[self.i2] * (self.alpha2_old - self.alpha[self.i2])
        
    @property
    def k(self):
        return self.kernel(self.i1, self.i1) + self.kernel(self.i2, self.i2) - 2*self.kernel(self.i1, self.i2)
    
    def run(self):
        """
        Execute the SMO
        """
        self.i1 = self.calc_i1()
        self.i2 = self.calc_i2()
        
        # ai < epsilon, a1 <-- 0
        self.alpha = map(lambda x: x if x >= self.epsilon else 0, self.alpha)
        
        # TODO select ai > 0, calculate b
        
        # TODO test for classification and repeat until classified
