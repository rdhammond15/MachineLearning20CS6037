"""
@file SMO algorithm implementation for 20CS6037

python SMO.py <full path to dataset> <epsilon>

@authors: Grace Gamstetter, Michael Gentile, and Richard Hammond
"""

import numpy as np
import sys


class Smo(object):
    def __init__(self, x, y, epsilon):
        """
        Initialize the algorithm with the training data
        
        @param x: The vector of training data
        @param y: The vector of labels
        """
        self.size = np.shape(x)[0]
        self.alpha2_old = None
        self.i1 = None
        self.i2 = None
        self.b = 0
        self.x = x
        self.y = y
        self.epsilon = epsilon
        self.alpha = self.init_alpha()
        self.w = self.calc_weight()

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
        index = np.where(self.y == adjust_label)[0]
        for i in range(size):
            if self.y == adjust_label:
                index = i
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
        for j in range(self.size):
            sum += self.alpha[j] * self.y[j] * (self.kernel(self.i1, j) - self.kernel(i, j) + self.y[i]) - self.y[self.i1]
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

    def calculate_b_from_kkt(self, i):
        """
        Get b for the desired index.
        @return b
        """
        # Get kkt for the function because we are solving for b
        # KKT(i) = a(i){yi(<w,xi> + b) - 1}
        # KKT(i)/a(i) = {yi(<w,xi> + b) - 1}
        # (KKT(i)/a(i)) + 1  = {yi(<w,xi> + b)
        # ((KKT(i)/a(i)) + 1) / yi  =  <w, xi> + b
        # (((KKT(i)/a(i)) + 1) / yi) - <w, xi>  = b
        """temp_kkt = self.kkt(i)
        temp_1 = (temp_kkt / self.alpha[i]) + 1
        temp_2 = temp_1 / self.y[i]
        temp_3 = temp_2 - np.dot(self.w, self.x[i])
        return temp_3
        """
        self.b = self.y[i] - np.matmul(self.w, self.x[i])
        return self.b

    @property
    def k(self):
        return self.kernel(self.i1, self.i1) + self.kernel(self.i2, self.i2) - 2*self.kernel(self.i1, self.i2)
    
    def run(self):
        """
        Execute the SMO
        """

        self.i1 = self.calc_i1()
        self.i2 = self.calc_i2()
        
        #find new alphas
        self.update_alpha2()
        self.update_alpha1()
        
        # ai < epsilon, a1 <-- 0
        self.alpha = list(map(lambda x: x if x >= self.epsilon else 0, self.alpha))

        self.w = self.calc_weight()

        # Select ai > 0, calculate b, Step 7
        # Didn't want to try lambda because this might get large.
        for i in range(len(self.alpha)):
            if self.alpha[i] > 0:
                self.b = self.calculate_b_from_kkt(i)
                break

    def is_classified(self):
        """
        Check to see if the training data is correctly classified
        
        uses the step function for w*x + b
        """
        for i in range(self.size):
            prediction = np.sign(np.dot(self.w.T, self.x[i]) + self.b)

            if prediction == self.y[i]:
                continue
            else:
                return False
        return True

if __name__ == '__main__':

    x = []
    y = []

    with open(sys.argv[1], 'r') as dataset:
        lines = dataset.readlines()
        for line in lines:
            data = [int(line.strip().strip('{}')) for line in line.split()]
            x.append(np.array([data[0], data[1]]))
            y.append(data[2])

    epsilon = float(sys.argv[2])
    smo_obj = Smo(x,y,epsilon)
    iter = 0
    while not smo_obj.is_classified() and iter < 10000:
        smo_obj.run()
        print(iter)
        iter = iter + 1
        
