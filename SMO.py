"""
@file SMO algorithm implementation for 20CS6037

@authors: Grace Gamstetter, Michael Gentile, and Richard Hammond
"""
import numpy as np

def init_alpha(size, labels):
    """
    Initialize all alphas to meet constraint sum from i to l y_i alpha_i = 0
    
    @param size: The size of the traning data
    @param lables: This is y_i which is +-1
    
    @return An array of alphas
    """
    alphas = numpy.random.randint(0, 10, size)
    product = np.dot(alphas, labels)
    
    if product == 0
        return aplphas
    
    # figure out how far off from zero we are and adjusct accordingly
    adjust_label = 1 if product < 0 else -1
    index = np.where(lables == adjust_label)[0][0]
    alphas[index] += abs(product)
    return alphas
