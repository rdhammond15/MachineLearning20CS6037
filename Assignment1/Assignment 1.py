from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(v_hat):
    """
    Apply the sigmoid to the target function
    """
    return 1.0 / (1.0 + np.exp(-v_hat))

def predict(features, coefs):
    """
    Apply coefficients to features and apply the sigmoid to it
    """
    return sigmoid(np.dot(features, coefs))

def classify(features, coefs):
    """
    Apply binary classification to the data based on the sigmoid
    """
    return map(lambda x: 1 if x >= 0.5 else 0, predict(features, coefs))

def cost(hypothesis, expected):
    """
    Apply the cost function to the hypothesis based on the expected output
    """
    return (-expected * np.log(hypothesis) - (1 - expected) * np.log(1 - hypothesis)).mean()

def train(features, expected, rate=.1):
    """
    Find the coeficients by minimizing the cost funciton
    """
    
    # initialize a matrix of coefficients the same size as the input data
    coef = np.zeros(features.shape[1])
    
    for i in range(1000):
        hypothesis = predict(features, coef)
        gradient = np.dot(features.T, (hypothesis - expected)) / expected.size
        coef -= rate * gradient
    
    return coef
    
    
if __name__ == "__main__":
    iris = load_iris()

    # we are intrested in setosa or not so modify the label
    setosa = map(lambda x: 0 if not x else 1, iris.target)

    # split into training and test data randomly
    prop_train, prop_test, label_train, label_test = train_test_split(iris.data[:,:1], setosa, test_size=.5, random_state=5)
    np_prop_train = np.asarray(prop_train)
    np_label_train = np.asarray(label_train)
    np_prop_test = np.asarray(prop_test)
    np_label_test = np.asarray(label_test)

    # concatenate a row of ones for the bias term
    np_prop_train = np.concatenate((np.ones((np_prop_train.shape[0], 1)), np_prop_train), axis=1)
    np_prop_test = np.concatenate((np.ones((np_prop_test.shape[0], 1)), np_prop_test), axis=1)
    
    # find thetas
    theta = train(np_prop_train, np_label_train)
    
    # create an x axis for our sigmoid using theta
    x_axis = np.linspace(-5, 10, 3000)
    x_axis = [[x] for x in x_axis]
    x_axis = np.asarray(x_axis)
    x_axis = np.concatenate((np.ones((x_axis.shape[0], 1)), x_axis), axis=1)
    
    # find the sigmoid based on the x_axis
    sigout = predict(x_axis, theta)
    
    # plot sigmoid
    plt.plot(x_axis[:,1], sigout, color='blue', linewidth=2)
    
    # get the test data output and classify it
    y = classify(np_prop_train, theta)
    
    # find the percentage that we were off
    same_count = 0
    for predicted, actual in zip(y, np_label_test):
        if predicted == actual:
            same_count += 1
    
    print "Predicted {} out of {} values correctly".format(same_count, len(y))
        
    plt.scatter(np_prop_train[:,1], y, color='red')
    plt.ylabel('Setosa Probability')
    plt.xlabel('Sepal Length (cm)')
    
    plt.show()
    
    

