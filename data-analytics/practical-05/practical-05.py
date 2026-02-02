# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn import datasets, linear_model, neighbors, preprocessing
# Do not edit anything above this line

def load_data(filename):
    '''
    Load data from a CSV file
    
    Parameters
    ----------
    filename : str
        CSV file with 2-D input in first two columns and one output column
        
    Return
    ------
    X: (N,2) ndarray
        design matrix -- N samples with 2 features
    y: (N,) ndarray
        output vector
    '''
    
    # insert code here
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    
    X = data[:, :2]
    y = data[:, 2]
    
    return X, y

def plot_data(X, y, x_label, y_label, title):
    '''
    Plot a scatter plot of the given data and distinguish between the binary classes
    
    Parameters
    ----------
    X : (N,2) ndarray
        design matrix -- N samples with 2 features
    y : (N,) ndarray
        output vector
    x_label : str
        x-axis label
    y_label : str
        y-axis label
    title : str
        plot title
    '''
    
    # insert code here
    not_admitted = X[y == 0]
    admitted = X[y == 1]
    
    plt.scatter(not_admitted[:, 0], not_admitted[:, 1], c='red', marker='x', label='not admitted')
    plt.scatter(admitted[:, 0], admitted[:, 1], c='blue', marker='o', label='admitted')
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()
    
def normalize_input_data(X):
    '''
    Normalize each input sample by subtracting the mean and dividing by the standard deviation
    
    Parameters
    ----------
    X : (N,D) ndarray
        design matrix -- N samples with D features
        
    Return
    ------
    X_norm: (N,D) ndarray
        normalized design matrix -- N samples with D features
    '''
    
    # insert code here
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
   
    X_norm = (X - mean) / std
    
    return X_norm

def split_data(X, y, N_train, N_val):
    '''
    Retrieve the training, validation and test data from a data set
    
    Parameters
    ----------
    X : (N,D) ndarray
        design matrix -- N samples with D features
    y : (N,) ndarray
        output vector
    N_train : float
        number of data points used for training
    N_val : float
        number of data points used for validation
        
    Return
    ------
    X_train: (N_train,D) ndarray
        design matrix -- N_train samples with D features
    y_train: (N_train,) ndarray
        output vector
    X_val: (N_val,D) ndarray
        design matrix -- N_val samples with D features
    y_val: (N_val,) ndarray
        output vector
    X_test: (N_test,D) ndarray
        design matrix -- N_test samples with D features
    y_test: (N_test,) ndarray
        output vector
    '''
    
    # insert code here
    X_train = X[:N_train]
    y_train = y[:N_train]
    
    X_val = X[N_train:N_train + N_val]
    y_val = y[N_train:N_train + N_val]
    
    X_test = X[N_train + N_val:]
    y_test = y[N_train + N_val:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def sigmoid(x):
    '''
    Calculate the sigmoid of x
    
    Parameters
    ----------
    x : float
        input
        
    Return
    ------
    sig : float
        sigmoid
    '''
    # insert code here
    ans = 1 / (1 + np.exp(-x))
    return ans

def nll_loss(X, y, w):
    '''
    Calculate the negative log likelihood for logistic regression
    Make use of the sigmoid functon above
    
    Parameters
    ----------
    X : (N,D+1) ndarray
        design matrix -- N samples with D+1 features
    y : (N,) ndarray
        output vector
    w : (D+1,) ndarray
        weights
        
    Return
    ------
    J : float
        negative log likelihood
    '''

    # insert code here
    z = np.dot(X, w)
    predictions = sigmoid(z)
    J = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    
    return J

def nll_grad(X, y, w):
    '''
    Calculate the gradient of the negative log likelihood for logistic regression
    
    Parameters
    ----------
    X : (N,D+1) ndarray
        design matrix -- N samples with D+1 features
    y : (N,) ndarray
        output vector
    w : (D+1,) ndarray
        weights
        
    Return
    ------
    grad : (D+1,) ndarray
        negative log likelihood gradient
    '''

    # insert code here
    z = np.dot(X, w)
    predictions = sigmoid(z)

    errors = predictions - y
    grad = np.dot(X.T, errors) / len(y)

    return grad

def plot_linear_decision_boundary(X, y, w):
    '''
    Plot the linear decision boundary on top of the given 2-D data
    
    Parameters
    ----------
    X : (N,2) ndarray
        design matrix -- N samples with 2 features
    y : (N,) ndarray
        output vector
    w : (3,) ndarray
        weights [w_0, w_1, w_2]
    '''
    
    # insert code here
    admitted = (y == 1)
    not_admitted = (y == 0)
    plt.scatter(X[admitted, 0], X[admitted, 1], c='green', label='Admitted', marker='o')
    plt.scatter(X[not_admitted, 0], X[not_admitted, 1], c='red', label='Not Admitted', marker='x')

    x1_values = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2_values = -(w[0] + w[1] * x1_values) / w[2]

    plt.plot(x1_values, x2_values, 'blue', label='Decision Boundary')

    plt.xlabel("Exam 1")
    plt.ylabel("Exam 2")
    plt.title("Decision Boundary")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
    
def logistic_regression_predict(X_new, w):
    '''
    Calculate the predicted outputs for logistic regression

    Parameters
    ----------
    X_new : (N_new,D+1) ndarray
        test input matrix -- N_new samples with D+1 features
    w : (D+1,) ndarray
        weights

    Return
    ------
    y_new : (N_new,) ndarray
        predicted output vector
    '''
    
    # insert code here
    z = np.dot(X_new, w)

    probabilities = 1 / (1 + np.exp(-z))
    y_new = (probabilities >= 0.5).astype(int)
    
    return y_new

def accuracy(y_hat, y):
    '''
    Calculate the classification accuracy of the predicted outputs relative to the actual outputs
    
    Parameters
    ----------
    y_hat : (N,) ndarray
        predicted output vector
    y : (N,) ndarray
        actual output vector
        
    Return
    ------
    acc : float
        accuracy as a percentage
    '''
    
    # insert code here
    correct_predictions = np.sum(y_hat == y)
    acc = (correct_predictions / len(y)) * 100
    
    return acc
        
