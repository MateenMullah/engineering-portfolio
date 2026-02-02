# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def add_bias(X):
    '''
    Add a column of 1's to the start of the design matrix
    
    Parameters
    ----------
    X : (N,D) ndarray
        design matrix -- N samples with D features
        
    Return
    ------
    X_bias: (N,D+1) ndarray
        design matrix
    '''
    X_bias = np.concatenate((np.ones((X.shape[0],1)),X), axis = 1)
    return X_bias

def calculate_w(X, y):
    '''
    Calculate the least-squares weights using the normal equations
    
    Parameters
    ----------
    X : (N,K) ndarray
        design matrix -- N samples with K features
    y : (N,) ndarray
        output vector
        
    Return
    ------
    w_fit : (K,) ndarray
        weight vector
    '''
    
    inv_X_transposeX = np.linalg.inv(np.dot(X.T, X))
    X_transposeY = np.dot(X.T, y)
    w = np.dot(inv_X_transposeX, X_transposeY)
    return w

def add_quadratic(X):
    '''
    Add a column to the end of the design matrix containing the corresponding x^2 terms
    
    Parameters
    ----------
    X : (N,1) ndarray
        design matrix -- N samples with 1 feature
        
    Return
    ------
    X_quad: (N,2) ndarray
        design matrix
    '''
    
    # insert code here
    quad = np.square(X)
    X_quad = np.hstack([X, quad])
    return X_quad

def plot_quadratic_fit(x, y, w_fit):
    '''
    Plot the quadratic fit on top of the given data
    
    Parameters
    ----------
    x : (N,) ndarray
        input vector
    y : (N,) ndarray
        output vector
    w_fit : (3,) ndarray
        coefficients [w_0, w_1, w_2]
    '''
    x_range = np.linspace(np.min(x), np.max(x), 100) 
    X_range = np.c_[np.ones(x_range.shape[0]), x_range, x_range**2] 
    y_fit = np.dot(X_range, w_fit)  
    
    
    plt.scatter(x, y)
    plt.plot(x_range, y_fit, color='red')
  
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Quadratic Fit')
 
    plt.legend()
    plt.show()
    return

    
def fit_polynomial_regression(X, y, P):
    '''
    Calculate the least-squares weights for polynomial regression
    Make use of the PolynomialFeatures class
    
    Parameters
    ----------
    X : (N,D) ndarray
        design matrix -- N samples with D features
    y : (N,) ndarray
        output vector
    P : int
        degree of polynomial
        
    Return
    ------
    w_fit : (K,) ndarray
        weight vector
    '''
    
    # insert code here
    X = X[:, 0]
    X_design = np.vander(X, P + 1, increasing=True)
  
    inv_X_transposeX = np.linalg.inv(np.dot(X_design.T, X_design))
    X_transposeY = np.dot(X_design.T, y)
    w_fit = np.dot(inv_X_transposeX, X_transposeY)
    
    return w_fit

def get_polynomial_regression_predictions(X, y, P, x_new):
    '''
    Calculate the predicted outputs for polynomial regression
    Make use of the PolynomialFeatures and LinearRegression classes
    
    Parameters
    ----------
    X : (N,D) ndarray
        design matrix -- N samples with D features
    y : (N,) ndarray
        output vector
    P : int
        degree of polynomial
    x_new : (N_new,) ndarray
        test input vector
        
    Return
    ------
    y_new : (N_new,) ndarray
        predicted output vector
    '''
    
    # insert code here
    degree = PolynomialFeatures(P)
    X_poly = degree.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    x_new_poly = degree.transform(x_new.reshape(-1, 1))
    y_new = model.predict(x_new_poly)
    return y_new