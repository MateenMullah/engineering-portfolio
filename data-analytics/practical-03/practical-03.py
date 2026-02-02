# Imports
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
# Do not edit anything above this line

def select_feature(X, feature_names, feature_name):
    '''
    Select only one feature from higher-dimensional input data
    
    Parameters
    ----------
    X : (N,D) ndarray
        design matrix -- N samples with D features
    feature_names : list
        names of all D features
    feature_name : str
        name of feature to be selected
        
    Return
    ------
    x_i: (N,) ndarray
        samples of selected feature
    '''
    
    # insert code here
    index = feature_names.index(feature_name)
    return X[:, index]

def split_training_data(X, y, ratio_train):
    '''
    Retrieves the training data from a data set (from its start)
    
    Parameters
    ----------
    X : (N,D) ndarray
        design matrix -- N samples with D features
    y : (N,) ndarray
        output vector
    ratio_train : float
        fraction of data used for training
        
    Return
    ------
    X_train: (N_train,D) ndarray
        design matrix -- N_train samples with D features
    y_train: (N_train,) ndarray
        output vector
    '''
    
    split_index = int( len(X) * ratio_train)
    return X[:split_index], y[:split_index]
    
def split_test_data(X, y, ratio_train):
    '''
    Retrieves the test data from a data set (from its end)
    
    Parameters
    ----------
    X : (N,D) ndarray
        design matrix -- N samples with D features
    y : (N,) ndarray
        output vector
    ratio_train : float
        fraction of data used for training
        
    Return
    ------
    X_test: (N_test,D) ndarray
        design matrix -- N_test samples with D features
    y_test: (N_test,) ndarray
        output vector
    '''
    
    split_index = int( len(X) * ratio_train)
    return X[split_index:], y[split_index:]

def simple_linear_regression_predictions(x_train, y_train, x_new):
    '''
    Calculate the predicted outputs for simple linear regression
    Make use of the LinearRegression class
    
    Parameters
    ----------
    x_train : (N,) ndarray
        input vector
    y_train : (N,) ndarray
        output vector
    x_new : (N_new,) ndarray
        test input vector
        
    Return
    ------
    y_new : (N_new,) ndarray
        predicted output vector
    '''
    
    x_train_reshaped = x_train.reshape(-1, 1)
    x_new_reshaped = x_new.reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(x_train_reshaped, y_train)
    y_new = model.predict(x_new_reshaped)
    
    return y_new
 
    
def squared_loss(y_hat, y):
    '''
    Calculate the squared loss between the predicted and actual outputs
    
    Parameters
    ----------
    y_hat : (N,) ndarray
        predicted output vector
    y : (N,) ndarray
        actual output vector
        
    Return
    ------
    J : float
        squared loss
    '''
    
    # insert code here
    return np.sum((y - y_hat) ** 2)
    
def mean_squared_error(y_hat, y):
    '''
    Calculate the mean squared error between the predicted and actual outputs
    
    Parameters
    ----------
    y_hat : (N,) ndarray
        predicted output vector
    y : (N,) ndarray
        actual output vector
        
    Return
    ------
    MSE : float
        mean squared error
    '''
    
    # insert code here
    return np.mean((y - y_hat) ** 2)
    
def root_mean_square_error(y_hat, y):
    '''
    Calculate the root-mean-square error between the predicted and actual outputs
    
    Parameters
    ----------
    y_hat : (N,) ndarray
        predicted output vector
    y : (N,) ndarray
        actual output vector
        
    Return
    ------
    RMSE : float
        root-mean-square error
    '''
    
    # insert code here
    return np.sqrt(mean_squared_error(y_hat, y))

def simple_polynomial_regression_predictions(P, x_train, y_train, x_new):
    '''
    Calculate the predicted outputs for simple polynomial regression
    Make use of the PolynomialFeatures and LinearRegression classes
    
    Parameters
    ----------
    P : int
        degree of polynomial
    x_train : (N,) ndarray
        input vector
    y_train : (N,) ndarray
        output vector
    x_new : (N_new,) ndarray
        test input vector
        
    Return
    ------
    y_new : (N_new,) ndarray
        predicted output vector
    '''
    
    # insert code here

    x_train_reshaped = x_train.reshape(-1, 1)
    x_new_reshaped = x_new.reshape(-1, 1)

    poly = PolynomialFeatures(P)
    X_train_poly = poly.fit_transform(x_train_reshaped)
    X_new_poly = poly.transform(x_new_reshaped)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_new = model.predict(X_new_poly)
    
    return y_new

def multiple_linear_regression_predictions(X_train, y_train, X_new):
    '''
    Calculate the predicted outputs for multiple linear regression
    Make use of the LinearRegression class
    
    Parameters
    ----------
    X_train : (N,D) ndarray
        design matrix -- N samples with D features
    y_train : (N,) ndarray
        output vector
    X_new : (N_new,D) ndarray
        test input matrix -- N_new samples with D features
        
    Return
    ------
    y_new : (N_new,) ndarray
        predicted output vector
    '''

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_new = model.predict(X_new)
    
    return y_new    

def multiple_linear_regression_weights(X_train, y_train):
    '''
    Calculate the least-squares weights for multiple linear regression
    Make use of the LinearRegression class
    
    Parameters
    ----------
    X_train : (N,D) ndarray
        design matrix -- N samples with D features
    y_train : (N,) ndarray
        output vector
        
    Return
    ------
    w : (D+1,) ndarray
        weight vector
    '''
    model = LinearRegression()
    model.fit(X_train, y_train)
    w_0 = model.intercept_
    w_re = model.coef_
    w = np.concatenate(([w_0], w_rest))
    
    return w