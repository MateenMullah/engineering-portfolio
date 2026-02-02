# Imports
import numpy as np
# Do not edit anything above this line
# Any imports below this line will be ignored

class SoftmaxClassifier:
    '''
    Perform classification using softmax regression

    '''

    def __init__(self, K, n_iterations, learning_rate, W_init):
        '''
        Store the initial class attributes

        Parameters
        ----------
        K : int
            number of classes
        n_iterations : int
            number of iterations
        learning_rate : float
            learning rate
        W_init : ndarray(K,D+1)
            initial parameter matrix
        '''

        self.K = K
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.W = W_init.astype(float)

    def nll_loss(self, X, y, W):
        '''
        Calculate the negative log likelihood

        Parameters
        ----------
        X : (N,D) ndarray
            design matrix -- N samples with D features
        y : (N,) ndarray
            output vector
        W : (K,D+1) ndarray
            weight matrix

        Return
        ------
        J : float
            negative log likelihood
        '''

        # insert code here
        N = X.shape[0]
        X_bias = np.c_[np.ones(N), X]
        scores = np.dot(X_bias, W.T)
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        correct_logprobs = -np.log(probs[np.arange(N), y])
        return np.sum(correct_logprobs) / N

    def nll_grad(self, X, y, W):
        '''
        Calculate the gradient of the negative log likelihood

        Parameters
        ----------
        X : (N,D) ndarray
            design matrix -- N samples with D features
        y : (N,) ndarray
            output vector
        W : (K,D+1) ndarray
            weight matrix

        Return
        ------
        grad : (K,D+1) ndarray
            negative log likelihood gradient
        '''

        # insert code here
        N = X.shape[0]
        X_bias = np.c_[np.ones(N), X]
        scores = np.dot(X_bias, W.T)
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        probs[np.arange(N), y] -= 1
        return np.dot(probs.T, X_bias) / N


    def fit(self, X, y):
        '''
        Optimize and store the weight matrix using the initial class attributes

        Parameters
        ----------
        X : (N,D) ndarray
            design matrix -- N samples with D features
        y : (N,) ndarray
            output vector
        '''

        # insert code here
        self.W = self.W.astype(float)
        for _ in range(self.n_iterations):
            grad = self.nll_grad(X, y, self.W)
            self.W -= self.learning_rate * grad

    def parameters(self):
        '''
        Retrieve the weight matrix

        Return
        ------
        W : (K,D+1) ndarray
            weight matrix
        '''

        return self.W

    def predict(self, X_new):
        '''
        Calculate the predicted outputs for softmax regression

        Parameters
        ----------
        X_new : (N_new,D) ndarray
            test input matrix -- N_new samples with D features

        Return
        ------
        y_new : (N_new,) ndarray
            predicted output vector
        '''

        # insert code here
        N = X_new.shape[0]
        X_new_bias = np.c_[np.ones(N), X_new]
        scores = np.dot(X_new_bias, self.W.T)
        return np.argmax(scores, axis=1)

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
    acc = np.mean(y_hat == y) * 100
    return acc

def confusion_matrix(y_hat, y):
    '''
    Calculate the confusion matrix of the predicted outputs (rows) relative to the actual outputs (columns)

    Parameters
    ----------
    y_hat : (N,) ndarray
        predicted output vector
    y : (N,) ndarray
        actual output vector

    Return
    ------
    confusion_matrix : (K,K) ndarray
        confusion matrix
    '''

    # insert code here
    K = len(np.unique(np.concatenate((y_hat, y))))
    cm = np.zeros((K, K), dtype=int)
    for i in range(len(y)):
        cm[int(y_hat[i]), int(y[i])] += 1
    return cm

def precision(confusion_matrix):
    '''
    Calculate the per-class precision based on the confusion matrix

    Parameters
    ----------
    confusion_matrix : (K,K) ndarray
        confusion matrix

    Return
    ------
    precision : (K,) ndarray
        vector with per-class precision
    '''

    # insert code here
    preci = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
    return preci

def recall(confusion_matrix):
    '''
    Calculate the per-class recall based on the confusion matrix

    Parameters
    ----------
    confusion_matrix : (K,K) ndarray
        confusion matrix

    Return
    ------
    recall : (K,) ndarray
        vector with per-class recall
    '''

    # insert code here
    reca = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    return reca

