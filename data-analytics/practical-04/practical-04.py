# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
# Do not edit anything above this line

def knn_predict(X_new, X_train, y_train, K=1):
    '''
    Calculate the predicted outputs for binary K-NN classification
    
    Parameters
    ----------
    X_new : (N_new,D) ndarray
        test input matrix -- N_new samples with D features
    X_train : (N_train,D) ndarray
        design matrix -- NN_train samples with D features
    y_train : (N_train,) ndarray
        output vector
    K : int
        number of neighbours
        
    Return
    ------
    y_new : (N_new,) ndarray
        predicted output vector
    '''
    
    # insert code here
    distances = distance.cdist(X_new, X_train, metric='euclidean')
    nearest_neighbors = np.argsort(distances, axis=1)[:, :K]
    neighbor_labels = y_train[nearest_neighbors]
    predictions = np.round(np.mean(neighbor_labels, axis=1)).astype(int)
    return predictions

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
    correct_predictions = (y_hat == y).sum()
    total_predictions = len(y)
    acc = (correct_predictions / total_predictions) * 100
    
    return acc
   

def plot_knn_accuracy(X_val, y_val, X_train, y_train):
    '''
    Plot the training and validation accuracy of K-NN classification for K=1 to K=15
    
    Parameters
    ----------
    X_val : (N_val,D) ndarray
        validation design matrix -- N_val samples with D features
    y_val : (N_val,) ndarray
        validation output vector
    X_train : (N_train,D) ndarray
        training design matrix -- N_train samples with D features
    y_train : (N_train,) ndarray
        training output vector
    '''
    
    # insert code here
    training_accuracies = []
    validation_accuracies = []

    for K in range(1, 16):
        y_hat_val = knn_predict(X_val, X_train, y_train, K)
        accuracy_val = accuracy(y_hat_val, y_val)
        validation_accuracies.append(accuracy_val)

        y_hat_train = knn_predict(X_train[:len(y_val)], X_train, y_train, K)
        accuracy_train = accuracy(y_hat_train, y_train[:len(y_val)])
        training_accuracies.append(accuracy_train)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 16), training_accuracies, label='Training Accuracy', marker='o', linestyle='-', color='black')
    plt.plot(range(1, 16), validation_accuracies, label='Validation Accuracy', marker='o', linestyle='--', color='red')

    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.title('KNN Accuracy for Different K values')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def gaussian_pdf(x, mean, cov):
    '''
    Calculate the likelihood of a Gaussian PDF
    
    Parameters
    ----------
    x : (D,) ndarray
        input vector
    mean : (D,) ndarray
        mean vector
    cov : (D,D) ndarray
        covariance matrix
        
    Return
    ------
    p : float
        likelihood
    '''
    
    # insert code here
    det_cov = np.linalg.det(cov)
    inverse_cov = np.linalg.inv(cov)

    diff = x - mean
    exp = -0.5 * np.dot(np.dot(diff.T, inverse_cov), diff)
    
    norm = 1/(np.sqrt((2*np.pi)** len(x) * det_cov))
    p = norm * np.exp(exp)
    return p
    

    