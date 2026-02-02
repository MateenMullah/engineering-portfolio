# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from collections import Counter
from sklearn import cluster
# Do not edit anything above this line
# Any imports below this line will be ignored

class K_Means:
    '''
    Perform clustering using the K-means algorithm

    '''

    def __init__(self, X, K):
        '''
        Store the dataset and number of clusters as class attributes
        Initialize the cluster assignment and mean vectors

        Parameters
        ----------
        X : (N,D) ndarray
            design matrix -- N samples with D features
        K : int
            number of classes
        '''

        self.K = K
        self.X = X
        N, D = X.shape
        self.clusters = np.random.randint(0, self.K, N)
        self.means = np.zeros((self.K, D))

    def update_clusters(self, n_iterations):
        '''
        Update the cluster assignment and mean vectors

        Parameters
        ----------
        n_iterations : int
            number of iterations
        '''
        
        # insert code here
        for i in range(n_iterations):
            distances = distance.cdist(self.X, self.means, metric = 'euclidean')
            self.clusters = np.argmin(distances, axis = 1)
            for k in range(self.K):
                assigned_points = self.X[self.clusters == k]
                if len(assigned_points) > 0:
                    self.means[k] = np.mean(assigned_points, axis = 0)
                else:
                    self.means[k] = self.X[np.random.choice(len(self.X))]
                    
    def purity(self, y):
        '''
        Calculate the purity of the cluster assignment
        
        Parameters
        ----------
        y : (N,) ndarray
            output vector

        Returns
        -------
        p : float
            purity
        '''
        
        # insert code here
        total = 0
        for k in range(self.K):
            cluster_ind = np.where(self.clusters == k)[0]

            actual_class = y[cluster_ind]
            most_class, counter = Counter(actual_class).most_common(1)[0]
            total += counter
            p = total/len(y)
        return p
                    
    def get_clusters(self):
        '''
        Retrieve the cluster assignment

        Returns
        -------
        clusters : (N,) ndarray
            cluster assignment
        '''

        return self.clusters
                    
    def get_means(self):
        '''
        Retrieve the mean vectors

        Returns
        -------
        means : (K,D) ndarray
            mean vectors
        '''

        return self.means
    
    def plot_clusters(self, title=None):
        '''
        Plot the cluster assignment and mean vectors on top of the given data

        Parameters
        ----------
        title : str
            title of the plot
        '''
        
        plt.figure()
        for i_cluster in range(self.K):
            plt.scatter(self.X[self.clusters==i_cluster, 2], self.X[self.clusters==i_cluster, 3], label="Cluster {:d}".format(self.K), edgecolor="white")
        plt.scatter(self.means[:, 2], self.means[:, 3], marker="X", s=100, edgecolor="white", color="black")
        plt.xlabel("Petal length (cm)")
        plt.ylabel("Petal width (cm)")
        if title:
            plt.title(title)
        plt.show()
        
