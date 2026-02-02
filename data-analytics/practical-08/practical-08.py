# Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# Do not edit anything above this line
# Any imports below this line will be ignored

class TreeNode:
    '''
    Support class for the 'DecisionTree' class
    Contains references to its children if not a leaf node

    '''

    def __init__(self, X, y, K):
        '''
        Store the samples that belong to this node and set it as a leaf node

        Parameters
        ----------
        X : (N,D) ndarray
            design matrix -- N samples with D features
        y : (N,) ndarray
            output vector
        K : int
            number of classes
        '''

        self.X = X
        self.y = y
        self.K = K
        self.leaf_node = True

    def prediction(self):
        '''
        Calculate the predicted output (class) for this node

        Return
        ------
        y_hat : float
            predicted output
        '''

        # insert code here
        counts = np.bincount(self.y, minlength=self.K)
        y_hat = np.argmax(counts)
        return y_hat

    def gini_index(self):
        '''
        Calculate the Gini index for all the samples in this node

        Return
        ------
        G : float
            Gini index
        '''

        # insert code here
        counts = np.bincount(self.y, minlength=self.K)
        prob = counts / np.sum(counts)
        G = np.sum(prob * (1 - prob))
        return G

    def gini_index_if_split(self, j, s):
        '''
        Calculate the Gini index for this node if it were to be split at x_j = s
        Do not create any new 'TreeNode' objects

        Parameters
        ----------
        j : int
            dimension (starting from 0)
        s : float
            split point

        Return
        ------
        G : float
            Gini index
        '''
        
        # insert code here
        left_mask = self.X[:, j] < s
        right_mask = self.X[:, j] >= s

        left_split = self.y[left_mask]
        right_split = self.y[right_mask]


        left_counts = np.bincount(left_split, minlength=self.K)
        left_prob = left_counts / len(left_split)
        left_gini = np.sum(left_prob *(1 - left_prob))

        right_counts = np.bincount(right_split, minlength=self.K)
        right_prob = right_counts / len(right_split)
        right_gini = np.sum(right_prob *(1 - right_prob))

        return left_gini + right_gini

    def split(self, j, s):
        '''
        Split this node based on x_j < s, creating two new 'TreeNode' objects
        Store these as the class attributes 'true_child' and 'false_child'

        Parameters
        ----------
        j : int
            dimension (starting from 0)
        s : float
            split point
        '''

        self.leaf_node = False
        self.j = j
        self.s = s

        # insert code here

        left_split = self.X[:, j] < s
        right_split = self.X[:, j] >= s

        left_X = self.X[left_split]
        left_y = self.y[left_split]
        right_X = self.X[right_split]
        right_y = self.y[right_split]

        self.true_child = TreeNode(left_X, left_y, self.K)
        self.false_child = TreeNode(right_X, right_y, self.K)

        self.leaf_node = False

    def traverse(self, x):
        '''
        Return either of the class attributes 'true_child' or 'false_child' based on x_j < s

        Parameters
        ----------
        x : (N,) ndarray
            input vector

        Return
        ------
        node : TreeNode
            child node
        '''
        
        # insert code here
        if x[self.j] < self.s:
            return self.true_child
        else:
            return self.false_child

    def possible_split_points(self, j):
        '''
        Calculate all logical split points for the j'th feature

        Parameters
        ----------
        j : int
            dimension (starting from 0)

        Return
        ------
        split_points : (K,) ndarray
            split points
        '''
        
        split_points = (np.unique(self.X[:,j])[1:]+np.unique(self.X[:,j])[:-1])/2

        return split_points

class DecisionTree:
    '''
    Perform classification using trees

    '''

    def __init__(self, max_leaf_nodes, K):
        '''
        Store the initial class attributes

        Parameters
        ----------
        max_leaf_nodes : float
            maximum number of leaf nodes
        K : int
            number of classes
        '''

        self.max_leaf_nodes = max_leaf_nodes
        self.K = K

    def fit(self, X, y):
        '''
        Fit the decision tree using the tree growing algorithm
        Only store the root node as a class attribute
        All the other nodes will have references from their parents due to the 'split' member function
        Use the member functions and attributes of the 'TreeNode' class

        Parameters
        ----------
        X : (N,D) ndarray
            design matrix -- N samples with D features
        y : (N,) ndarray
            output vector
        '''

        self.root = TreeNode(X,y,self.K)
        # insert code here
        self.root = TreeNode(X, y, self.K)
        queue = [self.root]
        num_leaf_nodes = 1

        while queue and num_leaf_nodes < self.max_leaf_nodes:
            current_node = queue.pop(0)

            best_gini = np.inf
            best_j = None
            best_s = None

            for j in range(current_node.X.shape[1]):
                split_points = current_node.possible_split_points(j)
                for s in split_points:
                    gini = current_node.gini_index_if_split(j, s)
                    if gini < best_gini:
                        best_gini = gini
                        best_j = j
                        best_s = s
            if best_j is not None:
                current_node.split(best_j, best_s)
                queue.append(current_node.true_child)
                queue.append(current_node.false_child)
                num_leaf_nodes += 1

    def predict(self, X_new):
        '''
        Calculate the predicted outputs for the decision tree
        Use the member functions and attributes of the 'TreeNode' class

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
        y_new = np.empty(X_new.shape[0], dtype=self.root.y.dtype)
        for i, x in enumerate(X_new):
            node = self.root
            while not node.leaf_node:
                node = node.traverse(x)
            y_new[i] = node.prediction()
        return y_new
# The functions below this point are already completed
# They are used in the notebook for visualizations

def plot_decision_boundaries(X,y,model):
    # Make grid
    N = 150
    grid_x1 = np.linspace(np.min(X[:, 0]) - 0.1, np.max(X[:, 0]) + 0.1, N)
    grid_x2 = np.linspace(np.min(X[:, 1]) - 0.1, np.max(X[:, 1]) + 0.1, N)
    grid_x1, grid_x2 = np.meshgrid(grid_x1, grid_x2)
    X_grid = np.c_[grid_x1.ravel(), grid_x2.ravel()]
    # Predictions
    predictions = model.predict(X_grid)
    grid_predictions = predictions.reshape(grid_x1.shape)
    # Plot the decision boundary
    plt.contourf(grid_x1, grid_x2, grid_predictions, cmap=ListedColormap(["C0", "C1", "C2"]))
    plt.xlim([np.min(X[:, 0]) - 0.1, np.max(X[:, 0]) + 0.1])
    plt.ylim([np.min(X[:, 1]) - 0.1, np.max(X[:, 1]) + 0.1])
    # Plot original data
    plt.scatter(X[y==0, 0], X[y==0, 1], marker="s", edgecolor="white", label="setosa")
    plt.scatter(X[y==1, 0], X[y==1, 1], marker="o", edgecolor="white", label="versicolour")
    plt.scatter(X[y==2, 0], X[y==2, 1], marker="D", edgecolor="white", label="virginica")
    plt.xlabel("Petal length (cm)")
    plt.ylabel("Petal width (cm)")
    plt.legend()
    plt.show()
    
def print_decisions(model,features,class_names):
    root = model.root
    regions = [root]
    levels = [0]
    false_children = [False]
    while len(regions)>0:
        region = regions[-1]
        level = levels[-1]
        levels.pop()
        if false_children[-1]:
            print("|   " * (level-1) + "else:")
        false_children.pop()
        if region.leaf_node:
            print("|   " * level + "{} (N={})".format(class_names[region.prediction()],region.y.size))
        else:
            print("|   " * level + "if {} < {:.2f}:".format(features[region.j],region.s))
            regions.append(region.false_child)
            false_children.append(True)
            levels.append(level+1)
            regions.append(region.true_child)
            false_children.append(False)
            levels.append(level+1)
        regions.remove(region)
