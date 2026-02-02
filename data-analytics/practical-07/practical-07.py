# Imports
import numpy as np
import matplotlib.pyplot as plt
# Do not edit anything above this line
# Any imports below this line will be ignored

class TreeNode:
    '''
    Support class for the 'RegressionTree' class to represent regions
    Contains references to its children if not a leaf node

    '''

    def __init__(self, X, y):
        '''
        Store the samples that are in this region and set this region as a leaf node

        Parameters
        ----------
        X : (N,D) ndarray
            design matrix -- N samples with D features
        y : (N,) ndarray
            output vector
        '''

        self.X = X
        self.y = y
        self.leaf_node = True

    def value(self):
        '''
        Calculate the predicted value for this region

        Return
        ------
        c : float
            predicted value
        '''

        # insert code here
        if len(self.y) == 0:
            return 0
        return self.y.mean()

    def squared_loss(self):
        '''
        Calculate the squared loss for all the samples in this region

        Return
        ------
        J : float
            squared loss
        '''


        c = self.value()
        J = ((self.y - c) ** 2).sum()
        return J

    def squared_loss_if_split(self, j, s):
        '''
        Calculate the squared loss for this node if it were to be split at x_j = s
        Do not create any new 'TreeNode' objects

        Parameters
        ----------
        j : int
            dimension (starting from 0)
        s : float
            split point

        Return
        ------
        J : float
            squared loss
        '''

        # insert code here
        left_mask = self.X[:, j] <= s
        right_mask = ~left_mask

        y_left = self.y[left_mask]
        y_right = self.y[right_mask]

        if len(y_left) > 0:
            c_left = y_left.mean()
            loss_left = ((y_left - c_left) ** 2).sum()
        else:
            loss_left = 0

        if len(y_right) > 0:
            c_right = y_right.mean()
            loss_right = ((y_right - c_right) ** 2).sum()
        else:
            loss_right = 0

        J = loss_left + loss_right
        return J

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
        true_mask = self.X[:, j] < s
        false_mask = ~true_mask

        X_true, y_true = self.X[true_mask], self.y[true_mask]
        X_false, y_false = self.X[false_mask], self.y[false_mask]

        self.true_child = TreeNode(X_true, y_true)
        self.false_child = TreeNode(X_false, y_false)

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
        Assume that the input features are all integers

        Parameters
        ----------
        j : int
            dimension (starting from 0)

        Return
        ------
        split_points : (K,) ndarray
            split points
        '''

        split_points = np.arange(min(self.X[:,j]),max(self.X[:,j]))+0.5

        return split_points

class RegressionTree:
    '''
    Perform regression using trees

    '''

    def __init__(self, max_leaf_nodes):
        '''
        Store the maximum number of leaf nodes used in the 'fit' member function

        Parameters
        ----------
        max_leaf_nodes : float
            maximum number of leaf nodes
        '''

        self.max_leaf_nodes = max_leaf_nodes

    def fit(self, X, y):
        '''
        Fit the regression tree using the tree growing algorithm
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

        self.root = TreeNode(X,y)
        leaf_nodes = [self.root]

        while len(leaf_nodes) < self.max_leaf_nodes:
            best_loss = None
            best_node = None
            best_j = None
            best_s = None

            for node in leaf_nodes:
                X_node = node.X
                for j in range(X_node.shape[1]):
                    possible_splits = np.unique(X_node[:, j])
                    for s in possible_splits:
                        loss = node.squared_loss_if_split(j, s)
                        if best_loss is None or loss < best_loss:
                            best_loss = loss
                            best_node = node
                            best_j = j
                            best_s = s

            if best_node is None:
                break

            best_node.split(best_j, best_s)
            leaf_nodes.remove(best_node)
            leaf_nodes.append(best_node.true_child)
            leaf_nodes.append(best_node.false_child)

    def predict(self, X_new):
        '''
        Calculate the predicted outputs for the regression tree
        Use the member functions and attributes of the 'TreeNode' class

        Parameters
        ----------
        X_new : (N_new,D) ndarray
            test input matrix -- N_new samples with D features

        Return
        ------s
        y_new : (N_new,) ndarray
            predicted output vector
        '''

        # insert code here
        y_new = np.zeros(X_new.shape[0])
        for i in range(X_new.shape[0]):
            node = self.root
            while hasattr(node, 'true_child'):
                node = node.traverse(X_new[i])
            y_new[i] = node.value()
        return y_new

    def overall_loss(self, X, y):
        '''
        Calculate the overall squared loss for the regression tree
        Use the 'predict' member function

        Parameters
        ----------
        X : (N,D) ndarray
            design matrix -- N samples with D features
        y : (N,) ndarray
            output vector

        Return
        ------
        J : float
            squared loss
        '''

        # insert code here
        y_pred = self.predict(X)
        return ((y - y_pred) ** 2).sum()

# The functions below this point are already completed
# They are used in the notebook for visualizations

def plot_regression_3d(x_1,x_2,model):
    # Grid of values
    x_1_range = [min(x_1) - 1, max(x_1)]
    x_2_range = [min(x_2), max(x_2)]
    N = 100
    x_1_grid, x_2_grid = np.meshgrid(
        np.linspace(x_1_range[0], x_1_range[1], N),
        np.linspace(x_2_range[0], x_2_range[1], N)
    )
    y_hat = []
    for x_1_tmp, x_2_tmp in zip(x_1_grid.ravel(), x_2_grid.ravel()):
        y_hat.append(model.predict(np.array([[x_1_tmp, x_2_tmp]])))
    y_hat = np.array(y_hat).reshape(x_1_grid.shape)
    # Plot fit
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection="3d")
    ax.view_init(30, -130)
    ax.plot_surface(x_1_grid, x_2_grid, y_hat, cmap="viridis", edgecolor="none", alpha=0.7)
    ax.set_xlim3d(0, max(x_1))
    ax.set_ylim3d(0, max(x_2))
    ax.set_zlim3d(0, 1500)
    ax.set_zlabel("Salary ($1000s)")
    ax.set_xlabel("Years")
    ax.set_ylabel("Hits")
    plt.show()

def plot_regression_2d(df,model):
    X = df[["Years", "Hits"]]
    y = df.Salary
    y_hat = model.predict(X.values)
    fig = plt.figure(figsize=(6,10))
    # Plot original data
    ax1 = fig.add_subplot(2, 1, 1)
    df.plot("Years", "Hits", kind="scatter", ax=ax1, s=df["Salary"]*0.05)
    ax1.set_xlim(0, 25)
    ax1.set_ylim(-6, 250)
    ax1.set_title("Actual outputs")
    # Plot predictions
    ax2 = fig.add_subplot(2, 1, 2)
    df.plot("Years", "Hits", kind="scatter", ax=ax2, s=y_hat*0.05)
    ax2.set_xlim(0, 25)
    ax2.set_ylim(-6, 250)
    ax2.set_title("Predicted outputs")
    # Plot boundaries
    root = model.root
    regions = [root]
    while len(regions)>0:
        region = regions[0]
        if not region.leaf_node:
            if region.j==0:
                ax1.vlines(region.s, ymin=min(region.X[:,1]), ymax=max(region.X[:,1]), linestyle="dashed", color="k")
                ax2.vlines(region.s, ymin=min(region.X[:,1]), ymax=max(region.X[:,1]), linestyle="dashed", color="k")
            else:
                ax1.hlines(region.s, xmin=min(region.X[:,0]), xmax=max(region.X[:,0]), linestyle="dashed", color="k")
                ax2.hlines(region.s, xmin=min(region.X[:,0]), xmax=max(region.X[:,0]), linestyle="dashed", color="k")
            regions.append(region.true_child)
            regions.append(region.false_child)
        regions.remove(region)
    plt.show()

def print_decisions(model,features):
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
            print("|   " * level + "{:.2f} (N={})".format(region.value(),region.y.size))
        else:
            print("|   " * level + "if {} < {}:".format(features[region.j],region.s))
            regions.append(region.false_child)
            false_children.append(True)
            levels.append(level+1)
            regions.append(region.true_child)
            false_children.append(False)
            levels.append(level+1)
        regions.remove(region)
