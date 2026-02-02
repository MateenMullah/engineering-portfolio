import numpy as np
import matplotlib.pyplot as plt

def calculate_w_1(x, y):

    n = len(x)

    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x_y = np.sum(x_i * y_i for x_i, y_i in zip(x, y))
    sum_x_squared = np.sum(x_i**2 for x_i in x)

    num = n * sum_x_y - sum_x * sum_y
    denom = n * sum_x_squared - sum_x**2
    w_1 = num / denom

    return w_1

def calculate_w_0(x, y):

    w_1 = calculate_w_1(x, y)
    y_bar = np.mean(y)
    w_0 = y_bar - w_1 * np.mean(x)

    return w_0

def plot_linear_fit(x, y, w_0_fit, w_1_fit):
    x_min = np.min(x)
    x_max = np.max(x)
    
    y_min = w_0_fit + w_1_fit * x_min
    y_max = w_0_fit + w_1_fit * x_max
    
    plt.figure(figsize=(8,6))
    plt.scatter(x, y, color='blue', label='Data points')
    
    plt.plot([x_min, x_max], [y_min, y_max], color='black', label='Linear fit')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Linear Regression Graph')
    plt.legend()
    
    plt.show()
   