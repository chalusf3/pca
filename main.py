import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def generate_data(n_points, dimension, n_dof):
    X = np.random.normal(size = (n_points, n_dof))
    A = np.random.uniform(low = -1, high = 1, size = (n_dof, dimension))
    return np.dot(X, A) + np.random.normal(loc = np.random.uniform(dimension), scale = 0.2, size = (n_points, dimension))

def compute_main_axis(data, n_axis):
    eigenvalues, eigenvectors = np.linalg.eig(np.dot(data.T, data) / data.shape[0])
    sort_index = sorted(np.arange(eigenvalues.shape[0]), key = lambda i: eigenvalues[i], reverse = True)
    sort_index = sort_index[:n_axis]
    return eigenvectors[:, sort_index], eigenvalues[sort_index]

def main2d():
    data = generate_data(50, 2, 2)
    mean_vector = np.mean(data, axis = 0)
    data_zero_mean = data - mean_vector
    axis, eig_val = compute_main_axis(data_zero_mean, 1)

    proj_matrix = np.dot(axis, axis.T)
    proj_data = np.dot(data_zero_mean, proj_matrix)

    plt.figure()
    plt.scatter(data_zero_mean[:, 0], data_zero_mean[:, 1], c = 'b')
    plt.scatter(proj_data[:, 0], proj_data[:, 1], c = 'r')

    for i in range(data.shape[0]):
        plt.plot([proj_data[i, 0], data_zero_mean[i, 0]], \
            [proj_data[i, 1], data_zero_mean[i, 1]], \
            c = 'b', linewidth = 1, linestyle = '--')

    for i in range(eig_val.shape[0]):
        v = axis[:, i] * eig_val[i]
        plt.plot([0, v[0]], [0, v[1]], c = 'g', linewidth = 1)

    plt.show()

def main3d():
    data = generate_data(20, 3, 2)
    mean_vector = np.mean(data, axis = 0)
    data_zero_mean = data - mean_vector
    axis, eig_val = compute_main_axis(data_zero_mean, 2)
    
    proj_matrix = np.dot(axis, axis.T)
    proj_data = np.dot(data, proj_matrix)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    # plot original points (zero-meaned)
    ax.scatter(data_zero_mean[:, 0], data_zero_mean[:, 1], data_zero_mean[:, 2], c = 'b', alpha = 0.1)
    # plot projections
    ax.scatter(proj_data[:, 0], proj_data[:, 1], proj_data[:, 2], c = 'r', alpha = 0.1)
    # plot lines between points and projections
    for i in range(data.shape[0]):
        ax.plot([proj_data[i, 0], data_zero_mean[i, 0]], \
            [proj_data[i, 1], data_zero_mean[i, 1]], \
            c = 'b', linewidth = 1, linestyle = '--')
    # plot axes
    for i in range(2):
        v = axis[:, i] * eig_val[i] * 2
        ax.plot([0, v[0]], [0, v[1]], [0, v[2]], c = 'r', linewidth = 1)

    plt.show()

if __name__ == '__main__':
    main2d()

