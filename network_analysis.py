
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.linalg import pinv

def threshold(matrix : np.ndarray, threshold : float) -> np.ndarray:
    """
    Thresholds a matrix by setting all values below a threshold to 0 and all values above to 1.
    Args:
    matrix: np.ndarray: The matrix to threshold
    threshold: float: The threshold value
    Returns:
    np.ndarray: The thresholded matrix
    """
    return (matrix > threshold).astype(int)


def get_shortest_path_matrix(adjacency_matrix : np.ndarray) -> np.ndarray:
    """
    Returns the shortest path matrix of a graph.
    Args:
    adjacency_matrix: np.ndarray: The adjacency matrix of the graph
    Returns:
    np.ndarray: The shortest path matrix
    """
    graph = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph())
    return nx.floyd_warshall_numpy(graph)

def plot_heatmap(matrix : np.ndarray, title : str):
    """
    Plots a heatmap of a matrix.
    Args:
    matrix: np.ndarray: The matrix to plot
    title: str: The title of the plot
    """
    plt.figure(figsize=(10, 10))
    sns.heatmap(matrix, cmap='viridis')
    plt.title(title)
    plt.show()

def get_laplacian_matrix(adjacency_matrix : np.ndarray) -> np.ndarray:
    """
    Returns the Laplacian matrix of a graph.
    Args:
    adjacency_matrix: np.ndarray: The adjacency matrix of the graph
    Returns:
    np.ndarray: The Laplacian matrix
    """
    return nx.laplacian_matrix(nx.from_numpy_array(adjacency_matrix)).toarray()

def get_normalized_laplacian_matrix(adjacency_matrix : np.ndarray) -> np.ndarray:
    """
    Returns the normalized Laplacian matrix of a graph.
    Args:
    adjacency_matrix: np.ndarray: The adjacency matrix of the graph
    Returns:
    np.ndarray: The normalized Laplacian matrix
    """
    graph = nx.from_numpy_array(adjacency_matrix)
    return nx.normalized_laplacian_matrix(graph).toarray()

def get_random_walk_matrix(adjacency_matrix : np.ndarray) -> np.ndarray:
    """
    Returns the random walk matrix of a graph.
    Args:
    adjacency_matrix: np.ndarray: The adjacency matrix of the graph
    Returns:
    np.ndarray: The random walk matrix
    """
    graph = nx.from_numpy_array(adjacency_matrix)
    return nx.normalized_laplacian_matrix(graph).toarray()

def compute_commute_times(adjacency : np.ndarray) -> np.ndarray:
    """
    Computes the commute times of a graph.
    Args:
    matrix: np.ndarray: The matrix to compute the commute times of
    Returns:
    np.ndarray: The commute times
    """
    n = adjacency.shape[0]
    laplacian = get_laplacian_matrix(adjacency)
    pinv_laplacian = pinv(laplacian)

    commute_times = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            commute_times[i, j] = pinv_laplacian[i, i] + pinv_laplacian[j, j] - 2 * pinv_laplacian[i, j]

    return commute_times

def triangle_count(adjacency : np.ndarray) -> int:
    """
    Counts the number of triangles in a graph.
    Args:
    adjacency: np.ndarray: The adjacency matrix of the graph
    Returns:
    int: The number of triangles in the graph
    """
    graph = nx.from_numpy_array(adjacency)
    return sum(nx.triangles(graph).values()) // 3