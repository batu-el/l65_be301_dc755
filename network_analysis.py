
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

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

def get_degree_matrix(adjacency_matrix : np.ndarray) -> np.ndarray:
    """
    Returns the degree matrix of a graph.
    Args:
    adjacency_matrix: np.ndarray: The adjacency matrix of the graph
    Returns:
    np.ndarray: The degree matrix
    """
    return np.diag(np.sum(adjacency_matrix, axis=1))

def get_degree_distribution_table(adjacency_matrix : np.ndarray, title : str = None) -> pd.DataFrame:
    """
    Returns the degree distribution of a graph.
    Args:
    adjacency_matrix: np.ndarray: The adjacency matrix of the graph
    Returns:
    pd.DataFrame: The degree distribution
    """
    degree_distribution = np.sum(adjacency_matrix, axis=1)
    degree_distribution = pd.Series(degree_distribution).value_counts().sort_index()
    degree_distribution = degree_distribution.reset_index()
    degree_distribution.columns = ['Degree', 'Count']
    if title:
        degree_distribution = degree_distribution.set_index('Degree')
        degree_distribution.index.name = title

        safe_title = title.replace(' ', '_').replace(':', '').replace('/', '').replace('\\', '')
        filename = f"{safe_title}.csv"

        counter = 1
        while os.path.isfile(filename):
            filename = f"{safe_title}_{counter}.csv"
            counter += 1

        degree_distribution.to_csv(filename)
        print(f"Degree distribution saved to {filename}")
    return degree_distribution



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
    safe_title = title.replace(' ', '_').replace(':', '').replace('/', '').replace('\\', '')

    counter = 1
    filename = f'{safe_title}.png'
    while os.path.isfile(filename):
        filename = f'{safe_title}_{counter}.png'
        counter += 1

    plt.savefig(filename)
    plt.close()

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

def run_analysis(adjacency_matrix, model):
    """
    Runs the analysis on the given adjacency matrix and attention matrix of the model
    Args:
    adjacency_matrix: np.ndarray: The adjacency matrix of the graph
    model: The model to analyse
    """
    attention_matrix = model.attn_weights_list[0].detach().cpu().numpy().squeeze()
    return

if __name__ == "__main__":
    # Create a random graph
    graph = nx.erdos_renyi_graph(100, 0.1)
    adjacency = nx.to_numpy_array(graph)
    # Plot the graph
    nx.draw(graph, with_labels=True)
    plt.show()
    # Plot the adjacency matrix
    plot_heatmap(adjacency, 'Adjacency Matrix')
    # Get the degree distribution
    degree_distribution = get_degree_distribution_table(adjacency, 'Degree Distribution')
    print(degree_distribution)