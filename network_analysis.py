
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import time
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.datasets import Planetoid
from GNNModel import GNNModel, SparseGraphTransformerModel, DenseGraphTransformerModel, train_sparse, test_sparse, sparse_training_loop, dense_training_loop
from dgl.data import RomanEmpireDataset
from roman_empire import preprocess_roman_empire
from utils.web_kb import texas_data, cornell_data
from utils.save_matrix import save_matrix
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj

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


def get_shortest_path_matrix(adjacency_matrix : np.ndarray) -> torch.Tensor:
    """
    Returns the shortest path matrix of a graph.
    Args:
    adjacency_matrix: np.ndarray: The adjacency matrix of the graph
    Returns:
    np.ndarray: The shortest path matrix
    """
    graph = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph())
    return torch.tensor(nx.floyd_warshall_numpy(graph)).float()

def get_shortest_path_matrix_tensor(adjacency_matrix : torch.Tensor) -> torch.Tensor:
    """
    Returns the shortest path matrix of a graph.
    Args:
    adjacency_matrix: np.ndarray: The adjacency matrix of the graph
    Returns:
    np.ndarray: The shortest path matrix
    """
    return get_shortest_path_matrix(adjacency_matrix.cpu().numpy())
    # graph = nx.from_numpy_array(adjacency_matrix.cpu().numpy(), create_using=nx.DiGraph())
    # return torch.tensor(nx.floyd_warshall_numpy(graph)).float()

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

def get_normalized_laplacian_matrix(adjacency_matrix) -> np.ndarray:
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

def graph_edit_distance(adjacency : np.ndarray, attention : np.ndarray) -> float:
    """
    Computes the graph edit distance between two graphs.
    Args:
    adjacency1: np.ndarray: The adjacency matrix of the first graph
    adjacency2: np.ndarray: The adjacency matrix of the second graph
    Returns:
    float: The graph edit distance between the two graphs
    """
    graph1 = nx.from_numpy_array(adjacency)
    graph2 = nx.from_numpy_array(attention)
    graph_edit_distance = np.min([x for x in nx.optimize_graph_edit_distance(graph1, graph2)])
    with open('graph_edit_distance_2.txt', 'w') as file:
        file.write(f'Graph edit distance: {graph_edit_distance}')
    return graph_edit_distance

def run_analysis(adjacency_matrix, model, threshold_value=0.1, title="Cora", shortest_paths=True, load_save=False):
    """
    Runs the analysis on the given adjacency matrix and attention matrix of the model
    Args:
    adjacency_matrix: np.ndarray: The adjacency matrix of the graph
    model: The model to analyse
    """
    attention_matrix = model.attn_weights_list[0].detach().cpu().numpy().squeeze()

    # Threshold the attention matrix
    attention_matrix = threshold(attention_matrix, threshold_value)
    np.save(f'{title}_attention_matrix.npy', attention_matrix)
    # save_matrix(attention_matrix, f'{title}_attention_matrix')
    # print("Attention matrix saved")

    # Plot the attention matrix
    # plot_heatmap(attention_matrix, f'{title} Attention Matrix')
    # plot_heatmap(adjacency_matrix, f'{title} Adjacency Matrix')

    # print("Heatmaps saved")

    # Get the degree distributions
    # adjacency_degree_distribution = get_degree_distribution_table(adjacency_matrix, f'{title} Adjacency Degree Distribution')
    # print(adjacency_degree_distribution)
    # attention_degree_distribution = get_degree_distribution_table(attention_matrix, f'{title} Attention Degree Distribution')
    # print(attention_degree_distribution)

    # print("Degree distributions saved")

    # # Get the shortest path matrices
    # if shortest_paths:
    #     if load_save:
    #         adjacency_shortest_path_matrix = torch.load(f'roman_shortest_path_matrix.pt')
    #         attention_shortest_path_matrix = get_shortest_path_matrix(attention_matrix)
    #     else:
    #         adjacency_shortest_path_matrix = get_shortest_path_matrix(adjacency_matrix)
    #         attention_shortest_path_matrix = get_shortest_path_matrix(attention_matrix)
    #     plot_heatmap(adjacency_shortest_path_matrix, f'{title} Adjacency Shortest Path Matrix')
    #     plot_heatmap(attention_shortest_path_matrix, f'{title} Attention Shortest Path Matrix')

    # # Get the commute times
    # adjacency_commute_times = compute_commute_times(adjacency_matrix)
    # attention_commute_times = compute_commute_times(attention_matrix)
    # plot_heatmap(adjacency_commute_times, f'{title} Adjacency Commute Times')
    # plot_heatmap(attention_commute_times, f'{title} Attention Commute Times')

    ged = graph_edit_distance(adjacency_matrix, attention_matrix)
    print(f'Graph edit distance: {ged}')

    # # Get the number of triangles
    # adjacency_triangles = triangle_count(adjacency_matrix)
    # attention_triangles = triangle_count(attention_matrix)
    # # Save the values to a text file
    # with open('triangles.txt', 'w') as file:
    #     file.write(f'{title} Adjacency triangles: {adjacency_triangles}\n')
    #     file.write(f'{title} Attention triangles: {attention_triangles}\n')
    # print(f'{title} Adjacency triangles: {adjacency_triangles}')
    # print(f'{title} Attention triangles: {attention_triangles}')

def run_thresholded_attention_analysis(thresholded_attention_matrix, title):
    """
    Runs the analysis on the given thresholded attention matrix
    and saves the results to the given directory.
    Args:
    thresholded_attention_matrix: np.ndarray: The thresholded attention matrix
    """
    # Plot the attention matrix
    plot_heatmap(thresholded_attention_matrix, f'{title} Attention Matrix')

    # Get the degree distributions
    attention_degree_distribution = get_degree_distribution_table(thresholded_attention_matrix, f'{title} Attention Degree Distribution')
    print(attention_degree_distribution)

    print("Degree distributions saved")

    # Get the shortest path matrices
    attention_shortest_path_matrix = get_shortest_path_matrix(thresholded_attention_matrix)
    plot_heatmap(attention_shortest_path_matrix, f'{title} Attention Shortest Path Matrix')

    # Get the commute times
    attention_commute_times = compute_commute_times(thresholded_attention_matrix)
    plot_heatmap(attention_commute_times, f'{title} Attention Commute Times')

    ged = graph_edit_distance(adjacency_matrix, thresholded_attention_matrix)
    print(f'Graph edit distance: {ged}')

    # Get the number of triangles
    attention_triangles = triangle_count(thresholded_attention_matrix)
    # Save the values to a text file
    with open('triangles.txt', 'w') as file:
        file.write(f'{title} Attention triangles: {attention_triangles}\n')
    print(f'{title} Attention triangles: {attention_triangles}')


if __name__ == "__main__":
    # dataset = preprocess_roman_empire()
    dataset = 'Cora'
    dataset = Planetoid('/tmp/Cora', dataset)
    data = dataset[0]

    # data = preprocess_roman_empire()
    # print(data)
    # data.dense_adj = to_dense_adj(data.edge_index, max_num_nodes=data.x.shape[0])[0]
    # data.dense_sp_matrix = get_shortest_path_matrix_tensor(data.dense_adj).float()
    # adjacency_matrix = nx.to_numpy_array(nx.from_edgelist(data.edge_index.T.tolist()))
    # adjacency_matrix = data.dense_adj.cpu().numpy()
    # np.save('Cora_adjacency_matrix.npy', adjacency_matrix)
    # print("saving adjacency matrix")
    adjacency_matrix = np.load('Cora_adjacency_matrix.npy')

    # # data = texas_data()
    # # data.dense_adj = to_dense_adj(data.edge_index, max_num_nodes=data.x.shape[0])[0]
    # # data.train_mask = data.train_mask[:, 5]
    # # data.val_mask = data.val_mask[:, 5]
    # # data.test_mask = data.test_mask[:, 5]
    # # print(data.train_mask)
    # # data.dense_sp_matrix = get_shortest_path_matrix_tensor(data.dense_adj).float()
    # # adjacency_matrix = data.dense_adj.cpu().numpy()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # data = data.to(device)
    # model = DenseGraphTransformerModel(data=data).to(device)
    # # model = SparseGraphTransformerModel(data=data).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # sparse_training_loop(data=data, model=model, optimizer=optimizer)
    # dense_training_loop(data=data, model=model, optimizer=optimizer)
    # attention_weights = model.attn_weights_list[0].detach().cpu().numpy().squeeze()

    # graph_edit_distance(adjacency_matrix, attention_weights)
    # run_analysis(adjacency_matrix, model, title="Cora", threshold_value=0.01, shortest_paths=True, load_save=False)

    # run_analysis(adjacency_matrix, model, shortest_paths=True, title="Roman Empire")
    # run_analysis(adjacency_matrix=adjacency_matrix, model=model, shortest_paths=True, title="Texas", load_save=False)

    cora_attention_weights = np.load('Cora_attention_matrix.npy')
    print("Attention matrix loaded")
    ged = graph_edit_distance(adjacency_matrix, cora_attention_weights)
    print(f'Graph edit distance: {ged}')
