from dgl.data import RomanEmpireDataset
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
import networkx as nx
import numpy as np

def preprocess(graph, train_ratio = 0.7, val_ratio = 0.15, test_ratio = 0.15):
    x = graph.ndata['feat']
    y = graph.ndata['label']
    edge_index = torch.stack([graph.edges()[0], graph.edges()[1]])

    num_nodes = len(y)
    indices = torch.randperm(num_nodes)
    num_train, num_val = int(num_nodes * train_ratio), int(num_nodes * val_ratio)
    num_test = num_nodes - num_train - num_val

    train_mask, val_mask, test_mask = torch.zeros(num_nodes, dtype=torch.bool), torch.zeros(num_nodes, dtype=torch.bool), torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[indices[:num_train]] = True
    val_mask[indices[num_train:num_train+num_val]] = True
    test_mask[indices[num_train+num_val:]] = True
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    return data

print(RomanEmpireDataset)

def preprocess_roman_empire():
    dataset = RomanEmpireDataset()[0]
    return preprocess(graph=dataset)

print(preprocess_roman_empire())

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

dataset = preprocess_roman_empire()
dense_adj = to_dense_adj(dataset.edge_index, max_num_nodes=dataset.x.shape[0])[0]
# torch.save(get_shortest_path_matrix_tensor(dense_adj), 'roman_shortest_path_matrix.pt')

# load = torch.load('roman_shortest_path_matrix.pt')
loaded_attention = np.load('Roman Empire_attention_matrix.npy')
print("saving shortest path attention matrix")
torch.save(get_shortest_path_matrix(loaded_attention), 'roman_attention_shortest_path_matrix.pt')
print(loaded_attention)

