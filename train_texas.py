from utils.web_kb import texas_data, cornell_data, wisconsin_data
from torch_geometric.utils import to_dense_adj
from network_analysis import get_shortest_path_matrix_tensor, get_shortest_path_matrix, run_analysis
from GNNModel import SparseGraphTransformerModel, DenseGraphTransformerModel, sparse_training_loop, dense_training_loop
from utils.set_seed import set_seed
import torch
import os
import numpy as np



def set_masks(data, split):
    data.train_mask = data.train_mask[:, split]
    data.val_mask = data.val_mask[:, split]
    data.test_mask = data.test_mask[:, split]

def train_texas(split, sparse=False, threshold_value=0.1):
    data = texas_data()
    data.dense_adj = to_dense_adj(data.edge_index, max_num_nodes=data.x.shape[0])[0]
    data.dense_sp_matrix = get_shortest_path_matrix_tensor(data.dense_adj)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    adjacency_matrix = data.dense_adj.cpu().numpy()
    set_masks(data, split)

    if sparse:
        model = SparseGraphTransformerModel(data = data).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        sparse_training_loop(data=data, model=model, optimizer=optimizer)
    else:
        model = DenseGraphTransformerModel(data = data).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        dense_training_loop(data=data, model=model, optimizer=optimizer)

    # run_analysis(adjacency_matrix=adjacency_matrix, model=model, threshold_value=threshold_value, shortest_paths=True, title="Texas", load_save=False)

    return model


def train_cornell(split, sparse=False, threshold_value=0.1):
    data = cornell_data()
    data.dense_adj = to_dense_adj(data.edge_index, max_num_nodes=data.x.shape[0])[0]
    data.dense_sp_matrix = get_shortest_path_matrix_tensor(data.dense_adj)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    adjacency_matrix = data.dense_adj.cpu().numpy()
    set_masks(data, split)

    if sparse:
        model = SparseGraphTransformerModel(data = data).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        sparse_training_loop(data=data, model=model, optimizer=optimizer)
    else:
        model = DenseGraphTransformerModel(data = data).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        dense_training_loop(data=data, model=model, optimizer=optimizer)

    # run_analysis(adjacency_matrix=adjacency_matrix, model=model, threshold_value=threshold_value, shortest_paths=True, title="Cornell", load_save=False)

    return model

def train_wisconsin(split, sparse=False, threshold_value=0.1):
    data = wisconsin_data()
    data.dense_adj = to_dense_adj(data.edge_index, max_num_nodes=data.x.shape[0])[0]
    data.dense_sp_matrix = get_shortest_path_matrix_tensor(data.dense_adj)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    adjacency_matrix = data.dense_adj.cpu().numpy()
    set_masks(data, split)

    if sparse:
        model = SparseGraphTransformerModel(data = data).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        sparse_training_loop(data=data, model=model, optimizer=optimizer)
    else:
        model = DenseGraphTransformerModel(data = data).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        dense_training_loop(data=data, model=model, optimizer=optimizer)

    # run_analysis(adjacency_matrix=adjacency_matrix, model=model, threshold_value=threshold_value, shortest_paths=True, title="Wisconsin", load_save=False)

    return model



def save_attention_matrix(matrix, directory, title):
    """
    Given an attention matrix, save it to a directory.
    If the directory does not exist, create it.
    If the filename already exists, append a number to the filename.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = title + ".npy"
    if os.path.exists(os.path.join(directory, filename)):
        i = 1
        while os.path.exists(os.path.join(directory, title + str(i) + ".npy")):
            i += 1
        filename = title + str(i) + ".npy"
    np.save(os.path.join(directory, filename), matrix)
    print(f"Matrix saved as {filename}.")

def train_with_different_seeds(threshold_value=0.01, threshold_string="001"):
    """
    Train the model with different seeds and save the thresholded attention matrices
    to a corresponding directory.
    """
    seeds = [1, 2, 3, 4, 5]
    for seed in seeds:
        set_seed(seed)
        model = train_texas(5, sparse=False, threshold_value=threshold_value)
        attention_matrix = model.attn_weights_list[0].detach().cpu().detach().numpy().squeeze()
        # threshold the attention matrix
        thresholded_attention_matrix = np.where(attention_matrix > threshold_value, 1, 0)
        save_attention_matrix(thresholded_attention_matrix, F"texas_attention_matrices_{threshold_string}", "texas_attention_matrix_seed_" + str(seed))


def train_with_different_seeds_cornell(threshold_value=0.01, threshold_string="001"):
    """
    Train the model with different seeds and save the thresholded attention matrices
    to a corresponding directory.
    """
    seeds = [1, 2, 3, 4, 5]
    for seed in seeds:
        set_seed(seed)
        model = train_cornell(5, sparse=False, threshold_value=0.01)
        attention_matrix = model.attn_weights_list[0].detach().cpu().detach().numpy().squeeze()
        # threshold the attention matrix
        thresholded_attention_matrix = np.where(attention_matrix > threshold_value, 1, 0)
        save_attention_matrix(thresholded_attention_matrix, f"cornell_attention_matrices_{threshold_string}", "cornell_attention_matrix_seed_" + str(seed))

def train_with_different_seeds_wisconsin(threshold_value=0.01, threshold_string="001"):
    """
    Train the model with different seeds and save the thresholded attention matrices
    to a corresponding directory.
    """
    seeds = [1, 2, 3, 4, 5]
    for seed in seeds:
        set_seed(seed)
        model = train_wisconsin(5, sparse=False, threshold_value=threshold_value)
        attention_matrix = model.attn_weights_list[0].detach().cpu().detach().numpy().squeeze()
        # threshold the attention matrix
        thresholded_attention_matrix = np.where(attention_matrix > threshold_value, 1, 0)
        save_attention_matrix(thresholded_attention_matrix, f"wisconsin_attention_matrices_{threshold_string}", "wisconsin_attention_matrix_seed_" + str(seed))


# train_texas(5, sparse=False, threshold_value=0.01)
# train_cornell(5, sparse=False, threshold_value=0)
if __name__ == "__main__":
    train_with_different_seeds_wisconsin(threshold_value=0.01, threshold_string="001")