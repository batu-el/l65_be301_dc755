from utils.web_kb import texas_data, cornell_data
from torch_geometric.utils import to_dense_adj
from network_analysis import get_shortest_path_matrix_tensor, get_shortest_path_matrix, run_analysis
from GNNModel import SparseGraphTransformerModel, DenseGraphTransformerModel, sparse_training_loop, dense_training_loop
from utils.set_seed import seed
import torch



def set_masks(data, split):
    data.train_mask = data.train_mask[:, split]
    data.val_mask = data.val_mask[:, split]
    data.test_mask = data.test_mask[:, split]

def train_texas(split, sparse=False, threshold_value=0.1):
    seed(1)
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

    run_analysis(adjacency_matrix=adjacency_matrix, model=model, threshold_value=threshold_value, shortest_paths=True, title="Texas", load_save=False)

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

    run_analysis(adjacency_matrix=adjacency_matrix, model=model, threshold_value=threshold_value, shortest_paths=True, title="Cornell", load_save=False)
    
train_texas(5, sparse=False, threshold_value=0.01)
# train_cornell(5, sparse=False, threshold_value=0)