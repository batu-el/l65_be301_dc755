import torch
from torch_geometric.data import Data
import numpy as np

def preprocess(dataset, train_ratio = 0.7, val_ratio = 0.15, test_ratio = 0.15):
    g = dataset[0]
    y = g.ndata['label']
    feat = g.ndata['feat']

    num_nodes = len(y)
    indices = torch.randperm(num_nodes)

    num_train, num_val = int(num_nodes * train_ratio), int(num_nodes * val_ratio)
    num_test = num_nodes - num_train - num_val

    train_mask, val_mask, test_mask = torch.zeros(num_nodes, dtype=torch.bool), torch.zeros(num_nodes, dtype=torch.bool), torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[indices[:num_train]] = True
    val_mask[indices[num_train:num_train+num_val]] = True
    test_mask[indices[num_train+num_val:]] = True

    # Convert NetworkX graph to edge list
    src, dst = g.edges()
    edge_list = list(zip(src.tolist(), dst.tolist()))
    # Create a set for symmetric edges to avoid duplicates
    symmetric_edges = set()

    # Add each edge and its reverse to the set
    for u, v in edge_list:
        symmetric_edges.add((u, v))
        symmetric_edges.add((v, u))
    edge_list = list(symmetric_edges)

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    node_features = feat #[g.nodes[node]['feat'] for node in G.nodes()]
    # Create a Data object
    if len(np.array(node_features).shape) == 1:
      data = Data(x=torch.tensor(np.array(node_features)).unsqueeze(1), edge_index=torch.tensor(np.array(edge_index)), y=torch.tensor(np.array(y)), train_mask=torch.tensor(np.array(train_mask)), val_mask=torch.tensor(np.array(val_mask)), test_mask=torch.tensor(np.array(test_mask)))
    else:
      data = Data(x=torch.tensor(np.array(node_features, dtype=float)).float(), edge_index=torch.tensor(np.array(edge_index)), y=torch.tensor(np.array(y)), train_mask=torch.tensor(np.array(train_mask)), val_mask=torch.tensor(np.array(val_mask)), test_mask=torch.tensor(np.array(test_mask)))
    return data